//! Model types: `BonsaiModel` struct, constructors, accessors, and main forward pass.

use super::weight_loaders::{load_f32_tensor, load_output_weight, load_transformer_block};
use crate::block::TransformerBlock;
use crate::error::{ModelError, ModelResult};
use crate::kv_cache::KvCache;
use crate::layers::linear::{Linear1Bit, LinearTernary};
use crate::layers::rms_norm::RmsNorm;
use crate::layers::rope::RopeTable;
use crate::model_registry::ModelVariant;
use oxibonsai_core::config::Qwen3Config;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::tensor_info::tensor_names;
use oxibonsai_kernels::traits::OneBitKernel;

#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
mod forward_cuda;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod forward_metal;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod gpu_cache;

/// The complete Bonsai-8B model (Qwen3 architecture) with loaded weights.
///
/// Lifetime `'a` is tied to the memory-mapped GGUF data.
pub struct BonsaiModel<'a> {
    config: Qwen3Config,
    /// Token embedding table: [vocab_size × hidden_size] as FP32.
    token_embd: Vec<f32>,
    /// 36 Transformer blocks.
    pub(crate) blocks: Vec<TransformerBlock<'a>>,
    /// Final output RMSNorm.
    output_norm: RmsNorm,
    /// Output (LM head) weight blocks.
    output_weight: OutputWeight<'a>,
    /// RoPE precomputed tables.
    rope: RopeTable,
    /// KV cache.
    kv_cache: KvCache,
    /// Dominant tensor quantization type, detected at load time for variant identification.
    dominant_quant_type: oxibonsai_core::GgufTensorType,
    /// Cached GPU weight handles for zero-overhead Metal decode path.
    /// Populated on first GPU forward pass, reused on subsequent calls.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    gpu_weight_cache: std::sync::Mutex<Option<oxibonsai_kernels::CachedModelWeights>>,
    /// Cached per-layer QKV concatenated bytes for CUDA path (built once, reused).
    /// Avoids repeated heap allocation on every token during CUDA decode.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    cuda_qkv_cache: std::sync::Mutex<Option<std::sync::Arc<Vec<Vec<u8>>>>>,
}

impl<'a> BonsaiModel<'a> {
    /// Load a model from a parsed GGUF file.
    ///
    /// Extracts configuration from metadata, then maps all tensor data
    /// into the layer structures (zero-copy for 1-bit weights).
    pub fn from_gguf(gguf: &'a GgufFile<'a>, max_seq_len: usize) -> ModelResult<Self> {
        let mut config = Qwen3Config::from_metadata(&gguf.metadata)?;
        if let Some(embd_info) = gguf.tensors.get(tensor_names::TOKEN_EMBD) {
            if embd_info.shape.len() >= 2 {
                let tensor_vocab = embd_info.shape[1] as usize;
                if tensor_vocab != config.vocab_size {
                    tracing::warn!(
                        metadata_vocab = config.vocab_size, tensor_vocab,
                        "vocab_size mismatch: GGUF metadata says {} but token_embd tensor has {} rows; using tensor dimension",
                        config.vocab_size, tensor_vocab,
                    );
                    config.vocab_size = tensor_vocab;
                }
            }
        }
        let dominant_quant_type = {
            let counts = gguf.tensors.count_by_type();
            counts
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(ty, _)| ty)
                .unwrap_or(oxibonsai_core::GgufTensorType::Q1_0_g128)
        };
        tracing::info!(
            layers = config.num_layers,
            hidden = config.hidden_size,
            heads = config.num_attention_heads,
            kv_heads = config.num_kv_heads,
            vocab = config.vocab_size,
            "loading BonsaiModel from GGUF"
        );
        let token_embd = load_f32_tensor(gguf, tensor_names::TOKEN_EMBD)?;
        let output_norm_w = load_f32_tensor(gguf, tensor_names::OUTPUT_NORM)?;
        let output_norm = RmsNorm::new(output_norm_w, config.rms_norm_eps);
        let kernel = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
        let output_weight = load_output_weight(gguf, &config, &kernel)?;
        let mut blocks = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let block = load_transformer_block(gguf, &config, layer_idx, &kernel)?;
            blocks.push(block);
        }
        let rope = RopeTable::new(config.head_dim, max_seq_len, config.rope_freq_base);
        let kv_cache = KvCache::new(
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
            max_seq_len,
        );
        tracing::info!(
            blocks = blocks.len(),
            embd_size = token_embd.len(),
            max_seq_len,
            "model loaded successfully"
        );
        Ok(Self {
            config,
            token_embd,
            blocks,
            output_norm,
            output_weight,
            rope,
            kv_cache,
            dominant_quant_type,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            gpu_weight_cache: std::sync::Mutex::new(None),
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_qkv_cache: std::sync::Mutex::new(None),
        })
    }

    /// Create a model from configuration only (no weights), for testing.
    pub fn new(config: Qwen3Config) -> Self {
        let h = config.hidden_size;
        let kv_cache = KvCache::new(
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
            4096,
        );
        let rope = RopeTable::new(config.head_dim, 4096, config.rope_freq_base);
        Self {
            token_embd: vec![0.0; config.vocab_size * h],
            blocks: Vec::new(),
            output_norm: RmsNorm::new(vec![1.0; h], config.rms_norm_eps),
            output_weight: OutputWeight::Fp32 {
                weights: vec![0.0; config.vocab_size * h],
                out_features: config.vocab_size,
                in_features: h,
            },
            rope,
            kv_cache,
            dominant_quant_type: oxibonsai_core::GgufTensorType::Q1_0_g128,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            gpu_weight_cache: std::sync::Mutex::new(None),
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_qkv_cache: std::sync::Mutex::new(None),
            config,
        }
    }

    /// Get model configuration.
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    /// Get mutable reference to KV cache.
    pub fn kv_cache_mut(&mut self) -> &mut KvCache {
        &mut self.kv_cache
    }

    /// Reset the KV cache for a new conversation.
    pub fn reset(&mut self) {
        self.kv_cache.clear();
    }

    /// Reset the KV cache (alias for `reset`).
    pub fn reset_cache(&mut self) {
        self.kv_cache.clear();
    }

    /// Upload all weight matrices across every Transformer block to GPU memory.
    ///
    /// Should be called once after model loading and before the first
    /// forward pass. If the kernel does not support GPU caching (e.g. CPU
    /// tiers), this is a cheap no-op.
    pub fn upload_weights_to_gpu(&mut self, kernel: &dyn OneBitKernel) {
        let n_blocks = self.blocks.len();
        if n_blocks == 0 {
            return;
        }
        tracing::info!(blocks = n_blocks, "uploading model weights to GPU");
        for block in &mut self.blocks {
            block.upload_to_gpu(kernel);
        }
        match self.output_weight {
            OutputWeight::OneBit(ref mut linear) => linear.upload_to_gpu(),
            OutputWeight::Ternary(ref mut linear) => linear.upload_to_gpu(),
            OutputWeight::Fp32 { .. } => {}
        }
        tracing::info!("GPU weight upload complete");
    }

    /// Detect the model variant from the loaded configuration and dominant tensor type.
    pub fn variant(&self) -> ModelVariant {
        ModelVariant::from_config_and_sample_tensor_type(&self.config, self.dominant_quant_type)
    }

    /// Approximate total number of parameters in the model.
    pub fn num_parameters(&self) -> u64 {
        self.variant().param_count()
    }

    /// Approximate model size in bytes (on disk).
    pub fn model_size_bytes(&self) -> u64 {
        self.variant().expected_model_size_bytes()
    }

    /// Maximum context length from the configuration.
    pub fn context_length(&self) -> usize {
        self.config.max_context_length
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// Hidden dimension size.
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Current KV cache memory usage in bytes.
    pub fn kv_cache_memory_bytes(&self) -> usize {
        self.kv_cache.memory_bytes()
    }

    /// Load a model from GGUF with auto-detected variant.
    ///
    /// Same as `from_gguf` but also logs the detected model variant.
    pub fn from_gguf_auto(gguf: &'a GgufFile<'a>, max_seq_len: usize) -> ModelResult<Self> {
        let model = Self::from_gguf(gguf, max_seq_len)?;
        let variant = model.variant();
        tracing::info!(
            variant = variant.name(),
            params = variant.param_count(),
            "auto-detected model variant"
        );
        Ok(model)
    }

    /// Process multiple prompt tokens in a single batch forward pass on GPU.
    ///
    /// Uses GEMM instead of GEMV for projections (processing all tokens at once),
    /// with sequential per-token attention. Only the last token's logits are
    /// returned (for generation to start). The GPU KV cache is populated for
    /// all positions.
    ///
    /// Falls back to sequential single-token forward if the GPU batch path
    /// is unavailable.
    pub fn forward_prefill(
        &mut self,
        token_ids: &[u32],
        pos_start: usize,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(ModelError::MissingTensor {
                name: "forward_prefill: empty token_ids".into(),
            });
        }
        if token_ids.len() == 1 {
            return self.forward(token_ids[0], pos_start, kernel);
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if token_ids.len() <= 16 {
            let mut last_logits = Vec::new();
            for (i, &token_id) in token_ids.iter().enumerate() {
                last_logits = self.forward(token_id, pos_start + i, kernel)?;
            }
            return Ok(last_logits);
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            match self.try_metal_prefill_with_lm_head(token_ids, pos_start) {
                Ok(logits) => return Ok(logits),
                Err(e) => {
                    tracing::warn!(
                        error = % e,
                        "metal batch prefill failed, falling back to sequential"
                    );
                }
            }
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        {
            match self.try_cuda_prefill_with_lm_head(token_ids, pos_start) {
                Ok(logits) => return Ok(logits),
                Err(e) => {
                    let msg = e.to_string();
                    if msg.contains("LM head not supported on CUDA prefill path") {
                        tracing::debug!(
                            error = % e,
                            "cuda batch prefill skipped (LM head dtype not supported), using sequential"
                        );
                    } else {
                        tracing::warn!(
                            error = % e,
                            "cuda batch prefill failed, falling back to sequential"
                        );
                    }
                }
            }
        }
        let mut last_logits = Vec::new();
        for (i, &token_id) in token_ids.iter().enumerate() {
            last_logits = self.forward(token_id, pos_start + i, kernel)?;
        }
        Ok(last_logits)
    }

    /// Forward pass for speculative decode verification.
    ///
    /// Processes multiple tokens in batch via GPU prefill, then runs the LM head
    /// and argmax on ALL positions (not just the last). Returns the greedy
    /// argmax token ID for each input position.
    ///
    /// If GPU batch path is unavailable, falls back to sequential CPU forward
    /// with argmax at each position.
    pub fn forward_prefill_verify(
        &mut self,
        token_ids: &[u32],
        pos_start: usize,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<Vec<u32>> {
        if token_ids.is_empty() {
            return Ok(vec![]);
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            match self.try_metal_prefill_verify(token_ids, pos_start) {
                Ok(ids) => return Ok(ids),
                Err(e) => {
                    tracing::warn!(
                        error = % e,
                        "metal batch prefill verify failed, falling back to sequential"
                    );
                }
            }
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        {
            match self.try_cuda_prefill_verify(token_ids, pos_start) {
                Ok(ids) => return Ok(ids),
                Err(e) => {
                    tracing::warn!(
                        error = % e,
                        "cuda batch prefill verify failed, falling back to sequential"
                    );
                }
            }
        }
        let mut token_ids_out = Vec::with_capacity(token_ids.len());
        for (i, &token_id) in token_ids.iter().enumerate() {
            let logits = self.forward(token_id, pos_start + i, kernel)?;
            let mut best_idx = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for (j, &v) in logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = j as u32;
                }
            }
            token_ids_out.push(best_idx);
        }
        Ok(token_ids_out)
    }

    /// Forward pass for a single token at position `pos`.
    ///
    /// Returns logits over the vocabulary `[vocab_size]`.
    #[tracing::instrument(skip(self, kernel), fields(token_id, pos))]
    pub fn forward(
        &mut self,
        token_id: u32,
        pos: usize,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<Vec<f32>> {
        let h = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        if pos >= self.kv_cache.max_seq_len() {
            return Err(ModelError::SequenceTooLong {
                seq_len: pos + 1,
                max_ctx: self.kv_cache.max_seq_len(),
            });
        }
        let embd_start = token_id as usize * h;
        let embd_end = embd_start + h;
        if embd_end > self.token_embd.len() {
            return Err(ModelError::MissingTensor {
                name: format!(
                    "token_id {} out of range (vocab={})",
                    token_id,
                    self.token_embd.len() / h
                ),
            });
        }
        let mut hidden = self.token_embd[embd_start..embd_end].to_vec();
        let t_blocks_start = std::time::Instant::now();
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let mut fused_logits = vec![0.0f32; vocab];
            if self
                .try_metal_full_forward_with_lm_head(&mut hidden, pos, &mut fused_logits)
                .is_ok()
            {
                let t_elapsed = t_blocks_start.elapsed();
                tracing::debug!(
                    target : "fwd_profile",
                    "pos={pos} fused_gpu={:.1}ms (metal layers+norm+lm_head)", t_elapsed
                    .as_secs_f64() * 1000.0,
                );
                return Ok(fused_logits);
            }
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        {
            if let Ok(fused_logits) = self.try_cuda_full_forward_with_lm_head(&hidden, pos) {
                return Ok(fused_logits);
            }
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let did_full_forward = {
            let q1_ok = self.try_metal_full_forward_inner(&mut hidden, pos).is_ok();
            if q1_ok {
                true
            } else {
                self.try_metal_full_forward_ternary_inner(&mut hidden, pos)
                    .is_ok()
            }
        };
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        let did_full_forward = match self.try_cuda_full_forward_inner(&hidden, pos) {
            Ok(new_hidden) => {
                hidden = new_hidden;
                true
            }
            Err(_) => false,
        };
        #[cfg(not(any(
            all(feature = "metal", target_os = "macos"),
            all(
                feature = "native-cuda",
                not(all(feature = "metal", target_os = "macos")),
                any(target_os = "linux", target_os = "windows")
            )
        )))]
        let did_full_forward = false;
        if !did_full_forward {
            for block in &self.blocks {
                block.forward(&mut hidden, pos, &mut self.kv_cache, &self.rope, kernel)?;
            }
        }
        let t_blocks_elapsed = t_blocks_start.elapsed();
        let t_norm_start = std::time::Instant::now();
        let mut normed = vec![0.0f32; h];
        self.output_norm.forward(&hidden, &mut normed)?;
        let t_norm_elapsed = t_norm_start.elapsed();
        let t_lm_start = std::time::Instant::now();
        let mut logits = vec![0.0f32; vocab];
        match &self.output_weight {
            OutputWeight::OneBit(linear) => {
                linear.forward_vec(&normed, &mut logits)?;
            }
            OutputWeight::Ternary(linear) => {
                linear.forward(&normed, &mut logits)?;
            }
            OutputWeight::Fp32 {
                weights,
                out_features,
                in_features,
            } => {
                for (i, logit) in logits.iter_mut().enumerate().take(*out_features) {
                    let row_start = i * in_features;
                    let mut sum = 0.0f32;
                    for j in 0..*in_features {
                        sum += weights[row_start + j] * normed[j];
                    }
                    *logit = sum;
                }
            }
        }
        let t_lm_elapsed = t_lm_start.elapsed();
        tracing::debug!(
            target : "fwd_profile",
            "pos={pos} blocks={:.1}ms norm={:.1}ms lm_head={:.1}ms gpu={}",
            t_blocks_elapsed.as_secs_f64() * 1000.0, t_norm_elapsed.as_secs_f64() *
            1000.0, t_lm_elapsed.as_secs_f64() * 1000.0, did_full_forward,
        );
        Ok(logits)
    }
}

/// Output projection can be 1-bit, ternary, or FP32 depending on the model.
pub(super) enum OutputWeight<'a> {
    OneBit(Linear1Bit<'a>),
    Ternary(LinearTernary<'a>),
    Fp32 {
        weights: Vec<f32>,
        out_features: usize,
        in_features: usize,
    },
}
