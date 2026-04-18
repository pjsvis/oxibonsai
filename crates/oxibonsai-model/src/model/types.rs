//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::weight_loaders::{load_f32_tensor, load_output_weight, load_transformer_block};
#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    )
))]
use crate::block::blocks_as_bytes;
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
    /// Attempt to run all transformer layers in a single Metal command buffer.
    ///
    /// On success, `hidden` is updated in-place through all layers. The GPU
    /// manages its own KV cache. Returns `Err` if any precondition is not
    /// met or the dispatch fails.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn try_metal_full_forward_inner(
        &self,
        hidden: &mut [f32],
        pos: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use oxibonsai_kernels::FullForwardLayerParams;
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let max_seq_len = self.kv_cache.max_seq_len();
        for block in &self.blocks {
            if block.fused_qkv_gpu_handle().is_none()
                || block.attn_output_gpu_handle().is_none()
                || block.fused_gate_up_gpu_handle().is_none()
                || block.ffn_down_gpu_handle().is_none()
            {
                return Err("missing GPU handle".into());
            }
        }
        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        for block in &self.blocks {
            let q_bytes =
                blocks_as_bytes(block.attn_q_blocks().ok_or("attn_q: not a 1-bit layer")?);
            let k_bytes =
                blocks_as_bytes(block.attn_k_blocks().ok_or("attn_k: not a 1-bit layer")?);
            let v_bytes =
                blocks_as_bytes(block.attn_v_blocks().ok_or("attn_v: not a 1-bit layer")?);
            let mut concat = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            concat.extend_from_slice(q_bytes);
            concat.extend_from_slice(k_bytes);
            concat.extend_from_slice(v_bytes);
            qkv_concats.push(concat);
        }
        let mut layer_params: Vec<FullForwardLayerParams<'_>> = Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 1_000_000u64 + (block.layer_index() as u64) * 10;
            layer_params.push(FullForwardLayerParams {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: block.fused_qkv_gpu_handle().map(|h| h.id()).unwrap_or(0),
                fused_qkv_bytes: &qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: block.attn_output_gpu_handle().map(|h| h.id()).unwrap_or(0),
                attn_proj_bytes: blocks_as_bytes(
                    block
                        .attn_output_blocks()
                        .ok_or("attn_output: not a 1-bit layer")?,
                ),
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: block
                    .fused_gate_up_gpu_handle()
                    .map(|h| h.id())
                    .unwrap_or(0),
                gate_bytes: blocks_as_bytes(
                    block
                        .ffn_gate_blocks()
                        .ok_or("ffn_gate: not a 1-bit layer")?,
                ),
                up_bytes: blocks_as_bytes(
                    block.ffn_up_blocks().ok_or("ffn_up: not a 1-bit layer")?,
                ),
                down_handle: block.ffn_down_gpu_handle().map(|h| h.id()).unwrap_or(0),
                down_bytes: blocks_as_bytes(
                    block
                        .ffn_down_blocks()
                        .ok_or("ffn_down: not a 1-bit layer")?,
                ),
            });
        }
        let rope_cos = self.rope.cos_at(pos);
        let rope_sin = self.rope.sin_at(pos);
        oxibonsai_kernels::try_metal_full_forward(
            hidden,
            pos,
            n_layers,
            &layer_params,
            rope_cos,
            rope_sin,
            h,
            inter,
            nq,
            nkv,
            hd,
            eps,
            max_seq_len,
            None,
            None,
            eps,
            None,
            None,
            0,
            None,
            None,
        )
        .map_err(|e| {
            tracing::warn!(
                error = % e, "full-forward GPU dispatch failed, falling back"
            );
            Box::new(e) as Box<dyn std::error::Error>
        })
    }
    /// Attempt to run every transformer layer on Metal for a ternary
    /// (TQ2_0_g128) model, encoding all layers into a single command buffer.
    ///
    /// Mirrors [`try_metal_full_forward_inner`] but uses the TQ2 GEMV kernel
    /// and ternary block slices. Returns `Err` if any block is not ternary or
    /// the Metal dispatch fails — in which case the caller should fall back
    /// to the CPU per-layer path.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn try_metal_full_forward_ternary_inner(
        &self,
        hidden: &mut [f32],
        pos: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::block::blocks_as_bytes_ternary;
        use oxibonsai_kernels::FullForwardLayerParamsTernary;
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let max_seq_len = self.kv_cache.max_seq_len();
        for block in &self.blocks {
            if block.attn_q_blocks_ternary().is_none()
                || block.attn_k_blocks_ternary().is_none()
                || block.attn_v_blocks_ternary().is_none()
                || block.attn_output_blocks_ternary().is_none()
                || block.ffn_gate_blocks_ternary().is_none()
                || block.ffn_up_blocks_ternary().is_none()
                || block.ffn_down_blocks_ternary().is_none()
            {
                return Err("non-ternary block on ternary full-forward path".into());
            }
        }
        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        for block in &self.blocks {
            let q_bytes = blocks_as_bytes_ternary(
                block
                    .attn_q_blocks_ternary()
                    .ok_or("attn_q: not a ternary layer")?,
            );
            let k_bytes = blocks_as_bytes_ternary(
                block
                    .attn_k_blocks_ternary()
                    .ok_or("attn_k: not a ternary layer")?,
            );
            let v_bytes = blocks_as_bytes_ternary(
                block
                    .attn_v_blocks_ternary()
                    .ok_or("attn_v: not a ternary layer")?,
            );
            let mut concat = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            concat.extend_from_slice(q_bytes);
            concat.extend_from_slice(k_bytes);
            concat.extend_from_slice(v_bytes);
            qkv_concats.push(concat);
        }
        let mut layer_params: Vec<FullForwardLayerParamsTernary<'_>> = Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 2_000_000u64 + (block.layer_index() as u64) * 10;
            let weight_handle_base = 3_000_000u64 + (block.layer_index() as u64) * 10;
            layer_params.push(FullForwardLayerParamsTernary {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: weight_handle_base,
                fused_qkv_bytes: &qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: weight_handle_base + 1,
                attn_proj_bytes: blocks_as_bytes_ternary(
                    block
                        .attn_output_blocks_ternary()
                        .ok_or("attn_output: not a ternary layer")?,
                ),
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: weight_handle_base + 2,
                gate_bytes: blocks_as_bytes_ternary(
                    block
                        .ffn_gate_blocks_ternary()
                        .ok_or("ffn_gate: not a ternary layer")?,
                ),
                up_bytes: blocks_as_bytes_ternary(
                    block
                        .ffn_up_blocks_ternary()
                        .ok_or("ffn_up: not a ternary layer")?,
                ),
                down_handle: weight_handle_base + 3,
                down_bytes: blocks_as_bytes_ternary(
                    block
                        .ffn_down_blocks_ternary()
                        .ok_or("ffn_down: not a ternary layer")?,
                ),
            });
        }
        let rope_cos = self.rope.cos_at(pos);
        let rope_sin = self.rope.sin_at(pos);
        oxibonsai_kernels::try_metal_full_forward_ternary(
            hidden,
            pos,
            n_layers,
            &layer_params,
            rope_cos,
            rope_sin,
            h,
            inter,
            nq,
            nkv,
            hd,
            eps,
            max_seq_len,
            None,
            None,
            eps,
            None,
            None,
            0,
            None,
            None,
        )
        .map_err(|e| {
            tracing::warn!(
                error = % e, "ternary full-forward GPU dispatch failed, falling back"
            );
            Box::new(e) as Box<dyn std::error::Error>
        })
    }
    /// Attempt to run all transformer layers + final RMSNorm + LM head GEMV
    /// in a single Metal command buffer.
    ///
    /// On success, `logits` is filled with the output logits and `hidden` is
    /// NOT updated (the GPU handles everything end-to-end). Returns `Err` if
    /// any precondition is not met (missing GPU handles, FP32 LM head, etc.).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn try_metal_full_forward_with_lm_head(
        &self,
        hidden: &mut [f32],
        pos: usize,
        logits: &mut Vec<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use oxibonsai_kernels::FullForwardLayerParams;
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let lm_head_linear = match &self.output_weight {
            OutputWeight::OneBit(linear) => linear,
            OutputWeight::Ternary(_) => {
                return Err("ternary LM head not supported on fused GPU path".into());
            }
            OutputWeight::Fp32 { .. } => {
                return Err("FP32 LM head not supported on fused GPU path".into());
            }
        };
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let max_seq_len = self.kv_cache.max_seq_len();
        for block in &self.blocks {
            if block.fused_qkv_gpu_handle().is_none()
                || block.attn_output_gpu_handle().is_none()
                || block.fused_gate_up_gpu_handle().is_none()
                || block.ffn_down_gpu_handle().is_none()
            {
                return Err("missing GPU handle".into());
            }
        }
        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        for block in &self.blocks {
            let q_bytes =
                blocks_as_bytes(block.attn_q_blocks().ok_or("attn_q: not a 1-bit layer")?);
            let k_bytes =
                blocks_as_bytes(block.attn_k_blocks().ok_or("attn_k: not a 1-bit layer")?);
            let v_bytes =
                blocks_as_bytes(block.attn_v_blocks().ok_or("attn_v: not a 1-bit layer")?);
            let mut concat = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            concat.extend_from_slice(q_bytes);
            concat.extend_from_slice(k_bytes);
            concat.extend_from_slice(v_bytes);
            qkv_concats.push(concat);
        }
        let mut layer_params: Vec<FullForwardLayerParams<'_>> = Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 1_000_000u64 + (block.layer_index() as u64) * 10;
            layer_params.push(FullForwardLayerParams {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: block.fused_qkv_gpu_handle().map(|h| h.id()).unwrap_or(0),
                fused_qkv_bytes: &qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: block.attn_output_gpu_handle().map(|h| h.id()).unwrap_or(0),
                attn_proj_bytes: blocks_as_bytes(
                    block
                        .attn_output_blocks()
                        .ok_or("attn_output: not a 1-bit layer")?,
                ),
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: block
                    .fused_gate_up_gpu_handle()
                    .map(|h| h.id())
                    .unwrap_or(0),
                gate_bytes: blocks_as_bytes(
                    block
                        .ffn_gate_blocks()
                        .ok_or("ffn_gate: not a 1-bit layer")?,
                ),
                up_bytes: blocks_as_bytes(
                    block.ffn_up_blocks().ok_or("ffn_up: not a 1-bit layer")?,
                ),
                down_handle: block.ffn_down_gpu_handle().map(|h| h.id()).unwrap_or(0),
                down_bytes: blocks_as_bytes(
                    block
                        .ffn_down_blocks()
                        .ok_or("ffn_down: not a 1-bit layer")?,
                ),
            });
        }
        let rope_cos = self.rope.cos_at(pos);
        let rope_sin = self.rope.sin_at(pos);
        let final_norm_handle = 2_000_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();
        let lm_head_handle = 3_000_000u64;
        let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
        let lm_head_out_features = lm_head_linear.out_features();
        oxibonsai_kernels::try_metal_full_forward(
            hidden,
            pos,
            n_layers,
            &layer_params,
            rope_cos,
            rope_sin,
            h,
            inter,
            nq,
            nkv,
            hd,
            eps,
            max_seq_len,
            Some(final_norm_handle),
            Some(final_norm_bytes),
            final_norm_eps,
            Some(lm_head_handle),
            Some(lm_head_bytes),
            lm_head_out_features,
            Some(logits),
            None,
        )
        .map_err(|e| {
            tracing::warn!(
                error = % e, "full-forward+lm_head GPU dispatch failed, falling back"
            );
            Box::new(e) as Box<dyn std::error::Error>
        })
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
    /// GPU batch prefill implementation: all layers + final norm + LM head.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn try_metal_prefill_with_lm_head(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        use oxibonsai_kernels::FullForwardLayerParams;
        let batch_size = token_ids.len();
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let lm_head_linear = match &self.output_weight {
            OutputWeight::OneBit(linear) => linear,
            OutputWeight::Ternary(_) => {
                return Err("ternary LM head not supported on GPU prefill path".into());
            }
            OutputWeight::Fp32 { .. } => {
                return Err("FP32 LM head not supported on GPU prefill path".into());
            }
        };
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let half_dim = hd / 2;
        let max_seq_len = self.kv_cache.max_seq_len();
        for block in &self.blocks {
            if block.fused_qkv_gpu_handle().is_none()
                || block.attn_output_gpu_handle().is_none()
                || block.fused_gate_up_gpu_handle().is_none()
                || block.ffn_down_gpu_handle().is_none()
            {
                return Err("missing GPU handle".into());
            }
        }
        let mut hidden_batch = vec![0.0f32; batch_size * h];
        for (t, &token_id) in token_ids.iter().enumerate() {
            let embd_start = token_id as usize * h;
            let embd_end = embd_start + h;
            if embd_end > self.token_embd.len() {
                return Err(format!(
                    "token_id {} out of range (vocab={})",
                    token_id,
                    self.token_embd.len() / h
                )
                .into());
            }
            hidden_batch[t * h..(t + 1) * h]
                .copy_from_slice(&self.token_embd[embd_start..embd_end]);
        }
        let mut cos_table = vec![0.0f32; batch_size * half_dim];
        let mut sin_table = vec![0.0f32; batch_size * half_dim];
        for t in 0..batch_size {
            let pos = pos_start + t;
            let cos_vals = self.rope.cos_at(pos);
            let sin_vals = self.rope.sin_at(pos);
            cos_table[t * half_dim..(t + 1) * half_dim].copy_from_slice(cos_vals);
            sin_table[t * half_dim..(t + 1) * half_dim].copy_from_slice(sin_vals);
        }
        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        for block in &self.blocks {
            let q_bytes =
                blocks_as_bytes(block.attn_q_blocks().ok_or("attn_q: not a 1-bit layer")?);
            let k_bytes =
                blocks_as_bytes(block.attn_k_blocks().ok_or("attn_k: not a 1-bit layer")?);
            let v_bytes =
                blocks_as_bytes(block.attn_v_blocks().ok_or("attn_v: not a 1-bit layer")?);
            let mut concat = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            concat.extend_from_slice(q_bytes);
            concat.extend_from_slice(k_bytes);
            concat.extend_from_slice(v_bytes);
            qkv_concats.push(concat);
        }
        let mut layer_params: Vec<FullForwardLayerParams<'_>> = Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 1_000_000u64 + (block.layer_index() as u64) * 10;
            layer_params.push(FullForwardLayerParams {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: block
                    .fused_qkv_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(0),
                fused_qkv_bytes: &qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: block
                    .attn_output_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(0),
                attn_proj_bytes: blocks_as_bytes(
                    block
                        .attn_output_blocks()
                        .ok_or("attn_output: not a 1-bit layer")?,
                ),
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: block
                    .fused_gate_up_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(0),
                gate_bytes: blocks_as_bytes(
                    block
                        .ffn_gate_blocks()
                        .ok_or("ffn_gate: not a 1-bit layer")?,
                ),
                up_bytes: blocks_as_bytes(
                    block.ffn_up_blocks().ok_or("ffn_up: not a 1-bit layer")?,
                ),
                down_handle: block.ffn_down_gpu_handle().map(|hnd| hnd.id()).unwrap_or(0),
                down_bytes: blocks_as_bytes(
                    block
                        .ffn_down_blocks()
                        .ok_or("ffn_down: not a 1-bit layer")?,
                ),
            });
        }
        let final_norm_handle = 2_000_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();
        let lm_head_handle = 3_000_000u64;
        let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
        let lm_head_out_features = lm_head_linear.out_features();
        let mut logits = vec![0.0f32; lm_head_out_features];
        oxibonsai_kernels::try_metal_full_forward_prefill(
            &hidden_batch,
            batch_size,
            pos_start,
            n_layers,
            &layer_params,
            &cos_table,
            &sin_table,
            h,
            inter,
            nq,
            nkv,
            hd,
            eps,
            max_seq_len,
            Some(final_norm_handle),
            Some(final_norm_bytes),
            final_norm_eps,
            Some(lm_head_handle),
            Some(lm_head_bytes),
            lm_head_out_features,
            Some(&mut logits),
            None,
        )
        .map_err(|e| {
            tracing::warn!(error = % e, "batch prefill GPU dispatch failed");
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        Ok(logits)
    }
    /// GPU batch prefill verify: all layers + final norm + LM head + per-position argmax.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn try_metal_prefill_verify(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        use oxibonsai_kernels::FullForwardLayerParams;
        let batch_size = token_ids.len();
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let lm_head_linear = match &self.output_weight {
            OutputWeight::OneBit(linear) => linear,
            OutputWeight::Ternary(_) => {
                return Err("ternary LM head not supported on GPU prefill verify path".into());
            }
            OutputWeight::Fp32 { .. } => {
                return Err("FP32 LM head not supported on GPU prefill verify path".into());
            }
        };
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let half_dim = hd / 2;
        let max_seq_len = self.kv_cache.max_seq_len();
        for block in &self.blocks {
            if block.fused_qkv_gpu_handle().is_none()
                || block.attn_output_gpu_handle().is_none()
                || block.fused_gate_up_gpu_handle().is_none()
                || block.ffn_down_gpu_handle().is_none()
            {
                return Err("missing GPU handle".into());
            }
        }
        let mut hidden_batch = vec![0.0f32; batch_size * h];
        for (t, &token_id) in token_ids.iter().enumerate() {
            let embd_start = token_id as usize * h;
            let embd_end = embd_start + h;
            if embd_end > self.token_embd.len() {
                return Err(format!(
                    "token_id {} out of range (vocab={})",
                    token_id,
                    self.token_embd.len() / h
                )
                .into());
            }
            hidden_batch[t * h..(t + 1) * h]
                .copy_from_slice(&self.token_embd[embd_start..embd_end]);
        }
        let mut cos_table = vec![0.0f32; batch_size * half_dim];
        let mut sin_table = vec![0.0f32; batch_size * half_dim];
        for t in 0..batch_size {
            let pos = pos_start + t;
            let cos_vals = self.rope.cos_at(pos);
            let sin_vals = self.rope.sin_at(pos);
            cos_table[t * half_dim..(t + 1) * half_dim].copy_from_slice(cos_vals);
            sin_table[t * half_dim..(t + 1) * half_dim].copy_from_slice(sin_vals);
        }
        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        for block in &self.blocks {
            let q_bytes =
                blocks_as_bytes(block.attn_q_blocks().ok_or("attn_q: not a 1-bit layer")?);
            let k_bytes =
                blocks_as_bytes(block.attn_k_blocks().ok_or("attn_k: not a 1-bit layer")?);
            let v_bytes =
                blocks_as_bytes(block.attn_v_blocks().ok_or("attn_v: not a 1-bit layer")?);
            let mut concat = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            concat.extend_from_slice(q_bytes);
            concat.extend_from_slice(k_bytes);
            concat.extend_from_slice(v_bytes);
            qkv_concats.push(concat);
        }
        let mut layer_params: Vec<FullForwardLayerParams<'_>> = Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 1_000_000u64 + (block.layer_index() as u64) * 10;
            layer_params.push(FullForwardLayerParams {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: block
                    .fused_qkv_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(0),
                fused_qkv_bytes: &qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: block
                    .attn_output_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(0),
                attn_proj_bytes: blocks_as_bytes(
                    block
                        .attn_output_blocks()
                        .ok_or("attn_output: not a 1-bit layer")?,
                ),
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: block
                    .fused_gate_up_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(0),
                gate_bytes: blocks_as_bytes(
                    block
                        .ffn_gate_blocks()
                        .ok_or("ffn_gate: not a 1-bit layer")?,
                ),
                up_bytes: blocks_as_bytes(
                    block.ffn_up_blocks().ok_or("ffn_up: not a 1-bit layer")?,
                ),
                down_handle: block.ffn_down_gpu_handle().map(|hnd| hnd.id()).unwrap_or(0),
                down_bytes: blocks_as_bytes(
                    block
                        .ffn_down_blocks()
                        .ok_or("ffn_down: not a 1-bit layer")?,
                ),
            });
        }
        let final_norm_handle = 2_000_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();
        let lm_head_handle = 3_000_000u64;
        let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
        let lm_head_out_features = lm_head_linear.out_features();
        let mut batch_token_ids: Vec<u32> = Vec::with_capacity(batch_size);
        oxibonsai_kernels::try_metal_full_forward_prefill_verify(
            &hidden_batch,
            batch_size,
            pos_start,
            n_layers,
            &layer_params,
            &cos_table,
            &sin_table,
            h,
            inter,
            nq,
            nkv,
            hd,
            eps,
            max_seq_len,
            Some(final_norm_handle),
            Some(final_norm_bytes),
            final_norm_eps,
            Some(lm_head_handle),
            Some(lm_head_bytes),
            lm_head_out_features,
            &mut batch_token_ids,
        )
        .map_err(|e| {
            tracing::warn!(error = % e, "batch prefill verify GPU dispatch failed");
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        Ok(batch_token_ids)
    }
    /// Greedy forward pass: runs all layers + LM head + argmax entirely on GPU.
    ///
    /// Instead of downloading the full logits vector (~607KB), this performs
    /// argmax on the GPU and downloads only the resulting token ID (4 bytes).
    /// This eliminates ~607KB of GPU→CPU bandwidth per token and removes
    /// CPU-side sampling overhead for greedy (temperature=0) decoding.
    ///
    /// On the first call, all weight handles are cached in `gpu_weight_cache`.
    /// Subsequent calls skip ALL byte concatenation, weight upload, and
    /// HashMap lookups — passing pre-cached handles directly to the GPU.
    ///
    /// Returns the token ID directly, or `Err` if the GPU path is not available.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn forward_greedy_gpu(
        &self,
        token_id: u32,
        pos: usize,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let max_seq_len = self.kv_cache.max_seq_len();
        let eps = if self.blocks.is_empty() {
            return Err("no blocks".into());
        } else {
            self.blocks[0].attn_norm_eps()
        };
        let final_norm_eps = self.output_norm.eps();
        let lm_head_out_features = match &self.output_weight {
            OutputWeight::OneBit(linear) => linear.out_features(),
            OutputWeight::Ternary(_) => {
                return Err("ternary LM head not supported on greedy GPU path".into());
            }
            OutputWeight::Fp32 { .. } => {
                return Err("FP32 LM head not supported on greedy GPU path".into());
            }
        };
        self.get_or_create_gpu_cache()?;
        let embd_start = token_id as usize * h;
        let embd_end = embd_start + h;
        if embd_end > self.token_embd.len() {
            return Err(format!(
                "token_id {} out of range (vocab={})",
                token_id,
                self.token_embd.len() / h
            )
            .into());
        }
        let mut hidden = self.token_embd[embd_start..embd_end].to_vec();
        let rope_cos = self.rope.cos_at(pos);
        let rope_sin = self.rope.sin_at(pos);
        let mut greedy_token_id: u32 = 0;
        let guard = self
            .gpu_weight_cache
            .lock()
            .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
        let cached = guard.as_ref().ok_or("GPU weight cache not populated")?;
        oxibonsai_kernels::try_metal_full_forward_cached(
            &mut hidden,
            pos,
            cached,
            rope_cos,
            rope_sin,
            h,
            inter,
            nq,
            nkv,
            hd,
            eps,
            max_seq_len,
            final_norm_eps,
            lm_head_out_features,
            None,
            Some(&mut greedy_token_id),
        )
        .map_err(|e| {
            tracing::warn!(error = % e, "cached greedy GPU forward failed");
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        Ok(greedy_token_id)
    }
    /// Build and cache GPU weight handles on first call; no-op on subsequent calls.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn get_or_create_gpu_cache(&self) -> Result<(), Box<dyn std::error::Error>> {
        use oxibonsai_kernels::FullForwardLayerParams;
        {
            let guard = self
                .gpu_weight_cache
                .lock()
                .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
            if guard.is_some() {
                return Ok(());
            }
        }
        let n_layers = self.blocks.len();
        if matches!(&self.output_weight, OutputWeight::Ternary(_)) {
            return Ok(());
        }
        if self.blocks.iter().any(|b| b.attn_q_blocks().is_none()) {
            return Ok(());
        }
        let lm_head_linear = match &self.output_weight {
            OutputWeight::OneBit(ref linear) => linear,
            OutputWeight::Fp32 { .. } => return Err("FP32 LM head not supported".into()),
            OutputWeight::Ternary(_) => unreachable!("ternary guard above"),
        };
        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        for block in &self.blocks {
            let q_bytes =
                blocks_as_bytes(block.attn_q_blocks().ok_or("attn_q: not a 1-bit layer")?);
            let k_bytes =
                blocks_as_bytes(block.attn_k_blocks().ok_or("attn_k: not a 1-bit layer")?);
            let v_bytes =
                blocks_as_bytes(block.attn_v_blocks().ok_or("attn_v: not a 1-bit layer")?);
            let mut concat = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            concat.extend_from_slice(q_bytes);
            concat.extend_from_slice(k_bytes);
            concat.extend_from_slice(v_bytes);
            qkv_concats.push(concat);
        }
        let mut layer_params: Vec<FullForwardLayerParams<'_>> = Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 1_000_000u64 + (block.layer_index() as u64) * 10;
            layer_params.push(FullForwardLayerParams {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: block
                    .fused_qkv_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(0),
                fused_qkv_bytes: &qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: block
                    .attn_output_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(0),
                attn_proj_bytes: blocks_as_bytes(
                    block
                        .attn_output_blocks()
                        .ok_or("attn_output: not a 1-bit layer")?,
                ),
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: block
                    .fused_gate_up_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(0),
                gate_bytes: blocks_as_bytes(
                    block
                        .ffn_gate_blocks()
                        .ok_or("ffn_gate: not a 1-bit layer")?,
                ),
                up_bytes: blocks_as_bytes(
                    block.ffn_up_blocks().ok_or("ffn_up: not a 1-bit layer")?,
                ),
                down_handle: block.ffn_down_gpu_handle().map(|hnd| hnd.id()).unwrap_or(0),
                down_bytes: blocks_as_bytes(
                    block
                        .ffn_down_blocks()
                        .ok_or("ffn_down: not a 1-bit layer")?,
                ),
            });
        }
        let final_norm_handle = 2_000_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let lm_head_handle = 3_000_000u64;
        let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
        let cached = oxibonsai_kernels::build_cached_weights(
            &layer_params,
            final_norm_handle,
            final_norm_bytes,
            lm_head_handle,
            lm_head_bytes,
        )
        .map_err(|e| format!("build_cached_weights: {e}"))?;
        let mut guard = self
            .gpu_weight_cache
            .lock()
            .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
        *guard = Some(cached);
        tracing::info!("GPU weight cache populated (all subsequent tokens use cached handles)");
        Ok(())
    }
    /// Get or build the cached per-layer QKV byte concatenations for the CUDA path.
    ///
    /// On first call the vectors are built and stored in `cuda_qkv_cache`.
    /// On subsequent calls the cached version is returned immediately.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    fn get_or_build_cuda_qkv_cache(
        &self,
    ) -> Result<std::sync::Arc<Vec<Vec<u8>>>, Box<dyn std::error::Error>> {
        let guard = self
            .cuda_qkv_cache
            .lock()
            .map_err(|e| format!("cuda_qkv_cache lock: {e}"))?;
        if let Some(ref cache) = *guard {
            return Ok(std::sync::Arc::clone(cache));
        }
        drop(guard);
        let n_layers = self.blocks.len();
        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        for block in &self.blocks {
            let q_bytes =
                blocks_as_bytes(block.attn_q_blocks().ok_or("attn_q: not a 1-bit layer")?);
            let k_bytes =
                blocks_as_bytes(block.attn_k_blocks().ok_or("attn_k: not a 1-bit layer")?);
            let v_bytes =
                blocks_as_bytes(block.attn_v_blocks().ok_or("attn_v: not a 1-bit layer")?);
            let mut concat = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            concat.extend_from_slice(q_bytes);
            concat.extend_from_slice(k_bytes);
            concat.extend_from_slice(v_bytes);
            qkv_concats.push(concat);
        }
        let mut guard = self
            .cuda_qkv_cache
            .lock()
            .map_err(|e| format!("cuda_qkv_cache lock: {e}"))?;
        let arc = std::sync::Arc::new(qkv_concats);
        *guard = Some(std::sync::Arc::clone(&arc));
        Ok(arc)
    }
    /// Build per-layer `CudaFullForwardLayerParams` using cached QKV bytes.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    fn build_cuda_layer_params<'b>(
        &'b self,
        qkv_concats: &'b [Vec<u8>],
    ) -> Result<Vec<oxibonsai_kernels::CudaFullForwardLayerParams<'b>>, Box<dyn std::error::Error>>
    {
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let mut layer_params: Vec<oxibonsai_kernels::CudaFullForwardLayerParams<'b>> =
            Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 1_000_000u64 + (block.layer_index() as u64) * 10;
            let weight_handle_base = 2_000_000u64 + (block.layer_index() as u64) * 4;
            layer_params.push(oxibonsai_kernels::CudaFullForwardLayerParams {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: block
                    .fused_qkv_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(weight_handle_base),
                fused_qkv_bytes: &qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: block
                    .attn_output_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(weight_handle_base + 1),
                attn_proj_bytes: blocks_as_bytes(
                    block
                        .attn_output_blocks()
                        .ok_or("attn_output: not a 1-bit layer")?,
                ),
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: block
                    .fused_gate_up_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(weight_handle_base + 2),
                gate_bytes: blocks_as_bytes(
                    block
                        .ffn_gate_blocks()
                        .ok_or("ffn_gate: not a 1-bit layer")?,
                ),
                up_bytes: blocks_as_bytes(
                    block.ffn_up_blocks().ok_or("ffn_up: not a 1-bit layer")?,
                ),
                down_handle: block
                    .ffn_down_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(weight_handle_base + 3),
                down_bytes: blocks_as_bytes(
                    block
                        .ffn_down_blocks()
                        .ok_or("ffn_down: not a 1-bit layer")?,
                ),
            });
        }
        Ok(layer_params)
    }
    /// Attempt to run all transformer layers (layers only, no LM head) on CUDA GPU.
    ///
    /// On success, returns the post-layers hidden state as a `Vec<f32>` which the
    /// caller should use to replace the CPU `hidden` buffer.  Returns `Err` if CUDA
    /// is unavailable or any precondition is not met.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    fn try_cuda_full_forward_inner(
        &self,
        hidden: &[f32],
        pos: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let heads_per_group = if nkv > 0 { nq / nkv } else { 1 };
        let max_seq_len = self.kv_cache.max_seq_len();
        let qkv_concats = self.get_or_build_cuda_qkv_cache()?;
        let layer_params = self.build_cuda_layer_params(&*qkv_concats)?;
        let rope_cos = self.rope.cos_at(pos);
        let rope_sin = self.rope.sin_at(pos);
        oxibonsai_kernels::try_cuda_full_forward(
            hidden,
            &layer_params,
            rope_cos,
            rope_sin,
            pos,
            nq,
            nkv,
            hd,
            heads_per_group,
            eps,
            h,
            inter,
            max_seq_len,
            None,
            0,
        )
        .ok_or_else(|| {
            tracing::warn!("CUDA full-forward (layers only) returned None, falling back");
            Box::<dyn std::error::Error>::from("CUDA layers-only forward returned None")
        })
    }
    /// Attempt to run all transformer layers + final RMSNorm + LM head on CUDA GPU.
    ///
    /// On success, returns the output logits vector directly (no intermediate allocation).
    /// Returns `Err` if any precondition is not met (no CUDA device, FP32 LM head,
    /// missing GPU handles, etc.).
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    fn try_cuda_full_forward_with_lm_head(
        &self,
        hidden: &[f32],
        pos: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let lm_head_linear = match &self.output_weight {
            OutputWeight::OneBit(linear) => linear,
            OutputWeight::Ternary(_) => {
                return Err("ternary LM head not supported on CUDA fused GPU path".into());
            }
            OutputWeight::Fp32 { .. } => {
                return Err("FP32 LM head not supported on CUDA fused GPU path".into());
            }
        };
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let heads_per_group = if nkv > 0 { nq / nkv } else { 1 };
        let max_seq_len = self.kv_cache.max_seq_len();
        let final_norm_handle = 2_000_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let lm_head_handle = 4_000_000u64;
        let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
        let vocab_size = lm_head_linear.out_features();
        let qkv_concats = self.get_or_build_cuda_qkv_cache()?;
        let layer_params = self.build_cuda_layer_params(&*qkv_concats)?;
        let rope_cos = self.rope.cos_at(pos);
        let rope_sin = self.rope.sin_at(pos);
        match oxibonsai_kernels::try_cuda_full_forward_with_gpu_lm_head(
            hidden,
            &layer_params,
            rope_cos,
            rope_sin,
            pos,
            nq,
            nkv,
            hd,
            heads_per_group,
            eps,
            h,
            inter,
            max_seq_len,
            Some(final_norm_bytes),
            final_norm_handle,
            lm_head_handle,
            &lm_head_bytes,
            vocab_size,
        ) {
            Some(gpu_logits) => Ok(gpu_logits),
            None => {
                tracing::warn!("CUDA full-forward+gpu_lm_head returned None, falling back");
                Err("CUDA full-forward+gpu_lm_head returned None".into())
            }
        }
    }
    /// GPU batch prefill implementation (CUDA): all layers + final norm + LM head.
    ///
    /// Returns the last token's logits.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    fn try_cuda_prefill_with_lm_head(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let batch_size = token_ids.len();
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let lm_head_linear = match &self.output_weight {
            OutputWeight::OneBit(linear) => linear,
            OutputWeight::Ternary(_) => {
                return Err("ternary LM head not supported on CUDA prefill path".into());
            }
            OutputWeight::Fp32 { .. } => {
                return Err("FP32 LM head not supported on CUDA prefill path".into());
            }
        };
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let half_dim = hd / 2;
        let heads_per_group = if nkv > 0 { nq / nkv } else { 1 };
        let max_seq_len = self.kv_cache.max_seq_len();
        let mut hidden_batch = vec![0.0f32; batch_size * h];
        for (t, &token_id) in token_ids.iter().enumerate() {
            let embd_start = token_id as usize * h;
            let embd_end = embd_start + h;
            if embd_end > self.token_embd.len() {
                return Err(format!(
                    "token_id {} out of range (vocab={})",
                    token_id,
                    self.token_embd.len() / h
                )
                .into());
            }
            hidden_batch[t * h..(t + 1) * h]
                .copy_from_slice(&self.token_embd[embd_start..embd_end]);
        }
        let mut cos_table = vec![0.0f32; batch_size * half_dim];
        let mut sin_table = vec![0.0f32; batch_size * half_dim];
        for t in 0..batch_size {
            let pos = pos_start + t;
            let cos_vals = self.rope.cos_at(pos);
            let sin_vals = self.rope.sin_at(pos);
            cos_table[t * half_dim..(t + 1) * half_dim].copy_from_slice(cos_vals);
            sin_table[t * half_dim..(t + 1) * half_dim].copy_from_slice(sin_vals);
        }
        let final_norm_handle = 2_000_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();
        let lm_head_handle = 3_000_000u64;
        let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
        let lm_head_out_features = lm_head_linear.out_features();
        let qkv_concats = self.get_or_build_cuda_qkv_cache()?;
        let layer_params = self.build_cuda_layer_params(&*qkv_concats)?;
        let mut logits = vec![0.0f32; lm_head_out_features];
        oxibonsai_kernels::try_cuda_prefill(
            &hidden_batch,
            batch_size,
            pos_start,
            n_layers,
            &layer_params,
            &cos_table,
            &sin_table,
            h,
            inter,
            nq,
            nkv,
            hd,
            heads_per_group,
            eps,
            max_seq_len,
            Some(final_norm_handle),
            Some(final_norm_bytes),
            final_norm_eps,
            Some(lm_head_handle),
            Some(lm_head_bytes),
            lm_head_out_features,
            Some(&mut logits),
            None,
        )
        .map_err(|e| {
            tracing::warn!(error = % e, "CUDA batch prefill dispatch failed");
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        Ok(logits)
    }
    /// GPU batch prefill verify (CUDA): all layers + final norm + LM head + argmax.
    ///
    /// Returns the greedy argmax token ID for each input position.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    fn try_cuda_prefill_verify(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let batch_size = token_ids.len();
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let lm_head_linear = match &self.output_weight {
            OutputWeight::OneBit(linear) => linear,
            OutputWeight::Ternary(_) => {
                return Err("ternary LM head not supported on CUDA prefill verify path".into());
            }
            OutputWeight::Fp32 { .. } => {
                return Err("FP32 LM head not supported on CUDA prefill verify path".into());
            }
        };
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let half_dim = hd / 2;
        let heads_per_group = if nkv > 0 { nq / nkv } else { 1 };
        let max_seq_len = self.kv_cache.max_seq_len();
        let mut hidden_batch = vec![0.0f32; batch_size * h];
        for (t, &token_id) in token_ids.iter().enumerate() {
            let embd_start = token_id as usize * h;
            let embd_end = embd_start + h;
            if embd_end > self.token_embd.len() {
                return Err(format!(
                    "token_id {} out of range (vocab={})",
                    token_id,
                    self.token_embd.len() / h
                )
                .into());
            }
            hidden_batch[t * h..(t + 1) * h]
                .copy_from_slice(&self.token_embd[embd_start..embd_end]);
        }
        let mut cos_table = vec![0.0f32; batch_size * half_dim];
        let mut sin_table = vec![0.0f32; batch_size * half_dim];
        for t in 0..batch_size {
            let pos = pos_start + t;
            let cos_vals = self.rope.cos_at(pos);
            let sin_vals = self.rope.sin_at(pos);
            cos_table[t * half_dim..(t + 1) * half_dim].copy_from_slice(cos_vals);
            sin_table[t * half_dim..(t + 1) * half_dim].copy_from_slice(sin_vals);
        }
        let final_norm_handle = 2_000_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();
        let lm_head_handle = 3_000_000u64;
        let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
        let lm_head_out_features = lm_head_linear.out_features();
        let qkv_concats = self.get_or_build_cuda_qkv_cache()?;
        let layer_params = self.build_cuda_layer_params(&*qkv_concats)?;
        let mut token_ids_out: Vec<u32> = Vec::with_capacity(batch_size);
        for t in 0..batch_size {
            let single_embd_start = token_ids[t] as usize * h;
            let single_hidden = self.token_embd[single_embd_start..single_embd_start + h].to_vec();
            let pos = pos_start + t;
            let t_half = half_dim;
            let cos_single = &cos_table[t * t_half..(t + 1) * t_half];
            let sin_single = &sin_table[t * t_half..(t + 1) * t_half];
            let mut greedy_id: u32 = 0;
            oxibonsai_kernels::try_cuda_prefill(
                &single_hidden,
                1,
                pos,
                n_layers,
                &layer_params,
                cos_single,
                sin_single,
                h,
                inter,
                nq,
                nkv,
                hd,
                heads_per_group,
                eps,
                max_seq_len,
                Some(final_norm_handle),
                Some(final_norm_bytes),
                final_norm_eps,
                Some(lm_head_handle),
                Some(lm_head_bytes),
                lm_head_out_features,
                None,
                Some(&mut greedy_id),
            )
            .map_err(|e| {
                tracing::warn!(
                    error = % e, "CUDA prefill verify dispatch failed at pos {pos}"
                );
                Box::new(e) as Box<dyn std::error::Error>
            })?;
            token_ids_out.push(greedy_id);
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
