//! Single Transformer block for Qwen3-8B.
//!
//! Each block contains:
//! 1. Pre-attention RMSNorm → Q/K/V projection → RoPE → GQA Attention → Output projection → Residual
//! 2. Pre-FFN RMSNorm → Gate/Up projection → SwiGLU → Down projection → Residual

use std::sync::Mutex;
use std::time::Instant;

use oxibonsai_kernels::traits::OneBitKernel;
use oxibonsai_kernels::GpuWeightHandle;

/// Convert a BlockQ1_0G128 slice to raw bytes (zero-copy).
///
/// # Safety
/// `BlockQ1_0G128` is `#[repr(C)]` with a well-defined 18-byte layout.
#[cfg(any(
    feature = "metal",
    all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    )
))]
pub(crate) fn blocks_as_bytes(blocks: &[oxibonsai_core::BlockQ1_0G128]) -> &[u8] {
    let ptr = blocks.as_ptr() as *const u8;
    let len = std::mem::size_of_val(blocks);
    // SAFETY: BlockQ1_0G128 is repr(C), so reinterpreting as bytes is valid.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

use crate::error::ModelResult;
use crate::kv_cache::KvCache;
use crate::layers::attention::attention_head;
use crate::layers::linear::Linear1Bit;
use crate::layers::rms_norm::RmsNorm;
use crate::layers::rope::RopeTable;
use crate::layers::sliding_window::SlidingWindowConfig;
use crate::layers::swiglu::swiglu as swiglu_fn;

/// Statistics collected during a single layer's forward pass.
#[derive(Debug, Clone)]
pub struct LayerStats {
    /// Layer index.
    pub layer_idx: usize,
    /// Time spent on attention norm + Q/K/V projection.
    pub projection_us: u64,
    /// Time spent on RoPE application.
    pub rope_us: u64,
    /// Time spent on attention computation (GQA).
    pub attention_us: u64,
    /// Time spent on FFN (MLP) sublayer.
    pub ffn_us: u64,
    /// Total forward time for this layer.
    pub total_us: u64,
}

impl LayerStats {
    /// Create empty stats for a given layer.
    fn new(layer_idx: usize) -> Self {
        Self {
            layer_idx,
            projection_us: 0,
            rope_us: 0,
            attention_us: 0,
            ffn_us: 0,
            total_us: 0,
        }
    }

    /// Fraction of time spent in attention (vs total).
    pub fn attention_fraction(&self) -> f64 {
        if self.total_us == 0 {
            return 0.0;
        }
        self.attention_us as f64 / self.total_us as f64
    }

    /// Fraction of time spent in FFN (vs total).
    pub fn ffn_fraction(&self) -> f64 {
        if self.total_us == 0 {
            return 0.0;
        }
        self.ffn_us as f64 / self.total_us as f64
    }
}

/// Pre-allocated scratch buffers for a single TransformerBlock's forward pass.
/// Eliminates per-token heap allocations in the hot path.
struct ScratchBuffers {
    normed: Vec<f32>,        // [hidden_size]
    q_all: Vec<f32>,         // [nq * head_dim]
    k_all: Vec<f32>,         // [nkv * head_dim]
    v_all: Vec<f32>,         // [nkv * head_dim]
    q_normed: Vec<f32>,      // [nq * head_dim]
    k_normed: Vec<f32>,      // [nkv * head_dim]
    q_rope: Vec<f32>,        // [nq * head_dim]
    k_rope: Vec<f32>,        // [nkv * head_dim]
    attn_out: Vec<f32>,      // [nq * head_dim]
    head_output: Vec<f32>,   // [head_dim]
    attn_proj: Vec<f32>,     // [hidden_size]
    gate_out: Vec<f32>,      // [intermediate_size]
    up_out: Vec<f32>,        // [intermediate_size]
    swiglu_out: Vec<f32>,    // [intermediate_size]
    down_out: Vec<f32>,      // [hidden_size]
    fused_qkv: Vec<f32>,     // [nq*hd + nkv*hd + nkv*hd]
    fused_gate_up: Vec<f32>, // [intermediate_size * 2]
}

impl ScratchBuffers {
    fn new(h: usize, nq: usize, nkv: usize, hd: usize, inter: usize) -> Self {
        Self {
            normed: vec![0.0; h],
            q_all: vec![0.0; nq * hd],
            k_all: vec![0.0; nkv * hd],
            v_all: vec![0.0; nkv * hd],
            q_normed: vec![0.0; nq * hd],
            k_normed: vec![0.0; nkv * hd],
            q_rope: vec![0.0; nq * hd],
            k_rope: vec![0.0; nkv * hd],
            attn_out: vec![0.0; nq * hd],
            head_output: vec![0.0; hd],
            attn_proj: vec![0.0; h],
            gate_out: vec![0.0; inter],
            up_out: vec![0.0; inter],
            swiglu_out: vec![0.0; inter],
            down_out: vec![0.0; h],
            fused_qkv: vec![0.0; nq * hd + nkv * hd + nkv * hd],
            fused_gate_up: vec![0.0; inter * 2],
        }
    }

    /// Zero all buffers before reuse.
    fn clear(&mut self) {
        self.normed.fill(0.0);
        self.q_all.fill(0.0);
        self.k_all.fill(0.0);
        self.v_all.fill(0.0);
        self.q_normed.fill(0.0);
        self.k_normed.fill(0.0);
        self.q_rope.fill(0.0);
        self.k_rope.fill(0.0);
        self.attn_out.fill(0.0);
        self.head_output.fill(0.0);
        self.attn_proj.fill(0.0);
        self.gate_out.fill(0.0);
        self.up_out.fill(0.0);
        self.swiglu_out.fill(0.0);
        self.down_out.fill(0.0);
        self.fused_qkv.fill(0.0);
        self.fused_gate_up.fill(0.0);
    }
}

/// A single Qwen3 Transformer block.
///
/// Holds references to weight data (zero-copy from GGUF mmap).
pub struct TransformerBlock<'a> {
    /// Layer index (0-based).
    layer_idx: usize,
    /// Pre-attention RMSNorm.
    attn_norm: RmsNorm,
    /// Q projection: [hidden_size → num_heads * head_dim].
    attn_q: Linear1Bit<'a>,
    /// K projection: [hidden_size → num_kv_heads * head_dim].
    attn_k: Linear1Bit<'a>,
    /// V projection: [hidden_size → num_kv_heads * head_dim].
    attn_v: Linear1Bit<'a>,
    /// Output projection: [num_heads * head_dim → hidden_size].
    attn_output: Linear1Bit<'a>,
    /// Per-head QK-norm on Q vectors (shape=[head_dim], shared across all Q heads).
    attn_q_norm: RmsNorm,
    /// Per-head QK-norm on K vectors (shape=[head_dim], shared across all KV heads).
    attn_k_norm: RmsNorm,
    /// Pre-FFN RMSNorm.
    ffn_norm: RmsNorm,
    /// Gate projection: [hidden_size → intermediate_size].
    ffn_gate: Linear1Bit<'a>,
    /// Up projection: [hidden_size → intermediate_size].
    ffn_up: Linear1Bit<'a>,
    /// Down projection: [intermediate_size → hidden_size].
    ffn_down: Linear1Bit<'a>,
    // Architecture parameters
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    /// Fused Q+K+V weight handle (single GPU dispatch).
    fused_qkv_handle: Option<GpuWeightHandle>,
    /// Fused gate+up weight handle (single GPU dispatch).
    fused_gate_up_handle: Option<GpuWeightHandle>,
    /// Pre-allocated scratch buffers (Mutex for Sync safety; uncontended in practice).
    scratch: Mutex<ScratchBuffers>,
}

impl<'a> TransformerBlock<'a> {
    /// Create a new Transformer block from loaded weights.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layer_idx: usize,
        attn_norm: RmsNorm,
        attn_q: Linear1Bit<'a>,
        attn_k: Linear1Bit<'a>,
        attn_v: Linear1Bit<'a>,
        attn_output: Linear1Bit<'a>,
        attn_q_norm: RmsNorm,
        attn_k_norm: RmsNorm,
        ffn_norm: RmsNorm,
        ffn_gate: Linear1Bit<'a>,
        ffn_up: Linear1Bit<'a>,
        ffn_down: Linear1Bit<'a>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_size: usize,
    ) -> Self {
        let inter = ffn_gate.out_features();
        let scratch = Mutex::new(ScratchBuffers::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            inter,
        ));
        Self {
            layer_idx,
            attn_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            attn_q_norm,
            attn_k_norm,
            ffn_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
            fused_qkv_handle: None,
            fused_gate_up_handle: None,
            scratch,
        }
    }

    /// Upload all weight matrices in this block to GPU memory.
    ///
    /// After calling this, all GEMV operations in [`forward`](Self::forward)
    /// will use GPU-resident weight buffers, eliminating per-call
    /// host→device copies.
    pub fn upload_to_gpu(&mut self, kernel: &dyn OneBitKernel) {
        self.attn_q.upload_to_gpu(kernel);
        self.attn_k.upload_to_gpu(kernel);
        self.attn_v.upload_to_gpu(kernel);
        self.attn_output.upload_to_gpu(kernel);
        self.ffn_gate.upload_to_gpu(kernel);
        self.ffn_up.upload_to_gpu(kernel);
        self.ffn_down.upload_to_gpu(kernel);

        // Fused QKV: concatenate Q, K, V blocks into a single weight buffer
        let mut qkv_blocks = Vec::with_capacity(
            self.attn_q.blocks().len() + self.attn_k.blocks().len() + self.attn_v.blocks().len(),
        );
        qkv_blocks.extend_from_slice(self.attn_q.blocks());
        qkv_blocks.extend_from_slice(self.attn_k.blocks());
        qkv_blocks.extend_from_slice(self.attn_v.blocks());
        self.fused_qkv_handle = kernel.upload_weights(&qkv_blocks);

        // Fused gate+up: concatenate gate and up blocks
        let mut gate_up_blocks =
            Vec::with_capacity(self.ffn_gate.blocks().len() + self.ffn_up.blocks().len());
        gate_up_blocks.extend_from_slice(self.ffn_gate.blocks());
        gate_up_blocks.extend_from_slice(self.ffn_up.blocks());
        self.fused_gate_up_handle = kernel.upload_weights(&gate_up_blocks);
    }

    /// Forward pass for a single token at position `pos`.
    ///
    /// - `hidden`: Input/output hidden state `[hidden_size]`. Modified in-place.
    /// - `pos`: Current token position in the sequence.
    /// - `kv_cache`: KV cache to store/retrieve K and V vectors.
    /// - `rope`: Precomputed RoPE table.
    /// - `kernel`: 1-bit kernel dispatcher.
    #[allow(clippy::needless_late_init)]
    #[tracing::instrument(skip_all, fields(layer = self.layer_idx))]
    pub fn forward(
        &self,
        hidden: &mut [f32],
        pos: usize,
        kv_cache: &mut KvCache,
        rope: &RopeTable,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<()> {
        // ═══════════════════════════════════════════════════════
        // Try full-layer GPU dispatch (single command buffer for entire layer)
        // ═══════════════════════════════════════════════════════
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            if let Some(Ok(())) = self.try_full_layer_gpu(hidden, pos, rope, kv_cache) {
                return Ok(());
            }
            // If it returns None (preconditions not met) or Some(Err(..)),
            // fall through to existing CPU/hybrid path.
        }

        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        {
            if let Some(Ok(())) = self.try_full_layer_cuda(hidden, pos, rope, kv_cache) {
                return Ok(());
            }
            // If it returns None (preconditions not met) or Some(Err(..)),
            // fall through to existing CPU/hybrid path.
        }

        let h = self.hidden_size;
        let hd = self.head_dim;
        let nq = self.num_heads;
        let nkv = self.num_kv_heads;
        let heads_per_group = nq / nkv;
        let total_start = Instant::now();

        let mut scratch = self.scratch.lock().map_err(|e| {
            crate::error::ModelError::Internal(format!("scratch lock poisoned: {e}"))
        })?;
        scratch.clear();

        // Destructure into disjoint mutable borrows so the borrow checker allows
        // simultaneous reads/writes to different fields.
        let ScratchBuffers {
            normed,
            q_all,
            k_all,
            v_all,
            q_normed,
            k_normed,
            q_rope,
            k_rope,
            attn_out,
            head_output,
            attn_proj,
            gate_out,
            up_out,
            swiglu_out,
            down_out,
            fused_qkv,
            fused_gate_up,
        } = &mut *scratch;

        let norm_us: u128;
        let qkv_us: u128;
        let qknorm_us: u128;
        let rope_us: u128;
        let cache_us: u128;
        let attn_us: u128;
        let ffn_us: u128;

        // ═══════════════════════════════════════════════════════
        // 1. ATTENTION SUBLAYER
        // ═══════════════════════════════════════════════════════

        // 1a-b. CPU RMSNorm + fused GPU GEMV (faster than GPU batch for 2-dispatch case)
        {
            // Fallback: individual dispatches
            // 1a. Pre-attention RMSNorm
            let norm_start = Instant::now();
            self.attn_norm.forward(hidden, normed)?;
            norm_us = norm_start.elapsed().as_micros();

            // 1b. Q/K/V projections (fused single dispatch when GPU handle available)
            let qkv_start = Instant::now();
            if let Some(fused_handle) = self.fused_qkv_handle {
                let q_rows = nq * hd;
                let k_rows = nkv * hd;
                let total_rows = q_rows + k_rows + k_rows;

                // Try MetalGraph direct dispatch first (single encoder, less overhead)
                #[cfg(all(feature = "metal", target_os = "macos"))]
                let metal_ok = {
                    let q_bytes = blocks_as_bytes(self.attn_q.blocks());
                    let k_bytes = blocks_as_bytes(self.attn_k.blocks());
                    let v_bytes = blocks_as_bytes(self.attn_v.blocks());
                    oxibonsai_kernels::try_metal_qkv(
                        normed,
                        fused_qkv,
                        fused_handle.id(),
                        q_bytes,
                        k_bytes,
                        v_bytes,
                        total_rows,
                        h,
                    )
                    .is_ok()
                };
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                let metal_ok = false;

                // Try native CUDA dispatch when Metal is not available
                #[cfg(all(
                    feature = "native-cuda",
                    not(all(feature = "metal", target_os = "macos")),
                    any(target_os = "linux", target_os = "windows")
                ))]
                let cuda_ok = if !metal_ok {
                    let q_bytes = blocks_as_bytes(self.attn_q.blocks());
                    let k_bytes = blocks_as_bytes(self.attn_k.blocks());
                    let v_bytes = blocks_as_bytes(self.attn_v.blocks());
                    oxibonsai_kernels::try_cuda_qkv(
                        normed,
                        fused_qkv,
                        fused_handle.id(),
                        q_bytes,
                        k_bytes,
                        v_bytes,
                        total_rows,
                        h,
                    )
                    .is_ok()
                } else {
                    false
                };
                #[cfg(not(all(
                    feature = "native-cuda",
                    not(all(feature = "metal", target_os = "macos")),
                    any(target_os = "linux", target_os = "windows")
                )))]
                let cuda_ok = false;

                if !metal_ok && !cuda_ok {
                    kernel.gemv_cached(fused_handle, normed, fused_qkv, total_rows, h)?;
                }
                q_all[..q_rows].copy_from_slice(&fused_qkv[..q_rows]);
                k_all[..k_rows].copy_from_slice(&fused_qkv[q_rows..q_rows + k_rows]);
                v_all[..k_rows].copy_from_slice(&fused_qkv[q_rows + k_rows..total_rows]);
            } else {
                self.attn_q.forward_vec(normed, q_all, kernel)?;
                self.attn_k.forward_vec(normed, k_all, kernel)?;
                self.attn_v.forward_vec(normed, v_all, kernel)?;
            }
            qkv_us = qkv_start.elapsed().as_micros();
        }

        // 1b'. QK-norm: apply per-head RMSNorm to Q and K before RoPE
        let qknorm_start = Instant::now();
        for head in 0..nq {
            let start = head * hd;
            self.attn_q_norm
                .forward(&q_all[start..start + hd], &mut q_normed[start..start + hd])?;
        }
        for head in 0..nkv {
            let start = head * hd;
            self.attn_k_norm
                .forward(&k_all[start..start + hd], &mut k_normed[start..start + hd])?;
        }
        qknorm_us = qknorm_start.elapsed().as_micros();

        // 1c. Apply RoPE to each Q and K head
        let rope_start = Instant::now();
        for head in 0..nq {
            let start = head * hd;
            rope.apply(
                &q_normed[start..start + hd],
                &mut q_rope[start..start + hd],
                pos,
            )?;
        }
        for head in 0..nkv {
            let start = head * hd;
            rope.apply(
                &k_normed[start..start + hd],
                &mut k_rope[start..start + hd],
                pos,
            )?;
        }
        rope_us = rope_start.elapsed().as_micros();

        // 1d. Store K and V in cache
        let cache_start = Instant::now();
        for head in 0..nkv {
            let start = head * hd;
            kv_cache.store_key(self.layer_idx, head, pos, &k_rope[start..start + hd]);
            kv_cache.store_value(self.layer_idx, head, pos, &v_all[start..start + hd]);
        }
        cache_us = cache_start.elapsed().as_micros();

        let seq_len = pos + 1;

        // 1e. GQA attention: 32 Q heads attend to 8 KV heads (4:1)
        let attn_start = Instant::now();
        for q_head in 0..nq {
            let kv_head = q_head / heads_per_group;
            let q_start = q_head * hd;

            let keys = kv_cache.keys_for(self.layer_idx, kv_head, seq_len);
            let values = kv_cache.values_for(self.layer_idx, kv_head, seq_len);

            attention_head(
                &q_rope[q_start..q_start + hd],
                keys,
                values,
                head_output,
                seq_len,
                hd,
            )?;

            attn_out[q_start..q_start + hd].copy_from_slice(head_output);
        }
        attn_us = attn_start.elapsed().as_micros();

        // 1f. Output projection + 1g. Residual + 2. FFN: try direct Metal → batch → CPU
        let ffn_start = Instant::now();
        let did_batch_ffn = if let (
            Some(attn_proj_handle),
            Some(gate_up_handle),
            Some(down_handle),
        ) = (
            self.attn_output.gpu_handle(),
            self.fused_gate_up_handle,
            self.ffn_down.gpu_handle(),
        ) {
            let inter = self.ffn_gate.out_features();

            // Try direct Metal graph dispatch first (single encoder, ~7x less overhead)
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                let attn_proj_blocks = self.attn_output.blocks();
                let gate_blocks = self.ffn_gate.blocks();
                let up_blocks = self.ffn_up.blocks();
                let down_blocks = self.ffn_down.blocks();

                let attn_proj_bytes = blocks_as_bytes(attn_proj_blocks);
                let gate_bytes = blocks_as_bytes(gate_blocks);
                let up_bytes = blocks_as_bytes(up_blocks);
                let down_bytes = blocks_as_bytes(down_blocks);

                let metal_result = oxibonsai_kernels::try_metal_ffn(
                    hidden,
                    attn_out,
                    self.ffn_norm.weight(),
                    self.ffn_norm.eps(),
                    attn_proj_handle.id(),
                    attn_proj_bytes,
                    gate_up_handle.id(),
                    gate_bytes,
                    up_bytes,
                    down_handle.id(),
                    down_bytes,
                    h,
                    inter,
                );

                if metal_result.is_ok() {
                    true
                } else {
                    tracing::warn!(error = ?metal_result.err(), "MetalGraph FFN failed, falling back");
                    // Fall back to scirs2-core batch dispatch
                    kernel.batch_ffn_phase(
                        hidden,
                        attn_out,
                        self.ffn_norm.weight(),
                        self.ffn_norm.eps(),
                        attn_proj_handle,
                        gate_up_handle,
                        down_handle,
                        h,
                        inter,
                        nq * hd,
                    )?
                }
            }

            #[cfg(all(
                feature = "native-cuda",
                not(all(feature = "metal", target_os = "macos")),
                any(target_os = "linux", target_os = "windows")
            ))]
            {
                let attn_proj_blocks = self.attn_output.blocks();
                let gate_blocks = self.ffn_gate.blocks();
                let up_blocks = self.ffn_up.blocks();
                let down_blocks = self.ffn_down.blocks();

                let attn_proj_bytes = blocks_as_bytes(attn_proj_blocks);
                let gate_bytes = blocks_as_bytes(gate_blocks);
                let up_bytes = blocks_as_bytes(up_blocks);
                let down_bytes = blocks_as_bytes(down_blocks);

                let cuda_result = oxibonsai_kernels::try_cuda_ffn(
                    hidden,
                    attn_out,
                    self.ffn_norm.weight(),
                    self.ffn_norm.eps(),
                    attn_proj_handle.id(),
                    attn_proj_bytes,
                    gate_up_handle.id(),
                    gate_bytes,
                    up_bytes,
                    down_handle.id(),
                    down_bytes,
                    h,
                    inter,
                );

                if cuda_result.is_ok() {
                    true
                } else {
                    tracing::warn!(error = ?cuda_result.err(), "CudaGraph FFN failed, falling back");
                    kernel.batch_ffn_phase(
                        hidden,
                        attn_out,
                        self.ffn_norm.weight(),
                        self.ffn_norm.eps(),
                        attn_proj_handle,
                        gate_up_handle,
                        down_handle,
                        h,
                        inter,
                        nq * hd,
                    )?
                }
            }

            #[cfg(not(any(
                all(feature = "metal", target_os = "macos"),
                all(
                    feature = "native-cuda",
                    any(target_os = "linux", target_os = "windows")
                )
            )))]
            {
                kernel.batch_ffn_phase(
                    hidden,
                    attn_out,
                    self.ffn_norm.weight(),
                    self.ffn_norm.eps(),
                    attn_proj_handle,
                    gate_up_handle,
                    down_handle,
                    h,
                    inter,
                    nq * hd,
                )?
            }
        } else {
            false
        };

        if !did_batch_ffn {
            // Fallback: individual dispatches
            // 1f. Output projection
            self.attn_output.forward_vec(attn_out, attn_proj, kernel)?;

            // 1g. Residual connection
            for i in 0..h {
                hidden[i] += attn_proj[i];
            }

            // ═══════════════════════════════════════════════════════
            // 2. FFN (MLP) SUBLAYER
            // ═══════════════════════════════════════════════════════

            // 2a. Pre-FFN RMSNorm
            self.ffn_norm.forward(hidden, normed)?;

            // 2b. Gate and Up projections (fused single dispatch when GPU handle available)
            if let Some(fused_handle) = self.fused_gate_up_handle {
                let inter = gate_out.len();
                let total_rows = inter * 2;
                kernel.gemv_cached(fused_handle, normed, fused_gate_up, total_rows, h)?;
                gate_out[..inter].copy_from_slice(&fused_gate_up[..inter]);
                up_out[..inter].copy_from_slice(&fused_gate_up[inter..total_rows]);
            } else {
                self.ffn_gate.forward_vec(normed, gate_out, kernel)?;
                self.ffn_up.forward_vec(normed, up_out, kernel)?;
            }

            // 2c. SwiGLU activation
            swiglu_fn(gate_out, up_out, swiglu_out);

            // 2d. Down projection
            self.ffn_down.forward_vec(swiglu_out, down_out, kernel)?;

            // 2e. Residual connection
            for i in 0..h {
                hidden[i] += down_out[i];
            }
        }
        ffn_us = ffn_start.elapsed().as_micros();

        let total_us = total_start.elapsed().as_micros();
        tracing::debug!(
            target: "block_profile",
            "L{layer}: norm={norm_us}µs qkv={qkv_us}µs qknorm={qknorm_us}µs rope={rope_us}µs cache={cache_us}µs attn={attn_us}µs ffn={ffn_us}µs total={total_us}µs",
            layer = self.layer_idx,
        );

        Ok(())
    }

    /// Forward pass with timing statistics collection.
    ///
    /// Same computation as `forward`, but records per-phase timing.
    #[tracing::instrument(skip_all, fields(layer = self.layer_idx))]
    pub fn forward_with_stats(
        &self,
        hidden: &mut [f32],
        pos: usize,
        kv_cache: &mut KvCache,
        rope: &RopeTable,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<LayerStats> {
        let total_start = Instant::now();
        let mut stats = LayerStats::new(self.layer_idx);

        let h = self.hidden_size;
        let hd = self.head_dim;
        let nq = self.num_heads;
        let nkv = self.num_kv_heads;
        let heads_per_group = nq / nkv;

        let mut scratch = self.scratch.lock().map_err(|e| {
            crate::error::ModelError::Internal(format!("scratch lock poisoned: {e}"))
        })?;
        scratch.clear();

        let ScratchBuffers {
            normed,
            q_all,
            k_all,
            v_all,
            q_normed,
            k_normed,
            q_rope,
            k_rope,
            attn_out,
            head_output,
            attn_proj,
            gate_out,
            up_out,
            swiglu_out,
            down_out,
            fused_qkv,
            fused_gate_up,
        } = &mut *scratch;

        // ═══════════════════════════════════════════════════════
        // 1. ATTENTION SUBLAYER
        // ═══════════════════════════════════════════════════════

        // 1a-b. Norm + Q/K/V projections
        let proj_start = Instant::now();

        // Try batch: RMSNorm + QKV in one Metal command buffer
        let batch_qkv = if let Some(fused_handle) = self.fused_qkv_handle {
            kernel.batch_attn_phase(
                hidden,
                self.attn_norm.weight(),
                self.attn_norm.eps(),
                fused_handle,
                nq * hd,
                nkv * hd,
                h,
            )?
        } else {
            None
        };

        if let Some((q_data, k_data, v_data)) = batch_qkv {
            q_all[..nq * hd].copy_from_slice(&q_data);
            k_all[..nkv * hd].copy_from_slice(&k_data);
            v_all[..nkv * hd].copy_from_slice(&v_data);
        } else {
            self.attn_norm.forward(hidden, normed)?;

            if let Some(fused_handle) = self.fused_qkv_handle {
                let q_rows = nq * hd;
                let k_rows = nkv * hd;
                let total_rows = q_rows + k_rows + k_rows;

                #[cfg(all(feature = "metal", target_os = "macos"))]
                let metal_ok = {
                    let q_bytes = blocks_as_bytes(self.attn_q.blocks());
                    let k_bytes = blocks_as_bytes(self.attn_k.blocks());
                    let v_bytes = blocks_as_bytes(self.attn_v.blocks());
                    oxibonsai_kernels::try_metal_qkv(
                        normed,
                        fused_qkv,
                        fused_handle.id(),
                        q_bytes,
                        k_bytes,
                        v_bytes,
                        total_rows,
                        h,
                    )
                    .is_ok()
                };
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                let metal_ok = false;

                if !metal_ok {
                    kernel.gemv_cached(fused_handle, normed, fused_qkv, total_rows, h)?;
                }
                q_all[..q_rows].copy_from_slice(&fused_qkv[..q_rows]);
                k_all[..k_rows].copy_from_slice(&fused_qkv[q_rows..q_rows + k_rows]);
                v_all[..k_rows].copy_from_slice(&fused_qkv[q_rows + k_rows..total_rows]);
            } else {
                self.attn_q.forward_vec(normed, q_all, kernel)?;
                self.attn_k.forward_vec(normed, k_all, kernel)?;
                self.attn_v.forward_vec(normed, v_all, kernel)?;
            }
        }

        // QK-norm: apply per-head RMSNorm to Q and K
        for head in 0..nq {
            let start = head * hd;
            self.attn_q_norm
                .forward(&q_all[start..start + hd], &mut q_normed[start..start + hd])?;
        }
        for head in 0..nkv {
            let start = head * hd;
            self.attn_k_norm
                .forward(&k_all[start..start + hd], &mut k_normed[start..start + hd])?;
        }
        stats.projection_us = proj_start.elapsed().as_micros() as u64;

        // 1c. Apply RoPE
        let rope_start = Instant::now();
        for head in 0..nq {
            let start = head * hd;
            rope.apply(
                &q_normed[start..start + hd],
                &mut q_rope[start..start + hd],
                pos,
            )?;
        }
        for head in 0..nkv {
            let start = head * hd;
            rope.apply(
                &k_normed[start..start + hd],
                &mut k_rope[start..start + hd],
                pos,
            )?;
        }
        stats.rope_us = rope_start.elapsed().as_micros() as u64;

        // 1d-f. Store KV, compute attention, output projection
        let attn_start = Instant::now();
        for head in 0..nkv {
            let start = head * hd;
            kv_cache.store_key(self.layer_idx, head, pos, &k_rope[start..start + hd]);
            kv_cache.store_value(self.layer_idx, head, pos, &v_all[start..start + hd]);
        }

        let seq_len = pos + 1;

        for q_head in 0..nq {
            let kv_head = q_head / heads_per_group;
            let q_start = q_head * hd;

            let keys = kv_cache.keys_for(self.layer_idx, kv_head, seq_len);
            let values = kv_cache.values_for(self.layer_idx, kv_head, seq_len);

            attention_head(
                &q_rope[q_start..q_start + hd],
                keys,
                values,
                head_output,
                seq_len,
                hd,
            )?;

            attn_out[q_start..q_start + hd].copy_from_slice(head_output);
        }

        // 1f + 2. Try batch FFN path: direct Metal → batch → CPU
        let did_batch_ffn =
            if let (Some(attn_proj_handle), Some(gate_up_handle), Some(down_handle)) = (
                self.attn_output.gpu_handle(),
                self.fused_gate_up_handle,
                self.ffn_down.gpu_handle(),
            ) {
                let inter = self.ffn_gate.out_features();

                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    let attn_proj_blocks = self.attn_output.blocks();
                    let gate_blocks = self.ffn_gate.blocks();
                    let up_blocks = self.ffn_up.blocks();
                    let down_blocks = self.ffn_down.blocks();

                    let attn_proj_bytes = blocks_as_bytes(attn_proj_blocks);
                    let gate_bytes = blocks_as_bytes(gate_blocks);
                    let up_bytes = blocks_as_bytes(up_blocks);
                    let down_bytes = blocks_as_bytes(down_blocks);

                    let metal_result = oxibonsai_kernels::try_metal_ffn(
                        hidden,
                        attn_out,
                        self.ffn_norm.weight(),
                        self.ffn_norm.eps(),
                        attn_proj_handle.id(),
                        attn_proj_bytes,
                        gate_up_handle.id(),
                        gate_bytes,
                        up_bytes,
                        down_handle.id(),
                        down_bytes,
                        h,
                        inter,
                    );

                    if metal_result.is_ok() {
                        true
                    } else {
                        kernel.batch_ffn_phase(
                            hidden,
                            attn_out,
                            self.ffn_norm.weight(),
                            self.ffn_norm.eps(),
                            attn_proj_handle,
                            gate_up_handle,
                            down_handle,
                            h,
                            inter,
                            nq * hd,
                        )?
                    }
                }

                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    kernel.batch_ffn_phase(
                        hidden,
                        attn_out,
                        self.ffn_norm.weight(),
                        self.ffn_norm.eps(),
                        attn_proj_handle,
                        gate_up_handle,
                        down_handle,
                        h,
                        inter,
                        nq * hd,
                    )?
                }
            } else {
                false
            };

        if !did_batch_ffn {
            self.attn_output.forward_vec(attn_out, attn_proj, kernel)?;

            for i in 0..h {
                hidden[i] += attn_proj[i];
            }
        }
        stats.attention_us = attn_start.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════
        // 2. FFN (MLP) SUBLAYER
        // ═══════════════════════════════════════════════════════

        let ffn_start = Instant::now();

        if !did_batch_ffn {
            self.ffn_norm.forward(hidden, normed)?;

            if let Some(fused_handle) = self.fused_gate_up_handle {
                let inter = gate_out.len();
                let total_rows = inter * 2;
                kernel.gemv_cached(fused_handle, normed, fused_gate_up, total_rows, h)?;
                gate_out[..inter].copy_from_slice(&fused_gate_up[..inter]);
                up_out[..inter].copy_from_slice(&fused_gate_up[inter..total_rows]);
            } else {
                self.ffn_gate.forward_vec(normed, gate_out, kernel)?;
                self.ffn_up.forward_vec(normed, up_out, kernel)?;
            }

            swiglu_fn(gate_out, up_out, swiglu_out);

            self.ffn_down.forward_vec(swiglu_out, down_out, kernel)?;

            for i in 0..h {
                hidden[i] += down_out[i];
            }
        }
        stats.ffn_us = ffn_start.elapsed().as_micros() as u64;

        stats.total_us = total_start.elapsed().as_micros() as u64;
        Ok(stats)
    }

    /// Forward pass with optional sliding window attention.
    ///
    /// When `sliding_window` is `Some`, attention is restricted to positions
    /// within the window, reducing compute for long sequences.
    #[tracing::instrument(skip_all, fields(layer = self.layer_idx))]
    pub fn forward_with_sliding_window(
        &self,
        hidden: &mut [f32],
        pos: usize,
        kv_cache: &mut KvCache,
        rope: &RopeTable,
        kernel: &dyn OneBitKernel,
        sliding_window: Option<&SlidingWindowConfig>,
    ) -> ModelResult<()> {
        let h = self.hidden_size;
        let hd = self.head_dim;
        let nq = self.num_heads;
        let nkv = self.num_kv_heads;
        let heads_per_group = nq / nkv;

        let mut scratch = self.scratch.lock().map_err(|e| {
            crate::error::ModelError::Internal(format!("scratch lock poisoned: {e}"))
        })?;
        scratch.clear();

        let ScratchBuffers {
            normed,
            q_all,
            k_all,
            v_all,
            q_normed,
            k_normed,
            q_rope,
            k_rope,
            attn_out,
            head_output,
            attn_proj,
            gate_out,
            up_out,
            swiglu_out,
            down_out,
            fused_qkv,
            fused_gate_up,
        } = &mut *scratch;

        // 1. Try batch: RMSNorm + QKV in one Metal command buffer
        let batch_qkv = if let Some(fused_handle) = self.fused_qkv_handle {
            kernel.batch_attn_phase(
                hidden,
                self.attn_norm.weight(),
                self.attn_norm.eps(),
                fused_handle,
                nq * hd,
                nkv * hd,
                h,
            )?
        } else {
            None
        };

        if let Some((q_data, k_data, v_data)) = batch_qkv {
            q_all[..nq * hd].copy_from_slice(&q_data);
            k_all[..nkv * hd].copy_from_slice(&k_data);
            v_all[..nkv * hd].copy_from_slice(&v_data);
        } else {
            // Fallback: individual dispatches
            self.attn_norm.forward(hidden, normed)?;

            if let Some(fused_handle) = self.fused_qkv_handle {
                let q_rows = nq * hd;
                let k_rows = nkv * hd;
                let total_rows = q_rows + k_rows + k_rows;

                #[cfg(all(feature = "metal", target_os = "macos"))]
                let metal_ok = {
                    let q_bytes = blocks_as_bytes(self.attn_q.blocks());
                    let k_bytes = blocks_as_bytes(self.attn_k.blocks());
                    let v_bytes = blocks_as_bytes(self.attn_v.blocks());
                    oxibonsai_kernels::try_metal_qkv(
                        normed,
                        fused_qkv,
                        fused_handle.id(),
                        q_bytes,
                        k_bytes,
                        v_bytes,
                        total_rows,
                        h,
                    )
                    .is_ok()
                };
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                let metal_ok = false;

                if !metal_ok {
                    kernel.gemv_cached(fused_handle, normed, fused_qkv, total_rows, h)?;
                }
                q_all[..q_rows].copy_from_slice(&fused_qkv[..q_rows]);
                k_all[..k_rows].copy_from_slice(&fused_qkv[q_rows..q_rows + k_rows]);
                v_all[..k_rows].copy_from_slice(&fused_qkv[q_rows + k_rows..total_rows]);
            } else {
                self.attn_q.forward_vec(normed, q_all, kernel)?;
                self.attn_k.forward_vec(normed, k_all, kernel)?;
                self.attn_v.forward_vec(normed, v_all, kernel)?;
            }
        }

        // 2'. QK-norm: apply per-head RMSNorm to Q and K before RoPE
        for head in 0..nq {
            let start = head * hd;
            self.attn_q_norm
                .forward(&q_all[start..start + hd], &mut q_normed[start..start + hd])?;
        }
        for head in 0..nkv {
            let start = head * hd;
            self.attn_k_norm
                .forward(&k_all[start..start + hd], &mut k_normed[start..start + hd])?;
        }

        // 3. RoPE
        for head in 0..nq {
            let start = head * hd;
            rope.apply(
                &q_normed[start..start + hd],
                &mut q_rope[start..start + hd],
                pos,
            )?;
        }
        for head in 0..nkv {
            let start = head * hd;
            rope.apply(
                &k_normed[start..start + hd],
                &mut k_rope[start..start + hd],
                pos,
            )?;
        }

        // 4. Store KV
        for head in 0..nkv {
            let start = head * hd;
            kv_cache.store_key(self.layer_idx, head, pos, &k_rope[start..start + hd]);
            kv_cache.store_value(self.layer_idx, head, pos, &v_all[start..start + hd]);
        }

        let full_seq_len = pos + 1;

        // 5. GQA attention with optional sliding window
        if let Some(sw_config) = sliding_window {
            let (positions, _count) =
                crate::layers::sliding_window::attention_range(pos, full_seq_len, sw_config);

            for q_head in 0..nq {
                let kv_head = q_head / heads_per_group;
                let q_start = q_head * hd;

                let all_keys = kv_cache.keys_for(self.layer_idx, kv_head, full_seq_len);
                let all_values = kv_cache.values_for(self.layer_idx, kv_head, full_seq_len);

                let windowed_keys: Vec<f32> = positions
                    .iter()
                    .flat_map(|&p| all_keys[p * hd..(p + 1) * hd].iter().copied())
                    .collect();
                let windowed_values: Vec<f32> = positions
                    .iter()
                    .flat_map(|&p| all_values[p * hd..(p + 1) * hd].iter().copied())
                    .collect();

                attention_head(
                    &q_rope[q_start..q_start + hd],
                    &windowed_keys,
                    &windowed_values,
                    head_output,
                    positions.len(),
                    hd,
                )?;

                attn_out[q_start..q_start + hd].copy_from_slice(head_output);
            }
        } else {
            for q_head in 0..nq {
                let kv_head = q_head / heads_per_group;
                let q_start = q_head * hd;

                let keys = kv_cache.keys_for(self.layer_idx, kv_head, full_seq_len);
                let values = kv_cache.values_for(self.layer_idx, kv_head, full_seq_len);

                attention_head(
                    &q_rope[q_start..q_start + hd],
                    keys,
                    values,
                    head_output,
                    full_seq_len,
                    hd,
                )?;

                attn_out[q_start..q_start + hd].copy_from_slice(head_output);
            }
        }

        // 6. Output projection + residual + 7. FFN: try direct Metal → batch → CPU
        let did_batch_ffn =
            if let (Some(attn_proj_handle), Some(gate_up_handle), Some(down_handle)) = (
                self.attn_output.gpu_handle(),
                self.fused_gate_up_handle,
                self.ffn_down.gpu_handle(),
            ) {
                let inter = self.ffn_gate.out_features();

                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    let attn_proj_blocks = self.attn_output.blocks();
                    let gate_blocks = self.ffn_gate.blocks();
                    let up_blocks = self.ffn_up.blocks();
                    let down_blocks = self.ffn_down.blocks();

                    let attn_proj_bytes = blocks_as_bytes(attn_proj_blocks);
                    let gate_bytes = blocks_as_bytes(gate_blocks);
                    let up_bytes = blocks_as_bytes(up_blocks);
                    let down_bytes = blocks_as_bytes(down_blocks);

                    let metal_result = oxibonsai_kernels::try_metal_ffn(
                        hidden,
                        attn_out,
                        self.ffn_norm.weight(),
                        self.ffn_norm.eps(),
                        attn_proj_handle.id(),
                        attn_proj_bytes,
                        gate_up_handle.id(),
                        gate_bytes,
                        up_bytes,
                        down_handle.id(),
                        down_bytes,
                        h,
                        inter,
                    );

                    if metal_result.is_ok() {
                        true
                    } else {
                        kernel.batch_ffn_phase(
                            hidden,
                            attn_out,
                            self.ffn_norm.weight(),
                            self.ffn_norm.eps(),
                            attn_proj_handle,
                            gate_up_handle,
                            down_handle,
                            h,
                            inter,
                            nq * hd,
                        )?
                    }
                }

                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    kernel.batch_ffn_phase(
                        hidden,
                        attn_out,
                        self.ffn_norm.weight(),
                        self.ffn_norm.eps(),
                        attn_proj_handle,
                        gate_up_handle,
                        down_handle,
                        h,
                        inter,
                        nq * hd,
                    )?
                }
            } else {
                false
            };

        if !did_batch_ffn {
            // Fallback: individual dispatches
            self.attn_output.forward_vec(attn_out, attn_proj, kernel)?;

            for i in 0..h {
                hidden[i] += attn_proj[i];
            }

            // 7. FFN sublayer
            self.ffn_norm.forward(hidden, normed)?;

            if let Some(fused_handle) = self.fused_gate_up_handle {
                let inter = gate_out.len();
                let total_rows = inter * 2;
                kernel.gemv_cached(fused_handle, normed, fused_gate_up, total_rows, h)?;
                gate_out[..inter].copy_from_slice(&fused_gate_up[..inter]);
                up_out[..inter].copy_from_slice(&fused_gate_up[inter..total_rows]);
            } else {
                self.ffn_gate.forward_vec(normed, gate_out, kernel)?;
                self.ffn_up.forward_vec(normed, up_out, kernel)?;
            }

            swiglu_fn(gate_out, up_out, swiglu_out);

            self.ffn_down.forward_vec(swiglu_out, down_out, kernel)?;

            for i in 0..h {
                hidden[i] += down_out[i];
            }
        }

        Ok(())
    }

    /// Get the layer index.
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Attempt full-layer GPU dispatch (attention + FFN in a single command buffer).
    ///
    /// Returns:
    /// - `Some(Ok(()))` if the full layer was successfully computed on GPU.
    /// - `Some(Err(..))` if GPU dispatch was attempted but failed.
    /// - `None` if preconditions are not met (handles not available).
    ///
    /// On `Some(Ok(()))`, `hidden` is modified in-place and the caller should
    /// return early, skipping the CPU path entirely. The GPU manages its own
    /// KV cache internally.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn try_full_layer_gpu(
        &self,
        hidden: &mut [f32],
        pos: usize,
        rope: &RopeTable,
        kv_cache: &KvCache,
    ) -> Option<ModelResult<()>> {
        // Check that all required GPU handles are available.
        let fused_qkv_handle = self.fused_qkv_handle?;
        let attn_proj_handle = self.attn_output.gpu_handle()?;
        let fused_gate_up_handle = self.fused_gate_up_handle?;
        let down_handle = self.ffn_down.gpu_handle()?;

        let h = self.hidden_size;
        let hd = self.head_dim;
        let nq = self.num_heads;
        let nkv = self.num_kv_heads;
        let inter = self.ffn_gate.out_features();
        let eps = self.attn_norm.eps();
        let n_layers = kv_cache.num_layers();
        let max_seq_len = kv_cache.max_seq_len();

        // Stable, unique handle IDs for f32 norm weights (outside Q1 handle range).
        let norm_handle_base = 1_000_000u64 + (self.layer_idx as u64) * 10;
        let attn_norm_handle_id = norm_handle_base;
        let q_norm_handle_id = norm_handle_base + 1;
        let k_norm_handle_id = norm_handle_base + 2;
        let ffn_norm_handle_id = norm_handle_base + 3;

        // Gather Q1 block bytes for weight matrices.
        let fused_qkv_bytes = blocks_as_bytes(self.attn_q.blocks());
        let fused_qkv_k_bytes = blocks_as_bytes(self.attn_k.blocks());
        let fused_qkv_v_bytes = blocks_as_bytes(self.attn_v.blocks());

        // Concatenate Q+K+V bytes for the fused QKV weight.
        let mut qkv_concat = Vec::with_capacity(
            fused_qkv_bytes.len() + fused_qkv_k_bytes.len() + fused_qkv_v_bytes.len(),
        );
        qkv_concat.extend_from_slice(fused_qkv_bytes);
        qkv_concat.extend_from_slice(fused_qkv_k_bytes);
        qkv_concat.extend_from_slice(fused_qkv_v_bytes);

        let attn_proj_bytes = blocks_as_bytes(self.attn_output.blocks());
        let gate_bytes = blocks_as_bytes(self.ffn_gate.blocks());
        let up_bytes = blocks_as_bytes(self.ffn_up.blocks());
        let down_bytes = blocks_as_bytes(self.ffn_down.blocks());

        // RoPE cos/sin for this position.
        let rope_cos = rope.cos_at(pos);
        let rope_sin = rope.sin_at(pos);

        let result = oxibonsai_kernels::try_metal_full_layer(
            hidden,
            pos,
            self.layer_idx,
            attn_norm_handle_id,
            self.attn_norm.weight(),
            fused_qkv_handle.id(),
            &qkv_concat,
            q_norm_handle_id,
            self.attn_q_norm.weight(),
            k_norm_handle_id,
            self.attn_k_norm.weight(),
            attn_proj_handle.id(),
            attn_proj_bytes,
            ffn_norm_handle_id,
            self.ffn_norm.weight(),
            fused_gate_up_handle.id(),
            gate_bytes,
            up_bytes,
            down_handle.id(),
            down_bytes,
            rope_cos,
            rope_sin,
            h,
            inter,
            nq,
            nkv,
            hd,
            eps,
            max_seq_len,
            n_layers,
        );

        match result {
            Ok(()) => {
                tracing::debug!(
                    target: "block_profile",
                    "L{layer}: full-layer GPU dispatch OK",
                    layer = self.layer_idx,
                );
                Some(Ok(()))
            }
            Err(e) => {
                tracing::warn!(
                    layer = self.layer_idx,
                    error = %e,
                    "full-layer GPU dispatch failed, falling back to CPU path"
                );
                Some(Err(crate::error::ModelError::Internal(format!(
                    "Metal full-layer dispatch failed: {e}"
                ))))
            }
        }
    }

    /// Attempt full-layer CUDA GPU dispatch (attention + FFN, no intermediate
    /// CPU round-trips between the two sublayers).
    ///
    /// Returns:
    /// - `Some(Ok(()))` if the full layer was successfully computed on GPU.
    /// - `Some(Err(..))` if GPU dispatch was attempted but failed.
    /// - `None` if preconditions are not met (handles not available).
    #[cfg(all(
        feature = "native-cuda",
        not(all(feature = "metal", target_os = "macos")),
        any(target_os = "linux", target_os = "windows")
    ))]
    fn try_full_layer_cuda(
        &self,
        hidden: &mut [f32],
        pos: usize,
        rope: &RopeTable,
        kv_cache: &KvCache,
    ) -> Option<ModelResult<()>> {
        // Check that all required GPU handles are available.
        let fused_qkv_handle = self.fused_qkv_handle?;
        let attn_proj_handle = self.attn_output.gpu_handle()?;
        let fused_gate_up_handle = self.fused_gate_up_handle?;
        let down_handle = self.ffn_down.gpu_handle()?;

        let h = self.hidden_size;
        let hd = self.head_dim;
        let nq = self.num_heads;
        let nkv = self.num_kv_heads;
        let heads_per_group = nq / nkv;
        let inter = self.ffn_gate.out_features();
        let eps = self.attn_norm.eps();
        let n_layers = kv_cache.num_layers();
        let max_seq_len = kv_cache.max_seq_len();

        // Stable, unique handle IDs for f32 norm weights (outside Q1 handle range).
        // Use an offset of 2_000_000 to avoid collisions with the Metal namespace.
        let norm_handle_base = 2_000_000u64 + (self.layer_idx as u64) * 10;
        let attn_norm_handle_id = norm_handle_base;
        let q_norm_handle_id = norm_handle_base + 1;
        let k_norm_handle_id = norm_handle_base + 2;
        let ffn_norm_handle_id = norm_handle_base + 3;

        // Gather Q1 block bytes for weight matrices.
        let fused_qkv_bytes = blocks_as_bytes(self.attn_q.blocks());
        let fused_qkv_k_bytes = blocks_as_bytes(self.attn_k.blocks());
        let fused_qkv_v_bytes = blocks_as_bytes(self.attn_v.blocks());

        // Concatenate Q+K+V bytes for the fused QKV weight.
        let mut qkv_concat = Vec::with_capacity(
            fused_qkv_bytes.len() + fused_qkv_k_bytes.len() + fused_qkv_v_bytes.len(),
        );
        qkv_concat.extend_from_slice(fused_qkv_bytes);
        qkv_concat.extend_from_slice(fused_qkv_k_bytes);
        qkv_concat.extend_from_slice(fused_qkv_v_bytes);

        let attn_proj_bytes = blocks_as_bytes(self.attn_output.blocks());
        let gate_bytes = blocks_as_bytes(self.ffn_gate.blocks());
        let up_bytes = blocks_as_bytes(self.ffn_up.blocks());
        let down_bytes = blocks_as_bytes(self.ffn_down.blocks());

        // RoPE cos/sin for this position.
        let rope_cos = rope.cos_at(pos);
        let rope_sin = rope.sin_at(pos);

        let result = oxibonsai_kernels::try_cuda_full_layer(
            hidden,
            pos,
            self.layer_idx,
            attn_norm_handle_id,
            self.attn_norm.weight(),
            fused_qkv_handle.id(),
            &qkv_concat,
            attn_proj_handle.id(),
            attn_proj_bytes,
            q_norm_handle_id,
            self.attn_q_norm.weight(),
            k_norm_handle_id,
            self.attn_k_norm.weight(),
            ffn_norm_handle_id,
            self.ffn_norm.weight(),
            fused_gate_up_handle.id(),
            gate_bytes,
            up_bytes,
            down_handle.id(),
            down_bytes,
            rope_cos,
            rope_sin,
            h,
            inter,
            nq,
            nkv,
            hd,
            heads_per_group,
            eps,
            max_seq_len,
            n_layers,
        );

        match result {
            Ok(()) => {
                tracing::debug!(
                    target: "block_profile",
                    "L{layer}: CUDA full-layer dispatch OK",
                    layer = self.layer_idx,
                );
                Some(Ok(()))
            }
            Err(e) => {
                tracing::warn!(
                    layer = self.layer_idx,
                    error = %e,
                    "CUDA full-layer dispatch failed, falling back to CPU path"
                );
                Some(Err(crate::error::ModelError::Internal(format!(
                    "CUDA full-layer dispatch failed: {e}"
                ))))
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Accessors for full-forward GPU path (model.rs integration)
    // ─────────────────────────────────────────────────────────────────────

    /// Attention norm weight slice.
    pub fn attn_norm_weight(&self) -> &[f32] {
        self.attn_norm.weight()
    }

    /// Attention norm epsilon.
    pub fn attn_norm_eps(&self) -> f32 {
        self.attn_norm.eps()
    }

    /// Q-norm weight slice.
    pub fn q_norm_weight(&self) -> &[f32] {
        self.attn_q_norm.weight()
    }

    /// K-norm weight slice.
    pub fn k_norm_weight(&self) -> &[f32] {
        self.attn_k_norm.weight()
    }

    /// FFN norm weight slice.
    pub fn ffn_norm_weight(&self) -> &[f32] {
        self.ffn_norm.weight()
    }

    /// Layer index.
    pub fn layer_index(&self) -> usize {
        self.layer_idx
    }

    /// Fused QKV GPU handle (if uploaded).
    pub fn fused_qkv_gpu_handle(&self) -> Option<GpuWeightHandle> {
        self.fused_qkv_handle
    }

    /// Attention output projection GPU handle (if uploaded).
    pub fn attn_output_gpu_handle(&self) -> Option<GpuWeightHandle> {
        self.attn_output.gpu_handle()
    }

    /// Fused gate+up GPU handle (if uploaded).
    pub fn fused_gate_up_gpu_handle(&self) -> Option<GpuWeightHandle> {
        self.fused_gate_up_handle
    }

    /// FFN down projection GPU handle (if uploaded).
    pub fn ffn_down_gpu_handle(&self) -> Option<GpuWeightHandle> {
        self.ffn_down.gpu_handle()
    }

    /// Q projection block slice.
    pub fn attn_q_blocks(&self) -> &[oxibonsai_core::BlockQ1_0G128] {
        self.attn_q.blocks()
    }

    /// K projection block slice.
    pub fn attn_k_blocks(&self) -> &[oxibonsai_core::BlockQ1_0G128] {
        self.attn_k.blocks()
    }

    /// V projection block slice.
    pub fn attn_v_blocks(&self) -> &[oxibonsai_core::BlockQ1_0G128] {
        self.attn_v.blocks()
    }

    /// Output projection block slice.
    pub fn attn_output_blocks(&self) -> &[oxibonsai_core::BlockQ1_0G128] {
        self.attn_output.blocks()
    }

    /// FFN gate block slice.
    pub fn ffn_gate_blocks(&self) -> &[oxibonsai_core::BlockQ1_0G128] {
        self.ffn_gate.blocks()
    }

    /// FFN up block slice.
    pub fn ffn_up_blocks(&self) -> &[oxibonsai_core::BlockQ1_0G128] {
        self.ffn_up.blocks()
    }

    /// FFN down block slice.
    pub fn ffn_down_blocks(&self) -> &[oxibonsai_core::BlockQ1_0G128] {
        self.ffn_down.blocks()
    }

    /// FFN gate output features (intermediate_size).
    pub fn ffn_gate_out_features(&self) -> usize {
        self.ffn_gate.out_features()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use oxibonsai_core::tensor::BlockQ1_0G128;

    fn make_blocks(n: usize, scale: f32, pattern: u8) -> Vec<BlockQ1_0G128> {
        (0..n)
            .map(|_| BlockQ1_0G128 {
                d: f16::from_f32(scale),
                qs: [pattern; 16],
            })
            .collect()
    }

    #[test]
    fn transformer_block_smoke_test() {
        // Minimal block: hidden=128, heads=2, kv_heads=1, head_dim=64, intermediate=256
        let h = 128;
        let hd = 64;
        let nq = 2;
        let nkv = 1;
        let inter = 256;
        let blocks_per_row = h / 128; // = 1

        // All-positive weights (0xFF = all bits set = all +scale)
        let q_blocks = make_blocks(nq * hd * blocks_per_row, 0.01, 0xFF);
        let k_blocks = make_blocks(nkv * hd * blocks_per_row, 0.01, 0xFF);
        let v_blocks = make_blocks(nkv * hd * blocks_per_row, 0.01, 0xFF);
        let o_blocks = make_blocks(h * blocks_per_row, 0.01, 0xFF);
        let gate_blocks = make_blocks(inter * blocks_per_row, 0.01, 0xFF);
        let up_blocks = make_blocks(inter * blocks_per_row, 0.01, 0xFF);
        let down_blocks = make_blocks(h * (inter / 128), 0.01, 0xFF);

        let block = TransformerBlock::new(
            0,
            RmsNorm::new(vec![1.0; h], 1e-6),
            Linear1Bit::new(&q_blocks, nq * hd, h),
            Linear1Bit::new(&k_blocks, nkv * hd, h),
            Linear1Bit::new(&v_blocks, nkv * hd, h),
            Linear1Bit::new(&o_blocks, h, nq * hd),
            RmsNorm::new(vec![1.0; hd], 1e-6),
            RmsNorm::new(vec![1.0; hd], 1e-6),
            RmsNorm::new(vec![1.0; h], 1e-6),
            Linear1Bit::new(&gate_blocks, inter, h),
            Linear1Bit::new(&up_blocks, inter, h),
            Linear1Bit::new(&down_blocks, h, inter),
            nq,
            nkv,
            hd,
            h,
        );

        let rope = RopeTable::new(hd, 16, 10000.0);
        let kernel = oxibonsai_kernels::KernelDispatcher::auto_detect();
        let mut kv_cache = KvCache::new(1, nkv, hd, 16);

        // Use non-uniform input to avoid cancellation
        let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let original = hidden.clone();
        block
            .forward(&mut hidden, 0, &mut kv_cache, &rope, &kernel)
            .expect("block forward should succeed");

        // After forward, hidden should differ from original
        let max_diff = hidden
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "forward should modify hidden state, max_diff={max_diff}"
        );
    }

    #[test]
    fn forward_with_stats_returns_timing() {
        let h = 128;
        let hd = 64;
        let nq = 2;
        let nkv = 1;
        let inter = 256;
        let blocks_per_row = h / 128;

        let q_blocks = make_blocks(nq * hd * blocks_per_row, 0.01, 0xFF);
        let k_blocks = make_blocks(nkv * hd * blocks_per_row, 0.01, 0xFF);
        let v_blocks = make_blocks(nkv * hd * blocks_per_row, 0.01, 0xFF);
        let o_blocks = make_blocks(h * blocks_per_row, 0.01, 0xFF);
        let gate_blocks = make_blocks(inter * blocks_per_row, 0.01, 0xFF);
        let up_blocks = make_blocks(inter * blocks_per_row, 0.01, 0xFF);
        let down_blocks = make_blocks(h * (inter / 128), 0.01, 0xFF);

        let block = TransformerBlock::new(
            0,
            RmsNorm::new(vec![1.0; h], 1e-6),
            Linear1Bit::new(&q_blocks, nq * hd, h),
            Linear1Bit::new(&k_blocks, nkv * hd, h),
            Linear1Bit::new(&v_blocks, nkv * hd, h),
            Linear1Bit::new(&o_blocks, h, nq * hd),
            RmsNorm::new(vec![1.0; hd], 1e-6),
            RmsNorm::new(vec![1.0; hd], 1e-6),
            RmsNorm::new(vec![1.0; h], 1e-6),
            Linear1Bit::new(&gate_blocks, inter, h),
            Linear1Bit::new(&up_blocks, inter, h),
            Linear1Bit::new(&down_blocks, h, inter),
            nq,
            nkv,
            hd,
            h,
        );

        let rope = RopeTable::new(hd, 16, 10000.0);
        let kernel = oxibonsai_kernels::KernelDispatcher::auto_detect();
        let mut kv_cache = KvCache::new(1, nkv, hd, 16);
        let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.01).collect();

        let stats = block
            .forward_with_stats(&mut hidden, 0, &mut kv_cache, &rope, &kernel)
            .expect("forward_with_stats should succeed");

        assert_eq!(stats.layer_idx, 0);
        // total_us should be non-negative (could be 0 on very fast runs)
        assert!(stats.total_us >= stats.projection_us.min(stats.attention_us));
    }

    #[test]
    fn forward_with_sliding_window_smoke() {
        let h = 128;
        let hd = 64;
        let nq = 2;
        let nkv = 1;
        let inter = 256;
        let blocks_per_row = h / 128;

        let q_blocks = make_blocks(nq * hd * blocks_per_row, 0.01, 0xFF);
        let k_blocks = make_blocks(nkv * hd * blocks_per_row, 0.01, 0xFF);
        let v_blocks = make_blocks(nkv * hd * blocks_per_row, 0.01, 0xFF);
        let o_blocks = make_blocks(h * blocks_per_row, 0.01, 0xFF);
        let gate_blocks = make_blocks(inter * blocks_per_row, 0.01, 0xFF);
        let up_blocks = make_blocks(inter * blocks_per_row, 0.01, 0xFF);
        let down_blocks = make_blocks(h * (inter / 128), 0.01, 0xFF);

        let block = TransformerBlock::new(
            0,
            RmsNorm::new(vec![1.0; h], 1e-6),
            Linear1Bit::new(&q_blocks, nq * hd, h),
            Linear1Bit::new(&k_blocks, nkv * hd, h),
            Linear1Bit::new(&v_blocks, nkv * hd, h),
            Linear1Bit::new(&o_blocks, h, nq * hd),
            RmsNorm::new(vec![1.0; hd], 1e-6),
            RmsNorm::new(vec![1.0; hd], 1e-6),
            RmsNorm::new(vec![1.0; h], 1e-6),
            Linear1Bit::new(&gate_blocks, inter, h),
            Linear1Bit::new(&up_blocks, inter, h),
            Linear1Bit::new(&down_blocks, h, inter),
            nq,
            nkv,
            hd,
            h,
        );

        let rope = RopeTable::new(hd, 16, 10000.0);
        let kernel = oxibonsai_kernels::KernelDispatcher::auto_detect();
        let mut kv_cache = KvCache::new(1, nkv, hd, 16);
        let sw_config = SlidingWindowConfig::new(8, 2);

        let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let original = hidden.clone();

        block
            .forward_with_sliding_window(
                &mut hidden,
                0,
                &mut kv_cache,
                &rope,
                &kernel,
                Some(&sw_config),
            )
            .expect("forward_with_sliding_window should succeed");

        let max_diff = hidden
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff > 1e-6);
    }

    #[test]
    fn layer_stats_fractions() {
        let mut stats = LayerStats::new(0);
        stats.total_us = 100;
        stats.attention_us = 60;
        stats.ffn_us = 30;

        assert!((stats.attention_fraction() - 0.6).abs() < 1e-10);
        assert!((stats.ffn_fraction() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn layer_stats_zero_total() {
        let stats = LayerStats::new(5);
        assert!((stats.attention_fraction() - 0.0).abs() < 1e-10);
        assert!((stats.ffn_fraction() - 0.0).abs() < 1e-10);
    }
}
