//! Full-layer GPU dispatch for OxiBonsai.
//!
//! This module implements end-to-end transformer layer execution on Metal GPU,
//! encoding the complete attention + FFN pipeline into a single command buffer.
//! This eliminates per-kernel CPU→GPU round-trips and maximises GPU occupancy.
//!
//! # Architecture
//!
//! A single `try_metal_full_layer()` call encodes all operations for one
//! transformer layer:
//!
//! **Attention sublayer:**
//! 1. Pre-attention RMSNorm
//! 2. Fused QKV projection (GEMV Q1)
//! 3. QK-norm (batched RMSNorm)
//! 4. RoPE (batched)
//! 5. KV cache store
//! 6. Attention scores + softmax + weighted sum
//!
//! **FFN sublayer:**
//! 7. Output projection (GEMV Q1) + residual add
//! 8. FFN RMSNorm
//! 9. Gate+Up GEMV + SwiGLU + Down GEMV + residual add

#![cfg(feature = "metal")]

use metal::{Buffer, MTLResourceOptions, MTLSize};
use std::sync::Arc;

use super::metal_graph::{
    alloc_buf, div_ceil, download_f32, set_scalar, upload_f32, MetalGraph, MetalGraphError,
    MetalWeightHandle,
};

// ═══════════════════════════════════════════════════════════════════════════
// Cached GPU weight handles (zero per-token overhead)
// ═══════════════════════════════════════════════════════════════════════════

/// Pre-cached GPU weight handles for a single transformer layer.
/// Eliminates per-token weight lookup and upload overhead.
pub struct CachedLayerWeights {
    pub attn_norm: Arc<MetalWeightHandle>,
    pub fused_qkv: Arc<MetalWeightHandle>,
    pub q_norm: Arc<MetalWeightHandle>,
    pub k_norm: Arc<MetalWeightHandle>,
    pub attn_proj: Arc<MetalWeightHandle>,
    pub ffn_norm: Arc<MetalWeightHandle>,
    pub gate_up: Arc<MetalWeightHandle>,
    pub down: Arc<MetalWeightHandle>,
}

/// Pre-cached GPU weight handles for the entire model.
/// After initial creation, no weight data needs to be copied or uploaded.
pub struct CachedModelWeights {
    pub layers: Vec<CachedLayerWeights>,
    pub final_norm: Arc<MetalWeightHandle>,
    pub lm_head: Arc<MetalWeightHandle>,
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU profiling (OXIBONSAI_PROFILE_GPU=1)
// ═══════════════════════════════════════════════════════════════════════════

mod gpu_profile {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    use std::time::Instant;

    static ENABLED: AtomicBool = AtomicBool::new(false);
    static INITIALIZED: AtomicBool = AtomicBool::new(false);

    pub struct GpuTimings {
        pub wall_ms: f64,
        pub cpu_encode_ms: f64,
        pub gpu_exec_ms: f64,
    }

    static STATS: Mutex<Vec<GpuTimings>> = Mutex::new(Vec::new());

    pub fn is_enabled() -> bool {
        if !INITIALIZED.load(Ordering::Relaxed) {
            INITIALIZED.store(true, Ordering::Relaxed);
            if std::env::var("OXIBONSAI_PROFILE_GPU").is_ok() {
                ENABLED.store(true, Ordering::Relaxed);
            }
        }
        ENABLED.load(Ordering::Relaxed)
    }

    /// Get GPU execution start/end times from a completed command buffer.
    ///
    /// Uses Objective-C `GPUStartTime` / `GPUEndTime` properties (CFTimeInterval,
    /// seconds since boot). Available on macOS 10.15+.
    ///
    /// # Safety
    /// Must be called only after `wait_until_completed()` returns.
    pub unsafe fn gpu_cmd_times(cmd_buf: &metal::CommandBufferRef) -> (f64, f64) {
        let start: f64 = msg_send![cmd_buf, GPUStartTime];
        let end: f64 = msg_send![cmd_buf, GPUEndTime];
        (start, end)
    }

    pub fn record_and_print(
        wall_start: Instant,
        encode_end: Instant,
        gpu_start: f64,
        gpu_end: f64,
    ) {
        let wall_end = Instant::now();
        let wall_ms = wall_end.duration_since(wall_start).as_secs_f64() * 1000.0;
        let cpu_encode_ms = encode_end.duration_since(wall_start).as_secs_f64() * 1000.0;
        let gpu_exec_ms = (gpu_end - gpu_start) * 1000.0;
        let overhead_ms = (wall_ms - cpu_encode_ms - gpu_exec_ms).max(0.0);

        if let Ok(mut stats) = STATS.lock() {
            let token_num = stats.len();
            eprintln!(
                "[GPU Profile] token={} wall={:.1}ms cpu_encode={:.1}ms gpu_exec={:.1}ms overhead={:.1}ms",
                token_num, wall_ms, cpu_encode_ms, gpu_exec_ms, overhead_ms,
            );
            stats.push(GpuTimings {
                wall_ms,
                cpu_encode_ms,
                gpu_exec_ms,
            });
        }
    }

    pub fn print_summary(model_size_bytes: u64) {
        if let Ok(stats) = STATS.lock() {
            if stats.is_empty() {
                return;
            }
            let n = stats.len() as f64;
            let avg_wall: f64 = stats.iter().map(|s| s.wall_ms).sum::<f64>() / n;
            let avg_cpu: f64 = stats.iter().map(|s| s.cpu_encode_ms).sum::<f64>() / n;
            let avg_gpu: f64 = stats.iter().map(|s| s.gpu_exec_ms).sum::<f64>() / n;
            let avg_overhead = (avg_wall - avg_cpu - avg_gpu).max(0.0);
            let gpu_bw = if avg_gpu > 0.0 {
                (model_size_bytes as f64) / (avg_gpu / 1000.0) / 1e9
            } else {
                0.0
            };
            eprintln!(
                "[GPU Profile Summary] tokens={} avg: wall={:.1}ms cpu={:.1}ms gpu={:.1}ms overhead={:.1}ms gpu_bw={:.1}GB/s",
                stats.len() as u64, avg_wall, avg_cpu, avg_gpu, avg_overhead, gpu_bw,
            );
        }
    }
}

/// Print the GPU profiling summary (call at end of generation).
///
/// `model_size_bytes` is the model file size, used to compute effective bandwidth.
/// This is a no-op if `OXIBONSAI_PROFILE_GPU` was not set.
pub fn print_gpu_profile_summary(model_size_bytes: u64) {
    if gpu_profile::is_enabled() {
        gpu_profile::print_summary(model_size_bytes);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU KV cache
// ═══════════════════════════════════════════════════════════════════════════

/// GPU-resident KV cache for all transformer layers.
///
/// Layout: `[n_layers × n_kv × max_seq × head_dim]` f16, contiguous.
/// Each layer occupies `n_kv * max_seq * head_dim` half-precision elements.
pub(crate) struct GpuKvCache {
    pub k_cache: Buffer,
    pub v_cache: Buffer,
    pub n_layers: usize,
    pub n_kv: usize,
    pub max_seq: usize,
    pub head_dim: usize,
}

impl GpuKvCache {
    /// Allocate the KV cache on the GPU.
    pub fn allocate(
        device: &metal::Device,
        n_layers: usize,
        n_kv: usize,
        max_seq: usize,
        head_dim: usize,
    ) -> Result<Self, MetalGraphError> {
        let total_elements = n_layers * n_kv * max_seq * head_dim;
        let byte_len = (total_elements * 2) as u64; // FP16: 2 bytes per element
        let opts = MTLResourceOptions::StorageModePrivate;

        Ok(Self {
            k_cache: alloc_buf(device, byte_len, opts)?,
            v_cache: alloc_buf(device, byte_len, opts)?,
            n_layers,
            n_kv,
            max_seq,
            head_dim,
        })
    }

    /// Element offset into the cache for a given layer.
    #[inline]
    pub fn layer_offset_elements(&self, layer_idx: usize) -> u32 {
        (layer_idx * self.n_kv * self.max_seq * self.head_dim) as u32
    }

    /// Check whether this cache matches the given dimensions.
    pub fn matches(&self, n_layers: usize, n_kv: usize, max_seq: usize, head_dim: usize) -> bool {
        self.n_layers == n_layers
            && self.n_kv == n_kv
            && self.max_seq == max_seq
            && self.head_dim == head_dim
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Full-layer intermediate buffers
// ═══════════════════════════════════════════════════════════════════════════

/// Lazily allocated intermediate buffers for full-layer dispatch.
///
/// These are allocated once and reused across all forward passes.
/// Each forward pass uploads new data into these buffers.
pub(crate) struct FullLayerBuffers {
    pub hidden_buf: Buffer,
    pub normed_buf: Buffer,
    pub qkv_buf: Buffer,
    pub q_rope_buf: Buffer,
    pub k_rope_buf: Buffer,
    pub cos_buf: Buffer,
    pub sin_buf: Buffer,
    pub scores_buf: Buffer,
    pub attn_out_buf: Buffer,
    pub swiglu_buf: Buffer,
    // Cached dimensions for reuse check.
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    max_seq: usize,
}

impl FullLayerBuffers {
    /// Allocate all intermediate buffers for the given dimensions.
    pub fn allocate(
        device: &metal::Device,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<Self, MetalGraphError> {
        let f32_size = std::mem::size_of::<f32>();
        let shared = MTLResourceOptions::StorageModeShared;
        let private = MTLResourceOptions::StorageModePrivate;

        let h_bytes = (hidden_size * f32_size) as u64;
        let qkv_total = nq * head_dim + 2 * nkv * head_dim;
        let qkv_bytes = (qkv_total * f32_size) as u64;
        let q_bytes = (nq * head_dim * f32_size) as u64;
        let k_bytes = (nkv * head_dim * f32_size) as u64;
        let half_dim = head_dim / 2;
        let rope_bytes = (half_dim * f32_size) as u64;
        let scores_bytes = (nq * max_seq * f32_size) as u64;
        let inter_bytes = (intermediate_size * f32_size) as u64;

        Ok(Self {
            hidden_buf: alloc_buf(device, h_bytes, shared)?, // CPU upload/download
            normed_buf: alloc_buf(device, h_bytes, private)?, // GPU-only intermediate
            qkv_buf: alloc_buf(device, qkv_bytes, private)?, // GPU-only intermediate
            q_rope_buf: alloc_buf(device, q_bytes, private)?, // GPU-only intermediate
            k_rope_buf: alloc_buf(device, k_bytes, private)?, // GPU-only intermediate
            cos_buf: alloc_buf(device, rope_bytes, shared)?, // CPU upload
            sin_buf: alloc_buf(device, rope_bytes, shared)?, // CPU upload
            scores_buf: alloc_buf(device, scores_bytes, private)?, // GPU-only intermediate
            attn_out_buf: alloc_buf(device, q_bytes, private)?, // GPU-only intermediate
            swiglu_buf: alloc_buf(device, inter_bytes, private)?, // GPU-only intermediate

            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq,
        })
    }

    /// Check whether existing buffers match the requested dimensions.
    pub fn matches(
        &self,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> bool {
        self.hidden_size == hidden_size
            && self.intermediate_size == intermediate_size
            && self.nq == nq
            && self.nkv == nkv
            && self.head_dim == head_dim
            && self.max_seq == max_seq
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MetalGraph extensions for full-layer dispatch
// ═══════════════════════════════════════════════════════════════════════════

impl MetalGraph {
    /// Get a cached weight or upload f32 data (e.g. norm weights) and cache it.
    ///
    /// Unlike `get_or_upload_weight` which uploads raw bytes (Q1 blocks),
    /// this reinterprets `&[f32]` as `&[u8]` for the Metal buffer.
    pub fn get_or_upload_f32_weight(
        &self,
        key: u64,
        data: &[f32],
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        let byte_slice = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        self.get_or_upload_weight(key, byte_slice)
    }

    /// Acquire the full-layer buffer set, allocating if needed.
    fn acquire_full_layer_buffers(
        &self,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<FullLayerBuffers>>, MetalGraphError> {
        let mut guard = self.full_layer_buffers.lock().map_err(|_| {
            MetalGraphError::ExecutionFailed("full_layer_buffers lock poisoned".into())
        })?;

        let needs_alloc = match guard.as_ref() {
            Some(b) => !b.matches(hidden_size, intermediate_size, nq, nkv, head_dim, max_seq),
            None => true,
        };

        if needs_alloc {
            *guard = Some(FullLayerBuffers::allocate(
                &self.device,
                hidden_size,
                intermediate_size,
                nq,
                nkv,
                head_dim,
                max_seq,
            )?);
        }

        Ok(guard)
    }

    /// Acquire the KV cache, allocating if needed.
    pub(crate) fn acquire_kv_cache(
        &self,
        n_layers: usize,
        n_kv: usize,
        max_seq: usize,
        head_dim: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<GpuKvCache>>, MetalGraphError> {
        let mut guard = self
            .kv_cache
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("kv_cache lock poisoned".into()))?;

        let needs_alloc = match guard.as_ref() {
            Some(c) => !c.matches(n_layers, n_kv, max_seq, head_dim),
            None => true,
        };

        if needs_alloc {
            *guard = Some(GpuKvCache::allocate(
                &self.device,
                n_layers,
                n_kv,
                max_seq,
                head_dim,
            )?);
        }

        Ok(guard)
    }

    /// Encode a single transformer layer's dispatches into an existing encoder.
    ///
    /// This is the core encoding logic extracted from `encode_full_layer`.
    /// It encodes all 18 GPU dispatches (attention + FFN) for one layer
    /// without creating/committing a command buffer. The hidden state is
    /// read from and written to `bufs.hidden_buf` in-place (via residual adds).
    #[allow(clippy::too_many_arguments)]
    fn encode_layer_into(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        bufs: &FullLayerBuffers,
        kv: &GpuKvCache,
        layer_idx: usize,
        pos: usize,
        attn_norm_w: &MetalWeightHandle,
        fused_qkv_w: &MetalWeightHandle,
        q_norm_w: &MetalWeightHandle,
        k_norm_w: &MetalWeightHandle,
        attn_proj_w: &MetalWeightHandle,
        ffn_norm_w: &MetalWeightHandle,
        gate_up_w: &MetalWeightHandle,
        down_w: &MetalWeightHandle,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
    ) -> Result<(), MetalGraphError> {
        let seq_len = (pos + 1) as u32;
        let inv_sqrt_hd = 1.0f32 / (head_dim as f32).sqrt();
        let heads_per_group = (nq / nkv) as u32;
        let h = hidden_size as u32;
        let inter = intermediate_size as u32;
        let qkv_total_rows = (nq * head_dim + 2 * nkv * head_dim) as u32;
        let cache_layer_offset = kv.layer_offset_elements(layer_idx);

        // ══════════════════════════════════════════════════════════════════
        // ATTENTION SUBLAYER
        // ══════════════════════════════════════════════════════════════════

        // Pre-attention RMSNorm
        self.dispatch_rmsnorm(
            encoder,
            &bufs.hidden_buf,
            &attn_norm_w.buffer,
            &bufs.normed_buf,
            eps,
            h,
        );

        // Fused QKV projection
        self.dispatch_gemv_q1(
            encoder,
            &fused_qkv_w.buffer,
            &bufs.normed_buf,
            &bufs.qkv_buf,
            qkv_total_rows,
            h,
        );

        // Fused QK-norm + QK-RoPE: normalise and apply rotary embedding in one dispatch
        {
            let q_offset: u64 = 0;
            let k_offset = (nq * head_dim * std::mem::size_of::<f32>()) as u64;
            self.dispatch_fused_qk_norm_rope(
                encoder,
                &bufs.qkv_buf,
                q_offset,
                &bufs.qkv_buf,
                k_offset,
                &bufs.q_rope_buf,
                &bufs.k_rope_buf,
                &q_norm_w.buffer,
                &k_norm_w.buffer,
                &bufs.cos_buf,
                &bufs.sin_buf,
                nq as u32,
                nkv as u32,
                head_dim as u32,
                eps,
            );
        }

        // Fused KV-Store: copy both K and V into the cache in a single dispatch
        {
            let v_offset = ((nq * head_dim + nkv * head_dim) * std::mem::size_of::<f32>()) as u64;
            self.dispatch_fused_kv_store(
                encoder,
                &bufs.k_rope_buf,
                &bufs.qkv_buf,
                v_offset,
                &kv.k_cache,
                &kv.v_cache,
                nkv as u32,
                head_dim as u32,
                max_seq_len as u32,
                pos as u32,
                cache_layer_offset,
            );
        }

        // 3-kernel attention: scores → softmax → weighted sum
        {
            // Attention scores V2
            {
                self.dispatch_attention_scores_v2(
                    encoder,
                    &bufs.q_rope_buf,
                    &kv.k_cache,
                    &bufs.scores_buf,
                    head_dim as u32,
                    nq as u32,
                    nkv as u32,
                    heads_per_group,
                    max_seq_len as u32,
                    seq_len,
                    inv_sqrt_hd,
                    cache_layer_offset,
                );
            }
            // Attention softmax
            {
                encoder.set_compute_pipeline_state(&self.pipelines.batched_softmax);
                encoder.set_buffer(0, Some(&bufs.scores_buf), 0);
                unsafe {
                    set_scalar(encoder, 1, &(nq as u32));
                    set_scalar(encoder, 2, &(max_seq_len as u32));
                    set_scalar(encoder, 3, &seq_len);
                }
                encoder
                    .dispatch_thread_groups(MTLSize::new(nq as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
            // Attention weighted sum
            {
                encoder.set_compute_pipeline_state(&self.pipelines.batched_attention_weighted_sum);
                encoder.set_buffer(0, Some(&bufs.scores_buf), 0);
                encoder.set_buffer(1, Some(&kv.v_cache), 0);
                encoder.set_buffer(2, Some(&bufs.attn_out_buf), 0);
                unsafe {
                    set_scalar(encoder, 3, &(head_dim as u32));
                    set_scalar(encoder, 4, &(nq as u32));
                    set_scalar(encoder, 5, &(nkv as u32));
                    set_scalar(encoder, 6, &heads_per_group);
                    set_scalar(encoder, 7, &(max_seq_len as u32));
                    set_scalar(encoder, 8, &seq_len);
                    set_scalar(encoder, 9, &cache_layer_offset);
                }
                let tg_x = div_ceil(head_dim, 64) as u64;
                encoder.dispatch_thread_groups(
                    MTLSize::new(tg_x, nq as u64, 1),
                    MTLSize::new(64, 1, 1),
                );
            }
        }

        // ══════════════════════════════════════════════════════════════════
        // FFN SUBLAYER
        // ══════════════════════════════════════════════════════════════════

        // Output projection + fused residual add: hidden = hidden + proj(attn_out)
        self.dispatch_gemv_q1_residual(
            encoder,
            &attn_proj_w.buffer,
            &bufs.attn_out_buf,
            &bufs.hidden_buf,
            h,
            (nq * head_dim) as u32,
            &bufs.hidden_buf,
        );

        // FFN RMSNorm
        self.dispatch_rmsnorm(
            encoder,
            &bufs.hidden_buf,
            &ffn_norm_w.buffer,
            &bufs.normed_buf,
            eps,
            h,
        );

        // Fused Gate+Up+SwiGLU
        self.dispatch_fused_gate_up_swiglu(
            encoder,
            &gate_up_w.buffer,
            &bufs.normed_buf,
            &bufs.swiglu_buf,
            inter,
            h,
        );

        // Down projection + fused residual add: hidden = hidden + down(swiglu)
        self.dispatch_gemv_q1_residual(
            encoder,
            &down_w.buffer,
            &bufs.swiglu_buf,
            &bufs.hidden_buf,
            h,
            inter,
            &bufs.hidden_buf,
        );

        Ok(())
    }

    /// Encode a complete transformer layer (attention + FFN) in one command buffer.
    ///
    /// All GPU dispatches share the same command buffer and encoder. Metal's
    /// automatic hazard tracking on shared-mode buffers ensures correct
    /// read-after-write dependencies between dispatches.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_full_layer(
        &self,
        hidden: &mut [f32],
        pos: usize,
        layer_idx: usize,
        attn_norm_w: &MetalWeightHandle,
        fused_qkv_w: &MetalWeightHandle,
        q_norm_w: &MetalWeightHandle,
        k_norm_w: &MetalWeightHandle,
        attn_proj_w: &MetalWeightHandle,
        ffn_norm_w: &MetalWeightHandle,
        gate_up_w: &MetalWeightHandle,
        down_w: &MetalWeightHandle,
        rope_cos: &[f32],
        rope_sin: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
        n_layers: usize,
    ) -> Result<(), MetalGraphError> {
        let half_dim = head_dim / 2;

        // ── Validate inputs ──────────────────────────────────────────────
        if hidden.len() < hidden_size {
            return Err(MetalGraphError::EncodingFailed(format!(
                "hidden too short: need {hidden_size}, got {}",
                hidden.len()
            )));
        }
        if rope_cos.len() < half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "rope_cos too short: need {half_dim}, got {}",
                rope_cos.len()
            )));
        }
        if rope_sin.len() < half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "rope_sin too short: need {half_dim}, got {}",
                rope_sin.len()
            )));
        }

        // ── Acquire buffers ──────────────────────────────────────────────
        let fl_guard = self.acquire_full_layer_buffers(
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq_len,
        )?;
        let bufs = fl_guard.as_ref().ok_or_else(|| {
            MetalGraphError::ExecutionFailed("full_layer_buffers not allocated".into())
        })?;

        let kv_guard = self.acquire_kv_cache(n_layers, nkv, max_seq_len, head_dim)?;
        let kv = kv_guard
            .as_ref()
            .ok_or_else(|| MetalGraphError::ExecutionFailed("kv_cache not allocated".into()))?;

        // ── Upload host data ─────────────────────────────────────────────
        unsafe {
            upload_f32(&bufs.hidden_buf, &hidden[..hidden_size]);
            upload_f32(&bufs.cos_buf, &rope_cos[..half_dim]);
            upload_f32(&bufs.sin_buf, &rope_sin[..half_dim]);
        }

        // ── Create command buffer + single compute encoder ───────────────
        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        self.encode_layer_into(
            encoder,
            bufs,
            kv,
            layer_idx,
            pos,
            attn_norm_w,
            fused_qkv_w,
            q_norm_w,
            k_norm_w,
            attn_proj_w,
            ffn_norm_w,
            gate_up_w,
            down_w,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            eps,
            max_seq_len,
        )?;

        // ── Commit and wait ──────────────────────────────────────────────
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // ── Read back result ─────────────────────────────────────────────
        unsafe {
            download_f32(&bufs.hidden_buf, &mut hidden[..hidden_size]);
        }

        Ok(())
    }

    /// Encode ALL transformer layers into a SINGLE Metal command buffer.
    ///
    /// This eliminates N-1 command buffer submissions (one per layer), reducing
    /// GPU scheduling overhead. The hidden state persists in `hidden_buf` across
    /// layers — each layer reads and writes it in-place via residual_add dispatches.
    ///
    /// Metal's automatic hazard tracking on shared-mode buffers with a
    /// non-concurrent compute encoder guarantees correct sequential ordering.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn encode_full_forward(
        &self,
        hidden: &mut [f32],
        pos: usize,
        n_layers: usize,
        layer_weights: &[(
            &Arc<MetalWeightHandle>, // attn_norm
            &Arc<MetalWeightHandle>, // fused_qkv
            &Arc<MetalWeightHandle>, // q_norm
            &Arc<MetalWeightHandle>, // k_norm
            &Arc<MetalWeightHandle>, // attn_proj
            &Arc<MetalWeightHandle>, // ffn_norm
            &Arc<MetalWeightHandle>, // gate_up
            &Arc<MetalWeightHandle>, // down
        )],
        rope_cos: &[f32],
        rope_sin: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
        // Optional final norm + LM head (fused into same command buffer)
        final_norm_w: Option<&Arc<MetalWeightHandle>>,
        final_norm_eps: f32,
        lm_head_w: Option<&Arc<MetalWeightHandle>>,
        lm_head_out_features: usize,
        logits_out: Option<&mut Vec<f32>>,
        // Greedy decoding: if Some, runs argmax on GPU and writes token_id
        // instead of downloading the full logits vector (~607KB → 4 bytes).
        greedy_token_id_out: Option<&mut u32>,
    ) -> Result<(), MetalGraphError> {
        let half_dim = head_dim / 2;

        // ── Validate inputs ──────────────────────────────────────────────
        if hidden.len() < hidden_size {
            return Err(MetalGraphError::EncodingFailed(format!(
                "hidden too short: need {hidden_size}, got {}",
                hidden.len()
            )));
        }
        if rope_cos.len() < half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "rope_cos too short: need {half_dim}, got {}",
                rope_cos.len()
            )));
        }
        if rope_sin.len() < half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "rope_sin too short: need {half_dim}, got {}",
                rope_sin.len()
            )));
        }
        if layer_weights.len() != n_layers {
            return Err(MetalGraphError::EncodingFailed(format!(
                "layer_weights length mismatch: need {n_layers}, got {}",
                layer_weights.len()
            )));
        }

        // ── Acquire buffers ──────────────────────────────────────────────
        let fl_guard = self.acquire_full_layer_buffers(
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq_len,
        )?;
        let bufs = fl_guard.as_ref().ok_or_else(|| {
            MetalGraphError::ExecutionFailed("full_layer_buffers not allocated".into())
        })?;

        let kv_guard = self.acquire_kv_cache(n_layers, nkv, max_seq_len, head_dim)?;
        let kv = kv_guard
            .as_ref()
            .ok_or_else(|| MetalGraphError::ExecutionFailed("kv_cache not allocated".into()))?;

        // ── Upload host data ─────────────────────────────────────────────
        unsafe {
            upload_f32(&bufs.hidden_buf, &hidden[..hidden_size]);
            upload_f32(&bufs.cos_buf, &rope_cos[..half_dim]);
            upload_f32(&bufs.sin_buf, &rope_sin[..half_dim]);
        }

        let profiling = std::env::var("OXIBONSAI_PROFILE").is_ok();
        let gpu_profiling = gpu_profile::is_enabled();

        if profiling {
            // ── PROFILE MODE: one command buffer per layer ───────────────
            let mut layer_times = Vec::with_capacity(n_layers);
            for (layer_idx, weights) in layer_weights.iter().enumerate() {
                let layer_cmd = self.command_queue.new_command_buffer();
                let layer_enc = layer_cmd.new_compute_command_encoder();
                self.encode_layer_into(
                    layer_enc,
                    bufs,
                    kv,
                    layer_idx,
                    pos,
                    weights.0,
                    weights.1,
                    weights.2,
                    weights.3,
                    weights.4,
                    weights.5,
                    weights.6,
                    weights.7,
                    hidden_size,
                    intermediate_size,
                    nq,
                    nkv,
                    head_dim,
                    eps,
                    max_seq_len,
                )?;
                layer_enc.end_encoding();
                layer_cmd.commit();
                let t = std::time::Instant::now();
                layer_cmd.wait_until_completed();
                let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;
                layer_times.push(elapsed_ms);
                eprintln!("[profile] layer {:2} = {:.3}ms", layer_idx, elapsed_ms);
            }
            let sum: f64 = layer_times.iter().sum();
            let min = layer_times.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = layer_times
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "[profile] layers total={:.3}ms  avg={:.3}ms  min={:.3}ms  max={:.3}ms",
                sum,
                sum / n_layers as f64,
                min,
                max,
            );

            // ── Final norm + LM head in a separate profiled command buffer ─
            let tail_cmd = self.command_queue.new_command_buffer();
            let tail_enc = tail_cmd.new_compute_command_encoder();
            self.encode_tail_and_commit(
                tail_enc,
                tail_cmd,
                bufs,
                hidden,
                hidden_size,
                final_norm_w,
                final_norm_eps,
                lm_head_w,
                lm_head_out_features,
                logits_out,
                greedy_token_id_out,
                true,
                None,
            )?;
        } else {
            // ── NORMAL MODE: single command buffer for all layers ─────────
            let wall_start = if gpu_profiling {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let cmd_buf = self.command_queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            for (layer_idx, weights) in layer_weights.iter().enumerate() {
                self.encode_layer_into(
                    encoder,
                    bufs,
                    kv,
                    layer_idx,
                    pos,
                    weights.0,
                    weights.1,
                    weights.2,
                    weights.3,
                    weights.4,
                    weights.5,
                    weights.6,
                    weights.7,
                    hidden_size,
                    intermediate_size,
                    nq,
                    nkv,
                    head_dim,
                    eps,
                    max_seq_len,
                )?;
            }
            self.encode_tail_and_commit(
                encoder,
                cmd_buf,
                bufs,
                hidden,
                hidden_size,
                final_norm_w,
                final_norm_eps,
                lm_head_w,
                lm_head_out_features,
                logits_out,
                greedy_token_id_out,
                false,
                wall_start,
            )?;
        }

        Ok(())
    }

    /// Shared tail: final RMSNorm + LM head + argmax, then commit + wait + download.
    ///
    /// When `profiling` is true, prints timing for the tail section.
    /// When `gpu_profile_wall_start` is Some, captures GPU timing breakdown.
    #[allow(clippy::too_many_arguments)]
    fn encode_tail_and_commit(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        cmd_buf: &metal::CommandBufferRef,
        bufs: &FullLayerBuffers,
        hidden: &mut [f32],
        hidden_size: usize,
        final_norm_w: Option<&Arc<MetalWeightHandle>>,
        final_norm_eps: f32,
        lm_head_w: Option<&Arc<MetalWeightHandle>>,
        lm_head_out_features: usize,
        logits_out: Option<&mut Vec<f32>>,
        greedy_token_id_out: Option<&mut u32>,
        profiling: bool,
        gpu_profile_wall_start: Option<std::time::Instant>,
    ) -> Result<(), MetalGraphError> {
        match (final_norm_w, lm_head_w) {
            (Some(fnorm_w), Some(lm_w)) if lm_head_out_features > 0 => {
                let h = hidden_size as u32;

                // 1. Final RMSNorm: hidden_buf → normed_buf
                self.dispatch_rmsnorm(
                    encoder,
                    &bufs.hidden_buf,
                    &fnorm_w.buffer,
                    &bufs.normed_buf,
                    final_norm_eps,
                    h,
                );

                // 2. Ensure logits buffer is allocated
                let mut lg = self.logits_buf.lock().map_err(|_| {
                    MetalGraphError::ExecutionFailed("logits_buf lock poisoned".into())
                })?;
                let needed_bytes = (lm_head_out_features * std::mem::size_of::<f32>()) as u64;
                if lg.as_ref().is_none_or(|b| b.length() < needed_bytes) {
                    *lg = Some(alloc_buf(
                        &self.device,
                        needed_bytes,
                        MTLResourceOptions::StorageModeShared,
                    )?);
                }
                let logits_buf = lg.as_ref().ok_or(MetalGraphError::BufferCreationFailed)?;

                // 3. LM head GEMV
                self.dispatch_gemv_q1(
                    encoder,
                    &lm_w.buffer,
                    &bufs.normed_buf,
                    logits_buf,
                    lm_head_out_features as u32,
                    h,
                );

                // 4. Greedy decoding: argmax on GPU
                if greedy_token_id_out.is_some() {
                    let mut tid_guard = self.token_id_buf.lock().map_err(|_| {
                        MetalGraphError::ExecutionFailed("token_id_buf lock poisoned".into())
                    })?;
                    let needed = std::mem::size_of::<u32>() as u64;
                    if tid_guard.as_ref().is_none_or(|b| b.length() < needed) {
                        *tid_guard = Some(alloc_buf(
                            &self.device,
                            needed,
                            MTLResourceOptions::StorageModeShared,
                        )?);
                    }
                    let token_id_buf_ref = tid_guard
                        .as_ref()
                        .ok_or(MetalGraphError::BufferCreationFailed)?;

                    self.dispatch_argmax(
                        encoder,
                        logits_buf,
                        token_id_buf_ref,
                        lm_head_out_features as u32,
                    );

                    let encode_end = std::time::Instant::now();
                    encoder.end_encoding();
                    cmd_buf.commit();
                    let t = std::time::Instant::now();
                    cmd_buf.wait_until_completed();
                    if profiling {
                        eprintln!(
                            "[profile] tail (norm+lmhead+argmax) = {:.3}ms",
                            t.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                    if let Some(ws) = gpu_profile_wall_start {
                        let (gs, ge) = unsafe { gpu_profile::gpu_cmd_times(cmd_buf) };
                        gpu_profile::record_and_print(ws, encode_end, gs, ge);
                    }

                    let token_id = unsafe { *(token_id_buf_ref.contents() as *const u32) };
                    if let Some(out) = greedy_token_id_out {
                        *out = token_id;
                    }
                } else {
                    let encode_end = std::time::Instant::now();
                    encoder.end_encoding();
                    cmd_buf.commit();
                    let t = std::time::Instant::now();
                    cmd_buf.wait_until_completed();
                    if profiling {
                        eprintln!(
                            "[profile] tail (norm+lmhead) = {:.3}ms",
                            t.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                    if let Some(ws) = gpu_profile_wall_start {
                        let (gs, ge) = unsafe { gpu_profile::gpu_cmd_times(cmd_buf) };
                        gpu_profile::record_and_print(ws, encode_end, gs, ge);
                    }

                    if let Some(out) = logits_out {
                        out.resize(lm_head_out_features, 0.0);
                        unsafe { download_f32(logits_buf, out) };
                    }
                }
            }
            _ => {
                let encode_end = std::time::Instant::now();
                encoder.end_encoding();
                cmd_buf.commit();
                let t = std::time::Instant::now();
                cmd_buf.wait_until_completed();
                if profiling {
                    eprintln!(
                        "[profile] tail (no lmhead) = {:.3}ms",
                        t.elapsed().as_secs_f64() * 1000.0
                    );
                }
                if let Some(ws) = gpu_profile_wall_start {
                    let (gs, ge) = unsafe { gpu_profile::gpu_cmd_times(cmd_buf) };
                    gpu_profile::record_and_print(ws, encode_end, gs, ge);
                }

                unsafe {
                    download_f32(&bufs.hidden_buf, &mut hidden[..hidden_size]);
                }
            }
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Attempt to run a full transformer layer via direct Metal dispatch.
///
/// This encodes the complete attention + FFN pipeline for one transformer
/// layer into a single Metal command buffer, eliminating per-kernel
/// CPU→GPU synchronisation overhead.
///
/// Returns `Ok(())` on success. Returns `Err(...)` if Metal is unavailable
/// or any dispatch step fails.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_layer(
    hidden: &mut [f32],
    pos: usize,
    layer_idx: usize,
    // Attention norm (f32 weights)
    attn_norm_handle_id: u64,
    attn_norm_bytes: &[f32],
    // Fused QKV (Q1 block bytes)
    fused_qkv_handle_id: u64,
    fused_qkv_bytes: &[u8],
    // Q-norm and K-norm (f32 weights, typically head_dim floats)
    q_norm_handle_id: u64,
    q_norm_bytes: &[f32],
    k_norm_handle_id: u64,
    k_norm_bytes: &[f32],
    // Attention output projection (Q1 block bytes)
    attn_proj_handle_id: u64,
    attn_proj_bytes: &[u8],
    // FFN norm (f32 weights)
    ffn_norm_handle_id: u64,
    ffn_norm_bytes: &[f32],
    // Gate+Up (Q1 block bytes, pre-fused)
    gate_up_handle_id: u64,
    gate_bytes: &[u8],
    up_bytes: &[u8],
    // Down projection (Q1 block bytes)
    down_handle_id: u64,
    down_bytes: &[u8],
    // RoPE tables for this position
    rope_cos: &[f32],
    rope_sin: &[f32],
    // Model dimensions
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    eps: f32,
    max_seq_len: usize,
    n_layers: usize,
) -> Result<(), MetalGraphError> {
    let graph = MetalGraph::global()?;

    // Upload/cache f32 norm weights
    let attn_norm_w = graph.get_or_upload_f32_weight(attn_norm_handle_id, attn_norm_bytes)?;
    let q_norm_w = graph.get_or_upload_f32_weight(q_norm_handle_id, q_norm_bytes)?;
    let k_norm_w = graph.get_or_upload_f32_weight(k_norm_handle_id, k_norm_bytes)?;
    let ffn_norm_w = graph.get_or_upload_f32_weight(ffn_norm_handle_id, ffn_norm_bytes)?;

    // Upload/cache Q1 block weights (SoA layout for GPU coalescing)
    let fused_qkv_w = graph.get_or_upload_q1_weight_soa(fused_qkv_handle_id, fused_qkv_bytes)?;
    let attn_proj_w = graph.get_or_upload_q1_weight_soa(attn_proj_handle_id, attn_proj_bytes)?;

    // Fuse gate+up on first upload (SoA layout)
    let gate_up_w = graph.get_or_upload_q1_weight_soa_lazy(gate_up_handle_id, || {
        let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
        fused.extend_from_slice(gate_bytes);
        fused.extend_from_slice(up_bytes);
        fused
    })?;

    let down_w = graph.get_or_upload_q1_weight_soa(down_handle_id, down_bytes)?;

    graph.encode_full_layer(
        hidden,
        pos,
        layer_idx,
        &attn_norm_w,
        &fused_qkv_w,
        &q_norm_w,
        &k_norm_w,
        &attn_proj_w,
        &ffn_norm_w,
        &gate_up_w,
        &down_w,
        rope_cos,
        rope_sin,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        eps,
        max_seq_len,
        n_layers,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Full-forward (all layers in one command buffer)
// ═══════════════════════════════════════════════════════════════════════════

/// Per-layer parameters for the full-forward path.
///
/// Contains weight handle IDs and raw byte slices for each layer's
/// weight matrices. These are used to upload/cache weights on the GPU.
pub struct FullForwardLayerParams<'a> {
    pub attn_norm_handle: u64,
    pub attn_norm_bytes: &'a [f32],
    pub fused_qkv_handle: u64,
    pub fused_qkv_bytes: &'a [u8],
    pub q_norm_handle: u64,
    pub q_norm_bytes: &'a [f32],
    pub k_norm_handle: u64,
    pub k_norm_bytes: &'a [f32],
    pub attn_proj_handle: u64,
    pub attn_proj_bytes: &'a [u8],
    pub ffn_norm_handle: u64,
    pub ffn_norm_bytes: &'a [f32],
    pub gate_up_handle: u64,
    pub gate_bytes: &'a [u8],
    pub up_bytes: &'a [u8],
    pub down_handle: u64,
    pub down_bytes: &'a [u8],
}

/// Attempt to run ALL transformer layers in a single Metal command buffer.
///
/// This encodes the complete attention + FFN pipeline for all `n_layers`
/// layers into one command buffer, eliminating N-1 GPU scheduling events
/// compared to the per-layer path.
///
/// When `final_norm_handle` and `lm_head_handle` are both `Some`, the final
/// RMSNorm and LM head GEMV are appended to the same command buffer,
/// eliminating an additional CPU→GPU round trip.  In that case, logits are
/// written to `logits_out` and `hidden` is NOT updated.
///
/// When `greedy_token_id_out` is `Some`, argmax is performed on the GPU after
/// the LM head GEMV and only the resulting token ID (4 bytes) is downloaded
/// instead of the full logits vector (~607KB), dramatically reducing PCIe/
/// memory bandwidth overhead for greedy (temperature=0) decoding.
///
/// Returns `Ok(())` on success. Returns `Err(...)` if Metal is unavailable
/// or any dispatch step fails.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_forward(
    hidden: &mut [f32],
    pos: usize,
    n_layers: usize,
    layer_params: &[FullForwardLayerParams<'_>],
    rope_cos: &[f32],
    rope_sin: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    eps: f32,
    max_seq_len: usize,
    // Optional final norm + LM head
    final_norm_handle: Option<u64>,
    final_norm_bytes: Option<&[f32]>,
    final_norm_eps: f32,
    lm_head_handle: Option<u64>,
    lm_head_bytes: Option<&[u8]>,
    lm_head_out_features: usize,
    logits_out: Option<&mut Vec<f32>>,
    // Greedy decoding: if Some, runs argmax on GPU and writes token_id
    greedy_token_id_out: Option<&mut u32>,
) -> Result<(), MetalGraphError> {
    if layer_params.len() != n_layers {
        return Err(MetalGraphError::EncodingFailed(format!(
            "layer_params length mismatch: need {n_layers}, got {}",
            layer_params.len()
        )));
    }

    let graph = MetalGraph::global()?;

    // Upload/cache all per-layer weights and collect into tuples
    #[allow(clippy::type_complexity)]
    let mut layer_weights: Vec<(
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
    )> = Vec::with_capacity(n_layers);

    for lp in layer_params {
        let attn_norm_w =
            graph.get_or_upload_f32_weight(lp.attn_norm_handle, lp.attn_norm_bytes)?;
        let q_norm_w = graph.get_or_upload_f32_weight(lp.q_norm_handle, lp.q_norm_bytes)?;
        let k_norm_w = graph.get_or_upload_f32_weight(lp.k_norm_handle, lp.k_norm_bytes)?;
        let ffn_norm_w = graph.get_or_upload_f32_weight(lp.ffn_norm_handle, lp.ffn_norm_bytes)?;

        let fused_qkv_w =
            graph.get_or_upload_q1_weight_soa(lp.fused_qkv_handle, lp.fused_qkv_bytes)?;
        let attn_proj_w =
            graph.get_or_upload_q1_weight_soa(lp.attn_proj_handle, lp.attn_proj_bytes)?;

        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up_w = graph.get_or_upload_q1_weight_soa_lazy(lp.gate_up_handle, || {
            let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
            fused.extend_from_slice(gate_bytes);
            fused.extend_from_slice(up_bytes);
            fused
        })?;

        let down_w = graph.get_or_upload_q1_weight_soa(lp.down_handle, lp.down_bytes)?;

        layer_weights.push((
            attn_norm_w,
            fused_qkv_w,
            q_norm_w,
            k_norm_w,
            attn_proj_w,
            ffn_norm_w,
            gate_up_w,
            down_w,
        ));
    }

    // Build reference tuples for encode_full_forward
    let weight_refs: Vec<_> = layer_weights
        .iter()
        .map(|(a, b, c, d, e, f, g, h)| (a, b, c, d, e, f, g, h))
        .collect();

    // Upload/cache optional final norm and LM head weights
    let final_norm_cached = match (final_norm_handle, final_norm_bytes) {
        (Some(handle), Some(bytes)) => Some(graph.get_or_upload_f32_weight(handle, bytes)?),
        _ => None,
    };

    let lm_head_cached = match (lm_head_handle, lm_head_bytes) {
        (Some(handle), Some(bytes)) => Some(graph.get_or_upload_q1_weight_soa(handle, bytes)?),
        _ => None,
    };

    graph.encode_full_forward(
        hidden,
        pos,
        n_layers,
        &weight_refs,
        rope_cos,
        rope_sin,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        eps,
        max_seq_len,
        final_norm_cached.as_ref(),
        final_norm_eps,
        lm_head_cached.as_ref(),
        lm_head_out_features,
        logits_out,
        greedy_token_id_out,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Cached weight builder + cached forward path
// ═══════════════════════════════════════════════════════════════════════════

/// Build the cached weight handles from layer params (called once on first token).
/// This does all the QKV concatenation, AoS→SoA conversion, and GPU upload.
pub fn build_cached_weights(
    layer_params: &[FullForwardLayerParams<'_>],
    final_norm_handle: u64,
    final_norm_bytes: &[f32],
    lm_head_handle: u64,
    lm_head_bytes: &[u8],
) -> Result<CachedModelWeights, MetalGraphError> {
    let graph = MetalGraph::global()?;

    let mut layers = Vec::with_capacity(layer_params.len());
    for lp in layer_params {
        let attn_norm = graph.get_or_upload_f32_weight(lp.attn_norm_handle, lp.attn_norm_bytes)?;
        let q_norm = graph.get_or_upload_f32_weight(lp.q_norm_handle, lp.q_norm_bytes)?;
        let k_norm = graph.get_or_upload_f32_weight(lp.k_norm_handle, lp.k_norm_bytes)?;
        let ffn_norm = graph.get_or_upload_f32_weight(lp.ffn_norm_handle, lp.ffn_norm_bytes)?;
        let fused_qkv =
            graph.get_or_upload_q1_weight_soa(lp.fused_qkv_handle, lp.fused_qkv_bytes)?;
        let attn_proj =
            graph.get_or_upload_q1_weight_soa(lp.attn_proj_handle, lp.attn_proj_bytes)?;

        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up = graph.get_or_upload_q1_weight_soa_lazy(lp.gate_up_handle, || {
            let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
            fused.extend_from_slice(gate_bytes);
            fused.extend_from_slice(up_bytes);
            fused
        })?;

        let down = graph.get_or_upload_q1_weight_soa(lp.down_handle, lp.down_bytes)?;

        layers.push(CachedLayerWeights {
            attn_norm,
            fused_qkv,
            q_norm,
            k_norm,
            attn_proj,
            ffn_norm,
            gate_up,
            down,
        });
    }

    let final_norm = graph.get_or_upload_f32_weight(final_norm_handle, final_norm_bytes)?;
    let lm_head = graph.get_or_upload_q1_weight_soa(lm_head_handle, lm_head_bytes)?;

    Ok(CachedModelWeights {
        layers,
        final_norm,
        lm_head,
    })
}

/// Like `try_metal_full_forward`, but uses pre-cached GPU weight handles.
/// Eliminates ALL per-token weight lookup, upload, and allocation overhead.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_forward_cached(
    hidden: &mut [f32],
    pos: usize,
    cached: &CachedModelWeights,
    rope_cos: &[f32],
    rope_sin: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    eps: f32,
    max_seq_len: usize,
    final_norm_eps: f32,
    lm_head_out_features: usize,
    logits_out: Option<&mut Vec<f32>>,
    greedy_token_id_out: Option<&mut u32>,
) -> Result<(), MetalGraphError> {
    let n_layers = cached.layers.len();
    let graph = MetalGraph::global()?;

    // Build weight refs from cached handles (no allocation, no upload, no HashMap lookup)
    let weight_refs: Vec<_> = cached
        .layers
        .iter()
        .map(|lw| {
            (
                &lw.attn_norm,
                &lw.fused_qkv,
                &lw.q_norm,
                &lw.k_norm,
                &lw.attn_proj,
                &lw.ffn_norm,
                &lw.gate_up,
                &lw.down,
            )
        })
        .collect();

    graph.encode_full_forward(
        hidden,
        pos,
        n_layers,
        &weight_refs,
        rope_cos,
        rope_sin,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        eps,
        max_seq_len,
        Some(&cached.final_norm),
        final_norm_eps,
        Some(&cached.lm_head),
        lm_head_out_features,
        logits_out,
        greedy_token_id_out,
    )
}
