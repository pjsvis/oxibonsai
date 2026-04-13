//! Full-layer GPU dispatch for OxiBonsai — CUDA backend.
//!
//! Mirrors [`metal_full_layer`] for Linux/Windows, encoding the complete
//! attention + FFN pipeline for one transformer layer on a single CUDA stream.
//! This eliminates CPU–GPU round-trips between the attention and FFN sublayers.
//!
//! # Pipeline (per token, decode path)
//!
//! **Attention sublayer:**
//! 1. Pre-attention RMSNorm (existing `rmsnorm_weighted_v2`)
//! 2. Fused QKV projection (V7/V8 GEMV)
//! 3. Fused QK-Norm + QK-RoPE (`fused_qk_norm_rope`)
//! 4. Fused KV-store (`fused_kv_store`) — writes FP16 into KV cache
//! 5. Batched attention scores V2 (`batched_attn_scores_v2`)
//! 6. Batched softmax (`batched_softmax`)
//! 7. Batched weighted sum (`batched_attn_weighted_sum`)
//!
//! **FFN sublayer:**
//! 8. Output projection + residual add (V7/V8 GEMV)
//! 9. FFN RMSNorm
//! 10. Gate+Up GEMV + SwiGLU
//! 11. Down GEMV + residual add
//!
//! # KV cache layout
//!
//! `[n_layers * nkv * max_seq * head_dim]` stored as FP16 (`u16` on Rust side).
//! Each layer's slice begins at `layer_idx * nkv * max_seq * head_dim` elements.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use cudarc::driver::sys;
use cudarc::driver::CudaGraph as CuDriverGraph;
use cudarc::driver::{CudaFunction, CudaSlice, CudaView, LaunchConfig, PushKernelArg};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use tracing::{debug, warn};

// ─── CUDA Driver Graph wrapper (cudarc::driver::CudaGraph is not Send) ───────
struct CuGraphHolder(CuDriverGraph);
// SAFETY: CUgraphExec is safe to send across threads when protected by a Mutex.
// We never call into the graph from multiple threads concurrently.
unsafe impl Send for CuGraphHolder {}

use super::cuda_attn_kernels::CUDA_ATTENTION_KERNELS_SRC;
use super::cuda_graph::{compile_or_load_ptx, CudaGraph, CudaGraphError};

// =============================================================================
// Compiled CUDA attention modules
// =============================================================================

/// Compiled CUDA function handles for the 7 attention kernels.
pub struct CudaAttnModules {
    pub fused_qk_norm: CudaFunction,
    pub fused_qk_rope: CudaFunction,
    pub fused_qk_norm_rope: CudaFunction,
    pub fused_kv_store: CudaFunction,
    pub batched_attn_scores_v2: CudaFunction,
    pub batched_softmax: CudaFunction,
    pub batched_attn_weighted_sum: CudaFunction,
}

// SAFETY: CudaFunction is Send in cudarc (wraps a raw function handle).
unsafe impl Send for CudaAttnModules {}
unsafe impl Sync for CudaAttnModules {}

// =============================================================================
// GPU KV cache
// =============================================================================

/// GPU-resident KV cache stored in FP16 to save VRAM.
///
/// Layout: `[n_layers * nkv * max_seq * head_dim]` as `u16` (FP16 bit pattern).
/// Element offset for layer `l`: `l * nkv * max_seq * head_dim`.
pub struct CudaKvCache {
    pub k_cache: CudaSlice<u16>,
    pub v_cache: CudaSlice<u16>,
    pub n_layers: usize,
    pub n_kv: usize,
    pub max_seq: usize,
    pub head_dim: usize,
}

// SAFETY: CudaSlice<u16> is Send in cudarc.
unsafe impl Send for CudaKvCache {}
unsafe impl Sync for CudaKvCache {}

impl CudaKvCache {
    /// Compute the element offset for a given layer index.
    #[inline]
    pub fn layer_offset_elements(&self, layer_idx: usize) -> u32 {
        (layer_idx * self.n_kv * self.max_seq * self.head_dim) as u32
    }

    /// Check whether this cache's dimensions match the given parameters.
    pub fn matches(&self, n_layers: usize, n_kv: usize, max_seq: usize, head_dim: usize) -> bool {
        self.n_layers == n_layers
            && self.n_kv == n_kv
            && self.max_seq == max_seq
            && self.head_dim == head_dim
    }
}

// =============================================================================
// Full-layer intermediate buffers
// =============================================================================

/// Pre-allocated GPU activation buffers for full-layer (attention + FFN) execution.
///
/// All buffers are allocated once and reused across forward passes.
/// Lazily resized when model dimensions change.
pub struct CudaFullLayerBuffers {
    /// [hidden_size] residual stream
    pub d_hidden: CudaSlice<f32>,
    /// [hidden_size] RMSNorm output / O-proj scratch
    pub d_normed: CudaSlice<f32>,
    /// [nq*hd + 2*nkv*hd] fused QKV GEMV output
    pub d_qkv: CudaSlice<f32>,
    /// [nq * head_dim] Q after norm+RoPE
    pub d_q_rope: CudaSlice<f32>,
    /// [nkv * head_dim] K after norm+RoPE
    pub d_k_rope: CudaSlice<f32>,
    /// [half_dim] RoPE cosines
    pub d_cos: CudaSlice<f32>,
    /// [half_dim] RoPE sines
    pub d_sin: CudaSlice<f32>,
    /// [nq * max_seq] attention scores
    pub d_scores: CudaSlice<f32>,
    /// [nq * head_dim] attention output
    pub d_attn_out: CudaSlice<f32>,
    /// [2 * intermediate_size] gate+up GEMV
    pub d_gate_up: CudaSlice<f32>,
    /// [intermediate_size] SwiGLU output
    pub d_swiglu: CudaSlice<f32>,
    /// [2] pos/seq_len for CUDA-graph-captured attention kernels: [pos, seq_len]
    pub d_pos_seqlen: CudaSlice<u32>,
    /// Dimension tracking.
    pub hidden_size: usize,
    pub nq: usize,
    pub nkv: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    pub intermediate_size: usize,
}

// SAFETY: CudaSlice<f32> is Send in cudarc.
unsafe impl Send for CudaFullLayerBuffers {}
unsafe impl Sync for CudaFullLayerBuffers {}

impl CudaFullLayerBuffers {
    /// Returns `true` when the buffer set matches all given dimensions.
    pub fn matches(
        &self,
        hidden_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
        intermediate_size: usize,
    ) -> bool {
        self.hidden_size == hidden_size
            && self.nq == nq
            && self.nkv == nkv
            && self.head_dim == head_dim
            && self.max_seq == max_seq
            && self.intermediate_size == intermediate_size
    }
}

// =============================================================================
// Pre-cached GPU weight handles for one transformer layer
// =============================================================================

/// Weights for one transformer layer, already uploaded to GPU device memory.
///
/// Q1_0_G128 projection weights are stored in SoA layout (`Arc<CudaSlice<u8>>`).
/// Norm weights are stored as plain FP32 (`Arc<CudaSlice<f32>>`).
pub struct CudaCachedLayerWeights {
    /// Q projection (Q1 SoA)
    pub q_weight: Arc<CudaSlice<u8>>,
    /// K projection (Q1 SoA)
    pub k_weight: Arc<CudaSlice<u8>>,
    /// V projection (Q1 SoA)
    pub v_weight: Arc<CudaSlice<u8>>,
    /// O projection (Q1 SoA)
    pub o_weight: Arc<CudaSlice<u8>>,
    /// Gate+Up concatenated (Q1 SoA)
    pub gate_up_weight: Arc<CudaSlice<u8>>,
    /// Down projection (Q1 SoA)
    pub down_weight: Arc<CudaSlice<u8>>,
    /// Pre-attention RMSNorm weights
    pub pre_attn_norm: Arc<CudaSlice<f32>>,
    /// Post-attention (FFN) RMSNorm weights
    pub post_attn_norm: Arc<CudaSlice<f32>>,
    /// QK-norm for Q heads
    pub q_norm: Arc<CudaSlice<f32>>,
    /// QK-norm for K heads
    pub k_norm: Arc<CudaSlice<f32>>,
}

// SAFETY: CudaSlice is Send in cudarc; Arc provides Sync.
unsafe impl Send for CudaCachedLayerWeights {}
unsafe impl Sync for CudaCachedLayerWeights {}

// =============================================================================
// Per-process cached model weights  (avoids 288+ HashMap lookups per token)
// =============================================================================

/// All GPU weight handles for the whole model, built once and reused across tokens.
///
/// On the first decode token, `get_or_build_model_weights` uploads weights and
/// caches them here.  Subsequent tokens clone three `Arc`s (O(1)) instead of
/// running 288+ `HashMap::get` calls under a Mutex.
pub struct CudaCachedModelWeights {
    pub graph: Arc<CudaGraph>,
    /// Dummy 1-byte device slice shared across layers (k / v weight fields are
    /// unused — the fused QKV weight is stored in `q_weight`).
    pub dummy_weight: Arc<CudaSlice<u8>>,
    /// Per-layer weight handles, wrapped in Arc so cloning is O(1).
    pub layers: Arc<Vec<CudaCachedLayerWeights>>,
    /// Number of layers — used as a cheap validity token.
    pub n_layers: usize,
}

unsafe impl Send for CudaCachedModelWeights {}
unsafe impl Sync for CudaCachedModelWeights {}

// =============================================================================
// Per-process singleton for attention-layer extended state
// =============================================================================

/// Process-wide singleton holding state for the full-layer path.
struct CudaFullLayerState {
    attn_modules: Mutex<Option<Arc<CudaAttnModules>>>,
    full_layer_buffers: Mutex<Option<CudaFullLayerBuffers>>,
    kv_cache: Mutex<Option<CudaKvCache>>,
    /// Cache for FP32 norm weights (separate from the Q1 u8 weight cache).
    f32_weight_cache: Mutex<HashMap<u64, Arc<CudaSlice<f32>>>>,
    /// Cached GPU model weights — rebuilt only when the model changes.
    cached_model_weights: Mutex<Option<CudaCachedModelWeights>>,
    /// Captured CUDA driver graph for replaying the 36-layer pipeline.
    ///
    /// Three-state:
    /// - `None`             → capture not yet attempted
    /// - `Some(None)`       → capture was attempted but failed (no retry)
    /// - `Some(Some(h))`    → capture succeeded; `h` is the exec graph
    ///
    /// Reset to `None` when activation-buffer dimensions change (model switch),
    /// so a new capture is attempted with the fresh buffers.
    cuda_driver_graph: Mutex<Option<Option<CuGraphHolder>>>,
}

unsafe impl Send for CudaFullLayerState {}
unsafe impl Sync for CudaFullLayerState {}

static FULL_LAYER_STATE: OnceLock<CudaFullLayerState> = OnceLock::new();

fn full_layer_state() -> &'static CudaFullLayerState {
    FULL_LAYER_STATE.get_or_init(|| CudaFullLayerState {
        attn_modules: Mutex::new(None),
        full_layer_buffers: Mutex::new(None),
        kv_cache: Mutex::new(None),
        f32_weight_cache: Mutex::new(HashMap::new()),
        cached_model_weights: Mutex::new(None),
        // None = not yet attempted; Some(None) = tried & failed; Some(Some(h)) = active
        cuda_driver_graph: Mutex::new(None),
    })
}

// =============================================================================
// Profiling helper  (gated by CUDA_PROFILE env var)
// =============================================================================

static PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();

#[inline(always)]
pub(super) fn profiling() -> bool {
    *PROFILE_ENABLED.get_or_init(|| std::env::var("CUDA_PROFILE").is_ok())
}

// =============================================================================
// F32 weight upload and caching
// =============================================================================

/// Upload f32 weights and cache them for reuse.
///
/// On the first call for `key`, the slice is uploaded to GPU device memory
/// and cached.  Subsequent calls return the cached `Arc<CudaSlice<f32>>`.
pub fn get_or_upload_f32_weight(
    graph: &CudaGraph,
    key: u64,
    data: &[f32],
) -> Result<Arc<CudaSlice<f32>>, CudaGraphError> {
    let state = full_layer_state();
    {
        let cache = state
            .f32_weight_cache
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        if let Some(existing) = cache.get(&key) {
            return Ok(Arc::clone(existing));
        }
    }

    // Upload outside the lock to avoid holding it during H2D copy.
    let d_slice = graph
        .stream_arc()
        .clone_htod(data)
        .map_err(|e| CudaGraphError::DriverError(format!("clone_htod f32: {e}")))?;
    let arc = Arc::new(d_slice);

    let mut cache = state
        .f32_weight_cache
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;
    cache.insert(key, Arc::clone(&arc));
    Ok(arc)
}

// =============================================================================
// Per-process model weight cache
// =============================================================================

/// Build (or return the already-cached) GPU weight handles for all transformer layers.
///
/// On the **first call** this uploads all Q1/FP32 weights to GPU memory, wraps them in
/// `Arc<Vec<CudaCachedLayerWeights>>`, and stores the result in `FULL_LAYER_STATE`.
///
/// On **subsequent calls** — i.e., every decode token after the first — only three
/// `Arc::clone()` operations are performed (O(1)).  This replaces the previous
/// `try_cuda_full_forward` behaviour of doing 288+ `HashMap` lookups + mutex
/// acquisitions every token.
fn get_or_build_model_weights(
    layer_params: &[CudaFullForwardLayerParams<'_>],
) -> Option<(Arc<CudaGraph>, Arc<Vec<CudaCachedLayerWeights>>)> {
    let n_layers = layer_params.len();
    let state = full_layer_state();

    // Fast path: cache hit — three Arc::clones, no HashMap access.
    {
        let guard = state.cached_model_weights.lock().ok()?;
        if let Some(ref cmw) = *guard {
            if cmw.n_layers == n_layers {
                return Some((Arc::clone(&cmw.graph), Arc::clone(&cmw.layers)));
            }
        }
    }

    // Slow path: first call (or model changed).  Build and cache.
    let graph = CudaGraph::global().ok()?;
    let dummy_weight = Arc::new(graph.stream_arc().alloc_zeros::<u8>(1).ok()?);

    let mut cached: Vec<CudaCachedLayerWeights> = Vec::with_capacity(n_layers);
    for lp in layer_params {
        let q_weight = graph
            .get_or_upload_weight_soa(lp.fused_qkv_handle, lp.fused_qkv_bytes)
            .ok()?;
        let o_weight = graph
            .get_or_upload_weight_soa(lp.attn_proj_handle, lp.attn_proj_bytes)
            .ok()?;
        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up_weight = graph
            .get_or_upload_weight_soa_lazy(lp.gate_up_handle, || {
                let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
                fused.extend_from_slice(gate_bytes);
                fused.extend_from_slice(up_bytes);
                fused
            })
            .ok()?;
        let down_weight = graph
            .get_or_upload_weight_soa(lp.down_handle, lp.down_bytes)
            .ok()?;
        let pre_attn_norm =
            get_or_upload_f32_weight(&graph, lp.attn_norm_handle, lp.attn_norm_bytes).ok()?;
        let post_attn_norm =
            get_or_upload_f32_weight(&graph, lp.ffn_norm_handle, lp.ffn_norm_bytes).ok()?;
        let q_norm = get_or_upload_f32_weight(&graph, lp.q_norm_handle, lp.q_norm_bytes).ok()?;
        let k_norm = get_or_upload_f32_weight(&graph, lp.k_norm_handle, lp.k_norm_bytes).ok()?;

        cached.push(CudaCachedLayerWeights {
            q_weight,
            k_weight: Arc::clone(&dummy_weight),
            v_weight: Arc::clone(&dummy_weight),
            o_weight,
            gate_up_weight,
            down_weight,
            pre_attn_norm,
            post_attn_norm,
            q_norm,
            k_norm,
        });
    }

    let layers = Arc::new(cached);
    let cmw = CudaCachedModelWeights {
        graph: Arc::clone(&graph),
        dummy_weight,
        layers: Arc::clone(&layers),
        n_layers,
    };

    if let Ok(mut guard) = state.cached_model_weights.lock() {
        *guard = Some(cmw);
    }

    Some((graph, layers))
}

// =============================================================================
// Attention module lazy init
// =============================================================================

/// Compile and cache the 7 CUDA attention kernels.
///
/// Idempotent: on the second call the already-compiled modules are returned
/// immediately from the `Mutex<Option<...>>` cache.
pub fn init_attn_modules(graph: &CudaGraph) -> Result<Arc<CudaAttnModules>, CudaGraphError> {
    let state = full_layer_state();
    let mut guard = state
        .attn_modules
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    if let Some(ref m) = *guard {
        return Ok(Arc::clone(m));
    }

    // Compile all 7 attention kernels in a single NVRTC call (disk-cached after first run).
    let ptx = compile_or_load_ptx(CUDA_ATTENTION_KERNELS_SRC, "attn_kernels")?;

    let module = graph
        .context_arc()
        .load_module(ptx)
        .map_err(|e| CudaGraphError::DriverError(format!("load_module attn: {e}")))?;

    let load = |name: &str| -> Result<CudaFunction, CudaGraphError> {
        module
            .load_function(name)
            .map_err(|e| CudaGraphError::DriverError(format!("load_function({name}): {e}")))
    };

    let modules = Arc::new(CudaAttnModules {
        fused_qk_norm: load("fused_qk_norm")?,
        fused_qk_rope: load("fused_qk_rope")?,
        fused_qk_norm_rope: load("fused_qk_norm_rope")?,
        fused_kv_store: load("fused_kv_store")?,
        batched_attn_scores_v2: load("batched_attn_scores_v2")?,
        batched_softmax: load("batched_softmax")?,
        batched_attn_weighted_sum: load("batched_attn_weighted_sum")?,
    });

    *guard = Some(Arc::clone(&modules));
    Ok(modules)
}

// =============================================================================
// Buffer / cache acquisition helpers
// =============================================================================

/// Acquire or (re-)allocate the full-layer activation buffers.
fn acquire_full_layer_buffers(
    graph: &CudaGraph,
    hidden_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    max_seq: usize,
    intermediate_size: usize,
) -> Result<std::sync::MutexGuard<'static, Option<CudaFullLayerBuffers>>, CudaGraphError> {
    let state = full_layer_state();
    let mut guard = state
        .full_layer_buffers
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    let needs_alloc = match guard.as_ref() {
        Some(b) => !b.matches(hidden_size, nq, nkv, head_dim, max_seq, intermediate_size),
        None => true,
    };

    if needs_alloc {
        let alloc = |n: usize| -> Result<CudaSlice<f32>, CudaGraphError> {
            graph
                .stream_arc()
                .alloc_zeros::<f32>(n)
                .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros fl({n}): {e}")))
        };

        let qkv_total = nq * head_dim + 2 * nkv * head_dim;
        let half_dim = head_dim / 2;

        let alloc_u32 = |n: usize| -> Result<CudaSlice<u32>, CudaGraphError> {
            graph
                .stream_arc()
                .alloc_zeros::<u32>(n)
                .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros u32({n}): {e}")))
        };

        *guard = Some(CudaFullLayerBuffers {
            d_hidden: alloc(hidden_size)?,
            d_normed: alloc(hidden_size)?,
            d_qkv: alloc(qkv_total)?,
            d_q_rope: alloc(nq * head_dim)?,
            d_k_rope: alloc(nkv * head_dim)?,
            d_cos: alloc(half_dim)?,
            d_sin: alloc(half_dim)?,
            d_scores: alloc(nq * max_seq)?,
            d_attn_out: alloc(nq * head_dim)?,
            d_gate_up: alloc(2 * intermediate_size)?,
            d_swiglu: alloc(intermediate_size)?,
            d_pos_seqlen: alloc_u32(2)?,
            hidden_size,
            nq,
            nkv,
            head_dim,
            max_seq,
            intermediate_size,
        });
        // Buffer dimensions changed → invalidate any captured CUDA driver graph.
        if let Ok(mut g) = full_layer_state().cuda_driver_graph.lock() {
            *g = None;
        }
    }

    Ok(guard)
}

/// Acquire or (re-)allocate the GPU KV cache.
fn acquire_kv_cache(
    graph: &CudaGraph,
    n_layers: usize,
    n_kv: usize,
    max_seq: usize,
    head_dim: usize,
) -> Result<std::sync::MutexGuard<'static, Option<CudaKvCache>>, CudaGraphError> {
    let state = full_layer_state();
    let mut guard = state
        .kv_cache
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    let needs_alloc = match guard.as_ref() {
        Some(c) => !c.matches(n_layers, n_kv, max_seq, head_dim),
        None => true,
    };

    if needs_alloc {
        let total_elements = n_layers * n_kv * max_seq * head_dim;
        let k_cache = graph
            .stream_arc()
            .alloc_zeros::<u16>(total_elements)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc kv k_cache: {e}")))?;
        let v_cache = graph
            .stream_arc()
            .alloc_zeros::<u16>(total_elements)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc kv v_cache: {e}")))?;

        *guard = Some(CudaKvCache {
            k_cache,
            v_cache,
            n_layers,
            n_kv,
            max_seq,
            head_dim,
        });
    }

    Ok(guard)
}

// =============================================================================
// Attention kernel launchers
// =============================================================================

/// Launch `fused_qk_norm`.
///
/// Grid `(nq + nkv, 1, 1)`, block `(256, 1, 1)`.
///
/// # Safety
/// All slices must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments, dead_code)]
unsafe fn launch_fused_qk_norm(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_q_in: &CudaSlice<f32>,
    d_k_in: &CudaSlice<f32>,
    d_q_out: &mut CudaSlice<f32>,
    d_k_out: &mut CudaSlice<f32>,
    d_q_weight: &CudaSlice<f32>,
    d_k_weight: &CudaSlice<f32>,
    nq: u32,
    nkv: u32,
    head_dim: u32,
    eps: f32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (nq + nkv, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_qk_norm)
        .arg(d_q_in)
        .arg(d_k_in)
        .arg(d_q_out)
        .arg(d_k_out)
        .arg(d_q_weight)
        .arg(d_k_weight)
        .arg(&nq)
        .arg(&nkv)
        .arg(&head_dim)
        .arg(&eps)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("fused_qk_norm launch: {e}")))
}

/// Launch `fused_qk_rope`.
///
/// Grid `(ceil(half_dim/64), nq + nkv, 1)`, block `(64, 1, 1)`.
///
/// # Safety
/// All slices must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments, dead_code)]
unsafe fn launch_fused_qk_rope(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_q_in: &CudaSlice<f32>,
    d_k_in: &CudaSlice<f32>,
    d_q_out: &mut CudaSlice<f32>,
    d_k_out: &mut CudaSlice<f32>,
    d_cos: &CudaSlice<f32>,
    d_sin: &CudaSlice<f32>,
    nq: u32,
    nkv: u32,
    half_dim: u32,
) -> Result<(), CudaGraphError> {
    let grid_x = half_dim.div_ceil(64);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, nq + nkv, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_qk_rope)
        .arg(d_q_in)
        .arg(d_k_in)
        .arg(d_q_out)
        .arg(d_k_out)
        .arg(d_cos)
        .arg(d_sin)
        .arg(&nq)
        .arg(&nkv)
        .arg(&half_dim)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("fused_qk_rope launch: {e}")))
}

/// Launch `fused_qk_norm_rope`.
///
/// Grid `(nq + nkv, 1, 1)`, block `(256, 1, 1)`.
///
/// `d_k_in_view` is a `CudaView` pointing at the K section of the QKV buffer.
///
/// # Safety
/// All slices/views must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_fused_qk_norm_rope(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_q_in: &CudaSlice<f32>,
    d_k_in_view: &CudaView<'_, f32>,
    d_q_out: &mut CudaSlice<f32>,
    d_k_out: &mut CudaSlice<f32>,
    d_q_weight: &CudaSlice<f32>,
    d_k_weight: &CudaSlice<f32>,
    d_cos: &CudaSlice<f32>,
    d_sin: &CudaSlice<f32>,
    nq: u32,
    nkv: u32,
    head_dim: u32,
    eps: f32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (nq + nkv, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_qk_norm_rope)
        .arg(d_q_in)
        .arg(d_k_in_view)
        .arg(d_q_out)
        .arg(d_k_out)
        .arg(d_q_weight)
        .arg(d_k_weight)
        .arg(d_cos)
        .arg(d_sin)
        .arg(&nq)
        .arg(&nkv)
        .arg(&head_dim)
        .arg(&eps)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("fused_qk_norm_rope launch: {e}")))
}

/// Launch `fused_kv_store`.
///
/// Grid `(ceil(head_dim/64), nkv, 1)`, block `(64, 1, 1)`.
///
/// `d_pos_seqlen[0]` = current position (read by the kernel from device memory).
///
/// # Safety
/// All slices/views must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_fused_kv_store(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_k_data: &CudaSlice<f32>,
    d_v_data_view: &CudaView<'_, f32>,
    d_k_cache: &mut CudaSlice<u16>,
    d_v_cache: &mut CudaSlice<u16>,
    head_dim: u32,
    nkv: u32,
    max_seq: u32,
    d_pos_seqlen: &CudaSlice<u32>,
    layer_offset: u32,
) -> Result<(), CudaGraphError> {
    let grid_x = head_dim.div_ceil(64);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, nkv, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_kv_store)
        .arg(d_k_data)
        .arg(d_v_data_view)
        .arg(d_k_cache)
        .arg(d_v_cache)
        .arg(&head_dim)
        .arg(&nkv)
        .arg(&max_seq)
        .arg(d_pos_seqlen)
        .arg(&layer_offset)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("fused_kv_store launch: {e}")))
}

/// Launch `batched_attn_scores_v2`.
///
/// Grid `(n_q, max_seq / BATCH_STRIDE, 1)`, block `(128, 1, 1)`.
///
/// The grid Y dimension is fixed at `max_seq / BATCH_STRIDE` (not `seq_len`) so the
/// kernel sequence can be captured as a CUDA driver graph once and replayed for any
/// position.  Blocks with `pos_start >= seq_len` (read from `d_pos_seqlen[1]`) exit
/// immediately via the existing loop condition, adding only negligible overhead.
///
/// # Safety
/// All slices must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_batched_attn_scores_v2(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_queries: &CudaSlice<f32>,
    d_k_cache: &CudaSlice<u16>,
    d_scores: &mut CudaSlice<f32>,
    head_dim: u32,
    n_q: u32,
    n_kv: u32,
    heads_per_group: u32,
    max_seq: u32,
    d_pos_seqlen: &CudaSlice<u32>,
    inv_sqrt_hd: f32,
    cache_layer_offset: u32,
) -> Result<(), CudaGraphError> {
    const BATCH_STRIDE: u32 = 4;
    // Fixed grid Y = max_seq / BATCH_STRIDE — constant across all decode positions,
    // allowing the kernel sequence to be captured as a replayable CUDA graph.
    let grid_y = max_seq.div_ceil(BATCH_STRIDE);
    let cfg = LaunchConfig {
        grid_dim: (n_q, grid_y, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.batched_attn_scores_v2)
        .arg(d_queries)
        .arg(d_k_cache)
        .arg(d_scores)
        .arg(&head_dim)
        .arg(&n_q)
        .arg(&n_kv)
        .arg(&heads_per_group)
        .arg(&max_seq)
        .arg(d_pos_seqlen)
        .arg(&inv_sqrt_hd)
        .arg(&cache_layer_offset)
        .arg(&BATCH_STRIDE)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("batched_attn_scores_v2 launch: {e}")))
}

/// Launch `batched_softmax`.
///
/// Grid `(n_q, 1, 1)`, block `(256, 1, 1)`.
///
/// `d_pos_seqlen[1]` = seq_len (read by the kernel from device memory).
///
/// # Safety
/// All slices must be valid device pointers allocated on the graph's stream.
unsafe fn launch_batched_softmax(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_scores: &mut CudaSlice<f32>,
    n_q: u32,
    max_seq: u32,
    d_pos_seqlen: &CudaSlice<u32>,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_q, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.batched_softmax)
        .arg(d_scores)
        .arg(&n_q)
        .arg(&max_seq)
        .arg(d_pos_seqlen)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("batched_softmax launch: {e}")))
}

/// Launch `batched_attn_weighted_sum`.
///
/// Grid `(ceil(head_dim/64), n_q, 1)`, block `(64, 1, 1)`.
///
/// `d_pos_seqlen[1]` = seq_len (read by the kernel from device memory).
///
/// # Safety
/// All slices must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_batched_attn_weighted_sum(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_scores: &CudaSlice<f32>,
    d_v_cache: &CudaSlice<u16>,
    d_attn_out: &mut CudaSlice<f32>,
    head_dim: u32,
    n_q: u32,
    n_kv: u32,
    heads_per_group: u32,
    max_seq: u32,
    d_pos_seqlen: &CudaSlice<u32>,
    cache_layer_offset: u32,
) -> Result<(), CudaGraphError> {
    let grid_x = head_dim.div_ceil(64);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, n_q, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.batched_attn_weighted_sum)
        .arg(d_scores)
        .arg(d_v_cache)
        .arg(d_attn_out)
        .arg(&head_dim)
        .arg(&n_q)
        .arg(&n_kv)
        .arg(&heads_per_group)
        .arg(&max_seq)
        .arg(d_pos_seqlen)
        .arg(&cache_layer_offset)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("batched_attn_weighted_sum launch: {e}")))
}

// =============================================================================
// encode_attn_phase
// =============================================================================

/// Encode the full attention sublayer on the CUDA stream (steps 1-7).
///
/// On return `bufs.d_attn_out` holds `[nq * head_dim]` attention output values.
///
/// # Safety
/// The function launches CUDA kernels.  The caller must ensure all GPU state
/// is valid and the stream is not concurrently used.
#[allow(clippy::too_many_arguments)]
pub unsafe fn encode_attn_phase(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_pre_norm_weight: &CudaSlice<f32>,
    d_fused_qkv_weight: &Arc<CudaSlice<u8>>,
    d_q_norm_weight: &CudaSlice<f32>,
    d_k_norm_weight: &CudaSlice<f32>,
    kv: &mut CudaKvCache,
    layer_idx: usize,
    _pos: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    heads_per_group: usize,
    norm_eps: f32,
    hidden_size: usize,
    bufs: &mut CudaFullLayerBuffers,
) -> Result<(), CudaGraphError> {
    let h_u32 = hidden_size as u32;
    let nq_u32 = nq as u32;
    let nkv_u32 = nkv as u32;
    let hd_u32 = head_dim as u32;
    let qkv_total_rows = (nq * head_dim + 2 * nkv * head_dim) as u32;
    let heads_per_group_u32 = heads_per_group as u32;
    let max_seq_u32 = bufs.max_seq as u32;
    let inv_sqrt_hd = 1.0f32 / (head_dim as f32).sqrt();
    let layer_offset = kv.layer_offset_elements(layer_idx);

    // Step 1: RMSNorm(d_hidden, norm_weight -> d_normed)
    graph.launch_rmsnorm_pub(
        &bufs.d_hidden,
        d_pre_norm_weight,
        &mut bufs.d_normed,
        h_u32,
        norm_eps,
    )?;

    // Step 2: Fused QKV GEMV (normed -> d_qkv)
    graph.launch_gemv_pub(
        d_fused_qkv_weight,
        &bufs.d_normed,
        &mut bufs.d_qkv,
        qkv_total_rows,
        h_u32,
    )?;

    // Step 3: Fused QK-Norm + RoPE
    // Q occupies elements [0 .. nq*head_dim] in d_qkv.
    // K occupies elements [nq*head_dim .. (nq+nkv)*head_dim] in d_qkv.
    let k_offset = nq * head_dim;
    let k_in_view = bufs.d_qkv.slice(k_offset..);
    launch_fused_qk_norm_rope(
        graph,
        mods,
        &bufs.d_qkv, // q_in: first nq*head_dim elements
        &k_in_view,  // k_in: starts at nq*head_dim
        &mut bufs.d_q_rope,
        &mut bufs.d_k_rope,
        d_q_norm_weight,
        d_k_norm_weight,
        &bufs.d_cos,
        &bufs.d_sin,
        nq_u32,
        nkv_u32,
        hd_u32,
        norm_eps,
    )?;

    // Step 4: Fused KV-Store — pos read from d_pos_seqlen[0] by the kernel
    let v_offset = (nq + nkv) * head_dim;
    let v_view = bufs.d_qkv.slice(v_offset..);
    launch_fused_kv_store(
        graph,
        mods,
        &bufs.d_k_rope,
        &v_view,
        &mut kv.k_cache,
        &mut kv.v_cache,
        hd_u32,
        nkv_u32,
        max_seq_u32,
        &bufs.d_pos_seqlen,
        layer_offset,
    )?;

    // Step 5: Batched attention scores V2 — seq_len read from d_pos_seqlen[1]
    launch_batched_attn_scores_v2(
        graph,
        mods,
        &bufs.d_q_rope,
        &kv.k_cache,
        &mut bufs.d_scores,
        hd_u32,
        nq_u32,
        nkv_u32,
        heads_per_group_u32,
        max_seq_u32,
        &bufs.d_pos_seqlen,
        inv_sqrt_hd,
        layer_offset,
    )?;

    // Step 6: Softmax — seq_len read from d_pos_seqlen[1]
    launch_batched_softmax(
        graph,
        mods,
        &mut bufs.d_scores,
        nq_u32,
        max_seq_u32,
        &bufs.d_pos_seqlen,
    )?;

    // Step 7: Weighted sum — seq_len read from d_pos_seqlen[1]
    launch_batched_attn_weighted_sum(
        graph,
        mods,
        &bufs.d_scores,
        &kv.v_cache,
        &mut bufs.d_attn_out,
        hd_u32,
        nq_u32,
        nkv_u32,
        heads_per_group_u32,
        max_seq_u32,
        &bufs.d_pos_seqlen,
        layer_offset,
    )
}

// =============================================================================
// encode_full_layer
// =============================================================================

/// Encode a complete transformer layer (attention + FFN) on the CUDA stream.
///
/// On entry `hidden[..hidden_size]` contains the current residual stream.
/// On return the same slice has been updated in-place.
#[allow(clippy::too_many_arguments)]
pub fn encode_full_layer(
    graph: &CudaGraph,
    hidden: &mut [f32],
    pos: usize,
    layer_idx: usize,
    d_pre_attn_norm: &CudaSlice<f32>,
    d_fused_qkv_weight: &Arc<CudaSlice<u8>>,
    d_o_weight: &Arc<CudaSlice<u8>>,
    d_q_norm: &CudaSlice<f32>,
    d_k_norm: &CudaSlice<f32>,
    d_post_attn_norm: &CudaSlice<f32>,
    d_gate_up_weight: &Arc<CudaSlice<u8>>,
    d_down_weight: &Arc<CudaSlice<u8>>,
    rope_cos: &[f32],
    rope_sin: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    heads_per_group: usize,
    norm_eps: f32,
    max_seq_len: usize,
    n_layers: usize,
    attn_mods: &CudaAttnModules,
) -> Result<(), CudaGraphError> {
    let h = hidden_size;
    let half_dim = head_dim / 2;

    if hidden.len() < h {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "hidden too short: need {h}, got {}",
            hidden.len()
        )));
    }
    if rope_cos.len() < half_dim {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "rope_cos too short: need {half_dim}, got {}",
            rope_cos.len()
        )));
    }
    if rope_sin.len() < half_dim {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "rope_sin too short: need {half_dim}, got {}",
            rope_sin.len()
        )));
    }

    let mut fl_guard =
        acquire_full_layer_buffers(graph, h, nq, nkv, head_dim, max_seq_len, intermediate_size)?;
    let bufs = fl_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("full_layer_buffers not allocated".into()))?;

    let mut kv_guard = acquire_kv_cache(graph, n_layers, nkv, max_seq_len, head_dim)?;
    let kv = kv_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("kv_cache not allocated".into()))?;

    // Upload host data to GPU.
    graph
        .stream_arc()
        .memcpy_htod(&hidden[..h], &mut bufs.d_hidden)
        .map_err(|e| CudaGraphError::DriverError(format!("upload hidden: {e}")))?;
    graph
        .stream_arc()
        .memcpy_htod(&rope_cos[..half_dim], &mut bufs.d_cos)
        .map_err(|e| CudaGraphError::DriverError(format!("upload cos: {e}")))?;
    graph
        .stream_arc()
        .memcpy_htod(&rope_sin[..half_dim], &mut bufs.d_sin)
        .map_err(|e| CudaGraphError::DriverError(format!("upload sin: {e}")))?;

    // Attention sublayer.
    unsafe {
        encode_attn_phase(
            graph,
            attn_mods,
            d_pre_attn_norm,
            d_fused_qkv_weight,
            d_q_norm,
            d_k_norm,
            kv,
            layer_idx,
            pos,
            nq,
            nkv,
            head_dim,
            heads_per_group,
            norm_eps,
            h,
            bufs,
        )?;
    }

    let h_u32 = h as u32;
    let inter_u32 = intermediate_size as u32;

    unsafe {
        // O-projection: d_attn_out -> d_normed (scratch)
        let attn_out_rows = (nq * head_dim) as u32;
        graph.launch_gemv_pub(
            d_o_weight,
            &bufs.d_attn_out,
            &mut bufs.d_normed,
            h_u32,
            attn_out_rows,
        )?;
        // residual_add: hidden += O_proj
        graph.launch_residual_add_pub(&mut bufs.d_hidden, &bufs.d_normed, h_u32)?;
        // FFN RMSNorm: hidden -> normed
        graph.launch_rmsnorm_pub(
            &bufs.d_hidden,
            d_post_attn_norm,
            &mut bufs.d_normed,
            h_u32,
            norm_eps,
        )?;
        // Gate+Up GEMV: normed -> d_gate_up (2*inter rows, k=h)
        graph.launch_gemv_pub(
            d_gate_up_weight,
            &bufs.d_normed,
            &mut bufs.d_gate_up,
            2 * inter_u32,
            h_u32,
        )?;
        // SwiGLU: d_gate_up -> d_swiglu
        graph.launch_swiglu_pub(&bufs.d_gate_up, &mut bufs.d_swiglu, inter_u32)?;
        // Down GEMV: swiglu -> d_normed (h rows, k=inter)
        graph.launch_gemv_pub(
            d_down_weight,
            &bufs.d_swiglu,
            &mut bufs.d_normed,
            h_u32,
            inter_u32,
        )?;
        // residual_add: hidden += down_output
        graph.launch_residual_add_pub(&mut bufs.d_hidden, &bufs.d_normed, h_u32)?;
    }

    // Synchronise and download result.
    graph
        .stream_arc()
        .synchronize()
        .map_err(|e| CudaGraphError::DriverError(format!("fl stream sync: {e}")))?;
    graph
        .stream_arc()
        .memcpy_dtoh(&bufs.d_hidden, &mut hidden[..h])
        .map_err(|e| CudaGraphError::DriverError(format!("download hidden fl: {e}")))?;
    graph
        .stream_arc()
        .synchronize()
        .map_err(|e| CudaGraphError::DriverError(format!("fl D2H sync: {e}")))?;

    Ok(())
}

// =============================================================================
// Public entry point
// =============================================================================

/// Attempt to run a full transformer layer (attention + FFN) via CUDA.
///
/// Mirrors `try_metal_full_layer` exactly.  Returns `Ok(())` on success,
/// `Err(CudaGraphError)` if CUDA is unavailable or any kernel launch fails.
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_full_layer(
    hidden: &mut [f32],
    pos: usize,
    layer_idx: usize,
    pre_attn_norm_handle_id: u64,
    pre_attn_norm_bytes: &[f32],
    fused_qkv_handle_id: u64,
    fused_qkv_bytes: &[u8],
    o_handle_id: u64,
    o_bytes: &[u8],
    q_norm_handle_id: u64,
    q_norm_bytes: &[f32],
    k_norm_handle_id: u64,
    k_norm_bytes: &[f32],
    post_attn_norm_handle_id: u64,
    post_attn_norm_bytes: &[f32],
    gate_up_handle_id: u64,
    gate_bytes: &[u8],
    up_bytes: &[u8],
    down_handle_id: u64,
    down_bytes: &[u8],
    rope_cos: &[f32],
    rope_sin: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    heads_per_group: usize,
    norm_eps: f32,
    max_seq_len: usize,
    n_layers: usize,
) -> Result<(), CudaGraphError> {
    let graph = CudaGraph::global()?;
    let attn_mods = init_attn_modules(&graph)?;

    let d_fused_qkv_weight =
        graph.get_or_upload_weight_soa(fused_qkv_handle_id, fused_qkv_bytes)?;
    let d_o_weight = graph.get_or_upload_weight_soa(o_handle_id, o_bytes)?;
    let d_gate_up_weight = graph.get_or_upload_weight_soa_lazy(gate_up_handle_id, || {
        let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
        fused.extend_from_slice(gate_bytes);
        fused.extend_from_slice(up_bytes);
        fused
    })?;
    let d_down_weight = graph.get_or_upload_weight_soa(down_handle_id, down_bytes)?;

    let d_pre_attn_norm =
        get_or_upload_f32_weight(&graph, pre_attn_norm_handle_id, pre_attn_norm_bytes)?;
    let d_post_attn_norm =
        get_or_upload_f32_weight(&graph, post_attn_norm_handle_id, post_attn_norm_bytes)?;
    let d_q_norm = get_or_upload_f32_weight(&graph, q_norm_handle_id, q_norm_bytes)?;
    let d_k_norm = get_or_upload_f32_weight(&graph, k_norm_handle_id, k_norm_bytes)?;

    encode_full_layer(
        &graph,
        hidden,
        pos,
        layer_idx,
        &d_pre_attn_norm,
        &d_fused_qkv_weight,
        &d_o_weight,
        &d_q_norm,
        &d_k_norm,
        &d_post_attn_norm,
        &d_gate_up_weight,
        &d_down_weight,
        rope_cos,
        rope_sin,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        heads_per_group,
        norm_eps,
        max_seq_len,
        n_layers,
        &attn_mods,
    )
}

// =============================================================================
// encode_layer_device  (pure GPU, no upload/download)
// =============================================================================

/// Run one transformer layer entirely on device, reading/writing `bufs.d_hidden`
/// in-place.  Unlike `encode_full_layer` this function performs **no** H2D/D2H
/// transfers — the caller is responsible for uploading before the first layer
/// and downloading after the last layer.
///
/// RoPE buffers (`bufs.d_cos` / `bufs.d_sin`) must already be populated by the
/// caller before the first layer call.
///
/// # Safety
/// All device pointers must be valid and allocated on the same CUDA stream.
#[allow(clippy::too_many_arguments)]
unsafe fn encode_layer_device(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    weights: &CudaCachedLayerWeights,
    kv: &mut CudaKvCache,
    layer_idx: usize,
    _pos: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    heads_per_group: usize,
    norm_eps: f32,
    hidden_size: usize,
    intermediate_size: usize,
    bufs: &mut CudaFullLayerBuffers,
) -> Result<(), CudaGraphError> {
    let h_u32 = hidden_size as u32;
    let nq_u32 = nq as u32;
    let nkv_u32 = nkv as u32;
    let hd_u32 = head_dim as u32;
    let inter_u32 = intermediate_size as u32;
    let qkv_total_rows = (nq * head_dim + 2 * nkv * head_dim) as u32;
    let heads_per_group_u32 = heads_per_group as u32;
    let max_seq_u32 = bufs.max_seq as u32;
    let inv_sqrt_hd = 1.0f32 / (head_dim as f32).sqrt();
    let layer_offset = kv.layer_offset_elements(layer_idx);

    // ── Attention sublayer ────────────────────────────────────────────────

    // Step 1: RMSNorm(d_hidden → d_normed)
    graph.launch_rmsnorm_pub(
        &bufs.d_hidden,
        &weights.pre_attn_norm,
        &mut bufs.d_normed,
        h_u32,
        norm_eps,
    )?;

    // Step 2: Fused QKV GEMV (d_normed → d_qkv)
    graph.launch_gemv_pub(
        &weights.q_weight,
        &bufs.d_normed,
        &mut bufs.d_qkv,
        qkv_total_rows,
        h_u32,
    )?;

    // Step 3: Fused QK-Norm + RoPE
    let k_offset = nq * head_dim;
    let k_in_view = bufs.d_qkv.slice(k_offset..);
    launch_fused_qk_norm_rope(
        graph,
        mods,
        &bufs.d_qkv,
        &k_in_view,
        &mut bufs.d_q_rope,
        &mut bufs.d_k_rope,
        &weights.q_norm,
        &weights.k_norm,
        &bufs.d_cos,
        &bufs.d_sin,
        nq_u32,
        nkv_u32,
        hd_u32,
        norm_eps,
    )?;

    // Step 4: Fused KV-Store — pos read from d_pos_seqlen[0] by the kernel
    let v_offset = (nq + nkv) * head_dim;
    let v_view = bufs.d_qkv.slice(v_offset..);
    launch_fused_kv_store(
        graph,
        mods,
        &bufs.d_k_rope,
        &v_view,
        &mut kv.k_cache,
        &mut kv.v_cache,
        hd_u32,
        nkv_u32,
        max_seq_u32,
        &bufs.d_pos_seqlen,
        layer_offset,
    )?;

    // Step 5: Batched attention scores V2 — seq_len read from d_pos_seqlen[1]
    launch_batched_attn_scores_v2(
        graph,
        mods,
        &bufs.d_q_rope,
        &kv.k_cache,
        &mut bufs.d_scores,
        hd_u32,
        nq_u32,
        nkv_u32,
        heads_per_group_u32,
        max_seq_u32,
        &bufs.d_pos_seqlen,
        inv_sqrt_hd,
        layer_offset,
    )?;

    // Step 6: Softmax — seq_len read from d_pos_seqlen[1]
    launch_batched_softmax(
        graph,
        mods,
        &mut bufs.d_scores,
        nq_u32,
        max_seq_u32,
        &bufs.d_pos_seqlen,
    )?;

    // Step 7: Weighted sum — seq_len read from d_pos_seqlen[1]
    launch_batched_attn_weighted_sum(
        graph,
        mods,
        &bufs.d_scores,
        &kv.v_cache,
        &mut bufs.d_attn_out,
        hd_u32,
        nq_u32,
        nkv_u32,
        heads_per_group_u32,
        max_seq_u32,
        &bufs.d_pos_seqlen,
        layer_offset,
    )?;

    // ── FFN sublayer ──────────────────────────────────────────────────────

    // O-projection + residual fused: GEMV(attn_out) + hidden (in-place).
    // V8 kernel handles k=nq*head_dim (4096) with shared-mem.
    let attn_out_rows = (nq * head_dim) as u32;
    graph.launch_gemv_residual_pub(
        &weights.o_weight,
        &bufs.d_attn_out,
        &mut bufs.d_hidden,
        h_u32,
        attn_out_rows,
    )?;
    // FFN RMSNorm: hidden → normed
    graph.launch_rmsnorm_pub(
        &bufs.d_hidden,
        &weights.post_attn_norm,
        &mut bufs.d_normed,
        h_u32,
        norm_eps,
    )?;
    // Fused gate+up GEMV + SwiGLU: normed → d_swiglu (1 kernel vs 2)
    graph.launch_fused_gate_up_swiglu_pub(
        &weights.gate_up_weight,
        &bufs.d_normed,
        &mut bufs.d_swiglu,
        inter_u32,
        h_u32,
    )?;
    // Down projection + residual fused: GEMV(swiglu) + hidden (in-place).
    // V9 kernel handles k=intermediate_size (14336) beyond shared-mem limit.
    graph.launch_gemv_residual_pub(
        &weights.down_weight,
        &bufs.d_swiglu,
        &mut bufs.d_hidden,
        h_u32,
        inter_u32,
    )?;

    Ok(())
}

// =============================================================================
// Per-layer parameter struct (mirrors metal_full_layer::FullForwardLayerParams)
// =============================================================================

/// Per-layer parameters for the CUDA full-forward path.
///
/// Mirrors [`FullForwardLayerParams`] in `metal_full_layer` so callers can
/// build params in a backend-agnostic fashion.
pub struct CudaFullForwardLayerParams<'a> {
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

// =============================================================================
// encode_full_forward  (all layers, no intermediate CPU syncs)
// =============================================================================

/// Run the full forward pass (all layers) on GPU without intermediate syncs.
///
/// # CUDA Graph acceleration
///
/// After the first token, the entire 36-layer kernel sequence is captured as a
/// replayable CUDA driver graph (`CUgraphExec`).  On every subsequent token:
/// 1. Per-token inputs (`hidden_init`, `rope_cos/sin`, `d_pos_seqlen`) are
///    uploaded via regular H2D on the same stream.
/// 2. `cuGraphLaunch` submits the pre-built execution graph to the stream;
///    CUDA guarantees the kernels execute after the preceding H2D ops.
/// 3. A single `cuStreamSynchronize` + D2H transfer yields the result.
///
/// This eliminates ~40ms of per-kernel scheduling overhead (468 launches × ~85 µs
/// at 300 MHz SM clock) for a projected **2× decode speedup**.
///
/// # Graph validity
///
/// The captured graph stores raw device pointers to `d_hidden`, `d_cos`,
/// `d_sin`, `d_pos_seqlen`, `d_scores`, the KV cache, and all weight slices.
/// `acquire_full_layer_buffers` invalidates the graph whenever buffer dimensions
/// change (model switch), so the pointers always remain valid during replay.
///
/// Returns the final hidden state (post-norm if `final_norm_weight` provided).
#[allow(clippy::too_many_arguments)]
pub fn encode_full_forward(
    graph: &Arc<CudaGraph>,
    hidden_init: &[f32],
    all_layer_weights: &[CudaCachedLayerWeights],
    rope_cos: &[f32],
    rope_sin: &[f32],
    pos: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    heads_per_group: usize,
    norm_eps: f32,
    hidden_size: usize,
    intermediate_size: usize,
    max_seq_len: usize,
    final_norm_weight: Option<&[f32]>,
    final_norm_handle: u64,
) -> Result<Vec<f32>, CudaGraphError> {
    let h = hidden_size;
    let half_dim = head_dim / 2;
    let n_layers = all_layer_weights.len();
    let h_u32 = h as u32;

    if hidden_init.len() < h {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "hidden_init too short: need {h}, got {}",
            hidden_init.len()
        )));
    }
    if rope_cos.len() < half_dim {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "rope_cos too short: need {half_dim}, got {}",
            rope_cos.len()
        )));
    }
    if rope_sin.len() < half_dim {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "rope_sin too short: need {half_dim}, got {}",
            rope_sin.len()
        )));
    }
    if n_layers == 0 {
        return Err(CudaGraphError::WeightLayoutError(
            "encode_full_forward: no layers provided".into(),
        ));
    }

    let attn_mods = init_attn_modules(graph)?;

    // Allocate / reuse activation buffers.
    let mut fl_guard =
        acquire_full_layer_buffers(graph, h, nq, nkv, head_dim, max_seq_len, intermediate_size)?;
    let bufs = fl_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("full_layer_buffers not allocated".into()))?;

    // Allocate / reuse KV cache.
    let mut kv_guard = acquire_kv_cache(graph, n_layers, nkv, max_seq_len, head_dim)?;
    let kv = kv_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("kv_cache not allocated".into()))?;

    let stream = graph.stream_arc();

    // Upload [pos, seq_len] to device buffer (8 bytes).  Written on every token
    // so the captured graph reads the correct values at replay time.
    // Upload [pos, seq_len] to the 2-element device buffer.  Updated every token
    // so the replayed graph reads the correct position at replay time.
    let pos_seqlen_host = [pos as u32, (pos + 1) as u32];
    unsafe {
        graph
            .raw_htod(&pos_seqlen_host, &mut bufs.d_pos_seqlen, 2)
            .map_err(|e| CudaGraphError::DriverError(format!("upload pos_seqlen: {e}")))?;
    }

    // ── Fast path: replay captured CUDA graph (every token after the first) ─
    {
        let graph_guard = full_layer_state()
            .cuda_driver_graph
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        if let Some(Some(ref holder)) = *graph_guard {
            // Upload per-token inputs on the same stream BEFORE graph launch.
            // CUDA stream ordering ensures these H2D copies complete before the
            // first kernel node in the replayed graph begins executing.
            unsafe {
                graph.raw_htod(&hidden_init[..h], &mut bufs.d_hidden, h)?;
                graph.raw_htod(&rope_cos[..half_dim], &mut bufs.d_cos, half_dim)?;
                graph.raw_htod(&rope_sin[..half_dim], &mut bufs.d_sin, half_dim)?;
            }

            // Submit the entire 36-layer kernel sequence as one driver call.
            holder
                .0
                .launch()
                .map_err(|e| CudaGraphError::DriverError(format!("graph launch: {e}")))?;

            // Enqueue the D2H copy on the same stream — it is automatically
            // ordered after the graph nodes.  One synchronise at the end is
            // sufficient; a separate sync before D2H is not needed.
            let mut result = vec![0.0f32; h];
            unsafe { graph.raw_dtoh(&bufs.d_hidden, &mut result, h)? }
            stream
                .synchronize()
                .map_err(|e| CudaGraphError::DriverError(format!("fast-path sync: {e}")))?;

            return Ok(result);
        }
    } // cuda_driver_graph lock released

    // ── Slow path: first call (or after buffer realloc) ──────────────────────
    // Execute all layers normally, then capture the kernel sequence for replay.

    // Upload initial hidden state and RoPE tables via raw H2D.
    unsafe {
        graph
            .raw_htod(&hidden_init[..h], &mut bufs.d_hidden, h)
            .map_err(|e| CudaGraphError::DriverError(format!("upload hidden_init: {e}")))?;
        graph
            .raw_htod(&rope_cos[..half_dim], &mut bufs.d_cos, half_dim)
            .map_err(|e| CudaGraphError::DriverError(format!("upload cos ff: {e}")))?;
        graph
            .raw_htod(&rope_sin[..half_dim], &mut bufs.d_sin, half_dim)
            .map_err(|e| CudaGraphError::DriverError(format!("upload sin ff: {e}")))?;
    }

    // Loop over all layers — pure device computation.
    for (layer_idx, weights) in all_layer_weights.iter().enumerate() {
        unsafe {
            encode_layer_device(
                graph,
                &attn_mods,
                weights,
                kv,
                layer_idx,
                pos,
                nq,
                nkv,
                head_dim,
                heads_per_group,
                norm_eps,
                h,
                intermediate_size,
                bufs,
            )?;
        }
    }

    // Optional final RMSNorm.
    if let Some(fnorm_data) = final_norm_weight {
        // Ensure weight is cached before capture (so capture sees only kernel launches).
        let d_fnorm = get_or_upload_f32_weight(graph, final_norm_handle, fnorm_data)?;
        unsafe {
            graph.launch_rmsnorm_pub(
                &bufs.d_hidden,
                &d_fnorm,
                &mut bufs.d_normed,
                h_u32,
                norm_eps,
            )?;
        }
        // Copy normed → hidden so d_hidden holds the final output for download.
        stream
            .memcpy_dtod(&bufs.d_normed, &mut bufs.d_hidden)
            .map_err(|e| CudaGraphError::DriverError(format!("dtod normed->hidden: {e}")))?;
    }

    // Enqueue D2H on the same stream — ordered after all preceding kernels.
    // One synchronise after the async copy is sufficient; no pre-D2H sync needed.
    let mut result = vec![0.0f32; h];
    unsafe { graph.raw_dtoh(&bufs.d_hidden, &mut result, h)? }
    stream
        .synchronize()
        .map_err(|e| CudaGraphError::DriverError(format!("ff D2H sync: {e}")))?;

    // ── Capture the kernel sequence as a replayable CUDA driver graph ─────────
    // Best-effort: failures here are non-fatal.  We already have a valid
    // `result` from the normal execution above — the graph only accelerates
    // subsequent tokens.
    //
    // Event tracking is permanently disabled (CudaGraph::new), so no
    // cuEventRecord/cuStreamWaitEvent calls are injected.  Capture sees only
    // clean kernel-launch nodes and D2D memcpy nodes.
    {
        if let Ok(ref mut graph_guard) = full_layer_state().cuda_driver_graph.lock() {
            // Only attempt capture if this is the first time (None).
            // Some(None) means a previous attempt failed — don't retry.
            if graph_guard.is_none() {
                let begin_ok = stream
                    .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
                    .is_ok();
                if !begin_ok {
                    warn!("CUDA graph: begin_capture failed — running without graph replay");
                    // Mark as tried-but-failed so we never retry.
                    **graph_guard = Some(None);
                } else {
                    // Record all layers.  Kernels are NOT executed during capture.
                    let record_ok: bool = (|| -> Result<(), CudaGraphError> {
                        for (layer_idx, weights) in all_layer_weights.iter().enumerate() {
                            unsafe {
                                encode_layer_device(
                                    graph,
                                    &attn_mods,
                                    weights,
                                    kv,
                                    layer_idx,
                                    pos,
                                    nq,
                                    nkv,
                                    head_dim,
                                    heads_per_group,
                                    norm_eps,
                                    h,
                                    intermediate_size,
                                    bufs,
                                )?;
                            }
                        }
                        // Record final norm if present (weight already cached).
                        if let Some(fnorm_data) = final_norm_weight {
                            let d_fnorm =
                                get_or_upload_f32_weight(graph, final_norm_handle, fnorm_data)?;
                            unsafe {
                                graph.launch_rmsnorm_pub(
                                    &bufs.d_hidden,
                                    &d_fnorm,
                                    &mut bufs.d_normed,
                                    h_u32,
                                    norm_eps,
                                )?;
                            }
                            // D2D copy → MemcpyNode in the captured graph.
                            stream
                                .memcpy_dtod(&bufs.d_normed, &mut bufs.d_hidden)
                                .map_err(|e| {
                                    CudaGraphError::DriverError(format!("dtod (capture): {e}"))
                                })?;
                        }
                        Ok(())
                    })()
                    .is_ok();

                    // Must call end_capture whenever begin_capture succeeded.
                    // Leaving the stream in capture mode corrupts future ops.
                    // SAFETY: 0 = "no special flags" for cuGraphInstantiateWithFlags.
                    //         CUgraphInstantiate_flags is #[repr(u32)]; transmuting safe.
                    let no_flags: sys::CUgraphInstantiate_flags =
                        unsafe { core::mem::transmute(0u32) };
                    match stream.end_capture(no_flags) {
                        Ok(Some(cu_graph)) if record_ok => match cu_graph.upload() {
                            Ok(()) => {
                                **graph_guard = Some(Some(CuGraphHolder(cu_graph)));
                                debug!("CUDA graph captured and uploaded successfully");
                            }
                            Err(e) => {
                                warn!("CUDA graph upload failed: {e} — disabling replay");
                                **graph_guard = Some(None);
                            }
                        },
                        Ok(_) => {
                            warn!("CUDA graph: end_capture returned no graph — disabling replay");
                            **graph_guard = Some(None);
                        }
                        Err(e) => {
                            warn!("CUDA graph: end_capture error: {e} — disabling replay");
                            **graph_guard = Some(None);
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

// =============================================================================
// try_cuda_full_forward  (public entry point)
// =============================================================================

/// Attempt to run the full inference forward pass (all N layers) on CUDA GPU.
///
/// Mirrors `try_metal_full_forward` for the CUDA backend.  All layer weights
/// are uploaded/cached on first call and reused on subsequent tokens.
///
/// Returns `None` on any error (callers fall back to the CPU path).
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_full_forward(
    hidden: &[f32],
    layer_params: &[CudaFullForwardLayerParams<'_>],
    rope_cos: &[f32],
    rope_sin: &[f32],
    pos: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    heads_per_group: usize,
    norm_eps: f32,
    hidden_size: usize,
    intermediate_size: usize,
    max_seq_len: usize,
    final_norm_bytes: Option<&[f32]>,
    final_norm_handle: u64,
) -> Option<Vec<f32>> {
    let _t0 = profiling().then(std::time::Instant::now);

    // Retrieve or build the cached model weights (O(1) Arc clones on warm path).
    let (graph, layer_weights) = get_or_build_model_weights(layer_params)?;

    let _t1 = profiling().then(std::time::Instant::now);
    if profiling() {
        eprintln!(
            "[cuda-prof] try_ff pos={pos}: weight_lookup={:.3}ms",
            (_t1.expect("profiling") - _t0.expect("profiling")).as_secs_f64() * 1000.0,
        );
    }

    let r = encode_full_forward(
        &graph,
        hidden,
        &*layer_weights,
        rope_cos,
        rope_sin,
        pos,
        nq,
        nkv,
        head_dim,
        heads_per_group,
        norm_eps,
        hidden_size,
        intermediate_size,
        max_seq_len,
        final_norm_bytes,
        final_norm_handle,
    );
    if profiling() {
        let elapsed = _t1.expect("profiling").elapsed().as_secs_f64() * 1000.0;
        let path = if pos == 0 { "slow" } else { "fast" };
        eprintln!("[cuda-prof] encode_ff pos={pos} path={path}: {elapsed:.1}ms");
    }
    if let Err(ref e) = r {
        warn!("CUDA full-forward error at pos={pos}: {e}");
    }
    r.ok()
}

// =============================================================================
// try_cuda_full_forward_with_gpu_lm_head  (GPU LM-head path)
// =============================================================================

/// Run all transformer layers + final RMSNorm + LM-head GEMV entirely on GPU.
///
/// This eliminates the CPU LM-head GEMV which takes ~20ms for large vocabularies.
/// The LM-head weight is uploaded/cached on first call.
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_full_forward_with_gpu_lm_head(
    hidden: &[f32],
    layer_params: &[CudaFullForwardLayerParams<'_>],
    rope_cos: &[f32],
    rope_sin: &[f32],
    pos: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    heads_per_group: usize,
    norm_eps: f32,
    hidden_size: usize,
    intermediate_size: usize,
    max_seq_len: usize,
    final_norm_bytes: Option<&[f32]>,
    final_norm_handle: u64,
    lm_head_handle: u64,
    lm_head_bytes: &[u8],
    vocab_size: usize,
) -> Option<Vec<f32>> {
    let normed = try_cuda_full_forward(
        hidden,
        layer_params,
        rope_cos,
        rope_sin,
        pos,
        nq,
        nkv,
        head_dim,
        heads_per_group,
        norm_eps,
        hidden_size,
        intermediate_size,
        max_seq_len,
        final_norm_bytes,
        final_norm_handle,
    )?;

    let graph = CudaGraph::global().ok()?;
    let _t_lm = profiling().then(std::time::Instant::now);
    let r = graph.encode_lm_head_gemv(
        &normed,
        lm_head_handle,
        lm_head_bytes,
        vocab_size,
        hidden_size,
    );
    if profiling() {
        eprintln!(
            "[cuda-prof] lm_head pos={pos}: {:.1}ms",
            _t_lm.expect("profiling").elapsed().as_secs_f64() * 1000.0
        );
    }
    r.ok()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
