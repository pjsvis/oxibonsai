//! Native CUDA dispatch engine for OxiBonsai Q1_0_G128 inference.
//!
//! Bypasses scirs2-core's broken CUDA stub and uses cudarc 0.19 directly.
//! Provides V7-quality SoA kernels (ported from the Metal/MSL implementation)
//! and a seven-kernel FFN pipeline that mirrors MetalGraph's command-buffer approach.
//!
//! # Architecture
//!
//! - **`CudaGraph`**: Owns the CUDA context, stream, compiled PTX modules, weight
//!   cache, and activation buffer pool.  Exposed as a process-wide singleton via
//!   [`CudaGraph::global`].
//! - **`try_cuda_ffn`** / **`try_cuda_qkv`**: Public convenience functions called
//!   directly from `block.rs`, mirroring `try_metal_ffn` / `try_metal_qkv`.
//!
//! # Weight cache
//!
//! On first use, Q1_0_G128 weight bytes are:
//! 1. Reformatted from AoS (`[scale|data]×N`) to SoA (`[all scales][all data]`)
//! 2. Uploaded to GPU device memory via `stream.clone_htod`
//! 3. Stored in `weight_cache` as `Arc<CudaSlice<u8>>` keyed by `u64` handle ID
//!
//! Subsequent calls look up the handle and skip the copy.
//!
//! # FFN pipeline (7 kernel launches on a single stream)
//!
//! 1. Upload `hidden`, `attn_out`, `norm_weight` → device
//! 2. `gemv_q1_g128_v7(attn_proj, attn_out → proj, h, h)`
//! 3. `residual_add(hidden, proj, h)`
//! 4. `rmsnorm_weighted_v2(hidden, norm_weight → normed, h, eps)`
//! 5. `gemv_q1_g128_v7(gate_up, normed → gate_up_buf, 2×inter, h)`
//! 6. `swiglu_fused(gate_up_buf → swiglu_buf, inter)`
//! 7. `gemv_q1_g128_v7(down, swiglu_buf → down_buf, h, inter)`
//! 8. `residual_add(hidden, down_buf, h)`
//! 9. Stream-synchronised download `hidden` → host

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use cudarc::driver::{
    result as cudarc_result, CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr,
    DevicePtrMut, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tracing::{debug, warn};

use super::cuda_kernels::CUDA_V7_KERNELS_SRC;

// ═══════════════════════════════════════════════════════════════════════════
// PTX disk cache — skip NVRTC on subsequent runs
// ═══════════════════════════════════════════════════════════════════════════

/// FNV-1a 64-bit hash of a byte string.
fn fnv1a_64(data: &[u8]) -> u64 {
    const BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = BASIS;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Try to load a compiled PTX from the disk cache.
fn load_ptx_cache(src_hash: u64, tag: &str) -> Option<cudarc::nvrtc::Ptx> {
    let path = format!("/tmp/oxibonsai_ptx_{src_hash:016x}_{tag}.ptx");
    let ptx_src = std::fs::read_to_string(&path).ok()?;
    Some(cudarc::nvrtc::Ptx::from_src(ptx_src))
}

/// Save a compiled PTX to the disk cache (best-effort; ignores write errors).
fn save_ptx_cache(ptx: &cudarc::nvrtc::Ptx, src_hash: u64, tag: &str) {
    let path = format!("/tmp/oxibonsai_ptx_{src_hash:016x}_{tag}.ptx");
    let _ = std::fs::write(&path, ptx.to_src());
}

/// Compile PTX from CUDA C source, using a disk cache keyed on the source hash.
///
/// First call: compiles via NVRTC (~5s), saves to /tmp.
/// Subsequent calls: loads from /tmp (~1ms), skips NVRTC entirely.
pub(crate) fn compile_or_load_ptx(
    src: &str,
    tag: &str,
) -> Result<cudarc::nvrtc::Ptx, CudaGraphError> {
    let hash = fnv1a_64(src.as_bytes());
    if let Some(cached) = load_ptx_cache(hash, tag) {
        debug!("PTX cache hit for tag={tag} hash={hash:016x}");
        return Ok(cached);
    }
    debug!("PTX cache miss for tag={tag}, compiling...");
    let ptx = compile_ptx(src).map_err(|e| CudaGraphError::CompilationFailed(format!("{e}")))?;
    save_ptx_cache(&ptx, hash, tag);
    debug!("PTX compiled and cached: tag={tag}");
    Ok(ptx)
}

// ═══════════════════════════════════════════════════════════════════════════
// Error type
// ═══════════════════════════════════════════════════════════════════════════

/// Errors from the CUDA graph dispatch engine.
#[derive(Debug)]
pub enum CudaGraphError {
    /// No CUDA-capable GPU found or driver not present.
    DeviceNotFound(String),
    /// NVRTC PTX compilation failed.
    CompilationFailed(String),
    /// CUDA driver API error.
    DriverError(String),
    /// A requested weight handle was not found in the cache.
    WeightNotFound(u64),
    /// The internal weight layout conversion failed (malformed bytes).
    WeightLayoutError(String),
    /// A mutex was poisoned.
    LockPoisoned,
}

impl fmt::Display for CudaGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceNotFound(s) => write!(f, "no CUDA device: {s}"),
            Self::CompilationFailed(s) => write!(f, "PTX compilation failed: {s}"),
            Self::DriverError(s) => write!(f, "CUDA driver error: {s}"),
            Self::WeightNotFound(id) => write!(f, "weight handle {id} not in cache"),
            Self::WeightLayoutError(s) => write!(f, "weight layout error: {s}"),
            Self::LockPoisoned => write!(f, "mutex lock poisoned"),
        }
    }
}

impl std::error::Error for CudaGraphError {}

// ═══════════════════════════════════════════════════════════════════════════
// Handle ID allocator
// ═══════════════════════════════════════════════════════════════════════════

static NEXT_HANDLE_ID: AtomicU64 = AtomicU64::new(1);

/// Allocate a new globally-unique weight handle ID.
pub(crate) fn alloc_handle_id() -> u64 {
    NEXT_HANDLE_ID.fetch_add(1, Ordering::Relaxed)
}

// ═══════════════════════════════════════════════════════════════════════════
// Compiled CUDA modules
// ═══════════════════════════════════════════════════════════════════════════

/// Handles to all compiled CUDA kernel functions used by `CudaGraph`.
#[allow(dead_code)]
struct CudaModules {
    gemv_q1_g128_v7: CudaFunction,
    gemv_q1_g128_v7_residual: CudaFunction,
    gemv_q1_g128_v8: CudaFunction,
    gemv_q1_g128_v8_residual: CudaFunction,
    gemv_q1_g128_v9: CudaFunction,
    gemv_q1_g128_v9_residual: CudaFunction,
    rmsnorm_weighted_v2: CudaFunction,
    residual_add: CudaFunction,
    swiglu_fused: CudaFunction,
    /// Fused gate+up Q1 GEMV with SwiGLU epilogue — halves dispatch count for FFN step 5+6.
    fused_gate_up_swiglu: CudaFunction,
    argmax_f32: CudaFunction,
}

// ═══════════════════════════════════════════════════════════════════════════
// Activation buffer pool
// ═══════════════════════════════════════════════════════════════════════════

/// Pre-allocated GPU activation buffers for a single FFN forward pass.
///
/// `d_scratch` is reused for both the attn_proj GEMV output (size h) and the
/// down GEMV output (size h) — they are never needed simultaneously.  This
/// saves one GPU buffer vs. keeping separate `d_proj` and `d_down`.
///
/// Resized lazily when `hidden_size` or `intermediate_size` changes.
#[allow(dead_code)]
struct CudaActivationBuffers {
    d_hidden: CudaSlice<f32>,      // h — residual stream (in/out)
    d_attn_out: CudaSlice<f32>,    // h — attention output (input to attn_proj GEMV)
    d_norm_weight: CudaSlice<f32>, // h — FFN pre-norm weights
    d_scratch: CudaSlice<f32>,     // h — scratch: attn_proj output OR down output
    d_normed: CudaSlice<f32>,      // h — RMSNorm output
    /// Intermediate gate+up GEMV output (2 × inter). Retained as a pre-allocated fallback
    /// buffer; not used in the fused `fused_gate_up_swiglu_q1` path.
    #[allow(dead_code)]
    d_gate_up: CudaSlice<f32>,
    d_swiglu: CudaSlice<f32>, // inter — SwiGLU output
    hidden_size: usize,
    intermediate_size: usize,
}

impl CudaActivationBuffers {
    fn matches(&self, h: usize, inter: usize) -> bool {
        self.hidden_size == h && self.intermediate_size == inter
    }
}

/// Pre-allocated GPU buffers for the LM-head GEMV.
struct LmHeadBuffers {
    d_input: CudaSlice<f32>,  // [hidden_size]
    d_output: CudaSlice<f32>, // [vocab_size]
    hidden_capacity: usize,
    vocab_capacity: usize,
}

impl LmHeadBuffers {
    fn fits(&self, hidden: usize, vocab: usize) -> bool {
        self.hidden_capacity >= hidden && self.vocab_capacity >= vocab
    }
}

/// Pre-allocated GPU buffers for the QKV projection.
///
/// Eliminates per-call `cuMemAlloc`/`cuMemFree` in `encode_qkv_phase`.
struct QkvBuffers {
    d_input: CudaSlice<f32>,  // h (normed hidden state)
    d_output: CudaSlice<f32>, // max_qkv_rows (Q + K + V)
    input_capacity: usize,
    output_capacity: usize,
}

impl QkvBuffers {
    fn fits(&self, input_len: usize, output_len: usize) -> bool {
        self.input_capacity >= input_len && self.output_capacity >= output_len
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CudaGraph — main engine
// ═══════════════════════════════════════════════════════════════════════════

/// Direct CUDA dispatch engine, mirroring [`MetalGraph`] for Linux/Windows.
///
/// Owns the CUDA context, stream, compiled kernels, weight cache, and
/// activation buffer pool. All state is protected by `Mutex` to satisfy
/// `Send + Sync` for `OnceLock<Arc<CudaGraph>>` storage.
pub struct CudaGraph {
    #[allow(dead_code)]
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    modules: CudaModules,
    buffers: Mutex<Option<CudaActivationBuffers>>,
    qkv_buffers: Mutex<Option<QkvBuffers>>,
    weight_cache: Mutex<HashMap<u64, Arc<CudaSlice<u8>>>>,
    /// Separate cache for f32 tensors (norm weights, RoPE buffers, etc.)
    f32_weight_cache: Mutex<HashMap<u64, Arc<CudaSlice<f32>>>>,
    /// Pre-allocated GPU buffers for the LM-head GEMV.
    lm_head_buffers: Mutex<Option<LmHeadBuffers>>,
}

// SAFETY: CudaContext/CudaStream/CudaSlice are all Send in cudarc.
// The Mutex guards provide Sync.
unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

// ── Singleton ──────────────────────────────────────────────────────────────

static GLOBAL_CUDA_GRAPH: OnceLock<Mutex<Option<Arc<CudaGraph>>>> = OnceLock::new();

impl CudaGraph {
    /// Access the process-wide `CudaGraph` singleton, initialising on first call.
    ///
    /// Returns `Err` if no CUDA device is present or PTX compilation fails.
    pub fn global() -> Result<Arc<CudaGraph>, CudaGraphError> {
        let mutex = GLOBAL_CUDA_GRAPH.get_or_init(|| Mutex::new(None));
        let mut guard = mutex.lock().map_err(|_| CudaGraphError::LockPoisoned)?;
        if let Some(ref cached) = *guard {
            return Ok(Arc::clone(cached));
        }
        let graph = Arc::new(Self::new()?);
        *guard = Some(Arc::clone(&graph));
        debug!("CudaGraph singleton initialised");
        Ok(graph)
    }

    /// Construct a new `CudaGraph` — heavy operation (device init + NVRTC compile).
    fn new() -> Result<Self, CudaGraphError> {
        let context =
            CudaContext::new(0).map_err(|e| CudaGraphError::DeviceNotFound(format!("{e}")))?;
        // Permanently disable cudarc's multi-stream event tracking.
        // We use exactly one stream throughout inference, so cross-stream
        // synchronisation events provide zero correctness benefit while adding
        // ~900 cuStreamWaitEvent / cuEventRecord calls per decode token
        // (~30 ms of wasted driver round-trips at 300 MHz SM clock).
        // Must be called BEFORE new_stream() so the context's event_tracking
        // flag is already false when num_streams increments to 1, preventing
        // cudarc from creating events on any CudaSlice.
        // SAFETY: single-stream single-threaded decode; no other thread reads
        // these slices from a different stream.
        unsafe {
            context.disable_event_tracking();
        }

        // Use a real non-blocking stream (not the legacy null stream) so that
        // cuStreamBeginCapture can be called later for CUDA Graph capture.
        // The legacy/null stream (from default_stream()) does not support capture.
        let stream = context
            .new_stream()
            .map_err(|e| CudaGraphError::DriverError(format!("create stream: {e}")))?;

        // Compile all V7 kernels in one PTX compilation (disk-cached after first run).
        let ptx = compile_or_load_ptx(CUDA_V7_KERNELS_SRC, "v7_kernels")?;
        let module = context
            .load_module(ptx)
            .map_err(|e| CudaGraphError::DriverError(format!("load_module: {e}")))?;

        let load = |name: &str| -> Result<CudaFunction, CudaGraphError> {
            module
                .load_function(name)
                .map_err(|e| CudaGraphError::DriverError(format!("load_function({name}): {e}")))
        };

        let modules = CudaModules {
            gemv_q1_g128_v7: load("gemv_q1_g128_v7")?,
            gemv_q1_g128_v7_residual: load("gemv_q1_g128_v7_residual")?,
            gemv_q1_g128_v8: load("gemv_q1_g128_v8")?,
            gemv_q1_g128_v8_residual: load("gemv_q1_g128_v8_residual")?,
            gemv_q1_g128_v9: load("gemv_q1_g128_v9")?,
            gemv_q1_g128_v9_residual: load("gemv_q1_g128_v9_residual")?,
            rmsnorm_weighted_v2: load("rmsnorm_weighted_v2")?,
            residual_add: load("residual_add")?,
            swiglu_fused: load("swiglu_fused")?,
            fused_gate_up_swiglu: load("fused_gate_up_swiglu_q1")?,
            argmax_f32: load("argmax_f32")?,
        };

        Ok(Self {
            context,
            stream,
            modules,
            buffers: Mutex::new(None),
            qkv_buffers: Mutex::new(None),
            weight_cache: Mutex::new(HashMap::new()),
            f32_weight_cache: Mutex::new(HashMap::new()),
            lm_head_buffers: Mutex::new(None),
        })
    }

    // ── AoS → SoA reformatter (identical to metal_graph.rs) ──────────────

    /// Reformat Q1_0_G128 weight bytes from AoS to SoA layout.
    ///
    /// AoS: `[scale₀|data₀][scale₁|data₁]...` (18 bytes/block)
    /// SoA: `[scale₀…scaleₙ][data₀…dataₙ]`
    ///
    /// Returns `None` if `aos_bytes.len()` is not a multiple of 18.
    fn reformat_q1_aos_to_soa(aos_bytes: &[u8]) -> Option<Vec<u8>> {
        const BLOCK_BYTES: usize = 18;
        const SCALE_BYTES: usize = 2;
        const DATA_BYTES: usize = 16;

        if aos_bytes.is_empty() || aos_bytes.len() % BLOCK_BYTES != 0 {
            return None;
        }
        let n_blocks = aos_bytes.len() / BLOCK_BYTES;
        let mut soa = vec![0u8; n_blocks * BLOCK_BYTES];
        let (scales_section, data_section) = soa.split_at_mut(n_blocks * SCALE_BYTES);

        for i in 0..n_blocks {
            let src = i * BLOCK_BYTES;
            scales_section[i * SCALE_BYTES..i * SCALE_BYTES + SCALE_BYTES]
                .copy_from_slice(&aos_bytes[src..src + SCALE_BYTES]);
            data_section[i * DATA_BYTES..i * DATA_BYTES + DATA_BYTES]
                .copy_from_slice(&aos_bytes[src + SCALE_BYTES..src + BLOCK_BYTES]);
        }
        Some(soa)
    }

    // ── Weight cache management ───────────────────────────────────────────

    /// Return a cached weight slice or upload it on demand.
    ///
    /// On first call for `handle_id`: converts `aos_bytes` to SoA, uploads to GPU,
    /// caches the slice.  On subsequent calls: returns the cached `Arc` immediately.
    pub fn get_or_upload_weight_soa(
        &self,
        handle_id: u64,
        aos_bytes: &[u8],
    ) -> Result<Arc<CudaSlice<u8>>, CudaGraphError> {
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        if let Some(existing) = cache.get(&handle_id) {
            return Ok(Arc::clone(existing));
        }
        let soa = Self::reformat_q1_aos_to_soa(aos_bytes).ok_or_else(|| {
            CudaGraphError::WeightLayoutError(format!(
                "AoS bytes length {} not divisible by 18",
                aos_bytes.len()
            ))
        })?;
        let d_weight = self
            .stream
            .clone_htod(&soa)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod weight: {e}")))?;
        let arc = Arc::new(d_weight);
        cache.insert(handle_id, Arc::clone(&arc));
        Ok(arc)
    }

    /// Like [`get_or_upload_weight_soa`] but accepts a lazy byte producer.
    ///
    /// The closure is only called on the first use of `handle_id`.  Useful when
    /// the caller needs to concatenate gate+up bytes without computing them on
    /// every token.
    pub fn get_or_upload_weight_soa_lazy<F>(
        &self,
        handle_id: u64,
        make_bytes: F,
    ) -> Result<Arc<CudaSlice<u8>>, CudaGraphError>
    where
        F: FnOnce() -> Vec<u8>,
    {
        {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            if let Some(existing) = cache.get(&handle_id) {
                return Ok(Arc::clone(existing));
            }
        }
        // Produce bytes outside the lock to avoid holding it during allocation.
        let aos_bytes = make_bytes();
        self.get_or_upload_weight_soa(handle_id, &aos_bytes)
    }

    /// Upload raw SoA bytes directly (used by `NativeCudaBackend::upload_weights_raw`).
    pub fn upload_weight_soa_new(
        &self,
        handle_id: u64,
        aos_bytes: &[u8],
    ) -> Result<(), CudaGraphError> {
        let _ = self.get_or_upload_weight_soa(handle_id, aos_bytes)?;
        Ok(())
    }

    // ── Activation buffer allocation ─────────────────────────────────────

    /// Ensure activation buffers are allocated for `(hidden_size, intermediate_size)`.
    /// Re-allocates if dimensions changed.
    fn acquire_buffers(
        &self,
        h: usize,
        inter: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<CudaActivationBuffers>>, CudaGraphError> {
        let mut guard = self
            .buffers
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;

        let needs_alloc = match guard.as_ref() {
            Some(b) => !b.matches(h, inter),
            None => true,
        };

        if needs_alloc {
            let alloc_f32 = |n: usize| -> Result<CudaSlice<f32>, CudaGraphError> {
                self.stream
                    .alloc_zeros::<f32>(n)
                    .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros({n}): {e}")))
            };
            *guard = Some(CudaActivationBuffers {
                d_hidden: alloc_f32(h)?,
                d_attn_out: alloc_f32(h)?,
                d_norm_weight: alloc_f32(h)?,
                d_scratch: alloc_f32(h)?,
                d_normed: alloc_f32(h)?,
                d_gate_up: alloc_f32(2 * inter)?,
                d_swiglu: alloc_f32(inter)?,
                hidden_size: h,
                intermediate_size: inter,
            });
        }

        Ok(guard)
    }

    /// Ensure QKV projection buffers are allocated for `(input_len, output_len)`.
    /// Re-allocates if the existing buffers are too small.
    fn acquire_qkv_buffers(
        &self,
        input_len: usize,
        output_len: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<QkvBuffers>>, CudaGraphError> {
        let mut guard = self
            .qkv_buffers
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;

        let needs_alloc = match guard.as_ref() {
            Some(b) => !b.fits(input_len, output_len),
            None => true,
        };

        if needs_alloc {
            let alloc = |n: usize| -> Result<CudaSlice<f32>, CudaGraphError> {
                self.stream
                    .alloc_zeros::<f32>(n)
                    .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros qkv({n}): {e}")))
            };
            *guard = Some(QkvBuffers {
                d_input: alloc(input_len)?,
                d_output: alloc(output_len)?,
                input_capacity: input_len,
                output_capacity: output_len,
            });
        }
        Ok(guard)
    }

    /// Shared-memory bytes required for the V8 kernel at given `k`.
    /// Returns `None` when `k` exceeds the 48 KB default shared-mem limit.
    #[inline]
    fn v8_shared_bytes(k: usize) -> Option<u32> {
        super::cuda_kernels::v8_shared_mem_bytes(k, 49_152)
    }

    // ── Low-level kernel launchers ────────────────────────────────────────

    /// Launch `gemv_q1_g128_v7` on the default stream.
    ///
    /// # Safety
    /// Caller must ensure all slices are valid device pointers on `self.stream`.
    unsafe fn launch_gemv_v7(
        &self,
        d_weight: &CudaSlice<u8>,
        d_input: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        let grid_x = n_rows.div_ceil(8);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.gemv_q1_g128_v7)
            .arg(d_weight)
            .arg(d_input)
            .arg(d_output)
            .arg(&n_rows)
            .arg(&k)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("gemv_v7 launch: {e}")))
    }

    /// Launch `gemv_q1_g128_v9`: vectorized 128-bit weight loads + `__ldg()` scales.
    ///
    /// No shared memory required; identical grid/block to V7.
    /// Use this for large `k` where V8 shared-mem would exceed 48 KB.
    ///
    /// # Safety
    /// Caller must ensure all slices are valid device pointers on `self.stream`.
    unsafe fn launch_gemv_v9(
        &self,
        d_weight: &CudaSlice<u8>,
        d_input: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        let grid_x = n_rows.div_ceil(8);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.gemv_q1_g128_v9)
            .arg(d_weight)
            .arg(d_input)
            .arg(d_output)
            .arg(&n_rows)
            .arg(&k)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("gemv_v9 launch: {e}")))
    }

    /// Launch `residual_add` on the default stream.
    unsafe fn launch_residual_add(
        &self,
        d_a: &mut CudaSlice<f32>,
        d_b: &CudaSlice<f32>,
        n: u32,
    ) -> Result<(), CudaGraphError> {
        let grid_x = n.div_ceil(256);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.residual_add)
            .arg(d_a)
            .arg(d_b)
            .arg(&n)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("residual_add launch: {e}")))
    }

    /// Launch `rmsnorm_weighted_v2` on the default stream.
    unsafe fn launch_rmsnorm(
        &self,
        d_input: &CudaSlice<f32>,
        d_weight: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n: u32,
        eps: f32,
    ) -> Result<(), CudaGraphError> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.rmsnorm_weighted_v2)
            .arg(d_input)
            .arg(d_weight)
            .arg(d_output)
            .arg(&n)
            .arg(&eps)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("rmsnorm launch: {e}")))
    }

    /// Launch `swiglu_fused` on the default stream.
    ///
    /// Kept as a fallback / building block; the hot path uses `launch_fused_gate_up_swiglu`
    /// which fuses the GEMV + SwiGLU steps into a single kernel dispatch.
    #[allow(dead_code)]
    unsafe fn launch_swiglu(
        &self,
        d_gate_up: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n: u32,
    ) -> Result<(), CudaGraphError> {
        let grid_x = n.div_ceil(256);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.swiglu_fused)
            .arg(d_gate_up)
            .arg(d_output)
            .arg(&n)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("swiglu launch: {e}")))
    }

    /// Launch `fused_gate_up_swiglu_q1`: fused gate+up Q1 GEMV with SwiGLU epilogue.
    ///
    /// Reads both the gate row and up row from the concatenated SoA weight matrix and
    /// writes `output[row] = SiLU(gate_dot) * up_dot` directly — no intermediate buffer.
    ///
    /// Grid: `(ceil(n_rows/8), 1, 1)`, Block: `(256, 1, 1)` (8 warps × 32 lanes).
    ///
    /// # Safety
    /// Caller must ensure all slices are valid device pointers on `self.stream`.
    unsafe fn launch_fused_gate_up_swiglu(
        &self,
        blocks: &CudaSlice<u8>,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        let grid_x = n_rows.div_ceil(8);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.fused_gate_up_swiglu)
            .arg(blocks)
            .arg(input)
            .arg(output)
            .arg(&n_rows)
            .arg(&k)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("fused_gate_up_swiglu launch: {e}")))
    }

    /// Launch `argmax_f32`: find the index of the maximum f32 value in `input`.
    ///
    /// # Safety
    /// Caller must ensure all slices are valid device pointers on `self.stream`.
    unsafe fn launch_argmax(
        &self,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<u32>,
        n: u32,
    ) -> Result<(), CudaGraphError> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.argmax_f32)
            .arg(input)
            .arg(output)
            .arg(&n)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("argmax launch: {e}")))
    }

    /// Upload `logits` to device, launch the argmax kernel, and return the index.
    ///
    /// Grid: `(1, 1, 1)`, Block: `(256, 1, 1)` — single-block reduction over 256 threads.
    pub fn encode_argmax(&self, logits: &[f32]) -> Result<u32, CudaGraphError> {
        let n = logits.len() as u32;
        let d_input = self
            .stream
            .clone_htod(logits)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod argmax input: {e}")))?;
        let mut d_output = self
            .stream
            .alloc_zeros::<u32>(1)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros argmax output: {e}")))?;

        unsafe {
            self.launch_argmax(&d_input, &mut d_output, n)?;
        }

        let result = self
            .stream
            .clone_dtoh(&d_output)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_dtoh argmax result: {e}")))?;
        Ok(result[0])
    }

    /// Launch `gemv_q1_g128_v7_residual`:  `output[row] = dot(weight[row], input) + residual[row]`.
    #[allow(dead_code)]
    unsafe fn launch_gemv_v7_residual(
        &self,
        d_weight: &CudaSlice<u8>,
        d_input: &CudaSlice<f32>,
        d_residual: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        let grid_x = n_rows.div_ceil(8);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.gemv_q1_g128_v7_residual)
            .arg(d_weight)
            .arg(d_input)
            .arg(d_residual)
            .arg(d_output)
            .arg(&n_rows)
            .arg(&k)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("gemv_v7_residual launch: {e}")))
    }

    /// Launch `gemv_q1_g128_v8` (shared-memory input cache, k ≤ 48 KB threshold).
    ///
    /// `shared_mem_bytes` must be `(k/128) * 129 * 4`; caller computes via [`Self::v8_shared_bytes`].
    unsafe fn launch_gemv_v8(
        &self,
        d_weight: &CudaSlice<u8>,
        d_input: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
        shared_mem_bytes: u32,
    ) -> Result<(), CudaGraphError> {
        let grid_x = n_rows.div_ceil(8);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes,
        };
        self.stream
            .launch_builder(&self.modules.gemv_q1_g128_v8)
            .arg(d_weight)
            .arg(d_input)
            .arg(d_output)
            .arg(&n_rows)
            .arg(&k)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("gemv_v8 launch: {e}")))
    }

    /// Launch `gemv_q1_g128_v8_residual`: V8 GEMV + fused residual add.
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_gemv_v8_residual(
        &self,
        d_weight: &CudaSlice<u8>,
        d_input: &CudaSlice<f32>,
        d_residual: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
        shared_mem_bytes: u32,
    ) -> Result<(), CudaGraphError> {
        let grid_x = n_rows.div_ceil(8);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes,
        };
        self.stream
            .launch_builder(&self.modules.gemv_q1_g128_v8_residual)
            .arg(d_weight)
            .arg(d_input)
            .arg(d_residual)
            .arg(d_output)
            .arg(&n_rows)
            .arg(&k)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("gemv_v8_residual launch: {e}")))
    }

    /// Launch `gemv_q1_g128_v9_residual`: V9 vectorised GEMV + fused residual add.
    ///
    /// Identical grid/block to V9; no shared memory.  Used when `k` exceeds the 48 KB
    /// shared-mem limit that V8 requires.
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_gemv_v9_residual(
        &self,
        d_weight: &CudaSlice<u8>,
        d_input: &CudaSlice<f32>,
        d_residual: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        let grid_x = n_rows.div_ceil(8);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.gemv_q1_g128_v9_residual)
            .arg(d_weight)
            .arg(d_input)
            .arg(d_residual)
            .arg(d_output)
            .arg(&n_rows)
            .arg(&k)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("gemv_v9_residual launch: {e}")))
    }

    // ── High-level dispatch: single GEMV ─────────────────────────────────

    /// Execute a single Q1 GEMV (`output = weight × input`) and return the result.
    pub fn encode_gemv(
        &self,
        weight_id: u64,
        weight_bytes: &[u8],
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, CudaGraphError> {
        let d_weight = self.get_or_upload_weight_soa(weight_id, weight_bytes)?;

        let d_input = self
            .stream
            .clone_htod(input)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod input: {e}")))?;
        let mut d_output = self
            .stream
            .alloc_zeros::<f32>(n_rows)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros output: {e}")))?;

        unsafe {
            self.launch_gemv_v7(&d_weight, &d_input, &mut d_output, n_rows as u32, k as u32)?;
        }

        self.stream
            .clone_dtoh(&d_output)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_dtoh output: {e}")))
    }

    /// Execute a single GEMV using a pre-cached weight (handle already in cache).
    pub fn encode_gemv_cached(
        &self,
        weight_id: u64,
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, CudaGraphError> {
        let d_weight = {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            cache
                .get(&weight_id)
                .map(Arc::clone)
                .ok_or(CudaGraphError::WeightNotFound(weight_id))?
        };

        let d_input = self
            .stream
            .clone_htod(input)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod input: {e}")))?;
        let mut d_output = self
            .stream
            .alloc_zeros::<f32>(n_rows)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros output: {e}")))?;

        unsafe {
            self.launch_gemv_v7(&d_weight, &d_input, &mut d_output, n_rows as u32, k as u32)?;
        }

        self.stream
            .clone_dtoh(&d_output)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_dtoh output: {e}")))
    }

    // ── High-level dispatch: full FFN pipeline ────────────────────────────

    /// Execute the optimised FFN phase pipeline (6 kernel launches: fused gate+up+SwiGLU).
    ///
    /// Improvements over the V7 two-step (GEMV → SwiGLU) pipeline:
    /// - Steps 2 uses `gemv_q1_g128_v8` (shared-memory padded input cache) when
    ///   `k = hidden_size ≤ 48 KB threshold` → eliminates non-coalesced global reads.
    /// - Steps 5+6 are **fused** into `fused_gate_up_swiglu_q1` — reads gate and up
    ///   rows simultaneously and applies `SiLU(gate)*up` in the epilogue, halving the
    ///   dispatch count for this step vs. the old GEMV + swiglu_fused pair.
    /// - Hardware fp16 scale decode (`cvt.f32.f16`) in all kernels.
    /// - `d_scratch` reused for both attn_proj and down outputs → 1 fewer GPU buffer.
    ///
    /// | Step | Op                                                              |
    /// |------|-----------------------------------------------------------------|
    /// | 1    | Upload hidden, attn_out, norm_weight → device                   |
    /// | 2    | GEMV_v8(attn_proj, attn_out → scratch)                          |
    /// | 3    | residual_add(hidden += scratch)                                  |
    /// | 4    | rmsnorm(hidden, norm_weight → normed)                            |
    /// | 5    | fused_gate_up_swiglu_q1(gate_up, normed → swiglu_buf)           |
    /// | 6    | GEMV_v7/v8(down, swiglu → scratch)                              |
    /// | 7    | residual_add(hidden += scratch)                                  |
    /// | 8    | Download hidden → host (stream-synchronised)                     |
    #[allow(clippy::too_many_arguments)]
    pub fn encode_ffn_phase(
        &self,
        hidden: &mut [f32],
        attn_out: &[f32],
        norm_weight: &[f32],
        eps: f32,
        attn_proj_w: &Arc<CudaSlice<u8>>,
        gate_up_w: &Arc<CudaSlice<u8>>,
        down_w: &Arc<CudaSlice<u8>>,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<(), CudaGraphError> {
        let h = hidden_size;
        let inter = intermediate_size;
        let h_u32 = h as u32;
        let i_u32 = inter as u32;

        // Shared-mem bytes for V8 at k=h (attn_proj and gate_up both have k=h).
        let h_v8_smem = Self::v8_shared_bytes(h);

        // ── Acquire / (re)allocate activation buffers ────────────────────
        let mut buf_guard = self.acquire_buffers(h, inter)?;
        let bufs = buf_guard
            .as_mut()
            .ok_or_else(|| CudaGraphError::DriverError("buffers not allocated".into()))?;

        // ── Step 1: Upload CPU → GPU (async, stream-ordered) ────────────
        self.stream
            .memcpy_htod(&hidden[..h], &mut bufs.d_hidden)
            .map_err(|e| CudaGraphError::DriverError(format!("upload hidden: {e}")))?;
        self.stream
            .memcpy_htod(&attn_out[..h], &mut bufs.d_attn_out)
            .map_err(|e| CudaGraphError::DriverError(format!("upload attn_out: {e}")))?;
        self.stream
            .memcpy_htod(&norm_weight[..h], &mut bufs.d_norm_weight)
            .map_err(|e| CudaGraphError::DriverError(format!("upload norm_weight: {e}")))?;

        unsafe {
            // ── Step 2: GEMV_v8(attn_proj, attn_out → scratch) ──────────
            match h_v8_smem {
                Some(smem) => self.launch_gemv_v8(
                    attn_proj_w,
                    &bufs.d_attn_out,
                    &mut bufs.d_scratch,
                    h_u32,
                    h_u32,
                    smem,
                )?,
                None => self.launch_gemv_v7(
                    attn_proj_w,
                    &bufs.d_attn_out,
                    &mut bufs.d_scratch,
                    h_u32,
                    h_u32,
                )?,
            }

            // ── Step 3: residual_add(hidden += scratch) ──────────────────
            self.launch_residual_add(&mut bufs.d_hidden, &bufs.d_scratch, h_u32)?;

            // ── Step 4: rmsnorm(hidden, norm_weight → normed) ────────────
            self.launch_rmsnorm(
                &bufs.d_hidden,
                &bufs.d_norm_weight,
                &mut bufs.d_normed,
                h_u32,
                eps,
            )?;

            // ── Step 5: fused_gate_up_swiglu_q1(gate_up, normed → swiglu_buf) ──
            // Reads gate row and up row simultaneously from the concatenated SoA weight
            // matrix, computes both dot products, and applies SiLU(gate)*up in the
            // epilogue — replaces the old GEMV_v8(→ gate_up_buf) + swiglu_fused pair,
            // halving the dispatch count for this step.
            self.launch_fused_gate_up_swiglu(
                gate_up_w,
                &bufs.d_normed,
                &mut bufs.d_swiglu,
                i_u32,
                h_u32,
            )?;

            // ── Step 6: GEMV_v8/v9(down, swiglu → scratch) ──────────────
            // Reuse d_scratch; down has k=inter (12 288); V8 if it fits, V9 otherwise.
            match Self::v8_shared_bytes(inter) {
                Some(smem) => self.launch_gemv_v8(
                    down_w,
                    &bufs.d_swiglu,
                    &mut bufs.d_scratch,
                    h_u32,
                    i_u32,
                    smem,
                )?,
                None => {
                    self.launch_gemv_v9(down_w, &bufs.d_swiglu, &mut bufs.d_scratch, h_u32, i_u32)?
                }
            }

            // ── Step 7: residual_add(hidden += scratch) ──────────────────
            self.launch_residual_add(&mut bufs.d_hidden, &bufs.d_scratch, h_u32)?;
        }

        // ── Step 8: Download result → host ───────────────────────────────
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("stream sync: {e}")))?;
        self.stream
            .memcpy_dtoh(&bufs.d_hidden, &mut hidden[..h])
            .map_err(|e| CudaGraphError::DriverError(format!("download hidden: {e}")))?;
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("stream sync D2H: {e}")))?;

        Ok(())
    }

    /// Execute a QKV projection using pre-allocated device buffers.
    ///
    /// Eliminates per-call `cuMemAlloc`/`cuMemFree` that penalised the V1 path.
    /// Uses V8 (shared-mem input cache) when `k ≤ 48 KB threshold`, V7 otherwise.
    pub fn encode_qkv_phase(
        &self,
        input: &[f32],
        output: &mut [f32],
        weight_w: &Arc<CudaSlice<u8>>,
        n_rows: usize,
        k: usize,
    ) -> Result<(), CudaGraphError> {
        // Acquire (or lazily allocate) pre-allocated QKV device buffers.
        let mut qkv_guard = self.acquire_qkv_buffers(k, n_rows)?;
        let qkv = qkv_guard
            .as_mut()
            .ok_or_else(|| CudaGraphError::DriverError("qkv buffers not allocated".into()))?;

        // H2D upload into the pre-allocated input buffer (no cuMemAlloc).
        self.stream
            .memcpy_htod(&input[..k], &mut qkv.d_input)
            .map_err(|e| CudaGraphError::DriverError(format!("upload qkv_input: {e}")))?;

        unsafe {
            match Self::v8_shared_bytes(k) {
                Some(smem) => self.launch_gemv_v8(
                    weight_w,
                    &qkv.d_input,
                    &mut qkv.d_output,
                    n_rows as u32,
                    k as u32,
                    smem,
                )?,
                None => self.launch_gemv_v7(
                    weight_w,
                    &qkv.d_input,
                    &mut qkv.d_output,
                    n_rows as u32,
                    k as u32,
                )?,
            }
        }

        // D2H download (stream-synchronised, no temporary Vec alloc).
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("qkv stream sync: {e}")))?;
        self.stream
            .memcpy_dtoh(&qkv.d_output, &mut output[..n_rows])
            .map_err(|e| CudaGraphError::DriverError(format!("download qkv_output: {e}")))?;
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("qkv D2H sync: {e}")))?;
        Ok(())
    }

    // ── Accessors for cuda_full_layer.rs ─────────────────────────────────

    /// Return the underlying `CudaStream` (shared reference).
    ///
    /// Used by `cuda_full_layer.rs` to perform uploads and synchronisation
    /// without going through the FFN-specific `encode_ffn_phase` API.
    pub fn stream_arc(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Raw H2D copy via `cuMemcpyHtoDAsync`.
    ///
    /// Event tracking is permanently disabled (see `CudaGraph::new`), so
    /// `device_ptr_mut` is a cheap pointer extraction with no injected waits.
    /// Using the raw driver API directly keeps the `unsafe` contract visible.
    ///
    /// # Safety
    /// - `src` must remain valid until the stream synchronises.
    /// - `dst` must be a valid device allocation on this graph's stream.
    /// - `count` must not exceed `dst.len()`.
    pub unsafe fn raw_htod<T: cudarc::driver::DeviceRepr>(
        &self,
        src: &[T],
        dst: &mut CudaSlice<T>,
        count: usize,
    ) -> Result<(), CudaGraphError> {
        let (dst_ptr, _rec) = dst.device_ptr_mut(&self.stream);
        cudarc_result::memcpy_htod_async(dst_ptr, &src[..count], self.stream.cu_stream())
            .map_err(|e| CudaGraphError::DriverError(format!("raw_htod: {e}")))
    }

    /// Raw D2H copy via `cuMemcpyDtoHAsync`.
    ///
    /// # Safety
    /// Caller must synchronise the stream before reading `dst`.
    pub unsafe fn raw_dtoh<T: cudarc::driver::DeviceRepr>(
        &self,
        src: &CudaSlice<T>,
        dst: &mut [T],
        count: usize,
    ) -> Result<(), CudaGraphError> {
        let (src_ptr, _rec) = src.device_ptr(&self.stream);
        cudarc_result::memcpy_dtoh_async(&mut dst[..count], src_ptr, self.stream.cu_stream())
            .map_err(|e| CudaGraphError::DriverError(format!("raw_dtoh: {e}")))
    }

    /// Return the underlying `CudaContext` (shared reference).
    ///
    /// Used by `cuda_full_layer.rs` to compile the attention PTX module.
    pub fn context_arc(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Upload `f32` weights and cache them under `key`.
    ///
    /// On the first call for `key`, the slice is copied to a device buffer and
    /// stored in `f32_weight_cache`.  Subsequent calls clone the cached `Arc`.
    ///
    /// Unlike [`get_or_upload_weight_soa`], no SoA reformatting is performed;
    /// the data is uploaded verbatim as typed `f32` device memory.
    pub fn get_or_upload_f32_weight(
        &self,
        key: u64,
        data: &[f32],
    ) -> Result<Arc<CudaSlice<f32>>, CudaGraphError> {
        // Fast path: check the f32 cache.
        {
            let cache = self
                .f32_weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            if let Some(existing) = cache.get(&key) {
                return Ok(Arc::clone(existing));
            }
        }
        // Slow path: upload and insert.
        let d_buf = self
            .stream
            .clone_htod(data)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod f32: {e}")))?;
        let arc = Arc::new(d_buf);
        let mut cache = self
            .f32_weight_cache
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        cache.insert(key, Arc::clone(&arc));
        Ok(arc)
    }

    // ── Public launchers for cuda_full_layer.rs ───────────────────────────

    /// Public wrapper around `launch_rmsnorm_weighted_v2`.
    ///
    /// # Safety
    /// All slices must be valid device pointers on `self.stream`.
    pub unsafe fn launch_rmsnorm_pub(
        &self,
        d_input: &CudaSlice<f32>,
        d_weight: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n: u32,
        eps: f32,
    ) -> Result<(), CudaGraphError> {
        self.launch_rmsnorm(d_input, d_weight, d_output, n, eps)
    }

    /// Public wrapper around `launch_gemv_v7`.
    ///
    /// # Safety
    /// All slices must be valid device pointers on `self.stream`.
    pub unsafe fn launch_gemv_v7_pub(
        &self,
        d_weight: &Arc<CudaSlice<u8>>,
        d_input: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        self.launch_gemv_v7(d_weight, d_input, d_output, n_rows, k)
    }

    /// Public wrapper around `launch_gemv_v8`.
    ///
    /// # Safety
    /// All slices must be valid device pointers on `self.stream`.
    pub unsafe fn launch_gemv_v8_pub(
        &self,
        d_weight: &Arc<CudaSlice<u8>>,
        d_input: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
        shared_mem_bytes: u32,
    ) -> Result<(), CudaGraphError> {
        self.launch_gemv_v8(d_weight, d_input, d_output, n_rows, k, shared_mem_bytes)
    }

    /// Public auto-dispatch GEMV: uses V8 when `k` fits in shared mem, V7 otherwise.
    ///
    /// # Safety
    /// All slices must be valid device pointers on `self.stream`.
    pub unsafe fn launch_gemv_pub(
        &self,
        d_weight: &Arc<CudaSlice<u8>>,
        d_input: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        match Self::v8_shared_bytes(k as usize) {
            Some(smem) => self.launch_gemv_v8(d_weight, d_input, d_output, n_rows, k, smem),
            None => self.launch_gemv_v9(d_weight, d_input, d_output, n_rows, k),
        }
    }

    /// Public auto-dispatch GEMV with fused in-place residual add.
    ///
    /// Computes `d_inout[row] = dot(weight[row], d_input) + d_inout[row]` for all rows.
    /// Uses V8 (shared-memory cache) when k fits in 49 KB; V9 (vectorised loads) otherwise.
    ///
    /// # Safety
    /// All slices must be valid device pointers on `self.stream`.
    /// `d_inout` is used as both the residual source and the output destination.
    /// Each output row is written exactly once by a single warp, so the read-before-write
    /// within the fused kernel is data-race-free even with aliased pointers.
    pub unsafe fn launch_gemv_residual_pub(
        &self,
        d_weight: &Arc<CudaSlice<u8>>,
        d_input: &CudaSlice<f32>,
        d_inout: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        // SAFETY: d_inout is used as both residual (immutable read) and output (mutable write).
        // The underlying GPU kernel reads residual[row] and writes output[row] for each row
        // exactly once in a single statement.  No warp-level data race occurs.
        let d_residual = &*(d_inout as *const CudaSlice<f32>);
        match Self::v8_shared_bytes(k as usize) {
            Some(smem) => self.launch_gemv_v8_residual(
                &**d_weight,
                d_input,
                d_residual,
                d_inout,
                n_rows,
                k,
                smem,
            ),
            None => {
                self.launch_gemv_v9_residual(&**d_weight, d_input, d_residual, d_inout, n_rows, k)
            }
        }
    }

    /// Public wrapper around `launch_residual_add`.
    ///
    /// # Safety
    /// All slices must be valid device pointers on `self.stream`.
    pub unsafe fn launch_residual_add_pub(
        &self,
        d_a: &mut CudaSlice<f32>,
        d_b: &CudaSlice<f32>,
        n: u32,
    ) -> Result<(), CudaGraphError> {
        self.launch_residual_add(d_a, d_b, n)
    }

    /// Public wrapper around `launch_swiglu_fused`.
    ///
    /// # Safety
    /// All slices must be valid device pointers on `self.stream`.
    pub unsafe fn launch_swiglu_pub(
        &self,
        d_gate_up: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n: u32,
    ) -> Result<(), CudaGraphError> {
        self.launch_swiglu(d_gate_up, d_output, n)
    }

    /// Public wrapper around `launch_fused_gate_up_swiglu`.
    ///
    /// # Safety
    /// All slices must be valid device pointers allocated on the graph's stream.
    pub unsafe fn launch_fused_gate_up_swiglu_pub(
        &self,
        blocks: &Arc<CudaSlice<u8>>,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        self.launch_fused_gate_up_swiglu(blocks, input, output, n_rows, k)
    }

    // ── LM-head GEMV buffer management ───────────────────────────────────

    fn acquire_lm_head_buffers(
        &self,
        hidden: usize,
        vocab: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<LmHeadBuffers>>, CudaGraphError> {
        let mut guard = self
            .lm_head_buffers
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        let needs_alloc = match guard.as_ref() {
            Some(b) => !b.fits(hidden, vocab),
            None => true,
        };
        if needs_alloc {
            let alloc = |n: usize| -> Result<CudaSlice<f32>, CudaGraphError> {
                self.stream
                    .alloc_zeros::<f32>(n)
                    .map_err(|e| CudaGraphError::DriverError(format!("alloc lm_head({n}): {e}")))
            };
            *guard = Some(LmHeadBuffers {
                d_input: alloc(hidden)?,
                d_output: alloc(vocab)?,
                hidden_capacity: hidden,
                vocab_capacity: vocab,
            });
        }
        Ok(guard)
    }

    /// Run the LM-head GEMV on GPU: `logits = lm_head_weight × normed`.
    ///
    /// Uploads `normed` (hidden_size floats) once, launches GEMV, downloads logits.
    /// The weight is cached on first call and reused across tokens.
    pub fn encode_lm_head_gemv(
        &self,
        normed: &[f32],
        handle_id: u64,
        weight_bytes: &[u8],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>, CudaGraphError> {
        let d_weight = self.get_or_upload_weight_soa(handle_id, weight_bytes)?;

        let mut buf_guard = self.acquire_lm_head_buffers(hidden_size, vocab_size)?;
        let bufs = buf_guard
            .as_mut()
            .ok_or_else(|| CudaGraphError::DriverError("lm_head buffers not allocated".into()))?;

        self.stream
            .memcpy_htod(&normed[..hidden_size], &mut bufs.d_input)
            .map_err(|e| CudaGraphError::DriverError(format!("upload lm_head input: {e}")))?;

        unsafe {
            self.launch_gemv_pub(
                &d_weight,
                &bufs.d_input,
                &mut bufs.d_output,
                vocab_size as u32,
                hidden_size as u32,
            )?;
        }

        let result = self
            .stream
            .clone_dtoh(&bufs.d_output)
            .map_err(|e| CudaGraphError::DriverError(format!("download logits: {e}")))?;

        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("lm_head D2H sync: {e}")))?;

        Ok(result)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public convenience entry points (mirroring try_metal_ffn / try_metal_qkv)
// ═══════════════════════════════════════════════════════════════════════════

/// Attempt to run the FFN phase via direct CUDA dispatch.
///
/// This is the primary entry point for `block.rs` on Linux/Windows.
/// It mirrors `try_metal_ffn` exactly:
///
/// 1. Get the global `CudaGraph` singleton.
/// 2. Upload/cache weights lazily (first call uploads; subsequent calls reuse).
/// 3. Encode the full 8-op FFN pipeline on the CUDA stream.
///
/// Returns `Ok(())` if the CUDA dispatch succeeded.
/// Returns `Err(...)` if no CUDA device is present or dispatch failed.
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_ffn(
    hidden: &mut [f32],
    attn_out: &[f32],
    norm_weight: &[f32],
    eps: f32,
    attn_proj_handle_id: u64,
    attn_proj_bytes: &[u8],
    gate_up_handle_id: u64,
    gate_bytes: &[u8],
    up_bytes: &[u8],
    down_handle_id: u64,
    down_bytes: &[u8],
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<(), CudaGraphError> {
    let graph = CudaGraph::global()?;

    let attn_proj_w = graph.get_or_upload_weight_soa(attn_proj_handle_id, attn_proj_bytes)?;

    // Gate+up: concatenate once, then cache under gate_up_handle_id
    let gate_up_w = graph.get_or_upload_weight_soa_lazy(gate_up_handle_id, || {
        let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
        fused.extend_from_slice(gate_bytes);
        fused.extend_from_slice(up_bytes);
        fused
    })?;

    let down_w = graph.get_or_upload_weight_soa(down_handle_id, down_bytes)?;

    graph.encode_ffn_phase(
        hidden,
        attn_out,
        norm_weight,
        eps,
        &attn_proj_w,
        &gate_up_w,
        &down_w,
        hidden_size,
        intermediate_size,
    )
}

/// Attempt to run a fused QKV projection via direct CUDA dispatch.
///
/// Mirrors `try_metal_qkv`:
///
/// 1. Get the global `CudaGraph` singleton.
/// 2. Upload/cache fused Q+K+V weight lazily.
/// 3. Encode a single GEMV on the CUDA stream.
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_qkv(
    input: &[f32],
    output: &mut [f32],
    weight_handle_id: u64,
    q_bytes: &[u8],
    k_bytes: &[u8],
    v_bytes: &[u8],
    n_rows: usize,
    k: usize,
) -> Result<(), CudaGraphError> {
    let graph = CudaGraph::global()?;

    let weight_w = graph.get_or_upload_weight_soa_lazy(weight_handle_id, || {
        let mut fused = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
        fused.extend_from_slice(q_bytes);
        fused.extend_from_slice(k_bytes);
        fused.extend_from_slice(v_bytes);
        fused
    })?;

    graph.encode_qkv_phase(input, output, &weight_w, n_rows, k)
}

// ═══════════════════════════════════════════════════════════════════════════
// NativeCudaBackend — GpuBackendTrait implementation
// ═══════════════════════════════════════════════════════════════════════════

use super::{CpuBackend, DeviceBuffer, GpuBackendTrait, GpuError};

/// Thin `GpuBackendTrait` wrapper around `CudaGraph`.
///
/// Returned by [`select_backend`](super::select_backend) when a CUDA device
/// is present and the `native-cuda` feature is enabled.
pub struct NativeCudaBackend {
    graph: Arc<CudaGraph>,
    cpu_fallback: CpuBackend,
}

impl NativeCudaBackend {
    /// Initialise the backend (may fail if no CUDA device).
    pub fn new() -> Result<Self, GpuError> {
        let graph = CudaGraph::global().map_err(|e| GpuError::NotAvailable(e.to_string()))?;
        Ok(Self {
            graph,
            cpu_fallback: CpuBackend::new(),
        })
    }
}

impl GpuBackendTrait for NativeCudaBackend {
    fn name(&self) -> &'static str {
        "native-cuda"
    }

    fn is_accelerated(&self) -> bool {
        true
    }

    fn device_count(&self) -> usize {
        1
    }

    // ── Primitive ops: delegate to CPU fallback (not on the hot path) ─────

    fn alloc(&self, size: usize, device_id: usize) -> Result<DeviceBuffer, GpuError> {
        self.cpu_fallback.alloc(size, device_id)
    }

    fn host_to_device(&self, src: &[f32], device_id: usize) -> Result<DeviceBuffer, GpuError> {
        self.cpu_fallback.host_to_device(src, device_id)
    }

    fn device_to_host(&self, buf: &DeviceBuffer) -> Result<Vec<f32>, GpuError> {
        self.cpu_fallback.device_to_host(buf)
    }

    fn matvec(
        &self,
        a: &DeviceBuffer,
        x: &DeviceBuffer,
        m: usize,
        k: usize,
        device_id: usize,
    ) -> Result<DeviceBuffer, GpuError> {
        self.cpu_fallback.matvec(a, x, m, k, device_id)
    }

    fn relu(&self, x: &DeviceBuffer, device_id: usize) -> Result<DeviceBuffer, GpuError> {
        self.cpu_fallback.relu(x, device_id)
    }

    fn softmax(
        &self,
        x: &DeviceBuffer,
        size: usize,
        device_id: usize,
    ) -> Result<DeviceBuffer, GpuError> {
        self.cpu_fallback.softmax(x, size, device_id)
    }

    fn synchronize(&self, _device_id: usize) -> Result<(), GpuError> {
        self.graph
            .stream
            .synchronize()
            .map_err(|e| GpuError::SyncFailed(e.to_string()))
    }

    fn memory_info(&self, _device_id: usize) -> Result<(usize, usize), GpuError> {
        cudarc::driver::result::mem_get_info().map_err(|e| GpuError::NotAvailable(e.to_string()))
    }

    // ── Accelerated Q1 GEMV ───────────────────────────────────────────────

    fn gemv_q1_g128(
        &self,
        block_bytes: &[u8],
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, GpuError> {
        // Use a transient handle ID based on the pointer to the bytes.
        // This is safe because block_bytes is borrowed from static model memory.
        let handle_id = block_bytes.as_ptr() as u64;
        self.graph
            .encode_gemv(handle_id, block_bytes, input, n_rows, k)
            .map_err(|e| GpuError::KernelLaunch(e.to_string()))
    }

    // ── Weight cache ──────────────────────────────────────────────────────

    fn upload_weights_raw(
        &self,
        block_bytes: &[u8],
    ) -> Result<crate::weight_cache::GpuWeightHandle, GpuError> {
        let id = alloc_handle_id();
        self.graph
            .upload_weight_soa_new(id, block_bytes)
            .map_err(|e| GpuError::KernelLaunch(e.to_string()))?;
        Ok(crate::weight_cache::GpuWeightHandle(id))
    }

    fn gemv_q1_g128_cached(
        &self,
        handle: crate::weight_cache::GpuWeightHandle,
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, GpuError> {
        self.graph
            .encode_gemv_cached(handle.id(), input, n_rows, k)
            .map_err(|e| GpuError::KernelLaunch(e.to_string()))
    }

    // ── Fused FFN pipeline ────────────────────────────────────────────────

    fn batch_ffn_phase(
        &self,
        hidden: &mut [f32],
        attn_out: &[f32],
        norm_weight: &[f32],
        norm_eps: f32,
        attn_proj_handle: crate::weight_cache::GpuWeightHandle,
        gate_up_handle: crate::weight_cache::GpuWeightHandle,
        down_handle: crate::weight_cache::GpuWeightHandle,
        h: usize,
        intermediate: usize,
        _attn_proj_k: usize,
    ) -> Result<bool, GpuError> {
        // Look up all three weights; if any is missing, fall through to CPU.
        let lookup = |id: u64| -> Result<Arc<CudaSlice<u8>>, GpuError> {
            self.graph
                .weight_cache
                .lock()
                .map_err(|_| GpuError::SyncFailed("weight cache lock poisoned".into()))?
                .get(&id)
                .map(Arc::clone)
                .ok_or_else(|| GpuError::NotAvailable(format!("weight {id} not cached")))
        };

        let attn_proj_w = match lookup(attn_proj_handle.id()) {
            Ok(w) => w,
            Err(e) => {
                warn!(error = %e, "NativeCudaBackend::batch_ffn_phase: missing attn_proj weight");
                return Ok(false);
            }
        };
        let gate_up_w = match lookup(gate_up_handle.id()) {
            Ok(w) => w,
            Err(e) => {
                warn!(error = %e, "NativeCudaBackend::batch_ffn_phase: missing gate_up weight");
                return Ok(false);
            }
        };
        let down_w = match lookup(down_handle.id()) {
            Ok(w) => w,
            Err(e) => {
                warn!(error = %e, "NativeCudaBackend::batch_ffn_phase: missing down weight");
                return Ok(false);
            }
        };

        self.graph
            .encode_ffn_phase(
                hidden,
                attn_out,
                norm_weight,
                norm_eps,
                &attn_proj_w,
                &gate_up_w,
                &down_w,
                h,
                intermediate,
            )
            .map_err(|e| GpuError::KernelLaunch(e.to_string()))?;

        Ok(true)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Check that the singleton initialises without panicking.
    ///
    /// Skipped gracefully if no CUDA device is present (CI Linux without GPU).
    #[test]
    fn test_cuda_graph_global_init() {
        match CudaGraph::global() {
            Ok(_) => {
                // Success: CUDA device present, PTX compiled
            }
            Err(e) => {
                // No GPU in CI — that's fine, just log and skip
                eprintln!("CudaGraph::global() not available (expected in CPU-only CI): {e}");
            }
        }
    }

    /// Verify AoS → SoA reformatter preserves total byte count.
    #[test]
    fn test_reformat_aos_to_soa_round_trip() {
        const N: usize = 10;
        let mut aos = vec![0u8; N * 18];
        // Fill with recognisable pattern: scale = sequential u16, data = 0xAB
        for i in 0..N {
            let base = i * 18;
            let v = i as u16;
            aos[base] = (v & 0xff) as u8;
            aos[base + 1] = (v >> 8) as u8;
            for j in 2..18 {
                aos[base + j] = 0xABu8;
            }
        }
        let soa = CudaGraph::reformat_q1_aos_to_soa(&aos).expect("reformat failed");
        assert_eq!(soa.len(), aos.len());

        // Verify scales section (first N*2 bytes)
        for i in 0..N {
            let v = i as u16;
            assert_eq!(
                soa[i * 2],
                (v & 0xff) as u8,
                "scale byte 0 wrong at block {i}"
            );
            assert_eq!(
                soa[i * 2 + 1],
                (v >> 8) as u8,
                "scale byte 1 wrong at block {i}"
            );
        }

        // Verify data section (bytes N*2 .. end)
        for i in 0..N {
            let data_start = N * 2 + i * 16;
            for j in 0..16 {
                assert_eq!(
                    soa[data_start + j],
                    0xABu8,
                    "data wrong at block {i} byte {j}"
                );
            }
        }
    }

    /// Verify that alloc_handle_id() produces strictly increasing unique values.
    #[test]
    fn test_handle_id_uniqueness() {
        let ids: Vec<u64> = (0..64).map(|_| alloc_handle_id()).collect();
        for w in ids.windows(2) {
            assert!(w[1] > w[0], "handle IDs not strictly increasing");
        }
    }

    /// Verify that the CUDA_V7_KERNELS_SRC constant contains the fused kernel entry point.
    ///
    /// This test does NOT require a GPU — it only inspects the static source string.
    #[test]
    fn test_fused_gate_up_swiglu_source_has_entry_point() {
        assert!(
            super::super::cuda_kernels::CUDA_V7_KERNELS_SRC.contains("fused_gate_up_swiglu_q1"),
            "CUDA_V7_KERNELS_SRC must contain the fused_gate_up_swiglu_q1 kernel entry point"
        );
    }

    /// Verify that the fused kernel source contains the SiLU epilogue expression.
    ///
    /// This guards against regressions where the epilogue is accidentally removed.
    #[test]
    fn test_fused_gate_up_swiglu_source_has_silu_epilogue() {
        let src = super::super::cuda_kernels::CUDA_V7_KERNELS_SRC;
        assert!(
            src.contains("silu(gate_partial) * up_partial"),
            "fused kernel epilogue 'silu(gate_partial) * up_partial' not found in kernel source"
        );
    }

    /// Verify that the fused kernel source contains both gate and up partial accumulator names.
    ///
    /// Ensures the dual-accumulator pattern is present, not just a single-path kernel.
    #[test]
    fn test_fused_gate_up_swiglu_source_has_dual_accumulators() {
        let src = super::super::cuda_kernels::CUDA_V7_KERNELS_SRC;
        assert!(
            src.contains("gate_partial"),
            "fused kernel must have 'gate_partial' accumulator"
        );
        assert!(
            src.contains("up_partial"),
            "fused kernel must have 'up_partial' accumulator"
        );
    }

    /// Runtime test: initialise CudaGraph and verify the fused kernel compiles successfully.
    ///
    /// Skipped gracefully if no CUDA device is present (CPU-only CI).  When a GPU is
    /// available, confirms that `fused_gate_up_swiglu_q1` was loaded from the PTX module
    /// by checking that `CudaGraph::global()` succeeds (it would error on
    /// `load_function("fused_gate_up_swiglu_q1")` otherwise).
    #[test]
    fn test_fused_gate_up_swiglu_runtime_compile() {
        match CudaGraph::global() {
            Ok(_) => {
                // If global() succeeded, the fused kernel compiled and loaded without error.
            }
            Err(e) => {
                eprintln!(
                    "test_fused_gate_up_swiglu_runtime_compile: no CUDA device (expected in CPU-only CI): {e}"
                );
            }
        }
    }
}
