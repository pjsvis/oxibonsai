# oxibonsai-kernels

1-bit Q1_0_g128 compute kernels for OxiBonsai — dequantization, GEMV, GEMM.

Implements the full compute stack for 1-bit inference: scalar reference kernels,
SIMD-accelerated tiers (AVX2+FMA, AVX-512, NEON), tiled cache-blocked GEMM,
parallel Rayon dispatch, and GPU backend stubs (CUDA, Metal).

Part of the [OxiBonsai](https://github.com/cool-japan/oxibonsai) project.

## Features

- `dequant_1bit_g128` — dequantize Q1_0_g128 blocks to f32
- `gemv_1bit_g128` — fused 1-bit GEMV (matrix-vector multiply)
- `gemm_1bit_g128` — fused 1-bit GEMM (batched matrix multiply)
- `KernelDispatcher::auto_detect()` — selects the best SIMD tier at runtime
- Tiled GEMM with cache-line alignment and software prefetch hints
- Parallel dispatch via Rayon (`gemv_1bit_g128_par`, `gemm_1bit_g128_par`)
- Platform tuning: `PlatformProfile`, `TunedThresholds`
- GPU backend trait (`GpuBackend`) with CUDA and Metal stubs

## SIMD Tiers

| Tier | Feature Flag | Width | Platform |
|------|-------------|-------|----------|
| Reference (scalar) | *(default)* | N/A | All |
| AVX2+FMA | `simd-avx2` | 256-bit | x86-64 |
| AVX-512 | `simd-avx512` | 512-bit | x86-64 |
| NEON | `simd-neon` | 128-bit | AArch64 |

## Usage

```toml
[dependencies]
# Auto-detect at runtime:
oxibonsai-kernels = { version = "0.1.0", features = ["simd-avx2"] }
```

```rust
use oxibonsai_kernels::KernelDispatcher;

let dispatcher = KernelDispatcher::auto_detect();
// dispatcher selects AVX2, AVX-512, NEON, or scalar automatically
```

## License

Apache-2.0 — COOLJAPAN OU
