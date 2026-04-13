# oxibonsai-kernels TODO

> 1-bit quantized compute kernels with SIMD dispatch and parallelism
> 16 files, ~4,300 lines, 243 tests

## Status: ✅ All Features Complete

NEON, AVX-512, tiled GEMM, packing, property tests, cross-tier correctness, performance tuning, cache-line alignment, and prefetch hints all implemented.

## Done

- [x] Reference scalar dequant (`dequant_1bit_g128`)
- [x] Reference scalar GEMV (`gemv_1bit_g128`)
- [x] Reference scalar GEMM (`gemm_1bit_g128`)
- [x] `OneBitKernel` trait abstraction
- [x] `KernelDispatcher` with `KernelTier` enum
- [x] AVX2+FMA SIMD kernels — `simd_avx2.rs` (dequant, gemv, gemm, 426 lines)
- [x] SciRS2-Core-powered runtime SIMD detection (`scirs2_core::simd::detect::get_cpu_features()`)
- [x] Rayon parallel GEMV (row-parallel, threshold 64 rows) (`parallel.rs`)
- [x] Rayon parallel GEMM (batch-parallel, threshold 4) (`parallel.rs`)
- [x] Criterion benchmark suite (dequant, gemv, gemm — reference vs AVX2 vs parallel)
- [x] **AVX-512 kernels** — `simd_avx512.rs` (~701 lines), 512-bit operations, `KernelTier::Avx512` in dispatcher
- [x] **NEON (ARM) kernels** — `simd_neon.rs` (~768 lines), NEON intrinsics, `KernelTier::Neon` in dispatcher
- [x] **Tiled GEMM** — Cache-blocking tiled matrix multiply (`tiled.rs`, 622 lines)
- [x] **Packing** — Panel packing for cache efficiency (`packing.rs`, 452 lines)
- [x] **Parallel tiled GEMM** — Rayon-parallel tiled paths (`parallel_tiled.rs`)
- [x] **Property tests** — `proptest` roundtrip: dequant → requant identity, GEMV distributivity, GEMM associativity (`proptest_kernels.rs`)
- [x] **Cross-tier correctness** — Automated reference vs AVX2 vs AVX-512 output comparison (`cross_tier.rs`)
- [x] **Parallel GEMV tuning** — `tuning.rs` with PlatformProfile, TunedThresholds, auto-detection of optimal thresholds per platform
- [x] **Cache-line alignment** — `aligned.rs` with AlignedBuffer/AlignedBlocks, 64-byte aligned allocations for SIMD loads
- [x] **Prefetch hints** — `prefetch.rs` with PrefetchConfig, software prefetch via x86 _mm_prefetch / ARM _prefetch, platform-specific dispatch
