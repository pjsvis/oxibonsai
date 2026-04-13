//! Kernel benchmarks for Q1\_0\_g128 operations.
//!
//! Compares reference (scalar), AVX2 SIMD, and parallel kernel performance.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use half::f16;
use oxibonsai_core::tensor::BlockQ1_0G128;
use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_kernels::traits::OneBitKernel;
use oxibonsai_kernels::{dequant::dequant_1bit_g128, gemv::gemv_1bit_g128};
use std::hint::black_box;

fn make_blocks(count: usize) -> Vec<BlockQ1_0G128> {
    (0..count)
        .map(|i| BlockQ1_0G128 {
            d: f16::from_f32(0.5 + (i as f32) * 0.001),
            qs: [0xAA; 16], // alternating bits
        })
        .collect()
}

fn bench_dequant(c: &mut Criterion) {
    let blocks = make_blocks(32); // 4096 elements = 1 row of hidden_size
    let mut output = vec![0.0f32; 32 * 128];

    let mut group = c.benchmark_group("dequant_4096");

    group.bench_function("reference", |b| {
        b.iter(|| {
            dequant_1bit_g128(black_box(&blocks), black_box(&mut output))
                .expect("dequant should succeed");
        });
    });

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        group.bench_function("avx2", |b| {
            b.iter(|| unsafe {
                oxibonsai_kernels::simd_avx2::dequant_1bit_g128_avx2(
                    black_box(&blocks),
                    black_box(&mut output),
                )
                .expect("avx2 dequant should succeed");
            });
        });
    }

    #[cfg(target_arch = "aarch64")]
    {
        group.bench_function("neon", |b| {
            b.iter(|| unsafe {
                oxibonsai_kernels::simd_neon::dequant_1bit_g128_neon(
                    black_box(&blocks),
                    black_box(&mut output),
                )
                .expect("neon dequant should succeed");
            });
        });
    }

    group.finish();
}

fn bench_gemv_single_row(c: &mut Criterion) {
    let blocks = make_blocks(32); // 4096 input features
    let input = vec![1.0f32; 4096];
    let mut output = vec![0.0f32; 1];

    let mut group = c.benchmark_group("gemv_1row_4096in");

    group.bench_function("reference", |b| {
        b.iter(|| {
            gemv_1bit_g128(
                black_box(&blocks),
                black_box(&input),
                black_box(&mut output),
                1,
                4096,
            )
            .expect("gemv should succeed");
        });
    });

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        group.bench_function("avx2", |b| {
            b.iter(|| unsafe {
                oxibonsai_kernels::simd_avx2::gemv_1bit_g128_avx2(
                    black_box(&blocks),
                    black_box(&input),
                    black_box(&mut output),
                    1,
                    4096,
                )
                .expect("avx2 gemv should succeed");
            });
        });
    }

    #[cfg(target_arch = "aarch64")]
    {
        group.bench_function("neon", |b| {
            b.iter(|| unsafe {
                oxibonsai_kernels::simd_neon::gemv_1bit_g128_neon(
                    black_box(&blocks),
                    black_box(&input),
                    black_box(&mut output),
                    1,
                    4096,
                )
                .expect("neon gemv should succeed");
            });
        });
    }

    group.finish();
}

fn bench_gemv_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv_dispatch");

    for &n_rows in &[32, 256, 1024, 4096] {
        let k = 4096;
        let blocks = make_blocks(n_rows * (k / 128));
        let input = vec![0.5f32; k];
        let mut output = vec![0.0f32; n_rows];

        let ref_disp = KernelDispatcher::with_tier(KernelTier::Reference);
        group.bench_with_input(BenchmarkId::new("reference", n_rows), &n_rows, |b, &nr| {
            b.iter(|| {
                ref_disp
                    .gemv(
                        black_box(&blocks),
                        black_box(&input),
                        black_box(&mut output),
                        nr,
                        k,
                    )
                    .expect("dispatcher gemv should succeed");
            });
        });

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let avx_disp = KernelDispatcher::with_tier(KernelTier::Avx2);
            group.bench_with_input(BenchmarkId::new("avx2", n_rows), &n_rows, |b, &nr| {
                b.iter(|| {
                    avx_disp
                        .gemv(
                            black_box(&blocks),
                            black_box(&input),
                            black_box(&mut output),
                            nr,
                            k,
                        )
                        .expect("dispatcher gemv should succeed");
                });
            });

            // Parallel (only above threshold)
            if n_rows >= 64 {
                group.bench_with_input(
                    BenchmarkId::new("avx2+parallel", n_rows),
                    &n_rows,
                    |b, &nr| {
                        b.iter(|| {
                            oxibonsai_kernels::parallel::gemv_1bit_g128_par(
                                &avx_disp,
                                black_box(&blocks),
                                black_box(&input),
                                black_box(&mut output),
                                nr,
                                k,
                            )
                            .expect("dispatcher par gemv should succeed");
                        });
                    },
                );
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            let neon_disp = KernelDispatcher::with_tier(KernelTier::Neon);
            group.bench_with_input(BenchmarkId::new("neon", n_rows), &n_rows, |b, &nr| {
                b.iter(|| {
                    neon_disp
                        .gemv(
                            black_box(&blocks),
                            black_box(&input),
                            black_box(&mut output),
                            nr,
                            k,
                        )
                        .expect("dispatcher gemv should succeed");
                });
            });

            if n_rows >= 64 {
                group.bench_with_input(
                    BenchmarkId::new("neon+parallel", n_rows),
                    &n_rows,
                    |b, &nr| {
                        b.iter(|| {
                            oxibonsai_kernels::parallel::gemv_1bit_g128_par(
                                &neon_disp,
                                black_box(&blocks),
                                black_box(&input),
                                black_box(&mut output),
                                nr,
                                k,
                            )
                            .expect("dispatcher par gemv should succeed");
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

fn bench_gemm_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_dispatch");

    for &m in &[1, 4, 16] {
        let n_rows = 4096;
        let k = 4096;
        let blocks = make_blocks(n_rows * (k / 128));
        let input = vec![0.5f32; m * k];
        let mut output = vec![0.0f32; m * n_rows];

        let ref_disp = KernelDispatcher::with_tier(KernelTier::Reference);
        group.bench_with_input(
            BenchmarkId::new("reference", format!("m{m}")),
            &m,
            |b, &batch| {
                b.iter(|| {
                    ref_disp
                        .gemm(
                            black_box(&blocks),
                            black_box(&input),
                            black_box(&mut output),
                            batch,
                            n_rows,
                            k,
                        )
                        .expect("dispatcher gemm should succeed");
                });
            },
        );

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let avx_disp = KernelDispatcher::with_tier(KernelTier::Avx2);
            group.bench_with_input(
                BenchmarkId::new("avx2", format!("m{m}")),
                &m,
                |b, &batch| {
                    b.iter(|| {
                        avx_disp
                            .gemm(
                                black_box(&blocks),
                                black_box(&input),
                                black_box(&mut output),
                                batch,
                                n_rows,
                                k,
                            )
                            .expect("dispatcher gemm should succeed");
                    });
                },
            );

            if m >= 4 {
                group.bench_with_input(
                    BenchmarkId::new("avx2+parallel", format!("m{m}")),
                    &m,
                    |b, &batch| {
                        b.iter(|| {
                            oxibonsai_kernels::parallel::gemm_1bit_g128_par(
                                &avx_disp,
                                black_box(&blocks),
                                black_box(&input),
                                black_box(&mut output),
                                batch,
                                n_rows,
                                k,
                            )
                            .expect("dispatcher par gemm should succeed");
                        });
                    },
                );
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            let neon_disp = KernelDispatcher::with_tier(KernelTier::Neon);
            group.bench_with_input(
                BenchmarkId::new("neon", format!("m{m}")),
                &m,
                |b, &batch| {
                    b.iter(|| {
                        neon_disp
                            .gemm(
                                black_box(&blocks),
                                black_box(&input),
                                black_box(&mut output),
                                batch,
                                n_rows,
                                k,
                            )
                            .expect("dispatcher gemm should succeed");
                    });
                },
            );

            if m >= 4 {
                group.bench_with_input(
                    BenchmarkId::new("neon+parallel", format!("m{m}")),
                    &m,
                    |b, &batch| {
                        b.iter(|| {
                            oxibonsai_kernels::parallel::gemm_1bit_g128_par(
                                &neon_disp,
                                black_box(&blocks),
                                black_box(&input),
                                black_box(&mut output),
                                batch,
                                n_rows,
                                k,
                            )
                            .expect("dispatcher par gemm should succeed");
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_dequant,
    bench_gemv_single_row,
    bench_gemv_dispatch,
    bench_gemm_dispatch,
);
criterion_main!(benches);
