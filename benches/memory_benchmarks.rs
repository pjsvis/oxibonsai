//! Memory profiling benchmarks.
//!
//! Measures the overhead of memory sampling primitives, KV-cache allocation,
//! and model initialisation so that any regressions in these hot paths are
//! caught early in CI.
//!
//! All benchmarks use the tiny test configuration so that the suite completes
//! in seconds rather than minutes.
//!
//! Run with:
//! ```text
//! cargo bench --bench memory_benchmarks
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::memory::{get_rss_bytes, MemoryProfiler};
use oxibonsai_runtime::presets::SamplingPreset;
use std::hint::black_box;

// ─── bench_memory_snapshot ────────────────────────────────────────────────────

/// Benchmark the cost of a single `MemoryProfiler::take_snapshot()` call.
///
/// On macOS this exercises the Mach `task_info` syscall; on Linux it reads
/// `/proc/self/statm`. This helps ensure the profiler does not add meaningful
/// latency when called on the hot decode path.
fn bench_memory_snapshot(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_snapshot");

    // Single snapshot cost
    group.bench_function("take_snapshot", |b| {
        let profiler = MemoryProfiler::new();
        b.iter(|| {
            let snap = profiler.take_snapshot();
            black_box(snap.rss_bytes)
        });
    });

    // Raw RSS read cost (no profiler overhead, no peak tracking)
    group.bench_function("get_rss_bytes", |b| {
        b.iter(|| black_box(get_rss_bytes()));
    });

    // Peak tracking overhead: sample N times in a row
    for &n in &[10usize, 100, 1000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("batch_snapshots", n), &n, |b, &count| {
            let profiler = MemoryProfiler::new();
            b.iter(|| {
                for _ in 0..count {
                    let snap = profiler.take_snapshot();
                    let _ = black_box(snap.rss_bytes);
                }
                black_box(profiler.peak_rss_bytes())
            });
        });
    }

    group.finish();
}

// ─── bench_kv_cache_memory ────────────────────────────────────────────────────

/// Benchmark KV-cache allocation cost at various context lengths.
///
/// Measures the overhead of constructing a KV cache (allocating the backing
/// vectors) without running inference. A new model is built on each iteration
/// to ensure allocations are not reused across samples.
fn bench_kv_cache_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_memory");

    // We benchmark model construction because KV cache is allocated inside it.
    // Different max_context_length values exercise different allocation sizes.
    for &ctx_len in &[64usize, 128, 256, 512] {
        group.throughput(Throughput::Elements(ctx_len as u64));
        group.bench_with_input(
            BenchmarkId::new("model_new_with_context", ctx_len),
            &ctx_len,
            |b, &max_ctx| {
                // Build a tiny config with the specified context length
                let mut config = Qwen3Config::tiny_test();
                config.max_context_length = max_ctx;
                let before = get_rss_bytes();
                b.iter(|| {
                    let engine =
                        InferenceEngine::new(config.clone(), SamplingPreset::Greedy.params(), 42);
                    black_box(engine)
                });
                let after = get_rss_bytes();
                // Sanity: print delta once (visible only with `-- --nocapture`)
                let _ = black_box(after.saturating_sub(before));
            },
        );
    }

    group.finish();
}

// ─── bench_model_load_memory ──────────────────────────────────────────────────

/// Benchmark how much RAM is consumed by constructing models of various sizes.
///
/// Measures the RSS delta before and after creating a model, and benchmarks
/// the construction time itself. Uses tiny test configs scaled to different
/// "virtual" sizes to simulate loading overhead without requiring real weights.
fn bench_model_load_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_load_memory");

    // Tiny fixed config — baseline model construction cost
    group.bench_function("tiny_config_construct", |b| {
        let config = Qwen3Config::tiny_test();
        b.iter(|| {
            let engine = InferenceEngine::new(config.clone(), SamplingPreset::Greedy.params(), 42);
            black_box(engine)
        });
    });

    // Scale up the hidden size to simulate models with more embedding memory.
    // These still use tiny layer counts so the benchmark runs fast.
    for &hidden in &[64usize, 128, 256] {
        group.bench_with_input(BenchmarkId::new("hidden_size", hidden), &hidden, |b, &h| {
            let config = Qwen3Config {
                hidden_size: h,
                // Keep head_dim = h / num_heads; num_heads = 4 (from tiny_test)
                head_dim: h / 4,
                num_attention_heads: 4,
                num_kv_heads: 2,
                intermediate_size: h * 2,
                ..Qwen3Config::tiny_test()
            };
            b.iter(|| {
                let engine =
                    InferenceEngine::new(config.clone(), SamplingPreset::Greedy.params(), 42);
                black_box(engine)
            });
        });
    }

    // Also measure the cost of the RSS read itself bracketing a model build.
    group.bench_function("rss_delta_around_construct", |b| {
        let config = Qwen3Config::tiny_test();
        b.iter(|| {
            let before = get_rss_bytes();
            let engine = InferenceEngine::new(config.clone(), SamplingPreset::Greedy.params(), 42);
            let after = get_rss_bytes();
            let delta = after.saturating_sub(before);
            let _ = black_box(engine);
            black_box(delta)
        });
    });

    group.finish();
}

// ─── Criterion registration ───────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_memory_snapshot,
    bench_kv_cache_memory,
    bench_model_load_memory,
);
criterion_main!(benches);
