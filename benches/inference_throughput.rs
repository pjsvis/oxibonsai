//! End-to-end inference throughput benchmarks.
//!
//! All benchmarks use a tiny Qwen3 configuration so that the loops complete
//! in milliseconds rather than minutes.  The purpose is to measure relative
//! performance of the inference pipeline components — not absolute tokens/sec
//! on a real model.
//!
//! Run with:
//! ```text
//! cargo bench --bench inference_throughput
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::presets::SamplingPreset;
use oxibonsai_runtime::sampling::SamplingParams;
use std::hint::black_box;

// ─── Shared fixture helpers ────────────────────────────────────────────────────

/// Build a tiny-config inference engine seeded deterministically.
fn tiny_engine(seed: u64) -> InferenceEngine<'static> {
    let config = Qwen3Config::tiny_test();
    let params = SamplingPreset::Greedy.params();
    InferenceEngine::new(config, params, seed)
}

/// Build a simple prompt of `len` sequential token IDs in [1, 256).
fn make_prompt(len: usize) -> Vec<u32> {
    (1..=len as u32).map(|i| (i % 255) + 1).collect()
}

// ─── bench_generate_tokens ────────────────────────────────────────────────────

/// Benchmark end-to-end token generation throughput (tokens/sec).
///
/// Measures the cost of prefilling a short prompt and decoding N new tokens
/// using greedy sampling on the tiny test model.
fn bench_generate_tokens(c: &mut Criterion) {
    let mut group = c.benchmark_group("generate_tokens");

    for &max_tokens in &[4usize, 8, 16] {
        let prompt = make_prompt(4);
        group.throughput(Throughput::Elements(max_tokens as u64));
        group.bench_with_input(
            BenchmarkId::new("greedy", max_tokens),
            &max_tokens,
            |b, &n| {
                let mut engine = tiny_engine(42);
                b.iter(|| {
                    engine.reset();
                    engine
                        .generate(black_box(&prompt), black_box(n))
                        .expect("generation must succeed")
                });
            },
        );
    }

    group.finish();
}

// ─── bench_prefill_latency ────────────────────────────────────────────────────

/// Benchmark prompt-processing (prefill) latency for varying prompt lengths.
///
/// Uses `max_tokens = 1` so that decode cost is negligible and the measurement
/// reflects primarily the cost of feeding the prompt through the model.
fn bench_prefill_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefill_latency");

    for &prompt_len in &[1usize, 4, 16, 32] {
        let prompt = make_prompt(prompt_len);
        group.throughput(Throughput::Elements(prompt_len as u64));
        group.bench_with_input(
            BenchmarkId::new("prompt_tokens", prompt_len),
            &prompt_len,
            |b, _| {
                let mut engine = tiny_engine(42);
                b.iter(|| {
                    engine.reset();
                    engine
                        .generate(black_box(&prompt), 1)
                        .expect("generation must succeed")
                });
            },
        );
    }

    group.finish();
}

// ─── bench_sampling_methods ───────────────────────────────────────────────────

/// Compare the overhead of different sampling strategies on a tiny model.
///
/// All strategies produce one output token from a 4-token prompt so the
/// benchmark isolates the per-token sampling overhead.
fn bench_sampling_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_methods");
    let prompt = make_prompt(4);

    // Greedy (temperature ≈ 0) — argmax, no PRNG needed
    group.bench_function("greedy", |b| {
        let mut engine = InferenceEngine::new(
            Qwen3Config::tiny_test(),
            SamplingPreset::Greedy.params(),
            42,
        );
        b.iter(|| {
            engine.reset();
            engine
                .generate(black_box(&prompt), 1)
                .expect("generation must succeed")
        });
    });

    // Balanced (temperature=0.7, top_k=40, top_p=0.9)
    group.bench_function("balanced", |b| {
        let mut engine = InferenceEngine::new(
            Qwen3Config::tiny_test(),
            SamplingPreset::Balanced.params(),
            42,
        );
        b.iter(|| {
            engine.reset();
            engine
                .generate(black_box(&prompt), 1)
                .expect("generation must succeed")
        });
    });

    // Creative (higher temperature)
    group.bench_function("creative", |b| {
        let mut engine = InferenceEngine::new(
            Qwen3Config::tiny_test(),
            SamplingPreset::Creative.params(),
            42,
        );
        b.iter(|| {
            engine.reset();
            engine
                .generate(black_box(&prompt), 1)
                .expect("generation must succeed")
        });
    });

    // Top-p only (temperature=1.0, no top-k)
    group.bench_function("top_p_only", |b| {
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.9,
            repetition_penalty: 1.0,
            ..SamplingParams::default()
        };
        let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), params, 42);
        b.iter(|| {
            engine.reset();
            engine
                .generate(black_box(&prompt), 1)
                .expect("generation must succeed")
        });
    });

    group.finish();
}

// ─── bench_kv_cache_operations ───────────────────────────────────────────────

/// Benchmark KV-cache throughput by generating progressively longer sequences.
///
/// As more tokens are generated the KV cache grows, exercising the store and
/// retrieve paths on each decode step.
fn bench_kv_cache_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_operations");

    // Generate N tokens sequentially so the cache fills up to N entries.
    for &seq_len in &[8usize, 16, 32, 64] {
        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(
            BenchmarkId::new("fill_and_use", seq_len),
            &seq_len,
            |b, &n| {
                let mut engine = tiny_engine(7);
                let prompt = make_prompt(1);
                b.iter(|| {
                    engine.reset();
                    engine
                        .generate(black_box(&prompt), black_box(n))
                        .expect("generation must succeed")
                });
            },
        );
    }

    group.finish();
}

// ─── bench_batch_generation ───────────────────────────────────────────────────

/// Benchmark throughput for batched generation at varying batch sizes.
///
/// Uses the engine's `batch_generate` API which processes each prompt
/// sequentially after resetting state between requests.
fn bench_batch_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_generation");

    for &batch_size in &[1usize, 2, 4, 8] {
        // Build a batch of identical short prompts
        let prompts: Vec<Vec<u32>> = (0..batch_size).map(|_| make_prompt(4)).collect();
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, _| {
                let mut engine = tiny_engine(99);
                b.iter(|| {
                    let results = engine.batch_generate(black_box(&prompts), 4);
                    // Consume results to prevent dead-code elimination
                    for r in &results {
                        let _ = black_box(r.as_ref().map(|br| br.generated_tokens.len()));
                    }
                });
            },
        );
    }

    group.finish();
}

// ─── Criterion registration ───────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_generate_tokens,
    bench_prefill_latency,
    bench_sampling_methods,
    bench_kv_cache_operations,
    bench_batch_generation,
);
criterion_main!(benches);
