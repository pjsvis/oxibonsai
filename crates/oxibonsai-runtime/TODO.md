# oxibonsai-runtime TODO

> Inference engine, sampling, tokenizer, OpenAI-compatible server
> 19 files, ~2,700 lines, 78 tests

## Status: ✅ All Features Complete

Observability, TOML config, streaming SSE, circuit breaker, health checks, builders, presets, batch engine, async engine, integration tests, and sampling distribution tests all implemented.

## Done

- [x] `InferenceEngine` — prefill + autoregressive decode loop
- [x] `InferenceEngine::from_gguf()` — load model from GGUF file
- [x] `Sampler` — temperature, top-k, top-p, repetition penalty
- [x] `TokenizerBridge` — HuggingFace tokenizers wrapper (encode/decode)
- [x] OpenAI-compatible `/v1/chat/completions` endpoint (non-streaming)
- [x] `/v1/models` and `/health` endpoints
- [x] CLI subcommands: `run`, `chat`, `serve`, `info`
- [x] **Tracing upgrade** — `EnvFilter` + optional JSON layer (`tracing_setup.rs`)
- [x] **`#[instrument]` spans** — `generate()`, server handlers; span hierarchy: request → prefill → decode
- [x] **Prometheus metrics** — `/metrics` endpoint; tokens generated, requests, tokens/sec, prefill latency, decode latency, request latency (`metrics.rs`)
- [x] **TOML config struct** — Server settings, sampling defaults, model path, tokenizer path, observability settings (`config.rs`)
- [x] **Layered config** — defaults → TOML file → CLI args override (`config.rs`)
- [x] **Streaming chat completions** — SSE via `tokio-stream` (`server.rs`)
- [x] **Circuit breaker** — Fault isolation for engine errors (`circuit_breaker.rs`)
- [x] **Health checks** — Liveness and readiness probes (`health.rs`)
- [x] **Builders** — Ergonomic `InferenceEngineBuilder` and server builder (`builders.rs`)
- [x] **Presets** — Default configuration presets for common use cases (`presets.rs`)
- [x] **Batch engine** — Batch inference for throughput optimization (`batch_engine.rs`)
- [x] **Async engine** — Non-blocking async inference paths (`async_engine.rs`)
- [x] **Recovery** — Error recovery and retry strategies (`recovery.rs`)
- [x] **Convenience helpers** — High-level one-shot inference API (`convenience.rs`)
- [x] **Integration tests** — `tests/generate_pipeline_tests.rs` with 26 tests: full generate() pipeline, determinism, sampling params, edge cases, engine state
- [x] **Sampling distribution tests** — `tests/sampling_distribution_tests.rs` with 22 tests: chi-square goodness of fit, temperature/top-k/top-p/repetition penalty statistical validation
