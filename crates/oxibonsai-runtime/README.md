# oxibonsai-runtime

Inference runtime, sampling, and OpenAI-compatible server for OxiBonsai.

Ties together core, kernels, model, and tokenizer into a production-ready
inference stack with advanced sampling, SSE streaming, OpenAI API compatibility,
Prometheus metrics, circuit breaker, and comprehensive configuration.

Part of the [OxiBonsai](https://github.com/cool-japan/oxibonsai) project.

## Features

- `InferenceEngine` — prefill + autoregressive decode loop
- `EngineBuilder` / `ConfigBuilder` / `SamplerBuilder` — ergonomic builder API
- Sampling presets: Greedy, Balanced, Creative, Code
- Advanced samplers: Mirostat v1/v2, Locally Typical, Eta, Min-P, adaptive
- `SamplerChain` — composable sampling pipeline
- Speculative decoding with draft/verify loop
- Beam search with configurable width, length penalty, n-gram blocking
- Token healing and context window management
- `InferencePipeline` — high-level generation API with stop reasons
- OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`
- SSE streaming for real-time token output
- Rate limiting, circuit breaker, CORS, tower middleware
- Admin API: `/admin/status`, `/admin/config`, `/admin/cache-stats`
- Prometheus metrics (`/metrics`): tokens/s, latency, request counts
- Health endpoint (`/health`) with readiness probes
- TOML configuration with layered loading (defaults → file → CLI)

## Feature Flags

| Flag | Description | Default |
|------|-------------|---------|
| `server` | Axum HTTP server | ✅ enabled |
| `rag` | RAG server endpoints | disabled |
| `wasm` | WASM-safe build | disabled |
| `metal` | Metal GPU backend | disabled |

## Usage

```toml
[dependencies]
oxibonsai-runtime = "0.1.0"
```

```rust
use oxibonsai_runtime::{EngineBuilder, SamplingPreset};

let engine = EngineBuilder::new()
    .model_path("models/Bonsai-8B.gguf")
    .preset(SamplingPreset::Balanced)
    .max_seq_len(4096)
    .build()?;
```

## License

Apache-2.0 — COOLJAPAN OU
