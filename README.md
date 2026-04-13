# OxiBonsai
# (オキシ盆栽)

**Pure Rust 1-Bit LLM Inference Engine for PrismML Bonsai Models**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.86%2B-orange.svg)](https://www.rust-lang.org)

OxiBonsai is the world's first zero-FFI, zero-C/C++ inference engine capable of running PrismML's Bonsai-8B (Q1\_0\_g128) on commodity CPU hardware. Built entirely on the COOLJAPAN ecosystem -- SciRS2, OxiBLAS, OxiFFT -- it delivers sovereign AI inference in Pure Rust.

## Features

### 1-Bit Native Inference

- Purpose-built Q1\_0\_g128 kernels exploiting sign-bit + group-scale structure
- Bonsai-8B (1.15 GB model) with < 2 GB total runtime memory
- Full Qwen3-8B architecture: 36-layer decoder, GQA 32Q/8KV, SwiGLU, RoPE, RMSNorm

### SIMD Acceleration Tiers

| Tier | Target | Width | Feature Flag |
|------|--------|-------|--------------|
| Reference | All platforms | Scalar | (default) |
| AVX2+FMA | x86-64 | 256-bit | `simd-avx2` |
| AVX-512 | x86-64 | 512-bit | `simd-avx512` |
| NEON | AArch64 | 128-bit | `simd-neon` |

Auto-detection via `KernelDispatcher::auto_detect()` selects the best tier at runtime.

### Observability

- Structured logging via `tracing` with env-filter and JSON output
- Inference metrics: tokens/sec, prefill/decode latency, request counts
- Health endpoint (`/health`) with readiness checks
- Circuit breaker for overload protection

### OpenAI-Compatible API

- `/v1/chat/completions` endpoint (POST)
- Streaming SSE support for real-time token output
- `/v1/models` endpoint
- CORS and tower middleware

### Builder Pattern API

```rust
use oxibonsai_runtime::{EngineBuilder, SamplingPreset};

let engine = EngineBuilder::new()
    .model_path("models/Bonsai-8B.gguf")
    .preset(SamplingPreset::Creative)
    .max_seq_len(4096)
    .build()?;
```

### Sampling Presets

| Preset | Temperature | Top-K | Top-P | Use Case |
|--------|-------------|-------|-------|----------|
| Greedy | 0.0 | 1 | 1.0 | Deterministic |
| Balanced | 0.7 | 40 | 0.9 | General |
| Creative | 1.0 | 100 | 0.95 | Creative writing |
| Code | 0.2 | 10 | 0.8 | Code generation |

## Bonsai-8B Model

| Metric | Value |
|---|---|
| Architecture | Qwen3-8B (GQA 32Q/8KV, SwiGLU, RoPE, RMSNorm) |
| Parameters | 8.19 billion |
| Layers | 36 Transformer decoder blocks |
| Model Size | 1.15 GB (Q1\_0\_g128 GGUF) |
| Context | Up to 65,536 tokens |
| Vocabulary | 151,936 tokens |
| Weight Format | 1-bit sign + FP16 group scale (128 weights/group) |

## Installation

Add OxiBonsai to your `Cargo.toml`:

```toml
[dependencies]
oxibonsai = "0.1.0"
```

## Quick Start

```bash
# Build
cargo build --release

# Download the Bonsai-8B model (1.16 GB)
mkdir -p models
curl -L -o models/Bonsai-8B.gguf \
  https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/Bonsai-8B.gguf

# Print model info
oxibonsai info --model models/Bonsai-8B.gguf

# Run inference
oxibonsai run \
  --model models/Bonsai-8B.gguf \
  --prompt "Explain quantum computing in simple terms" \
  --max-tokens 512 \
  --temperature 0.7 \
  --top-p 0.9

# Interactive chat
oxibonsai chat \
  --model models/Bonsai-8B.gguf

# Start API server
oxibonsai serve \
  --model models/Bonsai-8B.gguf \
  --host 127.0.0.1 --port 8080
```

## CLI Smoke Test

A shell script is provided to build and run a quick end-to-end CLI test
(inference, model info, validation) against the Bonsai-8B model:

```bash
# CPU only (auto-detects AVX2/AVX-512/NEON)
./scripts/cli.sh

# Metal GPU (macOS Apple Silicon)
./scripts/cli.sh metal

# CUDA GPU (Linux / Windows with NVIDIA GPU)
./scripts/cli.sh cuda
```

The script:
1. Builds a `--release` binary with the specified feature flags
2. Runs inference (`oxibonsai run`) with the Bonsai-8B model
3. Prints model info (`oxibonsai info`)
4. Validates the GGUF file (`oxibonsai validate`)

> **Note**: The model file `models/Bonsai-8B.gguf` (1.16 GB) must be
> downloaded first — see [Quick Start](#quick-start).

## Configuration

OxiBonsai supports TOML configuration files with `--config`:

```toml
[model]
path = "models/Bonsai-8B.gguf"
max_seq_len = 4096

[sampling]
temperature = 0.7
top_k = 40
top_p = 0.9
repetition_penalty = 1.1

[server]
host = "127.0.0.1"
port = 8080

[observability]
log_level = "info"
json_logs = false
```

## Crate Structure

```
oxibonsai/
├── crates/
│   ├── oxibonsai-core/        GGUF loader, tensor types, config, error types
│   ├── oxibonsai-kernels/     1-bit kernels (dequant, GEMV, GEMM, SIMD tiers,
│   │                          tiled, parallel, GPU dispatch)
│   ├── oxibonsai-tokenizer/   Pure Rust BPE tokenizer, vocabulary, ChatTemplate
│   ├── oxibonsai-model/       Qwen3-8B Transformer (GQA, SwiGLU, RoPE, RMSNorm,
│   │                          paged KV-cache, weight loaders)
│   ├── oxibonsai-rag/         RAG pipeline (chunking, embedders, vector store)
│   ├── oxibonsai-runtime/     Inference engine, sampling, OpenAI-compatible server,
│   │                          SSE streaming, metrics, health, circuit breaker
│   ├── oxibonsai-eval/        Evaluation harness (ROUGE, perplexity, MMLU)
│   └── oxibonsai-serve/       Standalone server binary
├── src/main.rs                CLI entry point (run, chat, serve, info, benchmark,
│                              quantize, validate)
├── benches/                   Criterion kernel benchmarks
├── examples/                  Usage examples
├── tests/                     Integration + feature flag tests
└── scripts/                   Publish script, CLI smoke test
```

## Examples

See the `examples/` directory:

- `basic_inference.rs` -- Load a model and run single-shot inference
- `streaming.rs` -- Server-sent event streaming
- `custom_sampling.rs` -- Custom sampling parameters and presets

```bash
cargo run --example basic_inference -- --model models/Bonsai-8B.gguf
```

## COOLJAPAN Ecosystem

```
OxiBonsai (Pure Rust 1-bit LLM inference)
  ├── SciRS2 v0.4.x     (tensor primitives, activation functions)
  ├── OxiBLAS v0.2.x    (GEMM/GEMV + 1-bit compute kernels)
  ├── OxiFFT v0.1.x     (optional RoPE acceleration)
  └── NumRS2 v0.3.x     (N-dimensional array backend)
```

All dependencies are Pure Rust -- zero C/C++/Fortran, zero FFI, zero `unsafe` in business logic.

## Development Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Foundation (workspace, GGUF loader, metadata) | ✅ Complete |
| Phase 1 | 1-Bit Kernels (dequant, GEMV, GEMM) | ✅ Complete |
| Phase 2 | Transformer Engine (Qwen3-8B forward pass) | ✅ Complete |
| Phase 3 | Inference Runtime (KV cache, sampling, CLI) | ✅ Complete |
| Phase 4 | Production Hardening (SIMD, parallel, tests, observability) | ✅ Complete |
| Phase 5 | Ecosystem Integration (SSE streaming, WASM, API, Bonsai family) | ✅ Complete |
| Phase 6 | Advanced Infrastructure (Multi-GPU, CUDA/Metal, PagedAttention) | ✅ Complete |
| Phase 7 | Production Features (model merging, flash decoding, RAG, eval) | ✅ Complete |
| Phase 8 | Final Polish (K-quant, streaming GGUF, kernel tuning, tests) | ✅ Complete |

## Performance Targets

| Platform | llama.cpp (C++) | OxiBonsai Target |
|----------|----------------|-----------------|
| x86-64 AVX2 | ~25 t/s | >= 20 t/s |
| x86-64 AVX-512 | ~35 t/s | >= 30 t/s |
| ARM NEON | ~20 t/s | >= 18 t/s |
| WASM | N/A | >= 5 t/s |

## Sponsorship

OxiBonsai is developed and maintained by **COOLJAPAN OU (Team Kitasan)**.

The COOLJAPAN Ecosystem represents one of the largest Pure Rust scientific computing efforts in existence — spanning 40+ projects, 500+ crates, and millions of lines of Rust code across scientific computing, machine learning, quantum computing, geospatial analysis, legal technology, multimedia processing, and more. Every line is written and maintained by a small dedicated team committed to a C/Fortran-free future for scientific software.

If you find OxiBonsai or any COOLJAPAN project useful, please consider sponsoring to support continued development.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps us:
- Maintain and expand the COOLJAPAN ecosystem (40+ projects, 500+ crates)
- Keep the entire stack 100% Pure Rust — no C/Fortran/system library dependencies
- Develop production-grade alternatives to OpenCV, FFmpeg, SciPy, NumPy, scikit-learn, PyTorch, TensorFlow, GDAL, and more
- Provide long-term support, security updates, and documentation
- Fund research into novel Rust-native algorithms and optimizations

## License

Apache License, Version 2.0

Copyright 2026 COOLJAPAN OU (Team KitaSan)
