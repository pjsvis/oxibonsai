# OxiBonsai Agent Guide

## MANDATORY: Use td for Task Management

Run td usage --new-session at conversation start (or after /clear). This tells you what to work on next.

Sessions are automatic (based on terminal/agent context). Optional:
- td session "name" to label the current session
- td session --new to force a new session in the same context

Use td usage -q after first read.

## Project Overview

OxiBonsai is a Pure Rust sub-2-bit LLM inference engine for PrismML Bonsai models (Q1_0_g128 1-bit and TQ2_0_g128 ternary). It runs on CPU (SIMD), Apple Silicon (Metal), and NVIDIA (CUDA) without any C/C++ dependencies.

**Version:** 0.1.2 · **~111k lines of Rust** · 3,544 tests passing

## Crate Architecture

```
oxibonsai/
├── crates/
│   ├── oxibonsai-core/        GGUF loader, tensor types, config, error types
│   ├── oxibonsai-kernels/     Q1 + TQ2 kernels (dequant, GEMV, GEMM, SIMD, GPU)
│   ├── oxibonsai-tokenizer/   Pure Rust BPE tokenizer
│   ├── oxibonsai-model/       Qwen3 Transformer (GQA, SwiGLU, RoPE, RMSNorm)
│   ├── oxibonsai-runtime/     Inference engine, sampling, server
│   ├── oxibonsai-rag/         RAG pipeline
│   ├── oxibonsai-eval/        Evaluation harness
│   └── oxibonsai-serve/       Standalone server binary
├── src/main.rs                CLI entry point
├── tests/                     Integration tests
└── examples/                  Usage examples
```

## Key Data Structures

### SamplingConfig (runtime/config.rs)
```rust
pub struct SamplingConfig {
    pub temperature: f32,        // 0.7 default
    pub top_k: usize,            // 40 default
    pub top_p: f32,              // 0.9 default
    pub repetition_penalty: f32, // 1.1 default
    pub max_tokens: usize,       // 512 default
}
```

### SamplingParams (runtime/sampling.rs)
```rust
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub max_tokens: usize,
}
```

## Configuration System

OxiBonsaiConfig::load_or_default(path) loads TOML with sections:
- `[model]` — model_path, tokenizer_path, max_seq_len
- `[sampling]` — temperature, top_k, top_p, repetition_penalty, max_tokens
- `[server]` — host, port
- `[observability]` — log_level, json_logs

**Precedence chain:** `defaults < TOML < CLI`

## CLI Commands (src/main.rs)

| Command | Description |
|---------|-------------|
| `run` | Single inference run |
| `chat` | Interactive multi-turn conversation |
| `serve` | OpenAI-compatible API server |
| `info` | Display model metadata |
| `benchmark` | Throughput benchmark |
| `convert` | HuggingFace safetensors → GGUF |
| `quantize` | Quantize model |
| `validate` | Validate GGUF file |

## Build Features

```bash
cargo build --release \
  --features "simd-neon metal native-tokenizer"
```

| Feature | Description |
|---------|-------------|
| `simd-neon` | ARM NEON SIMD kernels |
| `simd-avx2` | x86 AVX2 + FMA |
| `simd-avx512` | x86 AVX-512 |
| `metal` | Apple Silicon GPU |
| `native-cuda` | NVIDIA native CUDA |
| `cuda` | scirs2-core CUDA |
| `server` | Enable serve command |

## Common Tasks

### Run inference
```bash
cargo run --release -- run \
  --model models/Bonsai-8B.gguf \
  --prompt "Hello" \
  --max-tokens 256
```

### Run tests
```bash
cargo test --features "simd-neon metal native-tokenizer"
```

### Create a new branch for a fix
```bash
git checkout -b fix/your-fix-name
```

## Known Issues / Pending Work

See `briefs/` directory for implementation briefs (e.g., `briefs/01-*-sampling-toml.md`).
See `debriefs/` directory for completed work summaries.
See `TODO.md` for general roadmap items.