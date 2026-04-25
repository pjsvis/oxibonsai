# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### BREAKING
- CLI now **requires** `--tokenizer` for `run` and `chat` commands.
  The fallback behavior (using hardcoded token 151644) has been removed.
  Update scripts/automations to include `--tokenizer <path/to/tokenizer.json>`.

### Added
- Qwen3 chat template rendering with system prompt support via new `--system` flag for both Run and Chat commands
- Interactive Chat REPL with multi-turn conversation history, `/system`, `/help`, `/reset`, and `/quit` commands
- Repetition penalty now actually applied during sampling (previously threaded but unused)
- TOML `[sampling]` config values are now validated before use

### Fixed
- Chat history now correctly stores both user and assistant messages for proper multi-turn context
- Repetition penalty CLI flag and TOML value now affect sampling (Bug #2 fix)
- TOML sampling config values are validated (Bug #3 fix)

## [0.1.2] - 2026-04-19

### Added
- ONNX MatMulNBits (bits=2) ingestion — `oxibonsai convert --onnx` reads onnx-community Ternary releases directly and repacks them as GGUF (TQ2_0_g128)
- Qwen3 ONNX tensor role mapping for the converter

### Changed
- Upgraded `oxionnx-proto` workspace dependency to 0.1.2
- Workspace version bump to 0.1.2 across all nine subcrates and `[workspace.dependencies]`
- Alpha → Stable uplift for `oxibonsai-tokenizer`, `oxibonsai-rag`, `oxibonsai-eval`, and `oxibonsai-serve`

## [0.1.1] - 2026-04-18

### Added
- Native CUDA NVRTC backend with fused Q1 + TQ2 full-forward path (~21.9 tok/s on Ternary-Bonsai-1.7B : RTX 3060 CUDA 12.8)
- Fused Metal TQ2 full-forward — single GPU command buffer per token, ~50 tok/s on Ternary-Bonsai-1.7B (~13× speedup)
- Ternary CPU SIMD tiers (NEON/AVX2/AVX-512 TQ2 GEMV)
- TQ2_0_g128 support in the Metal backend (per-kernel dispatch + `blocks_as_bytes_ternary` zero-copy upload)
- `scripts/bench_ternary.sh` — CPU vs Metal throughput bench (3-run average + best)
- `scripts/download_ternary.sh` — fetch + convert safetensors → GGUF

### Changed
- Version bump to 0.1.1
- Internal dependency version alignment across workspace
- CUDA full-forward layer parameter handling refactored for cleaner weight management
- Workspace Cargo.toml files unified on workspace dependencies for better crate compatibility

### Fixed
- Workspace version consistency across all subcrates
- `blocks_as_bytes` import gating for broader feature-flag compatibility

## [0.1.0] - 2026-04-13

### Added

- Pure Rust 1-bit LLM inference engine for PrismML Bonsai models
- GGUF Q1_0_g128 format support with streaming parser
- Optimized 1-bit kernel operations (dequantization, GEMV, GEMM)
- SIMD acceleration support (AVX2, AVX-512, NEON, WASM SIMD)
- Parallel kernel dispatch with Rayon
- Qwen3 transformer model implementation with paged KV-cache
- High-level inference runtime with autoregressive generation
- Sampling strategies (greedy, top-k, top-p, temperature)
- OpenAI-compatible REST API server (chat completions, completions, embeddings)
- Streaming token generation via SSE
- RAG pipeline with chunking and similarity search
- Pure Rust BPE tokenizer
- Model evaluation framework (accuracy, perplexity metrics)
- WASM compilation target support
- Speculative decoding support
- Comprehensive test suite (140 tests)
- Cross-platform support (macOS, Linux, Windows, WASM)

[0.1.2]: https://github.com/cool-japan/oxibonsai/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/cool-japan/oxibonsai/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/cool-japan/oxibonsai/releases/tag/v0.1.0
