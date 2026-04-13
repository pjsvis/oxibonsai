# oxibonsai-core TODO

> GGUF loader, tensor types, model config, error types
> 12 files, ~2,100 lines, 63 tests

## Status: ✅ All Features Complete

All Phase 0–1 functionality implemented and tested, including fuzz/property tests, K-quant formats, and streaming GGUF reader.

## Done

- [x] GGUF v3 header parsing
- [x] Metadata key-value extraction (string, u32, u64, f32, array)
- [x] Tensor info parsing with offset calculation
- [x] `BlockQ1_0G128` type (18 bytes: f16 scale + 16×u8 bits)
- [x] mmap reader with zero-copy tensor access
- [x] `ModelConfig` from GGUF metadata (Qwen3 architecture)
- [x] Error types (`CoreError`)
- [x] **Property tests** — GGUF roundtrip parsing, tensor block alignment assertions (`tensor_property.rs`)
- [x] **Fuzz testing** — Malformed GGUF headers, truncated files, invalid tensor offsets (`fuzz_gguf.rs`, `gguf_edge_cases.rs`)
- [x] **Additional quant formats** — Q2_K, Q4_K support implemented in `quant_k.rs` with BlockQ2K/BlockQ4K structs, dequant/quantize, 21 tests
- [x] **Streaming GGUF reader** — `gguf/streaming.rs` with GgufStreamParser state machine, progressive parsing, 22 tests
