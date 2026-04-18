# oxibonsai-tokenizer TODO

> Pure Rust BPE tokenizer: encode, decode, training, serialization
> 9 files, ~3,200 lines, 85 tests (all passing)
> Version: 0.1.1 · Last updated: 2026-04-18

## Status: Alpha — Core Feature Set Complete

Full BPE tokenizer with training, encoding/decoding, batch operations, special token handling, chat templating, and JSON serialization. Functional for OxiBonsai runtime needs, still maturing relative to HuggingFace `tokenizers`.

## Done

- [x] `OxiTokenizer` struct — encode, decode, batch encode/decode
- [x] BPE algorithm — `BpeMerges` table, `bpe_encode`, GPT-2 style pre-tokenization
- [x] Byte fallback tokens (`<0xHH>`) for unknown bytes
- [x] Special token handling (BOS, EOS, PAD, custom tokens)
- [x] Char-level stub mode (`char_level_stub`) for testing without trained vocab
- [x] `BpeTrainer` — learn merges from corpus with configurable vocab size
- [x] `TrainerConfig` — merge frequency thresholds, training statistics
- [x] `Vocabulary` — bidirectional token↔ID mapping
- [x] `ChatTemplate` — chatml-style formatting with user-message extraction
- [x] `BatchEncoder` — padding (`PaddingStrategy`) and truncation (`TruncationSide`)
- [x] `from_json(vocab_json, merges_json, config)` tokenizer loader
- [x] `TokenizerState::save` / `load` — base64 serialization format (FORMAT_MAGIC)
- [x] WASM-safe implementation (no filesystem dependency in core)
- [x] Comprehensive tests (85) — serialization roundtrip, trainer correctness, chat template, edge cases
