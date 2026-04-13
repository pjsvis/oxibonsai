# oxibonsai-tokenizer TODO

> Pure Rust BPE tokenizer: encode, decode, training, serialization
> 9 files, ~3,200 lines, 85 tests

## Status: ✅ All Features Complete

Full BPE tokenizer with training, encoding/decoding, batch operations, special token handling, and JSON serialization.

## Done

- [x] `OxiTokenizer` struct — encode, decode, batch encode/decode
- [x] BPE algorithm — `BpeMerges` table, `bpe_encode`, GPT-2 style pre-tokenization
- [x] Byte fallback tokens (`<0xHH>`) for unknown bytes
- [x] Special token handling (BOS, EOS, PAD, custom tokens)
- [x] Char-level stub mode for testing without trained vocab
- [x] `BpeTrainer` — learn merges from corpus with configurable vocab size
- [x] `TrainerConfig` — merge frequency thresholds, training statistics
- [x] `Vocabulary` — bidirectional token↔ID mapping
- [x] JSON/base64 serialization for trained tokenizers
- [x] WASM-safe implementation (no filesystem dependency in core)
- [x] Comprehensive tests — serialization roundtrip, trainer correctness, edge cases
