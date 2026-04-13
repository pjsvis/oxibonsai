# oxibonsai-core

GGUF Q1_0_g128 parser, tensor types, and configuration for OxiBonsai.

Provides the foundational data types and I/O layer: GGUF file loading (v1/v2/v3),
Q1_0_g128 block tensor deserialization, Qwen3 model configuration, streaming GGUF
parser, GGUF writer, model card generation, and all shared error types.

Part of the [OxiBonsai](https://github.com/cool-japan/oxibonsai) project.

## Features

- GGUF v1/v2/v3 reader with forward-compatibility layer
- `GgufStreamParser` — state-machine streaming parser for network-loaded models
- `GgufWriter` — produce valid GGUF byte streams with metadata and tensors
- `Qwen3Config` — model configuration for Bonsai-8B, 4B, and 1.7B variants
- `BlockQ1_0G128` / `OneBitTensor` — Q1_0_g128 block tensor types
- `ModelCard` — structured model card (author, license, tags) embedded in GGUF
- K-quant formats: `BlockQ2K`, `BlockQ4K`
- Memory budget estimation and tensor metadata lazy loading
- `mmap` feature for zero-copy model file access

## Feature Flags

| Flag | Description | Default |
|------|-------------|---------|
| `mmap` | Memory-mapped file access via `memmap2` | enabled |
| `wasm` | WASM-safe builds (no `memmap2`) | disabled |

## Usage

```toml
[dependencies]
oxibonsai-core = "0.1.0"
```

## License

Apache-2.0 — COOLJAPAN OU
