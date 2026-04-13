# oxibonsai-model

Qwen3 Transformer implementation for 1-bit Bonsai inference.

Implements the full autoregressive forward pass for the Qwen3 architecture family
(Bonsai-8B, 4B, 1.7B) using Q1_0_g128 1-bit quantised weights — token embedding,
Grouped Query Attention with RoPE, SwiGLU MLP, RMSNorm, and paged KV-cache.

Part of the [OxiBonsai](https://github.com/cool-japan/oxibonsai) project.

## Features

- `BonsaiModel` — full Qwen3-8B: 36 layers, GQA 32Q/8KV, SwiGLU, RoPE, RMSNorm
- `TransformerBlock` — attention + FFN sublayers with residual connections
- `PagedKvCache` — vLLM-style block-based KV cache for high utilization
- Sparse attention patterns: local window, BigBird, Longformer, dilated
- Flash attention and FP16 KV cache for reduced memory bandwidth
- RoPE scaling variants: YaRN, linear, DynamicNTK, LLaMA 3.1, LongRoPE
- Weight tying (`TiedEmbedding`) for embedding/LM head sharing
- Dynamic quantization: DynamicQ8_0, DynamicQ4_0, DynamicQ4_1
- Model merging: SLERP, TIES, DARE, task vector
- Checkpoint save/load (OXCK binary format)
- Architecture auto-detection from GGUF metadata

## Feature Flags

| Flag | Description |
|------|-------------|
| `wasm` | WASM-safe build |
| `metal` | Metal GPU backend (macOS) |
| `native-cuda` | CUDA GPU backend (NVIDIA) |

## Usage

```toml
[dependencies]
oxibonsai-model = "0.1.0"
```

## License

Apache-2.0 — COOLJAPAN OU
