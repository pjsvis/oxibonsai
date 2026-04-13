# oxibonsai-model TODO

> Qwen3 transformer model: layers, blocks, forward pass, KV cache
> 15 files, ~2,700 lines, 73 tests

## Status: ✅ All Features Complete

Flash attention, FP16 KV cache, multi-model registry, sliding window, tracing instrumentation, layer correctness tests, and numerical stability tests all implemented.

## Done

- [x] `BonsaiModel` — token embedding, 36 transformer blocks, final norm, LM head
- [x] `TransformerBlock` — attention sublayer + FFN sublayer with residuals (`block.rs`)
- [x] RMSNorm layer
- [x] Rotary Position Embeddings (RoPE, base=1M)
- [x] Grouped Query Attention (32 Q heads, 8 KV heads, head_dim=128)
- [x] `CausalMask` — causal attention mask support (`attention.rs`)
- [x] SwiGLU FFN (gate/up projection → SiLU × gate → down projection)
- [x] Sliding window attention (`sliding_window.rs`)
- [x] 1-bit linear layer (Q1_0_g128 weight projection via kernels)
- [x] KV cache with position-indexed storage
- [x] **KV cache FP16 optimization** — K/V stored in f16, halving cache memory (`kv_cache_fp16.rs`)
- [x] **Flash attention** — Fused attention kernel to reduce memory bandwidth (`attention_fused.rs`)
- [x] GGUF weight loading with tensor name mapping
- [x] **Bonsai-4B architecture** — config, layers, dims (`model_registry.rs`)
- [x] **Bonsai-1.7B architecture** — smaller variant config (`model_registry.rs`)
- [x] **Architecture auto-detection** — Infer model variant from GGUF metadata (`model_registry.rs`)
- [x] **LayerStats** — Per-layer statistics instrumentation (`block.rs`)
- [x] **#[instrument] tracing** — Span instrumentation on `forward()` and `TransformerBlock::forward()`
- [x] **Layer-level correctness** — `tests/layer_correctness_tests.rs` with 23 tests: RMSNorm, SwiGLU, RoPE, Attention, TransformerBlock reference comparisons
- [x] **Numerical stability tests** — `tests/numerical_stability_tests.rs` with 25 tests: extreme inputs, overflow/underflow, long sequences, KV cache stress
