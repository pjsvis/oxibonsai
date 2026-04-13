# OxiBonsai TODO

> Pure Rust 1-bit LLM inference engine for PrismML Bonsai models
> 295 source files, ~92,305 lines of Rust code, 1,298+ tests across workspace — verified 2026-04-13

## Phase Status

| Phase | Status | Summary |
|-------|--------|---------|
| Phase 0: Foundation | ✅ Complete | Workspace, GGUF loader, metadata extraction |
| Phase 1: 1-Bit Kernels | ✅ Complete | dequant/gemv/gemm + property tests (20 kernel tests) |
| Phase 2: Transformer Engine | ✅ Complete | RMSNorm, SwiGLU, RoPE, GQA, 36-layer Qwen3 forward pass |
| Phase 3: Inference Runtime | ✅ Complete | KV cache, sampling, tokenizer, CLI (run/chat/serve/info), OpenAI-compatible server |
| Phase 4: Production Hardening | ✅ Complete | AVX2 SIMD, NEON, AVX-512, rayon, benchmarks, tracing, TOML config, proptests, integration tests, circuit breaker, health, metrics, recovery, Prometheus metrics, memory profiling, fuzz testing |
| Phase 5: Ecosystem Integration | ✅ Complete | All major subsystems implemented (see below) |
| Phase 6: Advanced Infrastructure | ✅ Complete | Multi-GPU, CUDA/Metal stubs, PagedAttention, GGUF v3, dynamic quant, profiler |
| Phase 7: Production Features | ✅ Complete | Model merging, checkpointing, flash decoding, attention sink, compression, ROUGE, MoD, adaptive sampling, oxibonsai-serve |
| Phase 8: Final Polish | ✅ Complete | K-quant formats, streaming GGUF, kernel tuning, comprehensive testing |

## Phase 4 — Complete

- [x] **NEON (ARM) SIMD kernels** -- Implemented `simd_neon.rs` with dequant/gemv/gemm
- [x] **AVX-512 SIMD kernels** -- Implemented `simd_avx512.rs` (compile-checked, untested on ARM)
- [x] **Property-based tests** -- proptest roundtrip, linearity, cross-tier in `proptest_kernels.rs`
- [x] **Integration tests** -- `tests/integration_pipeline.rs` + `tests/cross_tier.rs` + `tests/feature_flags.rs`
- [x] **Tracing enhancement** -- EnvFilter + `#[instrument]` spans on generate, forward, block, server, sampling
- [x] **TOML configuration system** -- `OxiBonsaiConfig` with layered loading, `--config` CLI arg
- [x] **Circuit breaker** -- Overload protection with configurable thresholds
- [x] **Health checks** -- `/health` endpoint with readiness probes
- [x] **Inference metrics** -- tokens/sec, latency, request counts via `InferenceMetrics`
- [x] **Builder pattern API** -- `EngineBuilder`, `ConfigBuilder`, `SamplerBuilder`
- [x] **Sampling presets** -- Greedy, Balanced, Creative, Code presets
- [x] **Batch engine** -- Batched inference support
- [x] **Async engine** -- Tokio-based async inference
- [x] **Recovery strategies** -- Error classification and automatic recovery
- [x] **Prometheus metrics** -- Self-contained Prometheus-compatible metrics; `/metrics` endpoint via `InferenceMetrics` + `metrics_text()` renderer
- [x] **Memory profiling** -- `MemoryProfiler` + `MemorySnapshot`; RSS via Mach (macOS) / `/proc/self/statm` (Linux); peak tracking
- [x] **Fuzz testing** -- Property-based kernel + GGUF parsing fuzz tests; cargo-fuzz harnesses being added

## Phase 5 — Complete

- [x] **Streaming chat completions (SSE)** -- Implemented with `generate_streaming` + SSE handler
- [x] **WASM compilation target** -- Compiles to `wasm32-unknown-unknown`; `wasm_api.rs` provides WASM-safe generate API
- [x] **Bonsai-4B and Bonsai-1.7B model support** -- `Qwen3Config::bonsai_4b()` / `bonsai_1_7b()`, `ModelVariant::Bonsai4B` / `Bonsai1_7B`, `model_variants.rs` with full `ModelSpec` + `CapabilityProfile`
- [x] **OxiRAG integration** -- Pure Rust RAG pipeline (`oxibonsai-rag`): chunker, IdentityEmbedder, TfIdfEmbedder, VectorStore, Retriever, RagPipeline
- [x] **MeCrab/OxiTokenizer** -- Native BPE tokenizer stub (`oxibonsai-tokenizer`): Vocabulary, BpeMerges, OxiTokenizer, char-level fallback, ChatTemplate/chatml
- [x] **Advanced sampling** -- Mirostat v1/v2, Locally Typical, Eta, Min-P samplers; `SamplerChain` composable pipeline; `LcgRng`
- [x] **Speculative decoding** -- `SpeculativeDecoder` with draft/verify loop, acceptance rate tracking, `generate_speculative`
- [x] **LoRA adapter support** -- `LoraAdapter`, `LoraRegistry`, `BonsaiLoraSet`; apply/merge into weights
- [x] **Context window management** -- `ContextWindow` with TruncateLeft/Right/SlidingWindow/Summarize strategies; system-prompt preservation
- [x] **Beam search** -- `BeamSearchEngine` with configurable `beam_width`, length penalty, no-repeat n-gram, early stopping
- [x] **Token healing** -- `TokenHealer`, `TokenHealingConfig`, `HealingDecoder`
- [x] **Inference pipeline API** -- `InferencePipeline`, `PipelineBuilder`, `GenerationStrategy`, `StopReason`, `PipelineOutput`; convenience constructors (`chat_pipeline`, `code_pipeline`, `greedy_pipeline`)
- [x] **INT8 quantization** -- Per-tensor and per-channel INT8 quantization; `Int8Tensor` with `dequantize()` and `matvec()`
- [x] **GGUF writer** -- `GgufWriter` producing valid GGUF byte streams with metadata and tensor entries
- [x] **Model export** -- `export_to_gguf` supporting Float32, Q1_0_g128, Int8PerChannel formats; FP32 exception layers; `ExportStats`
- [x] **Prefix caching** -- `PrefixCache`, `CacheBlock`, `CacheSession`, `PrefixAwarePrefill` for KV reuse
- [x] **Semantic cache** -- `SemanticCache` with cosine-similarity lookup for repeated prompts
- [x] **MoE architecture** -- Mixture-of-Experts router and expert networks (`moe_router.rs`, `moe_expert.rs`)
- [x] **Tensor parallelism** -- `TensorParallelConfig`, column/row parallel linear; `AllReduceOp`
- [x] **Grammar-constrained decoding** -- `TokenConstraint` trait; `JsonConstraint`, `RegexConstraint`, `AllowListConstraint`, `SequenceConstraint`
- [x] **OpenAI API extensions** -- Function calling, logprobs, JSON mode, `n` completions, response format, frequency/presence penalties (`api_extensions.rs`, `api_types.rs`)
- [x] **Embeddings endpoint** -- `POST /v1/embeddings`; `EmbedderRegistry` (IdentityEmbedder + TfIdfEmbedder); float + base64 encoding; dimensions truncation
- [x] **Completions endpoint** -- `POST /v1/completions` with single/batch prompt, echo, logprobs stub
- [x] **RAG server endpoints** -- `/v1/rag/index`, `/v1/rag/query` (feature-gated)
- [x] **Rate limiting + middleware** -- `RateLimiter`, `MiddlewareConfig`; tower-based rate-limit and timeout layers
- [x] **Admin API** -- `/admin/status`, `/admin/config`, `/admin/reset-metrics`, `/admin/cache-stats`
- [x] **Evaluation harness** -- `oxibonsai-eval` crate: HellaSwag, MMLU, TruthfulQA benchmarks; `EvalRunner`, `EvalResult`
- [x] **Optimizer suite** -- SGD (with momentum, Nesterov, weight decay), Adam, AdamW; CosineAnnealing / WarmupConstant / Constant LR schedulers
- [x] **LoRA training scaffold** -- `LoraTrainer` with per-step Adam updates, gradient clipping, warmup; `TrainingStep` metrics
- [x] **Fine-grained gradient computation** -- `GradientContext`, `LayerGradients`, `backward_rms_norm`, `backward_swiglu`, `backward_linear`
- [x] **Continuous batching** -- `ContinuousBatchScheduler` for in-flight request management
- [x] **Model registry** -- `ModelVariant`, `ModelSpec`, `CapabilityProfile`; auto-detection from config dimensions
- [x] **Comprehensive integration tests** -- `tests/full_pipeline_tests.rs` (25 tests), `tests/quantization_pipeline_tests.rs` (10 tests), `tests/server_integration_tests.rs` (8 tests)

## CI / Publish

- [x] Publish script (`scripts/publish.sh`) -- dry-run by default, `--for-real` for actual publish
- [x] Feature flag tests (`tests/feature_flags.rs`) -- compile-time validation of feature propagation
- [x] GitHub Actions CI pipeline -- `.github/workflows/ci.yml` (build, test, clippy, WASM, feature matrix)
- [ ] crates.io publish (awaiting approval)

## Phase 6 — Complete

- [x] **True multi-GPU inference** -- NCCL-style all-reduce/gather/scatter; `DeviceMesh` (tp×pp); rayon-simulated collectives; column/row weight partition + merge helpers; `multi_gpu.rs` + 27 tests
- [x] **CUDA/Metal backend** -- GPU kernel dispatch stubs: `GpuBackend` trait, `CpuBackend` (always-on), `CudaBackend` (feature = "cuda"), `MetalBackend` (feature = "metal"); `select_backend`, `gpu_matmul`; 18 integration tests; cuBLAS / Metal Performance Shaders left as future implementation
- [x] **Production GGUF with real model weights** -- Streaming `TensorChunkIter`; memory budget estimation (`estimate_memory_bytes`, `fits_in_budget`); lazy tensor metadata loading; `validate_gguf_file`; `gguf_loader.rs` + 25 tests
- [x] **Web UI** -- Chat interface; model management dashboard
- [x] **Distributed serving** -- Multi-node coordinator; consistent hashing for request routing
- [x] **Full MeCrab tokenizer integration** -- `NativeTokenizerBridge` added to `oxibonsai-runtime`; wraps `OxiTokenizer` with encode/decode/chat-format API; pure Rust, WASM-compatible, zero C/FFI; 10 integration tests all passing
- [x] **PagedAttention / vLLM-style KV management** -- Block-based KV cache for higher utilisation; `PagedKvCache`, `BlockPool`, `BlockTable`, `KvPage` in `oxibonsai-model/src/paged_kv_cache.rs`
- [x] **GGMLv4 / GGUF v3 support** -- Forward compat layer: `GgufVersion`, `ExtendedQuantType`, `GgufCompatReport`, `check_gguf_header`, `build_compat_report` in `gguf/compat.rs`; 20 integration tests all passing
- [x] **Sparse attention patterns** -- `SparseAttentionMask` with Local Window, Strided, Bigbird, Longformer, and Dilated patterns; `sparse_attention_forward`; memory reduction estimate; `sparse_attention.rs` + tests in `oxibonsai-model`
- [x] **Weight tying** -- `TiedEmbedding` with tied embedding/unembedding pair; `embed()`, `lm_head_logits()`, `set_embedding()` API; `TyingError` type; `weight_tying.rs` + tests in `oxibonsai-model`
- [x] **Dynamic quantization** -- `DynamicQuant` trait; `DynamicQ8_0`, `DynamicQ4_0`, `DynamicQ4_1` quantizers; `QuantizedTensor`, `quantize_model_dynamic`, `calibrate_scale`; `dynamic_quant.rs` + 21 tests
- [x] **Runtime profiler** -- `InferenceProfiler` with span tracking, `SpanRecord`, `ProfileReport`, stats (mean/min/max/p95/p99); flame-graph text renderer; `profiler.rs` + 20 tests
- [x] **Advanced RAG chunking** -- `SemanticChunker` with cosine-similarity boundary detection; `HierarchicalChunker` with multi-level chunking; `SlideWindowChunker`; 31 tests in `oxibonsai-rag`
- [x] **Tokenizer serialization** -- `save_json`/`load_json` for `OxiTokenizer`; vocabulary and BPE merge round-trip; `TokenizerSerializer`; 16 tests in `oxibonsai-tokenizer`
- [x] **GGUF model card** -- `ModelCard` with metadata (author, license, description, tags, datasets); `CardFormat` enum; `render_card`/`parse_card`; embedded in GGUF metadata via `oxibonsai-core`
- [x] **Token streaming metrics** -- `StreamingSession` with per-token timing; `TokenEvent`, `StreamingStats` (mean/p50/p95 TTFT + TPOT); `StreamingProfiler`; 22 tests in `oxibonsai-runtime`
- [x] **KV cache quantization** -- `QuantizedKvCache` with Q8 and Q4 formats; per-head scale factors; `KvCacheQuantizer`, `store_quantized`, `load_dequantized`; `kv_cache_quant.rs` + 18 tests

## Phase 7 — Complete

- [x] **KV cache quantization (INT8)** -- `QuantizedKvCache` with Q8/Q4 formats; per-head scale factors; `kv_cache_quant.rs`
- [x] **Model merging (SLERP, TIES, DARE, task vector)** -- `model_merge.rs` with multi-strategy merge support
- [x] **Checkpoint save/load (OXCK binary format)** -- `checkpoint.rs` with serialization/deserialization
- [x] **YaRN extended RoPE** -- `yarn_rope.rs` with YaRN-style RoPE scaling for extended context
- [x] **Flash Decoding** -- `flash_decode.rs` for optimized single-token decoding
- [x] **Attention Sink / StreamingLLM** -- `attention_sink.rs` with sink tokens for infinite context windows
- [x] **Dynamic activation quantization** -- `dynamic_quant.rs` with DynamicQ8_0/Q4_0/Q4_1 quantizers
- [x] **Inference profiler with FLOP counting** -- `profiler.rs` with span tracking, p95/p99 stats, flame-graph renderer
- [x] **GGUF model card generation** -- `ModelCard` with metadata; `render_card`/`parse_card` in `oxibonsai-core`
- [x] **Token streaming metrics (TTFT, TBT)** -- `StreamingSession`, `StreamingStats`, `StreamingProfiler` in `oxibonsai-runtime`
- [x] **Weight importance analysis and pruning** -- `pruning.rs` with structured/unstructured pruning
- [x] **PTQ calibration pipeline** -- `calibration.rs` with post-training quantization calibration
- [x] **BPE tokenizer training** -- `trainer.rs` in `oxibonsai-tokenizer` with BPE merge learning
- [x] **Tokenizer serialization** -- `save_json`/`load_json` for `OxiTokenizer`; vocabulary and BPE merge round-trip
- [x] **Advanced RAG chunking** -- Sentence, recursive, sliding window, markdown chunking in `oxibonsai-rag`
- [x] **Loss functions (CE, focal, distillation, contrastive)** -- `losses.rs` with full training loss suite
- [x] **Advanced LR schedulers (OneCycle, ReducePlateau, CyclicLR)** -- `lr_schedulers.rs` with advanced scheduling
- [x] **Sparse attention patterns (local window, BigBird, strided)** -- `sparse_attention.rs` with multiple patterns
- [x] **Weight tying (embedding/LM head sharing)** -- `weight_tying.rs` with `TiedEmbedding`
- [x] **Model compression pipeline (prune+quantize)** -- `compression.rs` with end-to-end compression
- [x] **ROUGE evaluation metrics (ROUGE-1/2/L/S)** -- `rouge.rs` in `oxibonsai-eval`
- [x] **Mixture of Depths (adaptive compute)** -- `mixture_of_depths.rs` with `ModRouter`
- [x] **Token budget tracking** -- `token_budget.rs` in `oxibonsai-runtime`
- [x] **RoPE scaling variants (linear, DynamicNTK, LLaMA 3.1, LongRoPE)** -- `rope_scaling.rs` with multiple strategies
- [x] **N-best hypothesis tracking** -- `nbest.rs` in `oxibonsai-runtime`
- [x] **Adaptive sampling (entropy cooling, repetition adaptation)** -- `adaptive_sampling.rs` with `AdaptiveChain`
- [x] **Generation quality metrics (repetition, diversity, BLEU)** -- `quality_metrics.rs` in `oxibonsai-runtime`
- [x] **oxibonsai-serve standalone binary crate** -- `crates/oxibonsai-serve` with CLI args, banner, OpenAI-compatible server
- [x] **Cross-attention layers** -- `cross_attention.rs` for encoder-decoder architectures
- [x] **Gradient checkpointing** -- `gradient_checkpoint.rs` for memory-efficient training
- [x] **Request deduplication** -- `dedup.rs` with FNV-1a hashing for duplicate request detection
- [x] **Model hot-reload** -- `hot_reload.rs` for live model swapping without restart
- [x] **Performance auto-tuning engine** -- `auto_tuner.rs` with CPU detection, SIMD tier selection, memory budget estimation, kernel benchmarking
- [x] **JSON Schema structured output constraint** -- `json_schema.rs` with schema parsing, validation state machine, constrained token generation
- [x] **Multi-model serving with LoRA routing** -- `multi_model.rs` / `lora_router` with dynamic model selection and LoRA adapter dispatch

## Phase 8 — Complete

- [x] **Q2_K and Q4_K quantization formats** — `BlockQ2K`/`BlockQ4K` with dequant/quantize in `oxibonsai-core/quant_k.rs`; 21 tests
- [x] **Streaming GGUF reader** — `GgufStreamParser` state machine for progressive network-loaded GGUF parsing; 22 tests
- [x] **Kernel performance tuning** — `PlatformProfile` detection, `TunedThresholds` auto-computation, cache-line aligned buffers, software prefetch hints; 29 tests
- [x] **Layer-level correctness tests** — RMSNorm, SwiGLU, RoPE, Attention, TransformerBlock reference comparisons; 23 tests
- [x] **Numerical stability tests** — Extreme inputs, overflow/underflow, long sequences, KV cache stress; 25 tests
- [x] **Full generate() pipeline integration tests** — End-to-end generation, determinism, sampling parameter effects, edge cases; 26 tests
- [x] **Sampling distribution statistical tests** — Chi-square goodness-of-fit, temperature/top-k/top-p/repetition penalty validation; 22 tests

## Performance Targets

| Platform | Target | Status |
|----------|--------|--------|
| x86-64 AVX2 (Zen4) | >= 20 t/s | 🔶 Kernels done, not benchmarked end-to-end |
| x86-64 AVX-512 | >= 30 t/s | 🔶 Kernels done, not benchmarked end-to-end |
| ARM NEON | >= 18 t/s | 🔶 Kernels done, not benchmarked end-to-end |
| WASM | >= 5 t/s | 🔶 Compiles; runtime throughput not benchmarked |

## Memory Targets

| Metric | Target | Status |
|--------|--------|--------|
| Model weights (mmap) | 1.15 GB | ✅ mmap zero-copy implemented |
| KV cache (4k context) | ~256 MB | ✅ Implemented |
| Runtime overhead | < 50 MB | 🔶 Profiler available via `MemoryProfiler`; end-to-end not yet characterized |
| Total (4k context) | < 1.5 GB | 🔶 Not profiled end-to-end |
