# oxibonsai-eval TODO

> Model evaluation harness: perplexity, accuracy, ROUGE, throughput metrics
> 9 files, ~2,100 lines, 38 tests

## Status: ✅ All Features Complete

Full evaluation framework with perplexity, multiple-choice accuracy, ROUGE scoring, throughput benchmarking, and dataset loading.

## Done

- [x] `PerplexityEvaluator` — from logits, bits-per-byte (BPB) metric
- [x] Pure Rust log-softmax computation (no external math dependency)
- [x] `McEvaluator` — MMLU-style multiple-choice accuracy with per-subject breakdown
- [x] `ExactMatchEvaluator` — text matching evaluation
- [x] `EvalDataset` — JSONL loading, sampling, train/test splits
- [x] `McDataset` / `MultipleChoiceQuestion` — structured MC dataset support
- [x] Deterministic sampling with LCG RNG (no external rand dependency)
- [x] ROUGE-N/L/S scoring metrics
- [x] Throughput evaluator — tokens-per-second, latency benchmarking
- [x] JSON/Markdown report generation
- [x] Error types (`EvalError`)
- [x] Tests for ROUGE metrics, perplexity, accuracy scoring
