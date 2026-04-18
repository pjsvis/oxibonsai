# oxibonsai-eval

[![Version](https://img.shields.io/badge/version-0.1.1-blue.svg)](https://crates.io/crates/oxibonsai-eval)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)]()
[![Tests](https://img.shields.io/badge/tests-38%20passing-brightgreen.svg)]()

Model evaluation harness for OxiBonsai — ROUGE, perplexity, accuracy, throughput.

Provides perplexity measurement, MMLU-style multiple-choice accuracy,
ROUGE-N/L/S scoring, exact-match scoring, throughput benchmarking, JSONL
dataset loading, and JSON/Markdown report generation.

Part of the [OxiBonsai](https://github.com/cool-japan/oxibonsai) project.

## Status

**Alpha** (v0.1.1) — 38 tests passing. API may change.

## Features

- `PerplexityEvaluator` — from log-probs or logits; bits-per-byte metric
- `McEvaluator` — MMLU-style multiple-choice with per-subject breakdown
- `ExactMatchEvaluator` — text-match evaluation
- ROUGE scoring: `RougeNScore` (ROUGE-1/2), `RougeLScore`, `RougeSScore`, `CorpusRouge`
- `ThroughputBenchmark` — tokens/s, prefill/decode latency, p95/p99
- `EvalDataset` — JSONL loading, train/test splits, deterministic sampling
- `EvalReport` — JSON and Markdown report generation
- Zero external API dependencies — pure Rust

## Usage

```toml
[dependencies]
oxibonsai-eval = "0.1.1"
```

## License

Apache-2.0 — COOLJAPAN OU
