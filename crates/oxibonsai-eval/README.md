# oxibonsai-eval

Model evaluation harness for OxiBonsai — ROUGE, perplexity, accuracy, throughput.

Provides perplexity measurement, MMLU-style multiple-choice accuracy,
ROUGE-N/L/S scoring, exact-match scoring, throughput benchmarking, JSONL
dataset loading, and JSON/Markdown report generation.

Part of the [OxiBonsai](https://github.com/cool-japan/oxibonsai) project.

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
oxibonsai-eval = "0.1.0"
```

## License

Apache-2.0 — COOLJAPAN OU
