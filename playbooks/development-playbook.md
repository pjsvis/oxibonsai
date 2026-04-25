# OxiBonsai Development Playbook

This playbook is an operational guide for working on OxiBonsai locally — building, running, testing, debugging, benchmarking, and shipping changes. It assumes you have already cloned the repo to `~/dev/github/cool-japan/oxibonsai` and have a working internet connection for crate and model downloads.

## 1. Project at a glance
OxiBonsai is a zero-FFI, zero-C/C++, Pure-Rust 1-bit LLM inference engine for the PrismML Bonsai family (currently Bonsai-8B in `Q1_0_g128`). It targets CPU (NEON/AVX2/AVX-512), Apple Silicon (Metal via `scirs2-metal`), NVIDIA (CUDA), and WASM. It ships:
- A CLI binary `oxibonsai` (`run`, `chat`, `serve`, `info`, `benchmark`, `quantize`, `validate`, `convert`, `tokenizer`)
- An OpenAI-compatible HTTP server (`oxibonsai serve`)
- A library API (`oxibonsai_runtime::EngineBuilder`)
- An evaluation harness (`oxibonsai-eval`)

The single source of architectural truth is the workspace-root `README.md`. This playbook complements it.

## 2. Prerequisites
- macOS (Apple Silicon) or Linux on a modern x86-64 / AArch64 host
- Rust toolchain. **Today** (v0.1.2) the project requires nightly because `oxibonsai-kernels` uses `#![feature(stdarch_aarch64_prefetch)]`. See Brief 02 to make this stable-buildable.
- ~3 GB free disk for the model + tokenizer + build artifacts
- A `git` install and basic familiarity with cargo workspaces

Install nightly and pin it for this directory:

```bash
rustup toolchain install nightly
cd ~/dev/github/cool-japan/oxibonsai
rustup override set nightly
```

## 3. Repository layout
```
oxibonsai/
├── Cargo.toml                 workspace root + `oxibonsai-cli` package
├── src/main.rs                CLI entry point — clap definitions, command handlers
├── crates/
│   ├── oxibonsai/             umbrella library (re-exports common types)
│   ├── oxibonsai-core/        GGUF reader, tensor types, error types
│   ├── oxibonsai-kernels/     1-bit kernels (NEON/AVX2/AVX-512/Metal/CUDA dispatch)
│   ├── oxibonsai-tokenizer/   Pure-Rust BPE tokenizer + ChatTemplate
│   ├── oxibonsai-model/       Qwen3-8B Transformer (GQA, RoPE, RMSNorm, KV cache)
│   ├── oxibonsai-rag/         RAG pipeline (chunking, embedders, vector store)
│   ├── oxibonsai-runtime/     InferenceEngine, sampling, OpenAI-compatible server
│   ├── oxibonsai-eval/        Evaluation harness (ROUGE, perplexity, MMLU)
│   └── oxibonsai-serve/       Standalone server binary
├── benches/                   Criterion micro-benchmarks
├── examples/                  Library usage examples
├── tests/                     Integration tests
└── scripts/cli.sh             End-to-end smoke test
```

When in doubt about which crate owns a behavior:
- "How is a logit produced?" → `oxibonsai-runtime` and `oxibonsai-model`
- "How is a token decoded?" → `oxibonsai-tokenizer`
- "How is a tensor stored?" → `oxibonsai-core`
- "How is matmul computed?" → `oxibonsai-kernels`
- "How does the CLI flag map?" → `src/main.rs`

## 4. Models & tokenizer
The CLI expects a Q1_0_g128 GGUF and the Qwen3 tokenizer JSON. Place both under `models/`:

```bash
mkdir -p models
curl -L -o models/Bonsai-8B.gguf \
  https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/Bonsai-8B.gguf
curl -L -o models/tokenizer.json \
  https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer.json
# Or, after the binary is built:
./target/release/oxibonsai tokenizer download --output models/tokenizer.json
```

The repo's `.gitignore` excludes `models/*` and `*.gguf|*.bin|*.safetensors`, so symlinking models from a shared cache (`~/models/`) is safe and recommended.

## 5. Building

The minimal build:

```bash
cargo build
```

The performance build for Apple Silicon:

```bash
cargo build --release --features "simd-neon metal native-tokenizer"
```

Feature reference (declared in workspace-root `Cargo.toml` and `crates/oxibonsai-kernels/Cargo.toml`):

| Feature             | Purpose                                                           |
| ------------------- | ----------------------------------------------------------------- |
| `default = [server]` | Build the `serve` HTTP server                                     |
| `simd-avx2`         | x86-64 AVX2+FMA kernels                                           |
| `simd-avx512`       | x86-64 AVX-512 kernels                                            |
| `simd-neon`         | AArch64 NEON kernels                                              |
| `metal`             | Apple GPU acceleration (auto-selects on macOS)                    |
| `native-cuda`       | NVIDIA GPU acceleration (CUDA toolkit required)                   |
| `cuda`              | Build kernels with CUDA but no native libs                        |
| `gpu`               | Generic GPU dispatch                                              |
| `native-tokenizer`  | Bundle Qwen3 tokenizer crate (lets you skip `--tokenizer`)        |
| `rag`               | Enable the RAG pipeline crate                                     |
| `eval`              | Enable the evaluation harness                                     |
| `wasm`              | Build for `wasm32-unknown-unknown` (server is unavailable)        |

Pick exactly one CPU SIMD tier and at most one GPU tier per build.

## 6. Running

The release binary is at `./target/release/oxibonsai`. Common invocations:

```bash
# Inspect a model (no kernels, fast):
./target/release/oxibonsai info --model models/Bonsai-8B.gguf

# Single-shot inference:
./target/release/oxibonsai run \
  --model models/Bonsai-8B.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Explain quantum computing in simple terms." \
  --max-tokens 256 --temperature 0.5

# Interactive chat (only `/reset` supported in v0.1.2; see Brief 03 for upcoming /system, /help):
./target/release/oxibonsai chat \
  --model models/Bonsai-8B.gguf \
  --tokenizer models/tokenizer.json

# OpenAI-compatible server:
./target/release/oxibonsai serve \
  --model models/Bonsai-8B.gguf \
  --tokenizer models/tokenizer.json \
  --host 127.0.0.1 --port 8080

# Synthetic micro-benchmark (no real weights):
./target/release/oxibonsai benchmark
```

The repo also ships `scripts/cli.sh metal` which builds + runs `info` + `run` + `validate` against the local model in one go.

## 7. Testing

```bash
cargo test                                                  # all crates, debug
cargo test --release                                        # release timing
cargo test --features "simd-neon metal native-tokenizer"    # match prod features
cargo test -p oxibonsai-runtime                             # one crate only
cargo test --test end_to_end_inference -- --nocapture       # one integration suite
```

Where to put new tests:
- Unit tests next to the code they test (`#[cfg(test)] mod tests { ... }`)
- Integration tests in `tests/`
- Benchmarks in `benches/` using Criterion

## 8. Benchmarking

```bash
cargo bench --features "simd-neon metal" -- gemv
cargo bench --features simd-neon -- dequant
```

For end-to-end token throughput, prefer a real model run with `--temperature 0.0` (greedy GPU path on macOS uses 4-byte argmax downloads instead of full logits — much faster):

```bash
./target/release/oxibonsai run \
  --model models/Bonsai-8B.gguf --tokenizer models/tokenizer.json \
  --prompt "Hello! Briefly introduce yourself." \
  --max-tokens 128 --temperature 0.0
# Look for the trailing "X tokens in Y.YYs (Z.Z tok/s)" line
```

Reference numbers measured during evaluation on a recent Apple Silicon Mac (release build, Metal+NEON+native-tokenizer, Bonsai-8B Q1_0_g128):

- Greedy single-shot, ~10-token prompt: **15–17 tok/s**
- Greedy single-shot, ~50-token prompt: **~19.7 tok/s**
- Synthetic `oxibonsai benchmark` (no model load): **~470 tok/s** (not representative of real workload)

These are baseline numbers; the README's targets are `>=18 tok/s NEON` and the project roadmap aims higher with future kernel tuning.

## 9. Profiling

GPU profiling on macOS is built into the Metal feature:

```bash
OXIBONSAI_PROFILE_GPU=1 ./target/release/oxibonsai run \
  --model models/Bonsai-8B.gguf --tokenizer models/tokenizer.json \
  --prompt "..." --max-tokens 64 --temperature 0.0
```

Tracing logs (the most useful diagnostic):

```bash
RUST_LOG=info ./target/release/oxibonsai run ...
RUST_LOG=oxibonsai_runtime=debug,oxibonsai_kernels=info ./target/release/oxibonsai run ...
RUST_LOG=trace ./target/release/oxibonsai run ...        # very chatty
```

A TOML `[observability]` section is honored:

```toml
[observability]
log_level = "debug"
json_logs = true
```

## 10. Debugging common failures

- **"`#![feature]` may not be used on the stable release channel"** → You're on stable. Either install nightly (`rustup toolchain install nightly && rustup override set nightly`) or apply Brief 02.
- **`vocab_size mismatch: GGUF metadata says 151936 but token_embd tensor has 151669 rows`** → Benign warning; the loader uses the tensor dimension. Bonsai-8B's GGUF advertises a padded vocab.
- **"no tokenizer specified — token IDs printed instead of text"** → Pass `--tokenizer models/tokenizer.json` or rebuild with `native-tokenizer`.
- **"metal batch prefill failed, falling back to sequential error=no blocks"** → Appears during the `oxibonsai benchmark` synthetic run because no model is loaded; harmless. If it appears during real `run`/`chat`, file an issue with the full log.
- **Repetition loops or persona drift** → The CLI hardcodes `repetition_penalty = 1.1` and does not apply the chat template (see Briefs 01 + 03). For now: drive `oxibonsai serve` over HTTP, or use `EngineBuilder` directly.
- **Throughput far below README targets** → Likely missing `--features "simd-neon metal"` at build time. Confirm with `RUST_LOG=info ...` and look for `kernel="Q1_0_g128 GPU (accelerated)"`.

## 11. Contribution workflow

### Branching
- `master` is the default branch and always tagged at the latest release.
- Feature branches: `feat/<short-slug>` (e.g. `feat/qwen3-chat-template`).
- Fix branches: `fix/<short-slug>` (e.g. `fix/honor-sampling-toml`).

### Commits
- Conventional Commits style: `feat(scope): summary`, `fix(scope): summary`, `chore(scope): summary`, etc.
- Reference upstream issues in the body when applicable.
- End every commit message with the agent attribution trailer:

  ```
  Co-Authored-By: Oz <oz-agent@warp.dev>
  ```

### PRs
- Open against `cool-japan/oxibonsai` `master`.
- Include before/after measurements when changing kernels or sampling.
- Keep diffs scoped — see `briefs/` for canonical small-PR shapes.
- Do **not** rebase published branches force-push without coordination.

### Pre-PR checklist
1. `cargo fmt --all`
2. `cargo clippy --all-targets --features "simd-neon metal native-tokenizer" -- -D warnings`
3. `cargo test  --features "simd-neon metal native-tokenizer"`
4. `cargo build --release --features "simd-neon metal native-tokenizer"` succeeds
5. Real-model smoke test (see §6) still produces reasonable output

## 12. Releases
Release management is owned by COOLJAPAN OU; do not bump versions in PRs unless the maintainers ask. The release script lives at `scripts/publish.sh`. The `CHANGELOG.md` is updated as part of release commits.

## 13. Working with the briefs
Implementation briefs live in `briefs/`:
- `01-honor-sampling-toml-and-rep-penalty-flag.md`
- `02-stable-rust-support-for-oxibonsai-kernels.md`
- `03-qwen3-chat-template-and-system-prompt.md`

Each brief is self-contained: it includes context, the precise problem, acceptance criteria, file-level implementation sketches, verification commands, and a PR description boilerplate. They are sized to be picked up by an autonomous coding agent or a contributor in a single afternoon.

When you finish a brief:
1. Open the PR from the brief's named branch.
2. Mark the brief as `Status: In review` (edit the table).
3. After merge, rename or move the brief to `briefs/done/`.

## 14. Adding a new brief
Follow the structure of the existing three. The minimum required sections are:
1. Front-matter table (Status, Priority, Risk, Estimated PR, Branch name, Base commit)
2. Context
3. Problem
4. Acceptance criteria (numbered, testable)
5. Affected files
6. Implementation sketch
7. Out of scope
8. Verification commands
9. PR description boilerplate (with the `Co-Authored-By: Oz <oz-agent@warp.dev>` trailer)

Keep briefs small enough to land in one PR. If a brief grows past ~300 lines of implementation, split it.

## 15. Quick-reference cheat sheet

```bash
# Initial setup
rustup override set nightly
mkdir -p models
ln -sfn ~/models/Bonsai-8B.gguf models/Bonsai-8B.gguf
ln -sfn ~/models/tokenizer.json models/tokenizer.json

# Build + run loop
cargo build --release --features "simd-neon metal native-tokenizer"
./target/release/oxibonsai run \
  --model models/Bonsai-8B.gguf --tokenizer models/tokenizer.json \
  --prompt "Hi." --max-tokens 16 --temperature 0.0

# Test + lint loop
cargo test  --features "simd-neon metal native-tokenizer"
cargo clippy --all-targets --features "simd-neon metal native-tokenizer" -- -D warnings
cargo fmt --all

# Smoke
./scripts/cli.sh metal
```
