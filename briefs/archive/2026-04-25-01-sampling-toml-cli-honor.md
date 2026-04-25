# Brief 01 — Honor `[sampling]` TOML config and add `--repetition-penalty` CLI flag

| Field         | Value                                                              |
| ------------- | ------------------------------------------------------------------ |
| Status        | Ready for implementation                                           |
| Priority      | High (documented feature is silently broken)                       |
| Risk          | Low — touches only `src/main.rs` and adds tests                    |
| Estimated PR  | ~50–120 lines + tests                                              |
| Branch name   | `fix/honor-sampling-toml-and-add-rep-penalty-flag`                 |
| Base commit   | `b4dc18d` (tag `v0.1.2`)                                           |

## Context
OxiBonsai is a Pure-Rust 1-bit LLM inference engine. The CLI binary `oxibonsai` is built from the workspace-root package `oxibonsai-cli`, entry point `src/main.rs`. The CLI accepts a TOML config via `--config` and individual flags. The README documents a `[sampling]` section with `temperature`, `top_k`, `top_p`, `repetition_penalty`, and `max_tokens`. The runtime crate `oxibonsai-runtime` already implements all of these in `crates/oxibonsai-runtime/src/sampling.rs::SamplingParams` and `crates/oxibonsai-runtime/src/config.rs::SamplingConfig`.

## Problem
1. `src/main.rs` calls `OxiBonsaiConfig::load_or_default(...)` (~line 280) but only consumes the `[observability]` section. The `[sampling]` block is parsed and silently dropped.
2. Both `Commands::Run` (~line 318) and `Commands::Chat` (~line 493) build `SamplingParams` with a hardcoded literal `repetition_penalty: 1.1`, ignoring the TOML value.
3. There is no CLI flag for `repetition_penalty`, so users have no way to override it from the command line either.

Verbatim, both call sites currently look like:

```rust
let params = oxibonsai_runtime::sampling::SamplingParams {
    temperature,
    top_k,
    top_p,
    repetition_penalty: 1.1,
    ..oxibonsai_runtime::sampling::SamplingParams::default()
};
```

## Acceptance criteria
A reviewer must be able to verify all of the following:

1. With a config file containing `[sampling] repetition_penalty = 1.3`, running `oxibonsai run --config foo.toml ...` (no CLI override) uses `1.3`. Same for `chat`.
2. CLI flags `--temperature`, `--top-k`, `--top-p`, `--max-tokens` override the corresponding TOML value when both are present.
3. A new CLI flag `--repetition-penalty <FLOAT>` exists for both `Run` and `Chat`, validates `>= 1.0`, defaults to the TOML value if omitted (else to `SamplingConfig::default().repetition_penalty`), and overrides TOML when set.
4. Currently `--max-tokens`, `--top-k`, `--top-p`, `--temperature` are clap `default_value` fields, so the CLI value always shadows TOML. Convert these to `Option<T>` (no clap default) so absence vs. explicit value is distinguishable, and apply the precedence chain `defaults < TOML < CLI`.
5. Behavior is unchanged when neither config nor flag is supplied — defaults from `SamplingConfig::default()` apply (temperature 0.7, top_k 40, top_p 0.9, repetition_penalty 1.1, max_tokens 512).
6. Existing integration tests pass; new tests cover the precedence chain `defaults < TOML < CLI` for `repetition_penalty` specifically.
7. README's `[sampling]` example actually works as documented; add one sentence after the example explicitly stating the precedence rule.

## Affected files
- `src/main.rs` — clap `Run` and `Chat` variants, the two `SamplingParams { ... }` constructions, post-`OxiBonsaiConfig::load_or_default` glue.
- `tests/end_to_end_inference.rs` (or new file `tests/sampling_config_overrides.rs`) — precedence tests.
- `README.md` — one-line clarification of precedence under the `[sampling]` example.

Do **not** change the runtime crate's `SamplingParams` shape or defaults.

## Implementation sketch
1. Change the clap `Run`/`Chat` fields for `temperature`, `top_k`, `top_p`, `max_tokens` from `T` with `default_value` to `Option<T>` (no clap default). Add `repetition_penalty: Option<f32>` to both.
2. After `let config = OxiBonsaiConfig::load_or_default(...)`, build `SamplingParams` like:

```rust
let s = &config.sampling;
let params = oxibonsai_runtime::sampling::SamplingParams {
    temperature:        temperature.unwrap_or(s.temperature),
    top_k:              top_k.unwrap_or(s.top_k),
    top_p:              top_p.unwrap_or(s.top_p),
    repetition_penalty: repetition_penalty.unwrap_or(s.repetition_penalty),
    ..oxibonsai_runtime::sampling::SamplingParams::default()
};
let max_tokens = max_tokens.unwrap_or(s.max_tokens);
```

3. Validate `repetition_penalty >= 1.0` at clap parse time via `value_parser` so the error surfaces before model load.
4. Update `--help` text to mention TOML fallback for the affected flags.
5. Reuse the existing `OxiBonsaiConfig::validate()` for any other invariant checks already implemented in `oxibonsai-runtime`.

## Out of scope (do not include in this PR)
- Applying the Qwen3 chat template in the `chat` REPL — see Brief 03.
- Stable-Rust support for `oxibonsai-kernels` — see Brief 02.
- New sampling features (mirostat, presence/frequency penalty, etc.).
- Renaming or restructuring the TOML schema.

## Verification commands
From `~/dev/github/cool-japan/oxibonsai` with `nightly` toolchain pinned via `rustup override`:

```bash
cargo build --release --features "simd-neon metal native-tokenizer"
cargo test  --features "simd-neon metal native-tokenizer"

# TOML alone:
printf '[sampling]\nrepetition_penalty = 1.4\n' > /tmp/oxi.toml
RUST_LOG=oxibonsai_runtime=debug \
  ./target/release/oxibonsai run --config /tmp/oxi.toml \
    --model models/Bonsai-8B.gguf --tokenizer models/tokenizer.json \
    --prompt "Hello." --max-tokens 8 --temperature 0.0
# Confirm tracing log line shows repetition_penalty=1.4

# CLI override beats TOML:
RUST_LOG=oxibonsai_runtime=debug \
  ./target/release/oxibonsai run --config /tmp/oxi.toml --repetition-penalty 1.2 \
    --model models/Bonsai-8B.gguf --tokenizer models/tokenizer.json \
    --prompt "Hello." --max-tokens 8 --temperature 0.0
# Confirm tracing log line shows repetition_penalty=1.2
```

## PR description boilerplate
- **Title:** `fix(cli): honor [sampling] TOML config and add --repetition-penalty flag`
- **Sections:** Problem · Fix · Precedence chain · Tests · Manual verification · Closes #<issue> (if applicable)
- **Trailer:** `Co-Authored-By: Oz <oz-agent@warp.dev>`
