# Brief 03 — Apply Qwen3 chat template in `chat` REPL and add `--system` flag

| Field         | Value                                                              |
| ------------- | ------------------------------------------------------------------ |
| Status        | Ready for implementation                                           |
| Priority      | Medium-High (instruction following silently underperforms today)   |
| Risk          | Medium — changes prompt construction and conversation state        |
| Estimated PR  | ~150–300 lines + tests                                             |
| Branch name   | `feat/qwen3-chat-template-and-system-prompt`                       |
| Base commit   | `b4dc18d` (tag `v0.1.2`); preferably rebased on Brief 01 if landed |

## Context
Bonsai-8B is built on the Qwen3-8B architecture and is post-trained against the Qwen3 chat template (see the `chat_template_jinja` shipped in `prism-ml/Bonsai-8B-mlx-1bit`). The template wraps each turn in `<|im_start|>{role}\n...<|im_end|>\n` and prefixes assistant turns with a `<think>...</think>` reasoning block. Without this framing, instruction-following degrades sharply: in observed runs the model wandered, looped on partial sentences, and ignored explicit "answer in one sentence" directives.

OxiBonsai's `chat` REPL currently does **none** of this. In `src/main.rs` (~line 569):

```rust
let prompt_tokens = if let Some(tok) = &tok {
    tok.encode(input)?
} else {
    vec![151644]
};
```

Each turn is just the raw user line, encoded directly. There is no system prompt, no `<|im_start|>` framing, no conversation history concatenation, and no `<think>` block. The CLI also exposes no `--system` flag and the REPL only recognizes `/reset` (no `/system`, no `/help`).

The `oxibonsai-tokenizer` crate already exposes a `ChatTemplate` type per the workspace README, so the building blocks exist; they are simply not wired up to the CLI.

## Problem
1. `oxibonsai chat` does not apply the Qwen3 chat template. The model receives unframed user text, which damages instruction-following on a base it was specifically tuned against.
2. There is no way to set a system prompt from the CLI for either `chat` or `run`.
3. The REPL's only command is `/reset`. Users cannot inspect, set, or clear the system prompt mid-session.

## Acceptance criteria
1. `oxibonsai chat --model ... --tokenizer ...` (no `--system`) applies the Qwen3 chat template using a sane built-in default system prompt (e.g. `"You are a helpful assistant."`). Each user turn is rendered as:

   ```
   <|im_start|>system
   {system}<|im_end|>
   <|im_start|>user
   {turn1}<|im_end|>
   <|im_start|>assistant
   {turn1_reply}<|im_end|>
   <|im_start|>user
   {turn2}<|im_end|>
   <|im_start|>assistant
   ```

   …with the turn boundary correctly handed to the engine via `engine.reset()` + full re-encode, OR via incremental KV reuse (whichever is simplest given the existing `InferenceEngine` API; consult `crates/oxibonsai-runtime/src/engine.rs`).
2. `chat` and `run` both gain `--system <STRING>` flags that override the default system prompt.
3. New REPL commands inside `chat`:
   - `/system [prompt]` — show the current system prompt (no arg) or replace it (with arg). Replacing implicitly resets context.
   - `/help` — list available REPL commands.
   - Existing `/reset` is preserved.
4. `run` always renders the same template framing for its single `--prompt` argument so single-shot output matches chat behavior.
5. The CLI does NOT regress previously working flags (`--temperature`, `--max-tokens`, etc.).
6. New tests in `tests/chat_template_tests.rs`:
   - Encoding a `(system, user)` pair produces the expected `<|im_start|>...<|im_end|>` token sequence.
   - Multi-turn rendering preserves `assistant` blocks from prior turns.
   - `/system new-prompt` replaces and resets context.
7. README gains a "Chat templates and system prompts" section showing the new flags and REPL commands.

## Affected files
- `src/main.rs` — `Run` and `Chat` clap variants, REPL loop, prompt construction.
- `crates/oxibonsai-tokenizer/src/lib.rs` (or wherever `ChatTemplate` lives) — confirm public surface; add helpers if needed (e.g. `render_qwen3(system, history) -> String`).
- `tests/chat_template_tests.rs` — new file.
- `README.md` — usage section.

## Implementation sketch
1. Add `system: Option<String>` to both clap variants, with help text noting that omission uses a sensible default.
2. Centralize template rendering in a small helper:

   ```rust
   fn render_qwen3_prompt(system: &str, history: &[Turn]) -> String { ... }
   ```

   where `Turn` is `{ role: Role, content: String }` and `Role` is `User | Assistant`. Append a trailing `<|im_start|>assistant\n` to signal a generation start.
3. In the `chat` REPL, maintain `Vec<Turn>` plus a `system: String`. After each generation, append the assistant turn with the decoded text (you have the streaming `tok.decode(...)` stream already; concatenate decoded chunks into a `String`).
4. On each new user input, re-render the full prompt and re-encode. Call `engine.reset()` before re-encoding to wipe the KV cache. (Optimization for later: incremental KV reuse — out of scope for this PR.)
5. Implement `/system [prompt]` and `/help` slash commands by extending the existing `if input == "/reset"` ladder.
6. For `run`, do the same template render with a single user turn.
7. Suppress the existing "no tokenizer specified" warning's `vec![151644]` fallback when chat templating is active — it cannot work without a tokenizer; emit a clear error instead.

## Verification commands
```bash
cargo build --release --features "simd-neon metal native-tokenizer"

# Default system prompt:
./target/release/oxibonsai run \
  --model models/Bonsai-8B.gguf --tokenizer models/tokenizer.json \
  --prompt "What is 12 * 7? Answer with only the number." \
  --max-tokens 16 --temperature 0.0
# Expect output to start with "84"

# Custom system prompt forcing one-sentence behavior:
./target/release/oxibonsai run \
  --model models/Bonsai-8B.gguf --tokenizer models/tokenizer.json \
  --system "You are a terse expert. Reply in EXACTLY ONE short sentence." \
  --prompt "What is 12 * 7?" \
  --max-tokens 32 --temperature 0.0
# Expect a single, short sentence containing 84

# REPL system command:
./target/release/oxibonsai chat \
  --model models/Bonsai-8B.gguf --tokenizer models/tokenizer.json
> /system You only answer in haiku.
> Tell me about Rust.
# Verify response style changes
> /system
# Verify the current system prompt is printed back
> /help
# Verify command listing
```

Add a `cargo test --test chat_template_tests` step.

## Out of scope (defer to follow-up PRs)
- Incremental KV-cache reuse across turns (currently we'll re-prefill on each turn).
- `<think>...</think>` reasoning-block handling — mirror Qwen3's template literally for now; reasoning extraction is a later UX improvement.
- Tool-calling / function-calling support.
- Streaming the assistant's `<|im_end|>` token detection (just stop on EOS as today).
- Server-side template handling in `oxibonsai serve` — separate brief.

## PR description boilerplate
- **Title:** `feat(cli): apply Qwen3 chat template and add --system flag`
- **Sections:** Problem · Fix · Template rendering · REPL commands · Tests · Manual verification · Out of scope
- **Trailer:** `Co-Authored-By: Oz <oz-agent@warp.dev>`
