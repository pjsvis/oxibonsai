---
date: 2026-04-25
tags: [debrief, cli, sampling, config]
---

## Debrief: Brief 01 — Honor [sampling] TOML Config and Add --repetition-penalty Flag

## Accomplishments

- **[CLI TOML Parsing Fixed]:** The `OxiBonsaiConfig::load_or_default()` was parsing the `[sampling]` TOML section but the Run and Chat handlers were silently discarding it. This is now fixed with proper precedence chain implementation.
- **[Precedence Chain Implemented]:** Changed `temperature`, `top_k`, `top_p`, `max_tokens` from `T with default_value` to `Option<T>` so CLI flags now properly override TOML values (previously broken).
- **[--repetition-penalty Flag Added]:** New CLI flag for both `Run` and `Chat` commands with `value_parser` validation ensuring `>= 1.0`.
- **[10 Precedence Tests Added]:** Comprehensive test coverage in `tests/sampling_config_overrides.rs` covering defaults, TOML parsing, CLI overrides, and edge cases.
- **[README Updated]:** Added precedence rule documentation under `[sampling]` example.
- **[Justfile Refactored]:** Moved implementation to `scripts/*.sh`, Justfile is now a clean facade.
- **[Modular Scripts Created]:** `scripts/build.sh`, `scripts/run.sh`, `scripts/chat.sh`, `scripts/test.sh`, `scripts/bench.sh`, `scripts/info.sh`, `scripts/validate.sh`, `scripts/smoke.sh` for clean separation of concerns.

## Problems

- **[Justfile Parser Issues]:** The Justfile had multiple issues with `just` parser:
  - Backtick command substitutions (`\`...\``) with regex patterns containing `.` caused parse errors
  - Dots in variable names (`MODEL ?= ...`) broke the parser
  - Dots in recipe names (`download-1.7b:`) caused errors
  - **Resolution:** Rewrote Justfile with `:=` assignment, replaced backticks with direct values, used underscores instead of dashes/dots in all names.

- **[tempfile Import Error]:** The new test file couldn't import `tempfile` crate because it wasn't in `dev-dependencies`.
  - **Resolution:** Added `tempfile.workspace = true` to workspace `Cargo.toml`.

## Lessons Learned

- **[Justfile Naming Rules]:** Never use dots (`.`) or hyphens (`-`) in Justfile variable or recipe names. Use underscores (`_`) instead. This is a `just` parser limitation, not obvious from documentation.
- **[Just Architecture]:** The Justfile should be a pure facade (list of verbs) that delegates to `scripts/*.sh` for implementation. This separation makes testing easier and errors more debuggable.
- **[Precedence Chain Pattern]:** When implementing CLI + TOML + defaults config systems, always use `Option<T>` for CLI values and apply `cli.unwrap_or(toml)` pattern. Using `default_value` on clap fields always shadows TOML.
- **[Validation Timing]:** Validation should happen at the earliest possible stage (clap parse time) so invalid input fails before model loading.

## Post-Debrief Checklist

- [x] **Archive Brief:** Move `briefs/01-honor-sampling-toml-and-rep-penalty-flag.md` to `briefs/archive/`
- [ ] **Update Changelog:** Add summary to `CHANGELOG.md` under `[Unreleased]`
- [ ] **Update Current Task:** Update `_CURRENT_TASK.md` (if exists)
- [x] **td Handoff:** Hand off to review via `td handoff`
- [x] **Create PR:** Push branch and create PR
