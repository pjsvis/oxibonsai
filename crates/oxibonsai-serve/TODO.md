# oxibonsai-serve TODO

> Standalone HTTP server: argument parsing, banner, server entry point
> 4 files, ~510 lines, 29 tests (all passing)
> Version 0.1.1 — last reviewed 2026-04-18

## Status: ✅ All Features Complete (Alpha)

Minimal standalone server binary with pure Rust argument parsing (zero clap dependency) and comprehensive flag handling. Delegates engine and HTTP stack to `oxibonsai-runtime`.

## Done

- [x] `ServerArgs` — host, port, model path, tokenizer, max_tokens, temperature, seed, log_level
- [x] Pure `std::env` argument parser (no clap/structopt dependency)
- [x] `--help` and `--version` flags
- [x] Comprehensive error messages for invalid arguments
- [x] Version banner display
- [x] Binary entry point (`main.rs`)
- [x] Library interface (`lib.rs`) with `args` and `banner` re-exports
- [x] Full test coverage for all flags and edge cases (`args_tests.rs`)
