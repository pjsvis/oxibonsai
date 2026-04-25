# Brief 02 — Stable-Rust support for `oxibonsai-kernels`

| Field         | Value                                                              |
| ------------- | ------------------------------------------------------------------ |
| Status        | Ready for implementation                                           |
| Priority      | High (blocks `cargo install` on stable toolchains)                 |
| Risk          | Medium — touches kernel hot path; needs a benchmark sanity check    |
| Estimated PR  | ~80–200 lines + benchmark numbers                                  |
| Branch name   | `feat/stable-rust-support-for-kernels`                             |
| Base commit   | `b4dc18d` (tag `v0.1.2`)                                           |

## Context
`crates/oxibonsai-kernels/src/lib.rs:1` currently begins with:

```rust
#![cfg_attr(target_arch = "aarch64", feature(stdarch_aarch64_prefetch))]
```

The `stdarch_aarch64_prefetch` feature is unstable, so on AArch64 (Apple Silicon, ARM Linux) any user attempting `cargo install oxibonsai-cli` on a default-stable toolchain hits:

```
error[E0554]: `#![feature]` may not be used on the stable release channel
```

The only workaround today is to install nightly and use `cargo +nightly install`. This is a steep onboarding cost for what is otherwise a published, indexed crate. The unstable intrinsic (`__prefetch` / `vld1q_u8` prefetch hints) is used to improve memory latency in NEON GEMV / dequant kernels; the perf gain is real but not load-bearing for correctness.

## Problem
The `oxibonsai-kernels` crate, and by transitive closure the entire workspace's published binary, requires nightly Rust on AArch64 because of a single crate-attribute pulling in an unstable intrinsic.

## Acceptance criteria
1. `cargo install oxibonsai-cli` on a stock stable Rust toolchain (>= MSRV declared in workspace `Cargo.toml`, currently `1.86`) succeeds on AArch64 macOS and AArch64 Linux without `#![feature(...)]` errors.
2. `cargo install oxibonsai-cli --features simd-neon` on stable also succeeds.
3. A new opt-in cargo feature `nightly` is added to `oxibonsai-kernels` (and surfaced from `oxibonsai-cli`) which, when enabled on a nightly toolchain, restores the current intrinsic-based prefetch path.
4. On stable (i.e. `nightly` feature off), all prefetch call sites use a stable fallback that compiles and runs correctly. Acceptable fallbacks, in order of preference:
   - inline `core::arch::asm!` emitting `prfm pldl1keep, [...]` on AArch64 (stable since 1.59)
   - a no-op (returns `()`)
5. A microbenchmark already in `crates/oxibonsai-kernels/benches/` (or a new one) is run on AArch64 with both `--features ""` (stable fallback) and `--features nightly` (intrinsic) and the delta is reported in the PR description. A regression of up to ~10% on the stable path is acceptable.
6. The README's installation section is updated to reflect that nightly is no longer required (and to describe the optional `nightly` feature for power users).
7. CI matrix (in `.github/workflows/`) gains a stable-toolchain build job for AArch64 macOS that runs `cargo build --features simd-neon` and `cargo test`.

## Affected files
- `crates/oxibonsai-kernels/src/lib.rs` — drop unconditional `#![feature(...)]`; introduce a `prefetch_l1` (or similar) abstraction.
- `crates/oxibonsai-kernels/Cargo.toml` — add `nightly` feature.
- `Cargo.toml` (workspace root, package `oxibonsai-cli`) — re-export `nightly` so callers can pass it through.
- `crates/oxibonsai-kernels/src/<wherever prefetch is invoked>.rs` — update call sites to use the new abstraction.
- `.github/workflows/*.yml` — add a stable-toolchain job.
- `README.md` — update install instructions.

## Implementation sketch
1. Replace the top-of-file `cfg_attr(...)` with a feature-gated equivalent:

   ```rust
   #![cfg_attr(
       all(target_arch = "aarch64", feature = "nightly"),
       feature(stdarch_aarch64_prefetch)
   )]
   ```

2. Introduce a single inlinable abstraction in a new module `prefetch.rs`:

   ```rust
   #[inline(always)]
   pub(crate) fn prefetch_l1<T>(p: *const T) {
       #[cfg(all(target_arch = "aarch64", feature = "nightly"))]
       unsafe {
           use core::arch::aarch64::_prefetch;
           use core::arch::aarch64::{_PREFETCH_LOCALITY3, _PREFETCH_READ};
           _prefetch(p as *const i8, _PREFETCH_READ, _PREFETCH_LOCALITY3);
       }
       #[cfg(all(target_arch = "aarch64", not(feature = "nightly")))]
       unsafe {
           core::arch::asm!(
               "prfm pldl1keep, [{x}]",
               x = in(reg) p,
               options(nostack, preserves_flags, readonly)
           );
       }
       #[cfg(not(target_arch = "aarch64"))]
       { let _ = p; }
   }
   ```

3. Audit `crates/oxibonsai-kernels/src/` for direct usage of `vld1q_u8`-style prefetch intrinsics or anything pulled in by `stdarch_aarch64_prefetch`; route them through `prefetch_l1`.
4. Add a `nightly = []` entry in `crates/oxibonsai-kernels/Cargo.toml`'s `[features]`.
5. In `Cargo.toml` (root), add a re-export feature: `nightly = ["oxibonsai-kernels/nightly"]`.
6. In CI, add an AArch64 macOS job pinning `dtolnay/rust-toolchain@stable`.

## Verification commands
On a stable toolchain (after running `rustup default stable`):

```bash
cargo build  --features simd-neon
cargo build  --release --features "simd-neon metal native-tokenizer"
cargo test   --features simd-neon
./target/release/oxibonsai info --model models/Bonsai-8B.gguf
./target/release/oxibonsai run --model models/Bonsai-8B.gguf \
  --tokenizer models/tokenizer.json --prompt "Hi." --max-tokens 16 --temperature 0.0
```

On a nightly toolchain (after `rustup override set nightly`):

```bash
cargo build --release --features "simd-neon metal native-tokenizer nightly"
# Should be functionally identical, slightly faster on prefetch-heavy benches
cargo bench --features "simd-neon nightly" -- gemv
```

Compare `cargo bench` numbers with and without the `nightly` feature on the same machine; include the delta in the PR description.

## Out of scope
- Changing how kernels dispatch (NEON / AVX2 / AVX-512 / GPU) at runtime.
- Adding new SIMD intrinsics for other operations.
- MSRV bumps beyond what is strictly required.

## PR description boilerplate
- **Title:** `feat(kernels): allow build on stable Rust by gating prefetch intrinsic behind nightly feature`
- **Sections:** Problem · Fix · Stable fallback details · Benchmark delta · CI changes · README updates
- **Trailer:** `Co-Authored-By: Oz <oz-agent@warp.dev>`
