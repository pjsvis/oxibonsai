---
date: 2026-04-25
tags: [debrief, kernels, stable-rust, aarch64]
---

## Debrief: Brief 02 — Stable-Rust Support for `oxibonsai-kernels`

## Accomplishments

- **[Stable Rust on AArch64]:** The crate now builds on stable Rust on AArch64 (Apple Silicon, ARM Linux) without `#![feature]` errors. This was blocking `cargo install oxibonsai-cli` on default toolchains.
- **[Nightly Feature Added]:** A new `nightly` feature enables the unstable `stdarch_aarch64_prefetch` intrinsic for power users who want the slightly faster path.
- **[Inline Assembly Fallback]:** On stable toolchains, the prefetch code uses `core::arch::asm!` with `prfm pldl1keep, [...]` — stable since Rust 1.59.
- **[Abstraction Layer]:** Created `crate::prefetch::prefetch_read()` and `prefetch_write()` that route to the appropriate implementation based on toolchain and feature flag.
- **[Updated All Call Sites]:** `simd_neon.rs` and `packing.rs` now use the abstraction layer instead of direct `_prefetch` calls.

## Problems

- **[Multiple Call Sites]:** The original brief showed only `lib.rs`, but `simd_neon.rs` had 6 direct `_prefetch` calls and `packing.rs` had 2 more. Solution: create a centralized `prefetch.rs` module and route all calls through it.
- **[Mutable vs Immutable Pointer]:** `prefetch_write` expects `*mut T` but `prefetch_read` expected `*const T`. Had to adjust cast in `packing.rs` to use `as *mut i8` for write operations.
- **[sed Substitution Collision]:** Two identical `_prefetch` calls existed in `simd_neon.rs` (lines 355 and 733). Solved by using sed with unique context identifiers.

## Lessons Learned

- **[Search Before Implementing]:** Always `rg "_prefetch"` across the whole crate before assuming only the documented file has the issue. The actual usage was spread across 3 files.
- **[Feature Gating Strategy]:** Use `#[cfg(all(target_arch = "aarch64", feature = "nightly"))]` to gate both the crate attribute AND the code path. This keeps the stable path clean.
- **[mod.rs Visibility]:** The `prefetch` module is already public, so `crate::prefetch::prefetch_read()` works from any submodule without additional exports.

## Post-Debrief Checklist

- [x] **Archive Brief:** Brief 02 should be moved to `briefs/archive/` after PR approval
- [ ] **Update Changelog:** Add summary to `CHANGELOG.md` under `[Unreleased]`
- [ ] **Update Current Task:** Brief 03 next (chat template)
- [x] **td Handoff:** Hand off to review via `td handoff`
- [ ] **Create PR:** Push branch and create PR
