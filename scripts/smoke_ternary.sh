#!/usr/bin/env bash
# OxiBonsai — Ternary Bonsai smoke test.
#
# Usage:
#   ./scripts/smoke_ternary.sh [OPTIONS]
#
#   --model <path>    Path to a TernaryBonsai GGUF (optional)
#   --skip-tests      Skip cargo test phase (default: run tests)
#   --release         Build in release mode (default: debug)
#   -h/--help         Print usage
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Color support ─────────────────────────────────────────────────────────────

if [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
    C_GREEN="$(tput setaf 2)"
    C_RED="$(tput setaf 1)"
    C_YELLOW="$(tput setaf 3)"
    C_RESET="$(tput sgr0)"
else
    C_GREEN=""
    C_RED=""
    C_YELLOW=""
    C_RESET=""
fi

pass()  { echo "${C_GREEN}PASS${C_RESET}  $*"; }
fail()  { echo "${C_RED}FAIL${C_RESET}  $*"; }
warn()  { echo "${C_YELLOW}WARN${C_RESET}  $*"; }

# ── Parse arguments ──────────────────────────────────────────────────────────

MODEL=""
SKIP_TESTS=0
RELEASE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --model requires a path argument" >&2
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=1
            shift
            ;;
        --release)
            RELEASE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "  --model <path>    Path to a TernaryBonsai GGUF (optional)"
            echo "  --skip-tests      Skip cargo test phase (default: run tests)"
            echo "  --release         Build in release mode (default: debug)"
            echo "  -h/--help         Print usage"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Usage: $0 [--model <path>] [--skip-tests] [--release] [-h]" >&2
            exit 1
            ;;
    esac
done

# ── Configuration ─────────────────────────────────────────────────────────────

if [[ "$RELEASE" -eq 1 ]]; then
    BUILD_MODE="release"
    BUILD_FLAG="--release"
else
    BUILD_MODE="debug"
    BUILD_FLAG=""
fi

DEFAULT_MODEL="models/Ternary-Bonsai-1.7B.gguf"
ALT_MODEL="models/Ternary-Bonsai-8B.gguf"

# Resolve model path: use --model if given, else the first default that exists.
if [[ -z "$MODEL" ]]; then
    if [[ -f "$DEFAULT_MODEL" ]]; then
        MODEL="$DEFAULT_MODEL"
    elif [[ -f "$ALT_MODEL" ]]; then
        MODEL="$ALT_MODEL"
    fi
fi

FAIL_COUNT=0

# ── Helper: nextest with cargo test fallback ──────────────────────────────────

# nextest_or_test <package> <filter>
#   Runs: cargo nextest run -p <package> --all-features -E 'test(<filter>)'
#   Falls back to: cargo test -p <package> --all-features -- <filter>
nextest_or_test() {
    local pkg="$1"
    local filter="$2"
    if cargo nextest --version >/dev/null 2>&1; then
        cargo nextest run -p "$pkg" --all-features -E "test($filter)"
    else
        cargo test -p "$pkg" --all-features -- "$filter"
    fi
}

# ── Banner ────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════════"
echo "  OxiBonsai Ternary smoke test  (mode: $BUILD_MODE)"
if [[ -n "$MODEL" ]]; then
    echo "  model: $MODEL"
else
    echo "  model: (none — skipping Phase 4)"
fi
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Phase 1 — Ternary unit/integration tests ──────────────────────────────────

if [[ "$SKIP_TESTS" -eq 0 ]]; then
    echo "── Phase 1: Ternary unit/integration tests ──────────────────"
    echo ""

    # 1a: oxibonsai-core ternary types
    echo "  [1/3] oxibonsai-core  (ternary types)..."
    if nextest_or_test "oxibonsai-core" "ternary" 2>&1; then
        pass "oxibonsai-core ternary"
    else
        fail "oxibonsai-core ternary"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""

    # 1b: oxibonsai-kernels ternary scalar + SIMD
    echo "  [2/3] oxibonsai-kernels  (scalar + SIMD ternary)..."
    if nextest_or_test "oxibonsai-kernels" "ternary" 2>&1; then
        pass "oxibonsai-kernels ternary"
    else
        fail "oxibonsai-kernels ternary"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""

    # 1c: oxibonsai-model ternary integration (ternary_integration.rs)
    echo "  [3/3] oxibonsai-model  (ternary integration)..."
    if nextest_or_test "oxibonsai-model" "ternary" 2>&1; then
        pass "oxibonsai-model ternary"
    else
        fail "oxibonsai-model ternary"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
else
    echo "── Phase 1: Ternary tests  (${C_YELLOW}SKIPPED${C_RESET} via --skip-tests) ───────────"
    echo ""
fi

# ── Phase 2 — Build check ─────────────────────────────────────────────────────

echo "── Phase 2: Build check  (mode: $BUILD_MODE) ────────────────"
echo ""

# shellcheck disable=SC2086
if cargo build $BUILD_FLAG 2>&1; then
    pass "Build OK"
else
    fail "Build FAILED"
    echo ""
    echo "ERROR: build failed — aborting remaining phases." >&2
    exit 1
fi
echo ""

# ── Phase 3 — CLI sanity ──────────────────────────────────────────────────────

echo "── Phase 3: CLI sanity ──────────────────────────────────────"
echo ""

BIN="./target/${BUILD_MODE}/oxibonsai"

if [[ ! -x "$BIN" ]]; then
    fail "Binary not found or not executable: $BIN"
    echo ""
    echo "ERROR: CLI binary missing after successful build." >&2
    exit 1
fi

# --help must succeed
echo "  Checking --help..."
if "$BIN" --help >/dev/null 2>&1; then
    pass "--help"
else
    fail "--help (exit code non-zero)"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# --version: print output but do not fail if subcommand is unimplemented
echo "  Checking --version..."
VERSION_OUT="$("$BIN" --version 2>&1 || true)"
if [[ -n "$VERSION_OUT" ]]; then
    echo "  version: $VERSION_OUT"
    pass "--version"
else
    warn "--version produced no output (may not be implemented yet)"
fi
echo ""

# ── Phase 4 — Model-file checks ───────────────────────────────────────────────

if [[ -n "$MODEL" ]]; then
    echo "── Phase 4: Model-file checks  ($MODEL) ──────────────"
    echo ""

    if [[ ! -f "$MODEL" ]]; then
        warn "Model file not found: $MODEL — skipping Phase 4"
        echo ""
    else
        # info — must print variant containing "Ternary" (case-insensitive)
        echo "  Running: $BIN info --model $MODEL"
        INFO_OUT="$("$BIN" info --model "$MODEL" 2>&1)"
        echo "$INFO_OUT"
        echo ""

        if echo "$INFO_OUT" | grep -qi "ternary"; then
            pass "info: variant name contains 'Ternary'"
        else
            warn "info: variant name does not contain 'Ternary' (model may be 1-bit)"
        fi

        # validate — must succeed
        echo "  Running: $BIN validate --model $MODEL"
        if "$BIN" validate --model "$MODEL" 2>&1; then
            pass "validate"
        else
            fail "validate"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        echo ""
    fi
else
    echo "── Phase 4: Model-file checks  (${C_YELLOW}SKIPPED${C_RESET} — no model provided) ──"
    echo ""
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════════"
if [[ "$FAIL_COUNT" -eq 0 ]]; then
    echo "  ${C_GREEN}All phases passed.${C_RESET}"
    echo "═══════════════════════════════════════════════════════════════"
    exit 0
else
    echo "  ${C_RED}${FAIL_COUNT} check(s) failed.${C_RESET}"
    echo "═══════════════════════════════════════════════════════════════"
    exit 1
fi
