#!/usr/bin/env bash
# OxiBonsai — CLI smoke test.
#
# Usage:
#   ./scripts/cli.sh             # build & run with default features (CPU SIMD)
#   ./scripts/cli.sh metal       # build & run with Metal GPU acceleration (macOS)
#   ./scripts/cli.sh cuda        # build & run with native CUDA backend (Linux/Windows)
#   ./scripts/cli.sh cuda-scirs  # build & run with scirs2-core CUDA backend
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Configuration ────────────────────────────────────────────────────────────

MODEL="models/Bonsai-8B.gguf"
TOKENIZER="models/tokenizer.json"
PROMPT="Explain 2+2 = ??"
MAX_TOKENS=100
TEMPERATURE=0.7
TOP_P=0.9
SEED=42

# ── Parse arguments ──────────────────────────────────────────────────────────

FEATURES=""
FEATURE_LABEL="default"

for arg in "$@"; do
    case "$arg" in
        metal)
            FEATURES="--features metal"
            FEATURE_LABEL="metal"
            ;;
        cuda)
            FEATURES="--features native-cuda,simd-avx2"
            FEATURE_LABEL="cuda"
            ;;
        cuda-scirs)
            FEATURES="--features cuda"
            FEATURE_LABEL="cuda-scirs"
            ;;
        --help|-h)
            echo "Usage: $0 [metal|cuda|cuda-scirs]"
            echo ""
            echo "  (no args)    Build & run with default features (CPU SIMD)"
            echo "  metal        Build & run with Metal GPU acceleration (macOS)"
            echo "  cuda         Build & run with native CUDA backend (Linux/Windows, cudarc 0.19)"
            echo "  cuda-scirs   Build & run with scirs2-core CUDA backend"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $arg" >&2
            echo "Usage: $0 [metal|cuda|cuda-scirs]" >&2
            exit 1
            ;;
    esac
done

# ── Pre-flight checks ───────────────────────────────────────────────────────

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model file not found: $MODEL" >&2
    echo "Download with:" >&2
    echo "  mkdir -p models && curl -L -o $MODEL \\" >&2
    echo "    https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/Bonsai-8B.gguf" >&2
    exit 1
fi

TOKENIZER_FLAG=""
if [[ -f "$TOKENIZER" ]]; then
    TOKENIZER_FLAG="--tokenizer $TOKENIZER"
else
    echo "WARN: tokenizer not found ($TOKENIZER), running without tokenizer"
fi

# ── Build ────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════════"
echo "  OxiBonsai CLI smoke test  (features: $FEATURE_LABEL)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Building release binary..."
# shellcheck disable=SC2086
cargo build --release $FEATURES
echo "Build OK"
echo ""

# ── Run inference ────────────────────────────────────────────────────────────

BIN="./target/release/oxibonsai"

echo "── run: basic inference ─────────────────────────────────────"
echo "  prompt:      \"$PROMPT\""
echo "  max_tokens:  $MAX_TOKENS"
echo "  temperature: $TEMPERATURE"
echo "  top_p:       $TOP_P"
echo "  seed:        $SEED"
echo ""

# shellcheck disable=SC2086
RUST_LOG=info "$BIN" run \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --seed "$SEED" \
    $TOKENIZER_FLAG

echo ""
echo "── info ─────────────────────────────────────────────────────"
"$BIN" info --model "$MODEL"

echo ""
echo "── validate ─────────────────────────────────────────────────"
"$BIN" validate --model "$MODEL"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All CLI checks passed  (features: $FEATURE_LABEL)"
echo "═══════════════════════════════════════════════════════════════"
