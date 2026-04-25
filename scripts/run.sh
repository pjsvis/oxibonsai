#!/usr/bin/env bash
# OxiBonsai — Run inference.
#
# Usage:
#   ./scripts/run.sh [metal|cuda] MODEL=<path> PROMPT=<text> [options]
#
# Options:
#   MODEL=<path>        GGUF model path (default: models/Ternary-Bonsai-1.7B.gguf)
#   PROMPT=<text>       Prompt text
#   MAX_TOKENS=<n>      Max tokens (default: 100)
#   TEMPERATURE=<f>    Temperature (default: 0.7)
#   TOP_P=<f>           Top-p (default: 0.9)
#   TOP_K=<n>          Top-k (default: 40)
#   SEED=<n>           Random seed (default: 42)
#   REP_PENALTY=<f>    Repetition penalty (default: 1.1)
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Backend selection ────────────────────────────────────────────────────────
BACKEND="${1:-cpu}"
shift 2>/dev/null || true

case "$BACKEND" in
    metal)      BIN="./target/release/oxibonsai" ;;
    cuda)       BIN="./target/release/oxibonsai" ;;
    *)          BIN="./target/release/oxibonsai" ;;
esac

# ── Defaults ────────────────────────────────────────────────────────────────
MODEL="${MODEL:-models/Ternary-Bonsai-1.7B.gguf}"
PROMPT="${PROMPT:-Explain quantum computing in simple terms}"
MAX_TOKENS="${MAX_TOKENS:-100}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-40}"
SEED="${SEED:-42}"
REP_PENALTY="${REP_PENALTY:-1.1}"
TOKENIZER="${TOKENIZER:-models/tokenizer.json}"

# ── Validate ────────────────────────────────────────────────────────────────
if [[ ! -f "$BIN" ]]; then
    echo "Error: Binary not found. Run './scripts/build.sh $BACKEND' first."
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Error: Model not found: $MODEL"
    exit 1
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Running inference..."
echo "  Model:      $MODEL"
echo "  Prompt:     $PROMPT"
echo "  Max tokens: $MAX_TOKENS"
echo "  Temp:       $TEMPERATURE, Top-P: $TOP_P, Top-K: $TOP_K"
echo "  Rep penalty: $REP_PENALTY"
echo ""

RUST_LOG=warn "$BIN" run \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --repetition-penalty "$REP_PENALTY" \
    --seed "$SEED" \
    --tokenizer "$TOKENIZER" \
    2>&1

echo ""
echo "✓ Inference complete"
