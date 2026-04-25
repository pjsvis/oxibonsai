#!/usr/bin/env bash
# OxiBonsai — Interactive chat mode.
#
# Usage:
#   ./scripts/chat.sh [metal|cuda] MODEL=<path> [options]
#
# Options:
#   MODEL=<path>        GGUF model path (default: models/Ternary-Bonsai-1.7B.gguf)
#   MAX_TOKENS=<n>      Max tokens per turn (default: 512)
#   TEMPERATURE=<f>    Temperature (default: 0.7)
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

BIN="./target/release/oxibonsai"

# ── Defaults ────────────────────────────────────────────────────────────────
MODEL="${MODEL:-models/Ternary-Bonsai-1.7B.gguf}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-40}"
SEED="${SEED:-42}"
REP_PENALTY="${REP_PENALTY:-1.1}"
TOKENIZER="${TOKENIZER:-models/tokenizer.json}"

# ── Validate ────────────────────────────────────────────────────────────────
if [[ ! -f "$BIN" ]]; then
    echo "Error: Binary not found. Run './scripts/build.sh' first."
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Error: Model not found: $MODEL"
    exit 1
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Starting interactive chat..."
echo "  Model:      $MODEL"
echo "  Max tokens: $MAX_TOKENS"
echo "  Temp:       $TEMPERATURE, Top-P: $TOP_P"
echo ""

exec "$BIN" chat \
    --model "$MODEL" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --repetition-penalty "$REP_PENALTY" \
    --seed "$SEED" \
    --tokenizer "$TOKENIZER"
