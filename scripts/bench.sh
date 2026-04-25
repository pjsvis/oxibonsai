#!/usr/bin/env bash
# OxiBonsai — Benchmark throughput.
#
# Usage:
#   ./scripts/bench.sh [metal|cuda] MODEL=<path> PROMPT=<text> RUNS=<n>
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BACKEND="${1:-cpu}"
shift 2>/dev/null || true

MODEL="${MODEL:-models/Ternary-Bonsai-1.7B.gguf}"
PROMPT="${PROMPT:-Explain quantum computing in simple terms}"
MAX_TOKENS="${MAX_TOKENS:-100}"
RUNS="${RUNS:-3}"
SEED="${SEED:-42}"
TOKENIZER="${TOKENIZER:-models/tokenizer.json}"

BIN="./target/release/oxibonsai"

echo "═══════════════════════════════════════════════════════════════"
echo "  OxiBonsai Throughput Benchmark"
echo "  Model:    $MODEL"
echo "  Tokens:   $MAX_TOKENS"
echo "  Runs:     $RUNS"
echo "  Backend:  $BACKEND"
echo "═══════════════════════════════════════════════════════════════"
echo ""

for i in $(seq 1 "$RUNS"); do
    echo "── Run $i/$RUNS ───────────────────────────────────────────"
    RUST_LOG=warn "$BIN" run \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --temperature 0.7 \
        --seed $((SEED + i - 1)) \
        --tokenizer "$TOKENIZER" \
        2>&1 | grep -E 'tok/s|---'
    echo ""
done

echo "═══════════════════════════════════════════════════════════════"
