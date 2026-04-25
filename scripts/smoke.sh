#!/usr/bin/env bash
# OxiBonsai — Smoke test (build + run + validate).
#
# Usage:
#   ./scripts/smoke.sh [metal|cuda] MODEL=<path>
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

echo "═══════════════════════════════════════════════════════════════"
echo "  OxiBonsai Smoke Test"
echo "  Backend: $BACKEND"
echo "  Model:   $MODEL"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Build
echo "── Build ────────────────────────────────────────────────────"
./scripts/build.sh "$BACKEND"
echo ""

# Info
echo "── Model Info ──────────────────────────────────────────────"
./scripts/info.sh MODEL="$MODEL"
echo ""

# Validate
echo "── Validate ────────────────────────────────────────────────"
./scripts/validate.sh MODEL="$MODEL"
echo ""

# Run
echo "── Run Inference ────────────────────────────────────────────"
PROMPT="What is 2+2?" MAX_TOKENS=20 ./scripts/run.sh "$BACKEND" MODEL="$MODEL"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ Smoke test complete"
echo "═══════════════════════════════════════════════════════════════"
