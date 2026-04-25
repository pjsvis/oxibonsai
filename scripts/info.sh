#!/usr/bin/env bash
# OxiBonsai — Show model info.
#
# Usage:
#   ./scripts/info.sh MODEL=<path>
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-models/Ternary-Bonsai-1.7B.gguf}"
BIN="./target/release/oxibonsai"

if [[ ! -f "$MODEL" ]]; then
    echo "Error: Model not found: $MODEL"
    exit 1
fi

"$BIN" info --model "$MODEL"
