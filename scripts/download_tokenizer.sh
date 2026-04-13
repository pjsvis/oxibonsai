#!/usr/bin/env bash
# Download Qwen3 tokenizer.json from HuggingFace.
#
# Usage:
#   ./scripts/download_tokenizer.sh
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

HF_REPO="Qwen/Qwen3-8B"
OUTPUT="$PROJECT_ROOT/models/tokenizer.json"

if [ -f "$OUTPUT" ]; then
    echo "✓ tokenizer.json already exists at $OUTPUT"
    exit 0
fi

mkdir -p "$PROJECT_ROOT/models"

echo "Downloading tokenizer.json from $HF_REPO ..."
curl -fSL "https://huggingface.co/${HF_REPO}/resolve/main/tokenizer.json" \
    -o "$OUTPUT"

echo "✓ Saved to $OUTPUT ($(wc -c < "$OUTPUT") bytes)"
