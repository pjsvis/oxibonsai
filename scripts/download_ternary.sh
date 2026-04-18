#!/usr/bin/env bash
# Download Ternary-Bonsai unpacked safetensors and convert to OxiBonsai GGUF.
#
# Usage: ./scripts/download_ternary.sh [8b|4b|1.7b]  (default: 1.7b)
#
# Requires: hf CLI (pip install huggingface_hub  — new command is `hf`, not `huggingface-cli`)
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SIZE="${1:-1.7b}"

case "$SIZE" in
  8b|8B)
    REPO="prism-ml/Ternary-Bonsai-8B-unpacked"
    OUT="models/Ternary-Bonsai-8B.gguf"
    ;;
  4b|4B)
    REPO="prism-ml/Ternary-Bonsai-4B-unpacked"
    OUT="models/Ternary-Bonsai-4B.gguf"
    ;;
  1.7b|1.7B|*)
    REPO="prism-ml/Ternary-Bonsai-1.7B-unpacked"
    OUT="models/Ternary-Bonsai-1.7B.gguf"
    ;;
esac

# ── Validate prerequisites ───────────────────────────────────────────────────
HF_CMD=""
if command -v hf >/dev/null 2>&1; then
    HF_CMD="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CMD="huggingface-cli"
else
    echo "ERROR: hf CLI not found." >&2
    echo "Install it with: pip install huggingface_hub" >&2
    exit 1
fi

mkdir -p models

LOCAL_DIR="models/${REPO##*/}"  # e.g. models/Ternary-Bonsai-1.7B-unpacked

# ── Download from HuggingFace ────────────────────────────────────────────────
echo "Downloading $REPO  (using: $HF_CMD)..."
"$HF_CMD" download "$REPO" \
    --local-dir "$LOCAL_DIR" \
    "model.safetensors" "model.safetensors.index.json" "config.json" "tokenizer.json" 2>/dev/null || \
"$HF_CMD" download "$REPO" \
    --local-dir "$LOCAL_DIR" \
    --include "model.safetensors*" "config.json" "tokenizer.json"

# ── Build the converter ──────────────────────────────────────────────────────
echo "Building converter..."
cargo build --release 2>&1 | tail -5

# ── Convert to GGUF ─────────────────────────────────────────────────────────
echo "Converting $LOCAL_DIR -> $OUT..."
./target/release/oxibonsai convert \
    --from "$LOCAL_DIR" \
    --to "$OUT" \
    --quant tq2_0_g128

echo "Done: $OUT"

# ── Copy tokenizer if not already present ────────────────────────────────────
if [[ -f "$LOCAL_DIR/tokenizer.json" && ! -f "models/tokenizer.json" ]]; then
    cp "$LOCAL_DIR/tokenizer.json" models/tokenizer.json
    echo "Copied tokenizer.json to models/"
fi

# ── Remove downloaded source files ───────────────────────────────────────────
echo "Removing $LOCAL_DIR..."
rm -rf "$LOCAL_DIR"
