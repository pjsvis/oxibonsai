#!/usr/bin/env bash
# OxiBonsai — Build release binary.
#
# Usage:
#   ./scripts/build.sh              # CPU SIMD (auto-detects AVX2/AVX-512/NEON)
#   ./scripts/build.sh metal        # Metal GPU (Apple Silicon)
#   ./scripts/build.sh cuda        # Native CUDA (Linux/Windows)
#   ./scripts/build.sh cuda-scirs  # scirs2-core CUDA
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

FEATURE="${1:-}"

case "$FEATURE" in
    metal)
        echo "Building with Metal GPU..."
        cargo build --release --features "metal native-tokenizer"
        ;;
    cuda)
        echo "Building with native CUDA..."
        cargo build --release --features "native-cuda simd-avx2 native-tokenizer"
        ;;
    cuda-scirs)
        echo "Building with scirs2 CUDA..."
        cargo build --release --features "cuda native-tokenizer"
        ;;
    *)
        echo "Building with CPU SIMD (auto-detect)..."
        cargo build --release --features "simd-neon native-tokenizer"
        ;;
esac

echo "✓ Build complete: ./target/release/oxibonsai"
