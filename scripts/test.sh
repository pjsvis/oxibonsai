#!/usr/bin/env bash
# OxiBonsai — Run tests.
#
# Usage:
#   ./scripts/test.sh [metal|cuda]
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BACKEND="${1:-cpu}"

case "$BACKEND" in
    metal) FEATURES="simd-neon metal native-tokenizer" ;;
    cuda)  FEATURES="native-cuda simd-avx2 native-tokenizer" ;;
    *)    FEATURES="simd-neon metal native-tokenizer" ;;
esac

echo "Running tests with features: $FEATURES"
cargo test --features "$FEATURES"
