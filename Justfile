# OxiBonsai — Just task runner (facade)
#
# All implementation is in scripts/*.sh
# This file is just a list of verbs (the API surface).
#
# Usage:
#   just              # list all recipes
#   just build        # build release binary
#   just run          # run inference
#   just chat         # interactive chat
#   just test         # run tests
#   just bench        # benchmark throughput
#   just info         # show model info
#   just validate     # validate GGUF file
#   just smoke        # smoke test (build + run + validate)
#
# Override with MODEL=... PROMPT=... just <verb>

MODEL       := 'models/Bonsai-8B.gguf'
PROMPT      := 'Explain quantum computing in simple terms'
MAX_TOKENS  := '100'
TEMP        := '0.7'
TOP_P       := '0.9'
TOP_K       := '40'
SEED        := '42'
REP_PENALTY := '1.1'
RUNS        := '3'

# ── Build ────────────────────────────────────────────────────────────────────

# Build release binary (CPU SIMD)
build:
    ./scripts/build.sh

# Build with Metal GPU
build_metal:
    ./scripts/build.sh metal

# Build with native CUDA
build_cuda:
    ./scripts/build.sh cuda

# Build with scirs2 CUDA
build_cuda_scirs:
    ./scripts/build.sh cuda-scirs

# ── Inference ────────────────────────────────────────────────────────────────

# Run inference (CPU)
run: build
    MODEL={{MODEL}} PROMPT='{{PROMPT}}' MAX_TOKENS={{MAX_TOKENS}} \
    TEMPERATURE={{TEMP}} TOP_P={{TOP_P}} TOP_K={{TOP_K}} \
    REP_PENALTY={{REP_PENALTY}} ./scripts/run.sh

# Run inference (Metal GPU)
run_metal: build_metal
    MODEL={{MODEL}} PROMPT='{{PROMPT}}' MAX_TOKENS={{MAX_TOKENS}} \
    TEMPERATURE={{TEMP}} TOP_P={{TOP_P}} TOP_K={{TOP_K}} \
    REP_PENALTY={{REP_PENALTY}} ./scripts/run.sh metal

# Run inference (CUDA)
run_cuda: build_cuda
    MODEL={{MODEL}} PROMPT='{{PROMPT}}' MAX_TOKENS={{MAX_TOKENS}} \
    TEMPERATURE={{TEMP}} TOP_P={{TOP_P}} TOP_K={{TOP_K}} \
    REP_PENALTY={{REP_PENALTY}} ./scripts/run.sh cuda

# Run inference (greedy/deterministic)
run_greedy: build
    MODEL={{MODEL}} PROMPT='{{PROMPT}}' MAX_TOKENS={{MAX_TOKENS}} \
    TEMPERATURE=0 TOP_P=1.0 TOP_K=1 ./scripts/run.sh

# Run inference on 1-bit Bonsai-8B
run_1bit: build
    MODEL='models/Bonsai-8B.gguf' PROMPT='{{PROMPT}}' MAX_TOKENS={{MAX_TOKENS}} \
    ./scripts/run.sh

# ── Chat ─────────────────────────────────────────────────────────────────────

# Interactive chat (CPU)
chat: build
    MODEL={{MODEL}} MAX_TOKENS=512 TEMPERATURE={{TEMP}} \
    TOP_P={{TOP_P}} TOP_K={{TOP_K}} REP_PENALTY={{REP_PENALTY}} \
    ./scripts/chat.sh

# Interactive chat (Metal GPU)
chat_metal: build_metal
    MODEL={{MODEL}} MAX_TOKENS=512 TEMPERATURE={{TEMP}} \
    TOP_P={{TOP_P}} TOP_K={{TOP_K}} REP_PENALTY={{REP_PENALTY}} \
    ./scripts/chat.sh metal

# ── Server ───────────────────────────────────────────────────────────────────

# Start HTTP server (CPU)
serve: build
    ./target/release/oxibonsai serve --model {{MODEL}}

# Start HTTP server (Metal GPU)
serve_metal: build_metal
    ./target/release/oxibonsai serve --model {{MODEL}}

# ── Testing ─────────────────────────────────────────────────────────────────

# Run tests (CPU + Metal)
test:
    ./scripts/test.sh

# Run tests (Metal GPU)
test_metal:
    ./scripts/test.sh metal

# Run tests (CUDA)
test_cuda:
    ./scripts/test.sh cuda

# ── Benchmark ────────────────────────────────────────────────────────────────

# Benchmark throughput (CPU vs Metal, 3 runs each)
bench: build build_metal
    MODEL={{MODEL}} PROMPT='{{PROMPT}}' MAX_TOKENS={{MAX_TOKENS}} \
    RUNS={{RUNS}} ./scripts/bench.sh

# Quick benchmark (1 run)
bench_quick: build build_metal
    MODEL={{MODEL}} PROMPT='{{PROMPT}}' MAX_TOKENS={{MAX_TOKENS}} \
    RUNS=1 ./scripts/bench.sh

# ── Info & Validation ────────────────────────────────────────────────────────

# Show model info
info:
    MODEL={{MODEL}} ./scripts/info.sh

# Validate GGUF file
validate:
    MODEL={{MODEL}} ./scripts/validate.sh

# Info + validate
check: info validate
    @echo "✓ Model check complete"

# ── Smoke Tests ──────────────────────────────────────────────────────────────

# Smoke test (CPU)
smoke:
    MODEL={{MODEL}} ./scripts/smoke.sh

# Smoke test (Metal GPU)
smoke_metal:
    MODEL={{MODEL}} ./scripts/smoke.sh metal

# Smoke test (CUDA)
smoke_cuda:
    MODEL={{MODEL}} ./scripts/smoke.sh cuda

# ── Download ────────────────────────────────────────────────────────────────

# Download Ternary Bonsai 1.7B
download_1_7b:
    ./scripts/download_ternary.sh 1.7b

# Download Ternary Bonsai 4B
download_4b:
    ./scripts/download_ternary.sh 4b

# Download Ternary Bonsai 8B
download_8b:
    ./scripts/download_ternary.sh 8b

# Download all ternary models
download_all: download_1_7b download_4b download_8b

# Download 1-bit Bonsai-8B
download_1bit:
    mkdir -p models && curl -L -o models/Bonsai-8B.gguf \
        https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/Bonsai-8B.gguf

# ── Conversion ──────────────────────────────────────────────────────────────

# Convert safetensors → GGUF (ternary)
convert_safetensors SRC:
    ./target/release/oxibonsai convert \
        --from {{SRC}} \
        --to 'models/converted.gguf' \
        --quant tq2_0_g128

# Convert safetensors → GGUF (1-bit)
convert_safetensors_1bit SRC:
    ./target/release/oxibonsai convert \
        --from {{SRC}} \
        --to 'models/converted.gguf' \
        --quant q1_0_g128

# Convert ONNX → GGUF
convert_onnx SRC:
    ./target/release/oxibonsai convert --onnx \
        --from {{SRC}} \
        --to 'models/converted.gguf'

# ── Development ─────────────────────────────────────────────────────────────

# Run all examples
examples: build
    cargo run --example basic_inference -- --model {{MODEL}}
    cargo run --example custom_sampling -- --model {{MODEL}}

# Clean build artifacts
clean:
    cargo clean

# Check prerequisites
prereqs:
    @echo "Checking prerequisites..."
    @which cargo && cargo --version
    @which rustc && rustc --version
    @echo ""
    @echo "Model files:"
    @ls -lh models/*.gguf 2>/dev/null || echo "  (none found)"
    @echo ""
    @echo "Tokenizer:"
    @ls -lh models/tokenizer.json 2>/dev/null || echo "  (not found)"

# Default: list recipes
default:
    @just --list
