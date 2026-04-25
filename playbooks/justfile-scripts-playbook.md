---
date: 2026-04-25
tags: [playbook, justfile, scripts, architecture]
---

# Justfile and Scripts Architecture Playbook

## Purpose
Define the correct architecture for task runners using `just` with implementation delegated to `scripts/*.sh`. This ensures separation of concerns, testability, and avoids common `just` parser pitfalls.

## Context & Prerequisites
- `just` installed (https://just.systems)
- Bash scripts with `#!/usr/bin/env bash` shebang
- Standard POSIX flags: `set -euo pipefail`

## The Protocol

### 1. Justfile as Facade (Not Implementation)
The `Justfile` should be a **pure API surface** — a list of verbs that call scripts. It contains NO implementation logic.

**RIGHT:**
```just
build:
    ./scripts/build.sh

run:
    MODEL={{MODEL}} PROMPT='{{PROMPT}}' ./scripts/run.sh
```

**WRONG (Implementation in Justfile):**
```just
build:
    cargo build --release --features "simd-neon metal"  # Don't do this
```

### 2. Scripts as Implementation
All implementation goes in `scripts/*.sh`. Each script:
- Is independently executable
- Handles its own defaults and validation
- Has a clear `--help` or docstring
- Uses `set -euo pipefail` for error safety

### 3. Justfile Naming Rules (CRITICAL)

| Element | Allowed | Forbidden | Example |
|---------|---------|-----------|---------|
| Variables | Alphanumeric, underscore | dots, hyphens, ?= | `MODEL := 'foo'` not `MODEL ?= 'foo'` |
| Recipes | Alphanumeric, underscore | dots, hyphens | `run_metal:` not `run-metal:` |
| Assignment | `:=` only | `=`, `?=`, `${}` | `VAR := 'value'` |

**Common Justfile Parse Errors:**
- `error: Unknown start of token '.'` → Recipe/variable has dot
- `error: Expected '=', ':', ...` → Used `?= ` or wrong assignment
- `error: Unknown start of token '-' → Recipe/variable has hyphen

### 4. Script Template

```bash
#!/usr/bin/env bash
# [Script Name] — One line description.
#
# Usage:
#   ./scripts/script.sh [arg] KEY=value KEY2=value2
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults ────────────────────────────────────────────────────────────────
ARG="${1:-default}"
KEY="${KEY:-default_value}"

# ── Validate ────────────────────────────────────────────────────────────────
if [[ ! -f "$BIN" ]]; then
    echo "Error: Binary not found. Run './scripts/build.sh' first."
    exit 1
fi

# ── Execute ─────────────────────────────────────────────────────────────────
exec "$BIN" command --arg "$ARG"
```

### 5. Environment Variable Passing

Pass variables BEFORE the script invocation in `just`:
```just
run:
    MODEL={{MODEL}} PROMPT='{{PROMPT}}' ./scripts/run.sh
```

Variables inside scripts use `${VAR:-default}` syntax.

## Standards & Patterns

- **Script names:** `verb.sh` (build.sh, run.sh, test.sh)
- **Script permissions:** Always `chmod +x scripts/*.sh`
- **Justfile location:** Project root
- **Scripts location:** `scripts/` directory
- **Justfile comments:** Include usage example at top

## Validation

To verify a Justfile is correctly structured:
```bash
just --list  # Should list all recipes without errors
./scripts/build.sh  # Should work independently
just run  # Should call ./scripts/run.sh with correct env
```

## Common Patterns

### Build + Run Pattern
```just
# Justfile
run: build
    MODEL={{MODEL}} ./scripts/run.sh

# scripts/run.sh
#!/usr/bin/env bash
set -euo pipefail
BIN="./target/release/oxibonsai"
MODEL="${MODEL:-models/default.gguf}"
exec "$BIN" run --model "$MODEL"
```

### Backend Selection Pattern
```bash
# scripts/run.sh
BACKEND="${1:-cpu}"
shift || true

case "$BACKEND" in
    metal) FEATURES="metal" ;;
    cuda)  FEATURES="native-cuda" ;;
    *)    FEATURES="simd-neon" ;;
esac
```

## Maintenance
If `just --list` fails with parse errors, check for:
1. Dots in variable/recipe names → Replace with underscores
2. Wrong assignment operator → Use `:=` not `=`
3. Backtick command substitutions with special chars → Move logic to scripts
