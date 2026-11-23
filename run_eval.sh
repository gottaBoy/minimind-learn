#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"

"$PYTHON" eval_llm.py \
  --device "$DEVICE" \
  --weight "${WEIGHT:-pretrain}" \
  --save_dir out
