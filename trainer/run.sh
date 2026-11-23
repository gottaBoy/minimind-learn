#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"

"$PYTHON" scripts/smoke_test.py --device "$DEVICE"
"$PYTHON" -m trainer.train_pretrain \
  --device "$DEVICE" \
  --data_path dataset/demo_pretrain.jsonl \
  --save_dir out \
  --from_weight none \
  --from_resume 0 \
  --epochs 1 \
  --batch_size 2 \
  --max_seq_len 128 \
  --accumulation_steps 1 \
  --log_interval 1 \
  --save_interval 1

"$PYTHON" eval_llm.py \
  --device "$DEVICE" \
  --weight pretrain \
  --save_dir out \
  --max_new_tokens 32 \
  --prompt "MiniMind is"
