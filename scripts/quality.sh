#!/usr/bin/env bash
# Serve a BF16 model with vLLM and run GSM8K through lm-eval's HF backend.
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-/mnt/nvme8tb/hf-ablate/.venv}
DIR=${1:?usage: quality.sh MODEL_DIR NAME}
NAME=${2:?}
OUT=work/quality-${NAME}.json

"$VENV/bin/lm_eval" --model hf \
  --model_args "pretrained=${DIR},dtype=bfloat16,trust_remote_code=True" \
  --tasks gsm8k --limit 200 --batch_size 8 --apply_chat_template --num_fewshot 5 --gen_kwargs max_gen_toks=2048 \
  --output_path "$OUT" > "work/quality-${NAME}.log" 2>&1

"$VENV/bin/python" - "$OUT" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
result = data["results"]["gsm8k"]
key = "exact_match,strict-match" if "exact_match,strict-match" in result else "exact_match,flexible-extract"
print(json.dumps({"gsm8k": round(result[key] * 100, 2)}))
PY
