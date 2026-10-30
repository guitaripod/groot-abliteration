#!/usr/bin/env bash
# Serve a candidate with vLLM in FP8 (fits beside whatever else holds the card),
# generate the held-out harmful responses, then shut the server down.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-/mnt/nvme8tb/hf-ablate/.venv}
MODEL_DIR=${1:?usage: eval_candidate.sh MODEL_DIR SERVED_NAME REPORT [extra vllm args]}
SERVED=${2:?}
REPORT=${3:?}
PORT=${PORT:-17800}
MAX_NEW=${MAX_NEW:-2048}

CONTAINER_DIR=${MODEL_DIR/\/mnt\/nvme8tb\/hf-ablate//ablate}
CONTAINER_DIR=${CONTAINER_DIR/\/mnt\/nvme8tb\/hf//hf}

docker rm -f vllm-cand >/dev/null 2>&1 || true
docker run -d --name vllm-cand --device nvidia.com/gpu=all --ipc=host --user 1000:1000 \
  --memory=64g --memory-swap=64g \
  -e HOME=/cache/home -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache/vllm \
  -e XDG_CACHE_HOME=/cache -e FLASHINFER_WORKSPACE_BASE=/cache -e TRITON_CACHE_DIR=/cache/triton \
  -e MAX_JOBS=3 \
  -v /mnt/nvme8tb/hf:/hf -v /mnt/nvme8tb/hf-ablate:/ablate -v /mnt/nvme8tb/vllm-cache:/cache \
  -p 127.0.0.1:${PORT}:8000 \
  vllm/vllm-openai:nightly-3f1bcfd82bb2e93fcd364fc7a5491718453a77e2 \
  "$CONTAINER_DIR" --served-model-name "$SERVED" --max-model-len 32768 \
  --gpu-memory-utilization ${UTIL:-0.25} --max-num-seqs 8 \
  ${QUANT:+--quantization "$QUANT"} > /dev/null 2>&1

ready=0
for _ in $(seq 1 60); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/v1/models")" = "200" ]; then
    ready=1
    break
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' vllm-cand 2>/dev/null)" != "true" ]; then
    break
  fi
  sleep 10
done

if [ "$ready" -ne 1 ]; then
  echo "eval server for ${SERVED} never became ready" >&2
  docker logs vllm-cand > "work/vllm-${SERVED}.log" 2>&1 || true
  tail -20 "work/vllm-${SERVED}.log" >&2 || true
  docker rm -f vllm-cand >/dev/null 2>&1 || true
  exit 1
fi

"$VENV/bin/python" -m groot_ablit.cli eval --model "$SERVED" \
  --endpoint "http://127.0.0.1:${PORT}/v1" --served-model "$SERVED" \
  --split eval --kind "${KIND:-harmful}" --out "$REPORT" --max-new-tokens "$MAX_NEW" --skip-judge

docker logs vllm-cand > "work/vllm-${SERVED}.log" 2>&1 || true
docker rm -f vllm-cand >/dev/null 2>&1 || true
echo "wrote $REPORT"
