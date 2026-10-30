#!/usr/bin/env bash
# Quantise the chosen build, verify the quantised weights, measure capability, publish.
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=/mnt/nvme8tb/hf-ablate/.venv
ABL=/mnt/nvme8tb/hf-ablate
SRC=${1:?usage: finish.sh CANDIDATE_DIR CANDIDATE_NAME}
NAME=${2:?}
FINAL=$ABL/final/nvfp4

restore() { systemctl --user start llama-swap.service 2>/dev/null || true; }
trap restore EXIT
systemctl --user stop llama-swap.service 2>/dev/null || true
sleep 4
docker rm -f vllm-cand vllm-hb >/dev/null 2>&1 || true

rm -rf "$FINAL"
./scripts/quantize.sh "$SRC" "$FINAL" || echo "quantise failed"

if [ -f "$FINAL/config.json" ]; then
  KIND=harmful_hard MAX_NEW=4096 UTIL=0.45 ./scripts/eval_candidate.sh "$FINAL" final reports/final_harmful.json
  KIND=xstest_safe MAX_NEW=768 UTIL=0.45 ./scripts/eval_candidate.sh "$FINAL" finalx reports/final_xstest.json
fi

./scripts/free_gpu.sh
./scripts/quality.sh "$SRC" abliterated > work/quality-abliterated.json 2>>work/quality.log || true
./scripts/free_gpu.sh
./scripts/quality.sh "$ABL/groot-bf16" parent > work/quality-parent.json 2>>work/quality.log || true
"$VENV/bin/python" - <<'PY'
import json, pathlib
merged = {}
for name in ("parent", "abliterated"):
    path = pathlib.Path(f"work/quality-{name}.json")
    if path.exists() and path.stat().st_size:
        try:
            merged[name] = json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
pathlib.Path("work/quality.json").write_text(json.dumps(merged, indent=1))
PY

systemctl --user stop llama-swap.service 2>/dev/null || true
sleep 4
docker rm -f vllm-hb >/dev/null 2>&1 || true
docker run -d --name vllm-hb --device nvidia.com/gpu=all --ipc=host --user 1000:1000 \
  --memory=32g --memory-swap=32g \
  -e HOME=/cache/home -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache/vllm \
  -e XDG_CACHE_HOME=/cache -e FLASHINFER_WORKSPACE_BASE=/cache -e TRITON_CACHE_DIR=/cache/triton \
  -v /mnt/nvme8tb/hf:/hf -v /mnt/nvme8tb/hf-ablate:/ablate -v /mnt/nvme8tb/vllm-cache:/cache \
  -p 127.0.0.1:17801:8000 \
  vllm/vllm-openai:nightly-3f1bcfd82bb2e93fcd364fc7a5491718453a77e2 \
  /ablate/harmbench --served-model-name harmbench --max-model-len 2048 \
  --gpu-memory-utilization 0.45 --quantization fp8 >/dev/null 2>&1

for _ in $(seq 1 40); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:17801/v1/models)" = "200" ] && break
  sleep 10
done

for pair in "reports/full_parent.json:work/harmbench-parent.json" \
            "reports/full_${NAME}.json:work/harmbench-winner.json" \
            "reports/final_harmful.json:work/harmbench-final.json"; do
  src=${pair%%:*}; dst=${pair#*:}
  [ -f "$src" ] || continue
  "$VENV/bin/python" -m groot_ablit.harmbench "$src" http://127.0.0.1:17801/v1 harmbench > "$dst" 2>>work/harmbench.log || true
done

docker rm -f vllm-hb >/dev/null 2>&1 || true
systemctl --user start llama-swap.service 2>/dev/null || true
sleep 5
"$VENV/bin/python" scripts/finalize.py
