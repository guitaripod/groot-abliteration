#!/usr/bin/env bash
# Unload every llama-swap model and wait until the card is idle enough for a full-model job.
set -euo pipefail
LIMIT_MIB=${LIMIT_MIB:-15000}
for model in $(curl -s http://127.0.0.1:8081/running | python3 -c "import json,sys;print(' '.join(m['model'] for m in json.load(sys.stdin).get('running',[])))"); do
  curl -s "http://127.0.0.1:8081/unload?model=${model}" -o /dev/null
done
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "$used" -lt "$LIMIT_MIB" ] && echo "gpu free: ${used} MiB" && exit 0
  sleep 3
done
echo "gpu still busy: ${used} MiB" >&2
exit 1
