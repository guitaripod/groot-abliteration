#!/usr/bin/env bash
# Launch Heretic in a detached tmux session. Output goes through tmux's pipe-pane
# so the interactive menus keep a real TTY while still being logged.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-/mnt/nvme8tb/hf-ablate/.venv}
SESSION=${SESSION:-heretic}
LOG=${LOG:-work/heretic.log}
mkdir -p work
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -c "$PWD" \
  "$VENV/bin/heretic --model /mnt/nvme8tb/hf-ablate/groot-bf16 --device-map cuda:0 \
    --n-trials ${TRIALS:-120} --n-startup-trials ${STARTUP:-40} \
    --max-response-length ${RESPONSE_LEN:-384} --batch-size ${BATCH:-16}"
tmux pipe-pane -t "$SESSION" -o "cat >> $LOG"
echo "started tmux session $SESSION (log: $LOG)"
