#!/usr/bin/env bash
# Score a generated report with the local judge and print the summary.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-/mnt/nvme8tb/hf-ablate/.venv}
REPORT=${1:?usage: judge_report.sh REPORT}
JUDGE_URL=${JUDGE_URL:-http://127.0.0.1:8081/v1} \
JUDGE_MODEL=${JUDGE_MODEL:-pennyroyal} \
  "$VENV/bin/python" -m groot_ablit.cli score --report "$REPORT"
