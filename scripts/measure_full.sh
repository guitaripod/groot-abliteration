#!/usr/bin/env bash
# Measure on the full held-out set with llama-swap stopped, so the evaluation
# server is not fighting a 50 GB model for the card. llama-swap comes back on exit.
set -uo pipefail
cd "$(dirname "$0")/.."

restore() {
  systemctl --user start llama-swap.service 2>/dev/null || true
}
trap restore EXIT

systemctl --user stop llama-swap.service 2>/dev/null || true
sleep 5

for spec in "$@"; do
  dir=${spec%%:*}
  name=${spec##*:}
  echo "== $name"
  KIND=harmful_hard MAX_NEW=4096 UTIL=0.45 ./scripts/eval_candidate.sh "$dir" "full-$name" "reports/full_${name}.json"
done
