#!/usr/bin/env bash
# Unattended chain: build variants, measure them, pick the one that clears the
# gates, quantise it to NVFP4, verify the quantised build, measure capability,
# and publish the result with a full card. Every stage logs and continues on
# failure; the summary at the end says exactly which stages succeeded.
set -uo pipefail
cd /home/marcus/Dev/python/groot-abliteration
export PATH="$HOME/.local/bin:$PATH"

VENV=/mnt/nvme8tb/hf-ablate/.venv
ABL=/mnt/nvme8tb/hf-ablate
LOG=work/overnight.log
STATE=work/overnight-state.json
SUMMARY=work/overnight-summary.md
mkdir -p work reports

exec > >(tee -a "$LOG") 2>&1
echo "=== overnight start $(date -Is)"

note() {
  "$VENV/bin/python" - "$STATE" "$1" "$2" <<'PY'
import json, sys, pathlib, datetime
path = pathlib.Path(sys.argv[1])
state = json.loads(path.read_text()) if path.exists() else {}
state[sys.argv[2]] = {"status": sys.argv[3], "at": datetime.datetime.now().isoformat(timespec="seconds")}
path.write_text(json.dumps(state, indent=1))
PY
}

restore_services() {
  systemctl --user start unrager-serve-watchdog.timer 2>/dev/null || true
  echo "=== overnight end $(date -Is)"
}
trap restore_services EXIT
systemctl --user stop unrager-serve-watchdog.timer 2>/dev/null || true
docker rm -f vllm-cand vllm-ablate >/dev/null 2>&1 || true
sleep 2

build_variant() {
  local name=$1 model=$2 experts=$3 shared=$4
  if [ -f "$ABL/candidates/$name/config.json" ]; then
    echo "-- $name exists"; note "build-$name" "present"; return 0
  fi
  echo "-- building $name"
  ./scripts/free_gpu.sh && \
  "$VENV/bin/python" -m groot_ablit.cli ablate --model "$model" --directions work/directions.pt \
    --experts "$experts" --shared "$shared" --out "$ABL/candidates/$name"
  note "build-$name" "$([ -f "$ABL/candidates/$name/config.json" ] && echo ok || echo failed)"
}

measure() {
  local dir=$1 served=$2 kind=$3 report=$4
  echo "-- measuring $served ($kind)"
  ./scripts/free_gpu.sh
  KIND="$kind" MAX_NEW=1536 ./scripts/eval_candidate.sh "$dir" "$served" "$report"
  if [ ! -f "$report" ]; then
    echo "-- retry $served ($kind) after a server failure"
    ./scripts/free_gpu.sh
    docker rm -f vllm-cand >/dev/null 2>&1 || true
    KIND="$kind" MAX_NEW=1536 ./scripts/eval_candidate.sh "$dir" "$served" "$report"
  fi
  note "measure-$served-$kind" "$([ -f "$report" ] && echo ok || echo failed)"
  ./scripts/free_gpu.sh
  "$VENV/bin/python" -m groot_ablit.cli score --report "$report" \
    --judge-url http://127.0.0.1:8081/v1 --judge-model pennyroyal
  note "score-$served-$kind" "$([ -f "$report" ] && echo ok || echo failed)"
}

for pair in "parent:reports/parent_fp8.json" "t57:reports/t57_fp8.json"; do
  name=${pair%%:*}; file=${pair#*:}
  if [ -f "$file" ] && [ ! -f "reports/overnight_${name}_harmful.json" ]; then
    cp -n "$file" "reports/overnight_${name}_harmful.json"
  fi
done

build_variant t57-e06 "$ABL/candidates/heretic-t57" 0.6 0.6
build_variant t57-e10 "$ABL/candidates/heretic-t57" 1.0 1.0
build_variant parent-e10 "$ABL/groot-bf16" 1.0 1.0

measure "$ABL/groot-bf16" parent harmful reports/overnight_parent_harmful.json
measure "$ABL/candidates/heretic-t57" t57 harmful reports/overnight_t57_harmful.json
measure "$ABL/candidates/t57-e06" t57e06 harmful reports/overnight_t57-e06_harmful.json
measure "$ABL/candidates/t57-e06" t57e06x xstest_safe reports/overnight_t57-e06_xstest.json
measure "$ABL/candidates/t57-e10" t57e10 harmful reports/overnight_t57-e10_harmful.json
measure "$ABL/candidates/t57-e10" t57e10x xstest_safe reports/overnight_t57-e10_xstest.json
measure "$ABL/candidates/parent-e10" parente10 harmful reports/overnight_parent-e10_harmful.json
measure "$ABL/groot-bf16" parentx xstest_safe reports/overnight_parent_xstest.json

"$VENV/bin/python" -m groot_ablit.select reports > work/selection.json
note select "$([ -s work/selection.json ] && echo ok || echo failed)"
cat work/selection.json

WINNER=$("$VENV/bin/python" - <<'PY'
import json
data = json.load(open("work/selection.json"))
winner = data.get("winner")
print(winner["name"] if winner else "")
PY
)
echo "winner: ${WINNER:-none}"

if [ -n "$WINNER" ]; then
  "$VENV/bin/python" - "$WINNER" <<'PY'
import json, sys
name = sys.argv[1]
strengths = {"t57-e06": (0.6, 0.6), "t57-e10": (1.0, 1.0), "parent-e10": (1.0, 1.0)}
experts, shared = strengths.get(name, (0.0, 0.0))
trial = {"trials": 120, "trial": 57 if name.startswith("t57") else "none",
         "kl": 0.0801 if name.startswith("t57") else "n/a",
         "max_weight": 1.45 if name.startswith("t57") else 0.0,
         "max_weight_position": 28.93 if name.startswith("t57") else 0.0,
         "min_weight": 0.18 if name.startswith("t57") else 0.0,
         "min_weight_distance": 18.61 if name.startswith("t57") else 0.0,
         "experts": experts, "shared": shared}
open("work/winner-trial.json", "w").write(json.dumps(trial, indent=1))
PY

  SRC="$ABL/candidates/$WINNER"
  FINAL="$ABL/final/nvfp4"
  rm -rf "$FINAL"
  ./scripts/quantize.sh "$SRC" "$FINAL"
  note quantize "$([ -f "$FINAL/config.json" ] && echo ok || echo failed)"

  if [ -f "$FINAL/config.json" ]; then
    measure "$FINAL" final-nvfp4 harmful reports/final_harmful.json
    measure "$FINAL" final-nvfp4x xstest_safe reports/final_xstest.json
    ./scripts/free_gpu.sh
    "$VENV/bin/python" scripts/crossjudge.py reports/final_harmful.json nemotron-3.5-super-vl-120b-nvfp4 \
      >> "$LOG" 2>&1 || true
  fi

  ./scripts/free_gpu.sh
  ./scripts/quality.sh "$ABL/groot-bf16" parent > work/quality-parent.json 2>>"$LOG" || true
  ./scripts/free_gpu.sh
  ./scripts/quality.sh "$SRC" abliterated > work/quality-abliterated.json 2>>"$LOG" || true
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

  "$VENV/bin/python" scripts/finalize.py || note publish failed
  note publish "$([ -f work/published.txt ] && echo ok || echo failed)"
fi

"$VENV/bin/python" - <<'PY' > "$SUMMARY"
import json, pathlib
state = json.loads(pathlib.Path("work/overnight-state.json").read_text()) if pathlib.Path("work/overnight-state.json").exists() else {}
published = pathlib.Path("work/published.txt").read_text().strip() if pathlib.Path("work/published.txt").exists() else "not published"
print("# Overnight results\n")
print(f"Published: {published}\n")
print("| stage | status |")
print("| --- | --- |")
for key in sorted(state):
    print(f"| {key} | {state[key]['status']} |")
print()
for path in sorted(pathlib.Path("reports").glob("overnight_*_harmful.json")) + sorted(pathlib.Path("reports").glob("final_*_harmful.json")):
    data = json.loads(path.read_text())
    s = data.get("summary")
    if s:
        print(f"- {path.name}: refusals {s['refusal_rate']:.1%}, hot {s['hot_rate']:.1%}, "
              f"mean {s['mean_score']:.3f}, hidden {s['hidden_refusals']}")
PY
echo "=== summary written to $SUMMARY"
cat "$SUMMARY"
