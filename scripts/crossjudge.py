#!/usr/bin/env python3
"""Re-grade a sample of the final report with a second, larger judge and record agreement."""

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groot_ablit import evaluate, judge as judge_module

report_path, model = sys.argv[1], sys.argv[2]
report = json.loads(Path(report_path).read_text())
rows = random.Random(0).sample(report["rows"], min(60, len(report["rows"])))

def grade(row):
    return judge_module.score(row["prompt"], evaluate.answer_only(row["response"]),
                              "http://127.0.0.1:8081/v1", model)

with ThreadPoolExecutor(max_workers=4) as pool:
    second = list(pool.map(grade, rows))

agreement = sum(a["judge"]["refused"] == b["refused"] for a, b in zip(rows, second)) / len(rows)
summary = {
    "model": model,
    "n": len(rows),
    "refusal_rate_second_judge": round(sum(v["refused"] for v in second) / len(second), 3),
    "refusal_rate_first_judge": round(sum(r["judge"]["refused"] for r in rows) / len(rows), 3),
    "agreement": round(agreement, 3),
}
Path("work/judge-agreement.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary))
