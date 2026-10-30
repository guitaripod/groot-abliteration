#!/usr/bin/env python3
"""Assemble the final card from the measured reports and publish the NVFP4 build."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groot_ablit import card as card_module
from groot_ablit import select as select_module

REPO_ID = "guitaripod/groot-19b-a3b-uncensored-nvfp4"
REPO_URL = "https://github.com/guitaripod/groot-abliteration"


def repo_line() -> str:
    import urllib.request

    try:
        request = urllib.request.Request(REPO_URL, method="HEAD")
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status < 400:
                return f"Pipeline, prompts, judge and quantisation recipe: {REPO_URL}."
    except Exception:
        pass
    return ("The pipeline is a self-contained local repository (prompts, direction extraction, "
            "ablation, quantisation recipe, judge harness); the quantisation plan is the parent's own "
            "`quantization_config`, reused verbatim.")


def summarise(path: str) -> dict | None:
    file = Path(path)
    if not file.exists():
        return None
    return json.loads(file.read_text()).get("summary")


def quality(path: str) -> dict:
    file = Path(path)
    if not file.exists():
        return {}
    return json.loads(file.read_text())


def main() -> None:
    reports = Path("reports")
    selection = select_module.select(str(reports))
    winner = selection["winner"]
    if not winner:
        print("no candidate measured; nothing to publish")
        return

    final_dir = Path("/mnt/nvme8tb/hf-ablate/final/nvfp4")
    verified = summarise("reports/final_harmful.json")
    verified_xstest = summarise("reports/final_xstest.json")
    if not verified:
        print("NVFP4 build not verified; not publishing")
        return

    quality_results = quality("work/quality.json")
    rows = []
    parent = summarise("reports/overnight_parent_harmful.json")
    if parent:
        rows.append({"name": "parent (BF16)", **{
            "refusals": parent["refusal_rate"], "hot": parent["hot_rate"],
            "mean": parent["mean_score"], "hidden": parent["hidden_refusals"],
            "xstest_refusals": (summarise("reports/overnight_parent_xstest.json") or {}).get("refusal_rate"),
        }})
    rows.append({"name": f"abliterated (BF16, {winner['name']})", **{
        "refusals": winner["harmful"]["refusal_rate"], "hot": winner["harmful"]["hot_rate"],
        "mean": winner["harmful"]["mean_score"], "hidden": winner["harmful"]["hidden_refusals"],
        "xstest_refusals": (winner.get("xstest") or {}).get("refusal_rate"),
    }})
    compliance = quality("work/harmbench-final.json").get("harmful_compliance_rate")
    for row in rows:
        if row["name"].startswith("parent"):
            row["compliance"] = quality("work/harmbench-parent.json").get("harmful_compliance_rate")
        elif row["name"].startswith("abliterated"):
            row["compliance"] = quality("work/harmbench-winner.json").get("harmful_compliance_rate")
    rows.append({"name": "this build (NVFP4)", **{
        "compliance": compliance,
        "refusals": verified["refusal_rate"], "hot": verified["hot_rate"],
        "mean": verified["mean_score"], "hidden": verified["hidden_refusals"],
        "xstest_refusals": (verified_xstest or {}).get("refusal_rate"),
    }})

    quality_rows = []
    if quality_results:
        for name in ("parent", "abliterated"):
            if name in quality_results:
                quality_rows.append({"name": name, **quality_results[name]})

    trial = json.loads(Path("work/winner-trial.json").read_text()) if Path("work/winner-trial.json").exists() else {}
    context = {
        "repo_name": REPO_ID.split("/")[-1],
        "base_model": "GrootTeam/Groot-19B-A3B-261023",
        "repo_url": REPO_URL,
        "repo_line": repo_line(),
        "train_harmful": 200,
        "train_harmless": 200,
        "mean_tokens": 22.8,
        "trials": trial.get("trials", 120),
        "trial": trial.get("trial", "–"),
        "kl": trial.get("kl", "–"),
        "max_weight": trial.get("max_weight", "–"),
        "max_weight_position": trial.get("max_weight_position", "–"),
        "min_weight": trial.get("min_weight", "–"),
        "min_weight_distance": trial.get("min_weight_distance", "–"),
        "experts": trial.get("experts", "–"),
        "shared": trial.get("shared", "–"),
        "judge": "pennyroyal (Qwen 3.8 27B, local vLLM)",
        "eval_harmful": (verified or {}).get("n", 120),
        "xstest": 225,
        "string_rate": verified.get("string_refusal_rate", 0.0),
        "rows": rows,
        "quality": quality_rows,
    }
    agreement = quality("work/judge-agreement.json")
    if agreement:
        context["repo_line"] = (
            f"Second judge cross-check on {agreement['n']} of the final prompts: "
            f"{agreement['agreement']:.0%} verdict agreement, "
            f"refusal rate {agreement['refusal_rate_second_judge']:.1%} vs "
            f"{agreement['refusal_rate_first_judge']:.1%}. " + context["repo_line"]
        )
    text = card_module.build(context)
    url = card_module.publish(str(final_dir), REPO_ID, text)
    print(f"published: {url}")
    Path("work/published.txt").write_text(url + "\n")


if __name__ == "__main__":
    main()
