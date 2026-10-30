"""Model card and upload.

The card is deliberately short: what changed, how, and every number that was
measured, including the ones that look bad.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi

TEMPLATE = """---
license: apache-2.0
base_model: GrootTeam/Groot-19B-A3B-261023
tags:
- abliteration
- uncensored
- nvfp4
- moe
- vllm
language:
- en
---

# {repo_name}

Uncensored build of [{base_model}](https://huggingface.co/{base_model}), shipped as
NVFP4 for Blackwell (sm120) and served with vLLM.

The parent is a 19B-parameter MoE (3B active, 128 experts top-8, 30 gated
delta-net + 10 full-attention layers, 256K context, 1 MTP layer). Nothing about
the architecture changed; refusal behaviour was removed by directional ablation
and the result was re-quantised to the parent's own mixed-precision plan.

## What was done

1. **Direction extraction.** First-generated-token residuals on {train_harmful}
   harmful and {train_harmless} harmless prompts, matched per prompt on token
   length ({mean_tokens} tokens mean each side) so the direction encodes
   refuse-versus-comply rather than length or framing. Per-layer directions as
   the normalised difference of means; the held-out evaluation prompts were
   never used for extraction.
2. **Attention ablation.** Heretic {trials}-trial Optuna search over per-layer
   directions and ablation-kernel shape on the attention output projections
   (`self_attn.o_proj`, `linear_attn.out_proj`), norm-preserving. Selected
   trial {trial}: KL divergence {kl} against the parent on benign text, with
   attention weights `max_weight={max_weight}` at layer `{max_weight_position}`,
   `min_weight={min_weight}`, `min_weight_distance={min_weight_distance}`.
3. **Expert ablation.** Heretic cannot reach this model's fused expert tensors
   (`down_proj` is a single 128x2048x512 parameter), so the routed experts and
   the shared expert were ablated separately: {experts} strength on all expert
   slices and {shared} on the shared expert, norm-preserving, all 40 layers.
4. **Quantisation.** The parent's own compressed-tensors plan, reused verbatim:
   FP8 W8A8 on attention and linear-attention projections, NVFP4 group-16
   weights on experts, shared expert and `lm_head`; vision tower untouched.
   Calibrated on 512 Alpaca samples. The MTP layer is not present in this build,
   so it runs without speculative decoding.

## Measurements

Compliance is graded by the HarmBench classifier (`cais/HarmBench-Llama-2-13b-cls`),
the metric validated against human labels; refusals and answer quality are graded
by a local rubric judge (`{judge}`), StrongREJECT-style, on the text after the
thinking block. Prompts: {eval_harmful} held-out harmful instructions, disjoint
from the extraction set; {xstest} XSTest safe prompts for over-refusal.

| Build | HarmBench compliance | Refusals (judge) | Hot (score >= 0.5) | Mean score | XSTest refusals |
| --- | --- | --- | --- | --- | --- |
{rows}

Capability, identical harness on both builds:

| Build | GSM8K | MMLU |
| --- | --- | --- |
{quality_rows}

String matching on refusal phrases was also recorded and is useless on this
model: it reports {string_rate:.1%} refusals on a build the judge scores
differently, because the refusals are paraphrased.

## What it is not

Abliteration removes a refusal direction; it does not add judgement, and it does
not make the model safe. Expect capability loss proportional to the numbers
above, and expect the model to comply with things the parent declined. The
parent's licence (Apache-2.0) carries over. No benchmark here measures whether
the output is correct or wise — only whether it complies and how much capability
survived.

## Reproduction

{repo_line} Direction extraction, ablation strengths, trial selection and
quantisation are all recorded, and the evaluation sets are held out from
extraction: 200 harmful and 200 harmless prompts for the direction, 120 harmful
plus 225 XSTest prompts for the measurement, no overlap.
"""


def _rate(value: float | None) -> str:
    return "–" if value is None else f"{value:.1%}"


def _score(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def build(context: dict) -> str:
    rows = []
    for row in context["rows"]:
        rows.append(
            "| {name} | {compliance} | {refusals} | {hot} | {mean} | {xstest} |".format(
                name=row["name"],
                compliance=_rate(row.get("compliance")),
                refusals=_rate(row["refusals"]),
                hot=_rate(row["hot"]),
                mean=_score(row["mean"]),
                xstest=_rate(row.get("xstest_refusals")),
            )
        )
    quality_rows = []
    for row in context["quality"]:
        quality_rows.append(
            f"| {row['name']} | {row.get('gsm8k', '–')} | {row.get('mmlu', '–')} |"
        )
    fields = {key: value for key, value in context.items() if key not in ("rows", "quality")}
    return TEMPLATE.format(
        rows="\n".join(rows),
        quality_rows="\n".join(quality_rows) or "| not measured | – | – |",
        **fields,
    )


def publish(folder: str, repo_id: str, card: str, private: bool = False) -> str:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
    Path(folder, "README.md").write_text(card)
    api.upload_folder(repo_id=repo_id, folder_path=folder, repo_type="model")
    return f"https://huggingface.co/{repo_id}"


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())
