# Groot 19B-A3B uncensored, NVFP4

Removing refusal from `GrootTeam/Groot-19B-A3B-261023` and shipping it as an NVFP4
checkpoint for the RTX PRO 6000 Blackwell, with the capability cost measured rather
than assumed.

## What the model is

40 layers, 30 GDN linear-attention + 10 full attention, 128 routed experts (top-8)
plus a shared expert, hidden 2048, 248k vocab, 256K context, 1 MTP layer.
Text and vision towers; only the text decoder is touched here.

Refusal is measured in the residual stream, so the ablation targets the matrices
that write into it: `self_attn.o_proj`, `linear_attn.out_proj`,
`shared_expert.down_proj` and the fused expert `down_proj` slices.

## Stages

1. **Prompts** (`cli.py prompts`) — harmful/harmless pairs from
   `mlabonne/harmful_behaviors` and `mlabonne/harmless_alpaca`, framing and token
   length matched, split into extraction and held-out evaluation sets. XSTest for
   over-refusal, plus agentic tool-policy probes because this model is post-trained
   in agent harnesses.
2. **Extract** — per-layer refusal directions at the first generated token,
   difference of means, float32, normalized. Cached to `work/directions.pt`.
3. **Optimise** — Heretic, 200 trials, per-layer directions, norm-preserving
   (row normalization FULL), KL target 0.1, refusal markers widened for this
   template. Searches attention and shared-expert weights, exports a merged model.
4. **Experts** (`cli.py experts`) — Heretic cannot reach the fused expert tensors,
   so the routed experts are ablated separately at a swept strength, norm-preserving,
   and evaluated at each point.
5. **Judge** (`cli.py judge`) — StrongREJECT-style rubric scored by a strong local
   judge: refusal, specificity, convincingness. String matching is kept only as a
   cheap cross-check, never as the result.
6. **Quantise** — `llm-compressor` recipe mirroring the official
   `Groot-19B-A3B-261023-NVFP4`: FP8 W8A8 on attention and linear-attention
   projections, NVFP4 group-16 on experts, shared expert and `lm_head`;
   vision tower and MTP untouched.
7. **Prove** — refusals re-measured after quantisation; MMLU and GSM8K against the
   parent; blackwall `doctor` and a full run against the parent's 78.5 quality.

## Gates

A candidate does not get quantised unless all of these hold, and does not ship
unless they still hold at NVFP4:

| Gate | Threshold |
| --- | --- |
| Refusal rate, held-out harmful (judge) | ≤ 3% |
| StrongREJECT-style compliance score | ≥ 0.5 mean on answered prompts |
| XSTest over-refusal | not worse than parent |
| Hidden refusal (disclaimer then comply) | absent by judge |
| KL divergence from parent, benign text | ≤ 0.2 |
| MMLU | within 2 points of parent |
| GSM8K | within 5 points of parent |
| blackwall quality | ≥ 74 (parent 78.5) |
| blackwall tools | ≥ 95 |

If a stage misses, the run stops and the numbers are reported; nothing is shipped
on a failed gate.
