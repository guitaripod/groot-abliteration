"""Command line for the abliteration pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from . import ablate, directions, evaluate, judge as judge_module, prompts as prompts_module
from transformers import AutoTokenizer


def parse_layers(spec: str | None, total: int) -> list[int]:
    if not spec:
        return list(range(total))
    start, _, end = spec.partition(":")
    return list(range(int(start or 0), int(end or total)))


def cmd_prompts(args) -> None:
    summary = prompts_module.build(args.model, args.out, seed=args.seed)
    print(json.dumps(summary, indent=1))


def cmd_extract(args) -> None:
    payload = json.loads(Path(args.prompts).read_text())
    model = directions.load_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    harmful = directions.residuals(model, tokenizer, payload["train"]["harmful"], batch_size=args.batch_size)
    harmless = directions.residuals(model, tokenizer, payload["train"]["harmless"], batch_size=args.batch_size)
    unit = directions.difference_of_means(harmful, harmless)
    extra = directions.principal_directions(harmful, harmless, top_k=args.top_k)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"difference_of_means": unit, "principal": extra}, path)
    print(f"directions: {tuple(unit.shape)} -> {path}")


def cmd_ablate(args) -> None:
    payload = torch.load(args.directions, map_location="cpu")
    stack = payload["difference_of_means"]
    model, tokenizer = evaluate.load_local(args.model)
    total = len(model.model.language_model.layers)
    layers = parse_layers(args.layers, total)
    strengths = {"attn": args.attn, "shared": args.shared, "experts": args.experts}
    ablate.apply(model, stack, strengths, layers=layers, preserve_norm=not args.no_preserve_norm)
    ablate.save(model, tokenizer, args.out)
    print(f"saved {args.out} with {strengths} over layers {layers[0]}..{layers[-1]}")


def cmd_eval(args) -> None:
    prompt_file = json.loads(Path(args.prompts).read_text())
    items = prompt_file[args.split][args.kind]
    if args.endpoint:
        responses = evaluate.generate_endpoint(items, args.endpoint, args.served_model or args.model,
                                               max_new_tokens=args.max_new_tokens)
    else:
        model, tokenizer = evaluate.load_local(args.model)
        responses = evaluate.generate_local(model, tokenizer, items, max_new_tokens=args.max_new_tokens)
    if args.skip_judge:
        report = {"rows": [{"prompt": prompt, "response": response} for prompt, response in zip(items, responses)]}
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=1))
        print(f"generated {len(responses)} responses -> {path}")
        return
    report = evaluate.evaluate(
        responses, items, args.judge_url, args.judge_model, os.environ.get("JUDGE_API_KEY")
    )
    report["model"] = args.model
    report["split"] = f"{args.split}/{args.kind}"
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["summary"], indent=1))


def cmd_score(args) -> None:
    report = json.loads(Path(args.report).read_text())
    rows = []
    for row in report["rows"]:
        verdict = judge_module.score(row["prompt"], evaluate.answer_only(row["response"]), args.judge_url, args.judge_model,
                                       os.environ.get("JUDGE_API_KEY"))
        rows.append({**row, "judge": verdict})
    summary = judge_module.summarise(rows)
    summary["string_refusal_rate"] = round(
        sum(evaluate.string_refusal(row["response"]) for row in rows) / max(1, len(rows)), 3
    )
    report["rows"] = rows
    report["summary"] = summary
    Path(args.report).write_text(json.dumps(report, indent=1))
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Abliterate Groot 19B-A3B")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prompts_parser = subparsers.add_parser("prompts")
    prompts_parser.add_argument("--model", required=True)
    prompts_parser.add_argument("--out", default="data")
    prompts_parser.add_argument("--seed", type=int, default=0)
    prompts_parser.set_defaults(func=cmd_prompts)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--model", required=True)
    extract_parser.add_argument("--prompts", default="data/prompts.json")
    extract_parser.add_argument("--out", default="work/directions.pt")
    extract_parser.add_argument("--batch-size", type=int, default=8)
    extract_parser.add_argument("--top-k", type=int, default=4)
    extract_parser.set_defaults(func=cmd_extract)

    ablate_parser = subparsers.add_parser("ablate")
    ablate_parser.add_argument("--model", required=True)
    ablate_parser.add_argument("--directions", default="work/directions.pt")
    ablate_parser.add_argument("--out", required=True)
    ablate_parser.add_argument("--attn", type=float, default=0.0)
    ablate_parser.add_argument("--shared", type=float, default=0.0)
    ablate_parser.add_argument("--experts", type=float, default=0.0)
    ablate_parser.add_argument("--layers", default=None)
    ablate_parser.add_argument("--no-preserve-norm", action="store_true")
    ablate_parser.set_defaults(func=cmd_ablate)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--model", required=True)
    eval_parser.add_argument("--prompts", default="data/prompts.json")
    eval_parser.add_argument("--split", default="eval")
    eval_parser.add_argument("--kind", default="harmful")
    eval_parser.add_argument("--out", required=True)
    eval_parser.add_argument("--max-new-tokens", type=int, default=512)
    eval_parser.add_argument("--skip-judge", action="store_true")
    eval_parser.add_argument("--endpoint", default=os.environ.get("EVAL_ENDPOINT"))
    eval_parser.add_argument("--served-model", default=os.environ.get("EVAL_SERVED_MODEL"))
    eval_parser.add_argument("--judge-url", default=os.environ.get("JUDGE_URL", "http://127.0.0.1:8081/v1"))
    eval_parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "nemotron-3.5-super-vl-120b-nvfp4"))
    eval_parser.set_defaults(func=cmd_eval)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--report", required=True)
    score_parser.add_argument("--judge-url", default=os.environ.get("JUDGE_URL", "http://127.0.0.1:8081/v1"))
    score_parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "nemotron-3.5-super-vl-120b-nvfp4"))
    score_parser.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
