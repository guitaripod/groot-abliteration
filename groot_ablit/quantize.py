"""Re-quantise an abliterated checkpoint to the official NVFP4 recipe.

The official `Groot-19B-A3B-261023-NVFP4` config is a compressed-tensors
mixed-precision plan: FP8 W8A8 on attention and linear-attention projections,
NVFP4 group-16 weights on the experts, shared expert and lm_head, vision tower
untouched. Its `config_groups` are reused verbatim so the abliterated build lands
in the same kernels and the same speed profile as the parent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

REFERENCE = os.environ.get(
    "QUANT_REFERENCE",
    "/mnt/nvme8tb/hf/hub/models--GrootTeam--Groot-19B-A3B-261023-NVFP4/snapshots/"
    "8ff73aca9300f4e9835972f0012ef5edbb016a8b/config.json",
)
CALIBRATION_DATASET = os.environ.get("CALIBRATION_DATASET", "tatsu-lab/alpaca")
CALIBRATION_SAMPLES = int(os.environ.get("CALIBRATION_SAMPLES", "512"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("CALIBRATION_MAX_LEN", "2048"))


def recipe() -> dict:
    reference = json.loads(Path(REFERENCE).read_text())["quantization_config"]
    groups = {}
    for name, group in reference["config_groups"].items():
        groups[name] = {
            "targets": group["targets"],
            "weights": group["weights"],
            "input_activations": group.get("input_activations"),
            "output_activations": group.get("output_activations"),
        }
    return {
        "config_groups": groups,
        "format": reference.get("format", "mixed-precision"),
        "ignore": reference.get("ignore", []),
    }


def quantize(model_path: str, out_path: str, num_samples: int = CALIBRATION_SAMPLES) -> None:
    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    plan = recipe()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")

    dataset = load_dataset(CALIBRATION_DATASET, split=f"train[:{num_samples}]")
    text_column = next(column for column in ("text", "prompt", "instruction") if column in dataset.column_names)
    dataset = dataset.map(
        lambda row: {"text": tokenizer.apply_chat_template(
            [{"role": "user", "content": str(row[text_column])[:4000]}],
            add_generation_prompt=True, tokenize=False,
        )},
        remove_columns=dataset.column_names,
    )

    modifier = QuantizationModifier(
        config_groups=plan["config_groups"],
        ignore=plan["ignore"],
    )
    oneshot(
        model=model,
        dataset=dataset,
        recipe=modifier,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=num_samples,
    )

    model.save_pretrained(out_path, max_shard_size="5GB")
    tokenizer.save_pretrained(out_path)
    try:
        AutoProcessor.from_pretrained(model_path, trust_remote_code=True).save_pretrained(out_path)
    except Exception:
        pass
    print(f"quantised -> {out_path}")


if __name__ == "__main__":
    import sys

    quantize(sys.argv[1], sys.argv[2])
