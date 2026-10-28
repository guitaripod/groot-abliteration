"""Refusal-direction extraction from residuals at the first generated token.

Directions are computed per layer as the normalized difference between the mean
residual for harmful prompts and the mean residual for harmless prompts. A PCA
base over the same residuals gives the extra directions needed when a single
direction is not enough (models 8B and above usually need more than one).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


def load_model(path: str, device: str = "cuda", dtype=torch.bfloat16):
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, device_map={"": device})
    model.eval()
    return model


def chat_ids(tokenizer, prompt: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True,
        return_dict=False,
    )


@torch.no_grad()
def residuals(model, tokenizer, prompts: list[str], batch_size: int = 8, max_tokens: int = 512) -> torch.Tensor:
    layers = model.model.language_model.layers
    captured: dict[int, list[torch.Tensor]] = {index: [] for index in range(len(layers))}
    handles = []

    def hook(index: int):
        def record(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[index].append(hidden[:, -1, :].to(torch.float32).cpu())
        return record

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(hook(index)))

    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            sequences = [chat_ids(tokenizer, prompt)[:max_tokens] for prompt in batch]
            width = max(len(sequence) for sequence in sequences)
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            input_ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=model.device)
            attention = torch.zeros_like(input_ids)
            for row, sequence in enumerate(sequences):
                input_ids[row, : len(sequence)] = torch.tensor(sequence, device=model.device)
                attention[row, : len(sequence)] = 1
            model(input_ids=input_ids, attention_mask=attention)
    finally:
        for handle in handles:
            handle.remove()

    return torch.stack([torch.cat(captured[index]).mean(dim=0) for index in range(len(layers))])


def difference_of_means(harmful: torch.Tensor, harmless: torch.Tensor) -> torch.Tensor:
    delta = harmful - harmless
    return delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def principal_directions(harmful: torch.Tensor, harmless: torch.Tensor, top_k: int = 4) -> torch.Tensor:
    base = harmless.mean(dim=0)
    deltas = harmful - base
    _, _, basis = torch.pca_lowrank(deltas, q=min(top_k, min(deltas.shape) - 1))
    return basis.transpose(0, 1)
