"""Norm-preserving directional ablation of the matrices that write into the residual stream.

Ablation removes the component of each row that points along the refusal direction
and rescales the row back to its original norm, so the model loses the direction
without losing weight magnitude. Supports dense matrices and the fused expert
tensors this architecture uses (`down_proj` of shape [experts, hidden, ffn]).
"""

from __future__ import annotations

import torch


def orthogonalize(weight: torch.Tensor, direction: torch.Tensor, strength: float = 1.0,
                  preserve_norm: bool = True, passes: int = 1) -> torch.Tensor:
    original = weight.detach().to(torch.float32)
    direction = direction.detach().to(torch.float32).flatten()
    if weight.dim() == 3:
        return torch.stack(
            [orthogonalize(original[index], direction, strength, preserve_norm, passes) for index in range(original.shape[0])]
        )
    if direction.numel() != original.shape[0]:
        raise ValueError(f"direction {tuple(direction.shape)} does not match weight {tuple(original.shape)}")
    unit = direction / direction.norm().clamp_min(1e-8)
    ablated = original
    for _ in range(max(1, passes)):
        projected = unit @ ablated
        ablated = ablated - strength * torch.outer(unit, projected)
        if not preserve_norm:
            break
        before = original.norm(dim=1, keepdim=True).clamp_min(1e-8)
        after = ablated.norm(dim=1, keepdim=True).clamp_min(1e-8)
        ablated = ablated * (before / after)
    return ablated


def layer_weights(layer, component: str):
    if component == "attn":
        if hasattr(layer, "self_attn"):
            return [("self_attn.o_proj", layer.self_attn.o_proj)]
        return [("linear_attn.out_proj", layer.linear_attn.out_proj)]
    if component == "shared":
        return [("mlp.shared_expert.down_proj", layer.mlp.shared_expert.down_proj)]
    if component == "experts":
        experts = layer.mlp.experts
        return [("mlp.experts.down_proj", experts)]
    raise ValueError(f"unknown component {component}")


def apply(model, directions: torch.Tensor, strengths: dict[str, float], layers: list[int] | None = None,
          preserve_norm: bool = True, log=print) -> None:
    stack = model.model.language_model.layers
    targets = layers if layers is not None else list(range(len(stack)))
    for index in targets:
        layer = stack[index]
        direction = directions[index] if directions.dim() == 2 else directions
        for component, strength in strengths.items():
            if strength <= 0:
                continue
            for name, module in layer_weights(layer, component):
                parameter = getattr(module, "weight", None) if component != "experts" else module.down_proj
                if parameter is None:
                    continue
                parameter.data = orthogonalize(parameter.data, direction, strength, preserve_norm).to(parameter.dtype)
        log(f"ablated layer {index}")


def save(model, tokenizer, out_dir: str, max_shard_size: str = "5GB") -> None:
    model.save_pretrained(out_dir, max_shard_size=max_shard_size)
    tokenizer.save_pretrained(out_dir)
