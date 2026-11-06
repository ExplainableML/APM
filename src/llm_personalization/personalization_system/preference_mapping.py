"""User-attribute -> response-attribute mapping (the world matrix).
Weighted-attribute outputs carry a `weight` per (attribute, side); weights are
normalized so sum(|w|) = 1."""
from __future__ import annotations

from typing import Literal

import torch


def generate_world_matrix(
    type: Literal["permutation", "dense"],
    shape: tuple[int, int],
    seed: int = 42,
    k: int = 1,
) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)

    if type == "permutation":
        if shape[0] != shape[1]:
            raise ValueError(f"Permutation matrix must be square, got shape {shape}")
        n = shape[0]
        if k < 1 or k > n:
            raise ValueError(f"permutation k must satisfy 1 <= k <= n; got k={k}, n={n}")
        perms: list[torch.Tensor] = []
        while len(perms) < k:
            cand = torch.randperm(n, generator=rng)
            if all(bool((cand != p).all().item()) for p in perms):
                perms.append(cand)
        matrix = torch.zeros(n, n)
        for p in perms:
            signs = torch.randint(0, 2, (n,), generator=rng) * 2 - 1
            matrix[torch.arange(n), p] = signs.float()
        return matrix

    if type == "dense":
        return torch.randn(shape, generator=rng)

    raise ValueError(f"Unknown type: {type}")


def user_attributes_to_vector(
    user_attributes: list[dict[str, str]],
    available_user_attributes: list[str],
) -> torch.Tensor:
    vector = torch.zeros(len(available_user_attributes))
    for attribute in user_attributes:
        assert attribute["attribute"] in available_user_attributes, attribute
        assert attribute["side"] in ("follow", "avoid")
        idx = available_user_attributes.index(attribute["attribute"])
        vector[idx] = 1.0 if attribute["side"] == "follow" else -1.0
    return vector


def map_to_response_vector(
    user_attribute_vector: torch.Tensor,
    world_matrix: torch.Tensor,
) -> torch.Tensor:
    return world_matrix @ user_attribute_vector


def response_vector_to_weighted_attributes(
    response_vector: torch.Tensor,
    available_response_attributes: list[str],
    max_attributes: int | None = None,
) -> list[dict]:
    """Returns [{"attribute", "side", "weight"}, ...] with sum(|weight|) = 1
    over the kept entries (or empty list if response_vector is all-zero).
    """
    out: list[dict] = []
    if max_attributes is None:
        for i, val in enumerate(response_vector):
            v = val.item()
            if v == 0:
                continue
            out.append({
                "attribute": available_response_attributes[i],
                "side": "follow" if v > 0 else "avoid",
                "weight": abs(v),
            })
    else:
        order = torch.argsort(response_vector.abs(), descending=True)
        for i in order[:max_attributes]:
            v = response_vector[i].item()
            if v == 0:
                continue
            out.append({
                "attribute": available_response_attributes[int(i)],
                "side": "follow" if v > 0 else "avoid",
                "weight": abs(v),
            })

    total = sum(x["weight"] for x in out)
    if total > 0:
        for x in out:
            x["weight"] /= total
    return out
