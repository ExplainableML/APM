"""Turn raw cached ratings into per-user candidate scores.

Methods consume a uniform `user_id_to_candidate_scores: dict[user_id, dict[(attr, side), float]]`,
mapping each cached `(gen_attr, gen_side)` candidate to its user-fit score.
"""
from __future__ import annotations

import math
from typing import Iterable


def weighted_mean(weighted_scores: Iterable[tuple[float, float | None]]) -> float:
    """sum(w * s) / sum(|w|) over the (weight, score) pairs whose score is not
    None; NaN if there are none. This is how a response is scored against a
    user's weighted target attributes everywhere in the benchmark."""
    num, denom = 0.0, 0.0
    for w, s in weighted_scores:
        if s is None:
            continue
        num += float(w) * s
        denom += abs(float(w))
    return num / denom if denom > 0 else math.nan


def attribute_candidate_scores(user, weighted_targets) -> dict[tuple[str, str], float]:
    """User-fit score of each cached candidate: the weighted mean of its cached
    ratings on the user's target attributes.
    `weighted_targets`: list of {"attribute", "side", "weight"} with weight >= 0."""
    out: dict[tuple[str, str], float] = {}
    for (a, s) in user.responses.keys():
        score = weighted_mean(
            (t["weight"], user.ratings.get((a, s, t["attribute"], t["side"]))) for t in weighted_targets
        )
        if not math.isnan(score):
            out[(a, s)] = score
    return out
