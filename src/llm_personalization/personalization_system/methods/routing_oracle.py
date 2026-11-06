"""Routing upper bound: at test time, pick per user the (attr, side) that
maximizes the cached candidate score. No training, no classifier — just
lookups + one batched generate."""
from __future__ import annotations

import math
from pathlib import Path

from llm_personalization.llm.llm_helper import LLMHelper
from llm_personalization.utils.gpu_monitor import log_gpu_usage

from ..prompts import format_system_prompt
from .base import PersonalizationSystem


class RoutingOracleSystem(PersonalizationSystem):
    def __init__(
        self,
        llm_helper_config: dict,
        attributes: list[str],
    ):
        self.attributes = list(attributes)
        self.llm_helper = LLMHelper(**llm_helper_config)

    def train(
        self,
        dataset,
        user_id_to_candidate_scores: dict[str, dict[tuple[str, str], float]],
        save_path: Path,
        val_dataset=None,
        val_user_id_to_candidate_scores: dict[str, dict[tuple[str, str], float]] | None = None,
        gt_for_debug=None,
        val_gt_for_debug=None,
    ) -> None:
        # No-op: oracle uses test-time scores directly.
        Path(save_path).mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        dataset,
        load_path: Path,
        user_id_to_candidate_scores: dict[str, dict[tuple[str, str], float]] | None = None,
        gt_for_debug=None,
    ) -> list[str]:
        if user_id_to_candidate_scores is None:
            raise ValueError("RoutingOracleSystem.evaluate requires user_id_to_candidate_scores")

        print("[RoutingOracleSystem] 1. Picking oracle (attr, side) per user from cache...")
        prompts = []
        fallback = 0
        for user in dataset:
            scores = user_id_to_candidate_scores.get(user.user_id, {}) or {}
            best_attr, best_side, best_score = None, None, -math.inf
            for attr in self.attributes:
                for side in ("follow", "avoid"):
                    s = scores.get((attr, side))
                    if s is None:
                        continue
                    if s > best_score:
                        best_attr, best_side, best_score = attr, side, s
            if best_attr is None:
                fallback += 1
                best_attr, best_side = self.attributes[0], "follow"
            prompts.append(
                [{"role": "system", "content": format_system_prompt(best_attr, best_side)}]
                + user.current_messages
            )
        if fallback:
            print(f"[RoutingOracleSystem] {fallback} users had no cached scores; used fallback attribute")

        print("[RoutingOracleSystem] 2. Generating responses (single batched vLLM call)...")
        log_gpu_usage("Before LLM load")
        self.llm_helper.load()
        responses = self.llm_helper.generate(prompts)
        self.llm_helper.unload()
        log_gpu_usage("After LLM unload")
        return [r.content for r in responses]
