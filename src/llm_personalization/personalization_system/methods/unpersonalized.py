"""Unpersonalized baseline: respond to the current prompt with no system
prompt and no history; training is a no-op."""
from __future__ import annotations

from pathlib import Path

from llm_personalization.llm.llm_helper import LLMHelper
from llm_personalization.utils.gpu_monitor import log_gpu_usage

from .base import PersonalizationSystem


class UnpersonalizedSystem(PersonalizationSystem):
    def __init__(
        self,
        llm_helper_config: dict,
    ):
        self.llm_helper = LLMHelper(**llm_helper_config)

    def train(
        self,
        dataset,
        user_id_to_candidate_scores=None,
        save_path: Path = None,
        val_dataset=None,
        val_user_id_to_candidate_scores=None,
        gt_for_debug=None,
        val_gt_for_debug=None,
    ) -> None:
        Path(save_path).mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        dataset,
        load_path: Path,
        user_id_to_candidate_scores=None,
        gt_for_debug=None,
    ) -> list[str]:
        prompts = [list(u.current_messages) for u in dataset]
        log_gpu_usage("Before LLM load")
        self.llm_helper.load()
        responses = self.llm_helper.generate(prompts)
        self.llm_helper.unload()
        log_gpu_usage("After LLM unload")
        return [r.content for r in responses]
