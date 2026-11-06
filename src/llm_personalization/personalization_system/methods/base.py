from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class PersonalizationSystem(ABC):
    """Methods consume `user_id_to_candidate_scores`: per user, a
    `dict[(attr, side), float]` of user-fit scores for the cached candidates."""

    @abstractmethod
    def train(
        self,
        dataset,
        user_id_to_candidate_scores: dict[str, dict[tuple[str, str], float]],
        save_path: Path,
        val_dataset=None,
        val_user_id_to_candidate_scores: dict[str, dict[tuple[str, str], float]] | None = None,
        gt_for_debug: dict[str, list[dict]] | None = None,
        val_gt_for_debug: dict[str, list[dict]] | None = None,
    ) -> None:
        ...

    @abstractmethod
    def evaluate(
        self,
        dataset,
        load_path: Path,
        user_id_to_candidate_scores: dict[str, dict[tuple[str, str], float]] | None = None,
        gt_for_debug: dict[str, list[dict]] | None = None,
    ) -> list[str]:
        """Returns one response string per user (in dataset order).
        `user_id_to_candidate_scores` is only required by the oracle;
        `gt_for_debug` is only used for accuracy logging."""
        ...
