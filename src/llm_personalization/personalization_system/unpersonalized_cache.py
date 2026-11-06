"""Loader for the unpersonalized response + per-attribute rating caches.
Both are world-independent, so caching once amortizes across all
(world x method) runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm_personalization.utils.io import iter_jsonl

from .methods.score_utils import weighted_mean


@dataclass
class UnpersonalizedUser:
    user_id: str
    response: str
    scores: dict[tuple[str, str], float]  # (attribute, side) -> score


class UnpersonalizedCache:
    """Loads the unpers response + per-attribute rating caches and joins on user_id."""

    def __init__(self, responses_path: Path | str, ratings_path: Path | str):
        responses_path = Path(responses_path)
        ratings_path = Path(ratings_path)

        if not responses_path.exists():
            raise FileNotFoundError(
                f"Unpersonalized response cache not found: {responses_path}\n"
                f"Build it with experiments/unpersonalized_responses."
            )
        if not ratings_path.exists():
            raise FileNotFoundError(
                f"Unpersonalized rating cache not found: {ratings_path}\n"
                f"Build it with experiments/unpersonalized_response_judging."
            )

        responses: dict[str, str] = {}
        for rec in iter_jsonl(responses_path):
            responses[str(rec["user_id"])] = rec["response"]

        users: dict[str, UnpersonalizedUser] = {}
        for rec in iter_jsonl(ratings_path):
            uid = str(rec["user_id"])
            if uid not in responses:
                continue
            users[uid] = UnpersonalizedUser(
                user_id=uid,
                response=responses[uid],
                scores={(s["attribute"], s["side"]): float(s["score"])
                        for s in rec["scores"] if s.get("score") is not None},
            )
        self._users = users
        print(
            f"[UnpersonalizedCache] loaded {len(users)} users "
            f"(responses={responses_path}, ratings={ratings_path})"
        )

    def __contains__(self, user_id: str) -> bool:
        return str(user_id) in self._users

    def __len__(self) -> int:
        return len(self._users)

    def get(self, user_id: str) -> UnpersonalizedUser | None:
        return self._users.get(str(user_id))

    def weighted_score(self, user_id: str, weighted_targets: list[dict]) -> float:
        """Same aggregation as WeightedAttributeJudge: sum w * s / sum |w|.
        The cache already stores the avoid side as 11 - follow, so no flip here."""
        user = self._users.get(str(user_id))
        if user is None:
            return float("nan")
        return weighted_mean((t["weight"], user.scores.get((t["attribute"], t["side"]))) for t in weighted_targets)
