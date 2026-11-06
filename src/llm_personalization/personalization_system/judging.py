from __future__ import annotations

from llm_personalization.judge.judge import AttributeJudge, avoid_score

from .methods.score_utils import weighted_mean


class WeightedAttributeJudge:
    """Scores responses against a user's weighted target attributes:
    the weighted mean of the judge's ratings, with `avoid` targets scored via
    `avoid_score`. One batched judge call across all (user, target) pairs."""

    def __init__(
        self,
        attribute_judge: AttributeJudge,
        user_id_to_weighted_attrs: dict[str, list[dict]],
    ):
        self.attribute_judge = attribute_judge
        self.user_id_to_weighted_attrs = user_id_to_weighted_attrs

    def load(self) -> None:
        self.attribute_judge.load()

    def unload(self) -> None:
        self.attribute_judge.unload()

    def judge(
        self,
        user_ids: list[str],
        conversations: list[list[dict[str, str]]],
    ) -> list[float]:
        judge_attributes: list[str] = []
        judge_requests: list[list[dict[str, str]]] = []

        for user_id, conversation in zip(user_ids, conversations):
            if user_id not in self.user_id_to_weighted_attrs:
                raise ValueError(f"Invalid user ID: {user_id}")
            for target in self.user_id_to_weighted_attrs[user_id]:
                judge_attributes.append(target["attribute"])
                judge_requests.append(conversation)

        raw_scores = self.attribute_judge.judge_response_attribute(judge_requests, judge_attributes)

        raw_iter = iter(raw_scores)
        out: list[float] = []
        for user_id in user_ids:
            pairs = []
            for target in self.user_id_to_weighted_attrs[user_id]:
                raw = next(raw_iter)
                pairs.append((target["weight"], avoid_score(raw) if target["side"] == "avoid" else raw))
            out.append(weighted_mean(pairs))
        return out
