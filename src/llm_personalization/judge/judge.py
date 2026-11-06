from abc import ABC, abstractmethod


type Messages = list[dict[str, str]]

# Ratings are the judge's 1-100 score divided by 10. The `avoid` side of an
# attribute is not judged separately but derived from the `follow` rating.
SCORE_MIDPOINT = 5.5


def avoid_score(follow_score: float | None) -> float | None:
    return None if follow_score is None else 11.0 - float(follow_score)


class AttributeJudge(ABC):
    @abstractmethod
    def judge_response_attribute(self, conversations: list[Messages], attributes: list[str]) -> list[float | None]:  # scores on 1-10 scale
        """
        Judge the compliance of a list of the last message in a conversation with a given attribute.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass
