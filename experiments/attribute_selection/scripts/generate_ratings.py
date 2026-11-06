import hydra
from omegaconf import DictConfig
from llm_personalization.data.load_ultrachat import load_ultrachat_conversations
from llm_personalization.judge import AttributeJudge
from hydra.utils import instantiate, get_original_cwd
import random
import numpy as np
from pathlib import Path

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    conversations = load_ultrachat_conversations(
        split="train_sft",
        limit=cfg.limit,
        seed=42,
    )
    candidate_attributes = list(cfg.candidate_attributes)

    prompt_attribute_mode = getattr(cfg, "mode", None) == "prompt_attribute"

    judge: AttributeJudge = instantiate(cfg.judge)
    judge.load()

    rng = random.Random(42)

    judge_requests_conversations = []
    judge_requests_attributes = []

    for messages in conversations:
        if len(messages) < 2:
            raise ValueError(f"Conversation has less than 2 messages: {messages}")
        number_of_messages = rng.randint(1, len(messages) // 2) * 2
        if prompt_attribute_mode:
            number_of_messages -= 1
        messages = messages[:number_of_messages]
        for attribute in candidate_attributes:
            judge_requests_conversations.append(messages)
            judge_requests_attributes.append(attribute)

    if prompt_attribute_mode:
        scores = judge.judge_user_prompt_attribute(
            conversations=judge_requests_conversations,
            attributes=judge_requests_attributes,
        )
    else:
        scores = judge.judge_response_attribute(
            conversations=judge_requests_conversations,
            attributes=judge_requests_attributes,
        )

    num_conversations = len(conversations)
    num_attributes = len(candidate_attributes)
    matrix = np.array(scores).reshape(num_conversations, num_attributes)
    out_path = Path(get_original_cwd()) / cfg.scores_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, matrix)


if __name__ == "__main__":
    main()