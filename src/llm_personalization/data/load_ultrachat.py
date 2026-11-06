from typing import Literal
from datasets import load_dataset
import random


def load_ultrachat_prompts(
    split: Literal["train_sft", "test_sft"],
    prefixes: tuple[str, ...] | None = None,
    limit: int | None = None,
    seed: int | None = None,
) -> list[str]:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)

    prompts = [
        row["prompt"] for row in ds
        if prefixes is None or row["prompt_id"].startswith(prefixes)
    ]

    if seed is not None:
        random.Random(seed).shuffle(prompts)

    return prompts[:limit] if limit else prompts


def load_ultrachat_prompt_response_pairs(
    split: Literal["train_sft", "test_sft"],
    prefixes: tuple[str, ...] | None = None,
    limit: int | None = None,
    seed: int | None = None,
) -> list[tuple[str, str]]:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)

    pairs = []
    for row in ds:
        if prefixes is not None and not row["prompt_id"].startswith(prefixes):
            continue
        messages = row["messages"]
        if len(messages) < 2:
            continue
        if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
            continue
        pairs.append((messages[0]["content"], messages[1]["content"]))

    if seed is not None:
        random.Random(seed).shuffle(pairs)

    return pairs[:limit] if limit else pairs

def load_ultrachat_conversations(
    split: Literal["train_sft", "test_sft"],
    prefixes: tuple[str, ...] | None = None,
    limit: int | None = None,
    seed: int | None = None,
) -> list[list[dict[str, str]]]:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)

    conversations = []
    for row in ds:
        if prefixes is not None and not row["prompt_id"].startswith(prefixes):
            continue
        messages = row["messages"]
        if len(messages) < 2:
            continue
        if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
            continue
        conversations.append(messages)

    if seed is not None:
        random.Random(seed).shuffle(conversations)

    return conversations[:limit] if limit else conversations


def load_ultrachat_conversations_with_ids(
    split: Literal["train_sft", "test_sft"],
    prefixes: tuple[str, ...] | None = None,
    limit: int | None = None,
    seed: int | None = None,
) -> list[tuple[int, list[dict[str, str]]]]:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)

    conversations_with_ids = []
    for row in ds:
        if prefixes is not None and not row["prompt_id"].startswith(prefixes):
            continue
        messages = row["messages"]
        if len(messages) < 2:
            continue
        if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
            continue
        conversations_with_ids.append((row["prompt_id"], messages))

    if seed is not None:
        random.Random(seed).shuffle(conversations_with_ids)

    return conversations_with_ids[:limit] if limit else conversations_with_ids