"""Generate one candidate response per (attribute, side) for every user prompt
in a synthetic-conversations JSONL. Slurm-array friendly: contiguous slices by
(array_task_id, array_task_count), one output JSONL per task."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from llm_personalization.llm.llm_helper import LLMHelper
from llm_personalization.personalization_system.prompts import (
    format_system_prompt as _format_system_prompt,
)
from llm_personalization.utils.io import read_jsonl, slice_for_task


def _extract_prompt_messages(record: dict) -> list[dict[str, str]]:
    """Only the first user turn of the held-out (last) conversation."""
    last_conversation = record["conversations"][-1]["messages"]
    return last_conversation[:1]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(get_original_cwd())

    input_path = project_root / cfg.input_path
    output_dir = project_root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    attributes = list(cfg.attributes)
    sides = list(cfg.sides)
    task_id = int(cfg.array_task_id)
    task_count = int(cfg.array_task_count)

    print(f"[attribute_responses] Loading users from {input_path}", flush=True)
    all_records = read_jsonl(input_path)
    print(f"[attribute_responses]   {len(all_records)} users total", flush=True)

    if cfg.get("input_limit") is not None:
        all_records = all_records[: int(cfg.input_limit)]
        print(
            f"[attribute_responses]   capped to {len(all_records)} users "
            f"(input_limit={int(cfg.input_limit)})",
            flush=True,
        )

    records = slice_for_task(all_records, task_id, task_count)
    if cfg.get("limit") is not None:
        records = records[: int(cfg.limit)]
    print(
        f"[attribute_responses] Task {task_id}/{task_count}: "
        f"processing {len(records)} users "
        f"x {len(attributes)} attributes x {len(sides)} sides "
        f"= {len(records) * len(attributes) * len(sides)} generations",
        flush=True,
    )

    out_path = (
        output_dir
        / f"{cfg.split_name}_part{task_id:04d}_of{task_count:04d}.jsonl"
    )
    print(f"[attribute_responses] Writing to {out_path}", flush=True)

    # One flat prompt list (single batched vLLM call); per-user (start, length)
    # ranges map responses back.
    prompts: list[list[dict[str, str]]] = []
    user_ranges: list[tuple[int, int]] = []
    user_combos: list[list[tuple[str, str]]] = []
    user_messages: list[list[dict[str, str]]] = []
    user_ids: list[str] = []
    user_gt_attrs: list[list[dict[str, str]]] = []

    for record in records:
        current_messages = _extract_prompt_messages(record)
        combos: list[tuple[str, str]] = []
        start = len(prompts)
        for attribute in attributes:
            for side in sides:
                prompts.append(
                    [{"role": "system", "content": _format_system_prompt(attribute, side)}]
                    + current_messages
                )
                combos.append((attribute, side))
        user_ranges.append((start, len(prompts) - start))
        user_combos.append(combos)
        user_messages.append(current_messages)
        user_ids.append(str(record["user_idx"]))
        user_gt_attrs.append(record.get("rewrite_style_attributes", []))

    llm: LLMHelper = instantiate(cfg.llm)
    llm.load()
    try:
        print(f"[attribute_responses] Generating {len(prompts)} responses...", flush=True)
        responses = llm.generate(prompts)
    finally:
        llm.unload()

    print(f"[attribute_responses] Writing results to {out_path}", flush=True)
    with open(out_path, "w") as f:
        for user_id, current_messages, gt_attrs, (start, length), combos in tqdm(
            list(zip(user_ids, user_messages, user_gt_attrs, user_ranges, user_combos)),
            desc="writing",
        ):
            user_responses = []
            for offset, (attribute, side) in enumerate(combos):
                resp = responses[start + offset]
                user_responses.append(
                    {
                        "attribute": attribute,
                        "side": side,
                        "response": resp.content,
                        "finish_reason_stop": resp.finish_reason_stop,
                    }
                )
            record_out = {
                "user_id": user_id,
                "gt_user_attributes": gt_attrs,
                "current_messages": current_messages,
                "responses": user_responses,
            }
            f.write(json.dumps(record_out) + "\n")

    print(
        f"[attribute_responses] Done. Wrote {len(records)} users to {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
