"""Judge each unpersonalized response against every response attribute.
One row per user with flat `scores: [{attribute, side, score, derived}, ...]`
of size 2K (avoid side derived as 11-x). Slurm-array friendly."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from llm_personalization.judge.judge import avoid_score
from llm_personalization.judge.parsed_rating_judge import ParsedRatingJudge
from llm_personalization.utils.io import read_jsonl, slice_for_task


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(get_original_cwd())

    input_path = project_root / cfg.input_path
    output_dir = project_root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    attributes = list(cfg.attributes)
    task_id = int(cfg.array_task_id)
    task_count = int(cfg.array_task_count)

    print(f"[judge_unpers] Loading from {input_path}", flush=True)
    all_records = read_jsonl(input_path)
    print(f"[judge_unpers]   {len(all_records)} users total", flush=True)

    records = slice_for_task(all_records, task_id, task_count)
    if cfg.get("limit") is not None:
        records = records[: int(cfg.limit)]

    out_path = output_dir / f"part{task_id:04d}_of{task_count:04d}.jsonl"
    print(f"[judge_unpers] Task {task_id}/{task_count}: {len(records)} users -> {out_path}", flush=True)

    # One judge call per (user, attribute_name); avoid side derived as 11-x.
    judge_conversations: list[list[dict[str, str]]] = []
    judge_attributes: list[str] = []
    for rec in records:
        conv = list(rec["current_messages"]) + [{"role": "assistant", "content": rec["response"]}]
        for attr in attributes:
            judge_conversations.append(conv)
            judge_attributes.append(attr)

    total_calls = len(judge_conversations)
    print(f"[judge_unpers] Total judge calls: {total_calls} (users * attributes)", flush=True)

    judge: ParsedRatingJudge = instantiate(cfg.judge)
    judge.load()
    try:
        scores = judge.judge_response_attribute(judge_conversations, judge_attributes)
    finally:
        judge.unload()

    n_attrs = len(attributes)
    none_count = sum(1 for s in scores if s is None)
    if none_count:
        print(f"[judge_unpers] WARNING: {none_count}/{total_calls} judge calls returned None", flush=True)

    print(f"[judge_unpers] Writing to {out_path}", flush=True)
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(temp_path, "w") as f:
        for u_i, rec in enumerate(tqdm(records, desc="writing")):
            base = u_i * n_attrs
            materialized = []
            for a_i, attr in enumerate(attributes):
                follow = scores[base + a_i]
                avoid = avoid_score(follow)
                materialized.append({"attribute": attr, "side": "follow", "score": follow, "derived": False})
                materialized.append({"attribute": attr, "side": "avoid", "score": avoid, "derived": True})
            f.write(json.dumps({
                "user_id": rec["user_id"],
                "current_messages": rec["current_messages"],
                "response": rec["response"],
                "scores": materialized,
            }) + "\n")
    temp_path.replace(out_path)

    print(f"[judge_unpers] Done. {len(records)} users, {total_calls} calls, {none_count} unparsed.", flush=True)


if __name__ == "__main__":
    main()
