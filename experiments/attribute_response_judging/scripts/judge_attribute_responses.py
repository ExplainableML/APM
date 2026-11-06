"""Judge each candidate response (from `experiments/attribute_responses`)
against every response attribute. One judge call per (response, attribute);
the avoid-side score is derived as `11 - follow_score` and marked
`"derived": true` in the materialized output. Slurm-array friendly."""

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

    output_dir = project_root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    attributes = list(cfg.attributes)
    task_id = int(cfg.array_task_id)
    task_count = int(cfg.array_task_count)

    # Load all input files in order, tagging each user with its split name.
    tagged_records: list[tuple[str, dict]] = []
    for entry in cfg.inputs:
        in_path = project_root / entry.path
        split_name = entry.split_name
        users = read_jsonl(in_path)
        print(
            f"[judge_attribute_responses] Loaded {len(users)} users from "
            f"{in_path} (split={split_name})",
            flush=True,
        )
        tagged_records.extend((split_name, r) for r in users)
    print(
        f"[judge_attribute_responses]   {len(tagged_records)} users total "
        f"across {len(cfg.inputs)} input file(s)",
        flush=True,
    )

    sliced = slice_for_task(tagged_records, task_id, task_count)
    if cfg.get("limit") is not None:
        sliced = sliced[: int(cfg.limit)]

    out_path = output_dir / f"ratings_part{task_id:04d}_of{task_count:04d}.jsonl"
    print(
        f"[judge_attribute_responses] Task {task_id}/{task_count}: "
        f"{len(sliced)} users -> {out_path}",
        flush=True,
    )

    # Flat list of judge calls; per-user offsets map scores back into shape.
    judge_conversations: list[list[dict[str, str]]] = []
    judge_attributes: list[str] = []
    user_offsets: list[int] = []
    user_response_counts: list[int] = []  # parallel to sliced

    for split_name, record in sliced:
        user_offsets.append(len(judge_conversations))
        responses = record.get("responses", [])
        user_response_counts.append(len(responses))
        for resp in responses:
            conv = list(record["current_messages"]) + [
                {"role": "assistant", "content": resp["response"]}
            ]
            for attribute_name in attributes:
                judge_conversations.append(conv)
                judge_attributes.append(attribute_name)

    total_calls = len(judge_conversations)
    print(
        f"[judge_attribute_responses] Total judge calls: {total_calls} "
        f"(users * responses * attributes)",
        flush=True,
    )

    judge: ParsedRatingJudge = instantiate(cfg.judge)
    judge.load()
    try:
        scores = judge.judge_response_attribute(judge_conversations, judge_attributes)
    finally:
        judge.unload()

    print(f"[judge_attribute_responses] Writing results to {out_path}", flush=True)
    n_attrs = len(attributes)
    none_count = sum(1 for s in scores if s is None)
    if none_count > 0:
        print(
            f"[judge_attribute_responses] WARNING: {none_count}/{total_calls} "
            f"judge calls failed to parse a score (returned None).",
            flush=True,
        )

    with open(out_path, "w") as f:
        for (split_name, record), user_off, n_responses in tqdm(
            list(zip(sliced, user_offsets, user_response_counts)),
            desc="writing",
        ):
            ratings_out = []
            for r_i, resp in enumerate(record.get("responses", [])):
                resp_off = user_off + r_i * n_attrs
                materialized = []
                for a_i, attribute_name in enumerate(attributes):
                    follow_score = scores[resp_off + a_i]
                    derived_avoid_score = avoid_score(follow_score)
                    materialized.append(
                        {
                            "attribute": attribute_name,
                            "side": "follow",
                            "score": follow_score,
                            "derived": False,
                        }
                    )
                    materialized.append(
                        {
                            "attribute": attribute_name,
                            "side": "avoid",
                            "score": derived_avoid_score,
                            "derived": True,
                        }
                    )
                ratings_out.append(
                    {
                        "gen_attribute": resp.get("attribute"),
                        "gen_side": resp.get("side"),
                        "response": resp.get("response"),
                        "scores": materialized,
                    }
                )

            out_record = {
                "user_id": record.get("user_id"),
                "split": split_name,
                "gt_user_attributes": record.get("gt_user_attributes", []),
                "current_messages": record.get("current_messages", []),
                "ratings": ratings_out,
            }
            f.write(json.dumps(out_record) + "\n")

    print(
        f"[judge_attribute_responses] Done. Wrote {len(sliced)} users "
        f"({total_calls} judge calls, {none_count} unparsed) to {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
