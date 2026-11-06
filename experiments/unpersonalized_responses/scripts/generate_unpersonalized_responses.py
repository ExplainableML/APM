"""Generate one unpersonalized response per user (no system prompt), read from
an attribute_responses JSONL. Slurm-array friendly: contiguous slices, one
output JSONL per task."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from llm_personalization.llm.llm_helper import LLMHelper
from llm_personalization.utils.io import read_jsonl, slice_for_task


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(get_original_cwd())

    input_path = project_root / cfg.input_path
    output_dir = project_root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    task_id = int(cfg.array_task_id)
    task_count = int(cfg.array_task_count)

    print(f"[unpers_responses] Loading users from {input_path}", flush=True)
    all_records = read_jsonl(input_path)
    print(f"[unpers_responses]   {len(all_records)} users total", flush=True)

    if cfg.get("input_limit") is not None:
        all_records = all_records[: int(cfg.input_limit)]
    records = slice_for_task(all_records, task_id, task_count)
    if cfg.get("limit") is not None:
        records = records[: int(cfg.limit)]

    out_path = output_dir / f"part{task_id:04d}_of{task_count:04d}.jsonl"
    print(
        f"[unpers_responses] Task {task_id}/{task_count}: {len(records)} users -> {out_path}",
        flush=True,
    )

    prompts = [list(r["current_messages"]) for r in records]
    user_ids = [str(r["user_id"]) for r in records]

    llm: LLMHelper = instantiate(cfg.llm)
    llm.load()
    try:
        responses = llm.generate(prompts)
    finally:
        llm.unload()

    print(f"[unpers_responses] Writing {len(records)} records to {out_path}", flush=True)
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(temp_path, "w") as f:
        for uid, msgs, resp in tqdm(list(zip(user_ids, prompts, responses)), desc="writing"):
            f.write(json.dumps({
                "user_id": uid,
                "current_messages": msgs,
                "response": resp.content,
                "finish_reason_stop": resp.finish_reason_stop,
            }) + "\n")
    temp_path.replace(out_path)

    print(f"[unpers_responses] Done.", flush=True)


if __name__ == "__main__":
    main()
