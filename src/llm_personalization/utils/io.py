"""JSONL I/O and slurm-array slicing shared by the pipeline stages."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def iter_jsonl(path: Path | str) -> Iterator[dict]:
    """Records of a .jsonl file, or of every *.jsonl in a directory (sorted)."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    for file in files:
        with open(file) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def read_jsonl(path: Path | str) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: Path | str, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"saved {len(records)} records to {path}")


def slice_for_task(records: list, task_id: int, task_count: int) -> list:
    """Contiguous slice for one array task, so concatenating the per-task
    outputs in task order preserves the input order."""
    if task_count <= 0:
        raise ValueError(f"array_task_count must be >= 1, got {task_count}")
    if not 0 <= task_id < task_count:
        raise ValueError(f"array_task_id={task_id} out of range for task_count={task_count}")
    n = len(records)
    return records[(n * task_id) // task_count : (n * (task_id + 1)) // task_count]
