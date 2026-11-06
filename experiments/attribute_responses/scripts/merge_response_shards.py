"""Validate and merge one split of array-generated response JSONL shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--expected-attributes", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.input_dir / f"{args.split}.jsonl"
    temp_path = output_path.with_suffix(".jsonl.tmp")
    shard_paths = [
        args.input_dir
        / f"{args.split}_part{index:04d}_of{args.num_shards:04d}.jsonl"
        for index in range(args.num_shards)
    ]

    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing response shards:\n" + "\n".join(missing))

    count = 0
    non_stop_count = 0
    user_ids: set[str] = set()
    expected_combos = [
        (attribute, side)
        for attribute in args.expected_attributes
        for side in ("follow", "avoid")
    ]
    with temp_path.open("w") as output_file:
        for shard_path in shard_paths:
            with shard_path.open() as shard_file:
                for line_number, line in enumerate(shard_file, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {shard_path}:{line_number}"
                        ) from exc
                    responses = record.get("responses", [])
                    if len(responses) != len(expected_combos):
                        raise ValueError(
                            f"{shard_path}:{line_number} has "
                            f"{len(responses)} responses, expected {len(expected_combos)}"
                        )
                    actual_combos = [
                        (response.get("attribute"), response.get("side"))
                        for response in responses
                    ]
                    if actual_combos != expected_combos:
                        raise ValueError(
                            f"{shard_path}:{line_number} has unexpected "
                            "attribute/side ordering"
                        )
                    if any(not response.get("response") for response in responses):
                        raise ValueError(
                            f"{shard_path}:{line_number} contains an empty response"
                        )
                    non_stop_count += sum(
                        not response.get("finish_reason_stop")
                        for response in responses
                    )
                    user_id = str(record.get("user_id"))
                    if user_id in user_ids:
                        raise ValueError(
                            f"Duplicate user_id {user_id!r} in {args.split} shards"
                        )
                    user_ids.add(user_id)
                    output_file.write(line if line.endswith("\n") else line + "\n")
                    count += 1

    if count != args.expected_records:
        temp_path.unlink(missing_ok=True)
        raise ValueError(
            f"Merged {count} {args.split} records, expected {args.expected_records}"
        )

    temp_path.replace(output_path)
    print(
        f"Merged {count} records into {output_path} "
        f"({non_stop_count} responses did not finish with stop)",
        flush=True,
    )


if __name__ == "__main__":
    main()
