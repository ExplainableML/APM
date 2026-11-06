"""Validate a complete set of attribute-rating shards and write a success marker."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--expected-train", type=int, required=True)
    parser.add_argument("--expected-test", type=int, required=True)
    parser.add_argument("--expected-attributes", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_paths = [
        args.input_dir / f"ratings_part{index:04d}_of{args.num_shards:04d}.jsonl"
        for index in range(args.num_shards)
    ]
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing rating shards:\n" + "\n".join(missing))

    expected_score_pairs = [
        (attribute, side)
        for attribute in args.expected_attributes
        for side in ("follow", "avoid")
    ]
    expected_gen_pairs = expected_score_pairs
    split_counts: Counter[str] = Counter()
    seen_users: set[tuple[str, str]] = set()
    none_scores = 0

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

                split = record.get("split")
                if split not in ("train", "test"):
                    raise ValueError(
                        f"{shard_path}:{line_number} has invalid split {split!r}"
                    )
                user_key = (split, str(record.get("user_id")))
                if user_key in seen_users:
                    raise ValueError(f"Duplicate user {user_key!r}")
                seen_users.add(user_key)
                split_counts[split] += 1

                ratings = record.get("ratings", [])
                actual_gen_pairs = [
                    (rating.get("gen_attribute"), rating.get("gen_side"))
                    for rating in ratings
                ]
                if actual_gen_pairs != expected_gen_pairs:
                    raise ValueError(
                        f"{shard_path}:{line_number} has unexpected generated "
                        "attribute/side pairs"
                    )
                for rating in ratings:
                    scores = rating.get("scores", [])
                    actual_score_pairs = [
                        (score.get("attribute"), score.get("side"))
                        for score in scores
                    ]
                    if actual_score_pairs != expected_score_pairs:
                        raise ValueError(
                            f"{shard_path}:{line_number} has unexpected score pairs"
                        )
                    none_scores += sum(
                        score.get("score") is None for score in scores
                    )

    expected_counts = Counter(
        train=args.expected_train,
        test=args.expected_test,
    )
    if split_counts != expected_counts:
        raise ValueError(
            f"Rating split counts are {dict(split_counts)}, "
            f"expected {dict(expected_counts)}"
        )
    if none_scores:
        raise ValueError(f"Found {none_scores} unparsed rating scores")

    success_path = args.input_dir / "_SUCCESS.json"
    temp_path = args.input_dir / "_SUCCESS.json.tmp"
    summary = {
        "status": "complete",
        "num_shards": args.num_shards,
        "split_counts": dict(split_counts),
        "attributes": args.expected_attributes,
        "unparsed_scores": none_scores,
    }
    with temp_path.open("w") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")
    temp_path.replace(success_path)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
