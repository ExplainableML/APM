"""Validate paired unpersonalized response and rating caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--ratings-dir", type=Path, required=True)
    parser.add_argument("--num-response-shards", type=int, required=True)
    parser.add_argument("--num-rating-shards", type=int, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--expected-attributes", nargs="+", required=True)
    return parser.parse_args()


def expected_paths(directory: Path, num_shards: int) -> list[Path]:
    return [
        directory / f"part{index:04d}_of{num_shards:04d}.jsonl"
        for index in range(num_shards)
    ]


def load_records(paths: list[Path], label: str) -> dict[str, dict]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {label} shards:\n" + "\n".join(missing))

    records: dict[str, dict] = {}
    for path in paths:
        with path.open() as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
                user_id = str(record.get("user_id"))
                if user_id in records:
                    raise ValueError(f"Duplicate {label} user_id {user_id!r}")
                records[user_id] = record
    return records


def main() -> None:
    args = parse_args()
    response_paths = expected_paths(args.responses_dir, args.num_response_shards)
    rating_paths = expected_paths(args.ratings_dir, args.num_rating_shards)
    responses = load_records(response_paths, "response")
    ratings = load_records(rating_paths, "rating")

    if len(responses) != args.expected_records:
        raise ValueError(
            f"Found {len(responses)} responses, expected {args.expected_records}"
        )
    if len(ratings) != args.expected_records:
        raise ValueError(
            f"Found {len(ratings)} ratings, expected {args.expected_records}"
        )
    if responses.keys() != ratings.keys():
        missing_ratings = sorted(responses.keys() - ratings.keys())
        extra_ratings = sorted(ratings.keys() - responses.keys())
        raise ValueError(
            f"Response/rating user mismatch: missing={missing_ratings[:10]}, "
            f"extra={extra_ratings[:10]}"
        )

    expected_pairs = [
        (attribute, side)
        for attribute in args.expected_attributes
        for side in ("follow", "avoid")
    ]
    unparsed_scores = 0
    non_stop_responses = 0
    for user_id, response in responses.items():
        if not isinstance(response.get("response"), str) or not response["response"].strip():
            raise ValueError(f"User {user_id!r} has an empty response")
        non_stop_responses += not bool(response.get("finish_reason_stop"))

        rating = ratings[user_id]
        if rating.get("response") != response.get("response"):
            raise ValueError(f"User {user_id!r} response differs between caches")
        actual_pairs = [
            (score.get("attribute"), score.get("side"))
            for score in rating.get("scores", [])
        ]
        if actual_pairs != expected_pairs:
            raise ValueError(f"User {user_id!r} has unexpected score pairs")
        unparsed_scores += sum(
            score.get("score") is None for score in rating["scores"]
        )

    if unparsed_scores:
        raise ValueError(f"Found {unparsed_scores} unparsed rating scores")

    summary = {
        "status": "complete",
        "response_shards": args.num_response_shards,
        "rating_shards": args.num_rating_shards,
        "records": len(responses),
        "attributes": args.expected_attributes,
        "unparsed_scores": unparsed_scores,
        "non_stop_responses": non_stop_responses,
    }
    args.ratings_dir.mkdir(parents=True, exist_ok=True)
    success_path = args.ratings_dir / "_SUCCESS.json"
    temp_path = success_path.with_suffix(success_path.suffix + ".tmp")
    with temp_path.open("w") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")
    temp_path.replace(success_path)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
