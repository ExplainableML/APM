"""Merge the robustness shards of one world of a benchmark run.

Checks that every shard was run with the same robustness config and that
together they cover each question exactly once, re-grades the full set, and
writes the result into the run exactly as an unsharded run would have:
`robustness_benchmark.world_<w>` in results/results.json, and
results/robustness_world_<w>_test_samples.jsonl.

    python experiments/run_benchmark/scripts/merge_robustness_shards.py --run-dir data/run_benchmark/<run> --world 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_personalization.personalization_system.robustness import (
    compare_to_unpersonalized,
    load_questions,
    shard_dir,
    shard_stem,
)
from llm_personalization.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--world", type=int, default=0)
    args = parser.parse_args()

    in_dir = shard_dir(args.run_dir, args.world)
    metas = [json.loads(p.read_text()) for p in sorted(in_dir.glob("shard_*.json"))]
    if not metas:
        raise FileNotFoundError(f"no shard summaries in {in_dir}")
    num_shards = {m["num_shards"] for m in metas}
    robustness_cfgs = {json.dumps(m["robustness"], sort_keys=True) for m in metas}
    if len(num_shards) != 1 or len(robustness_cfgs) != 1:
        raise ValueError(f"shards in {in_dir} disagree on num_shards {num_shards} or robustness config")
    (num_shards,) = num_shards
    robustness_cfg = metas[0]["robustness"]
    missing = sorted(set(range(num_shards)) - {m["shard_index"] for m in metas})
    if missing:
        raise FileNotFoundError(f"missing shards {missing} of {num_shards} in {in_dir}")

    questions = load_questions(robustness_cfg)
    by_index: dict[int, dict] = {}
    for i in range(num_shards):
        for rec in read_jsonl(in_dir / f"{shard_stem(i, num_shards)}.jsonl"):
            if rec["question_index"] in by_index:
                raise ValueError(f"question {rec['question_index']} appears in more than one shard")
            by_index[rec["question_index"]] = rec
    if set(by_index) != set(range(len(questions))):
        raise ValueError(f"shards cover {len(by_index)} of {len(questions)} questions")
    records = [by_index[i] for i in range(len(questions))]
    for rec, q in zip(records, questions):
        if rec["question_id"] != q.question_id:
            raise ValueError(f"question order mismatch at {rec['question_index']}: {rec['question_id']} != {q.question_id}")

    summary = compare_to_unpersonalized(
        [r["personalized_response"] for r in records],
        [r["unpersonalized_response"] for r in records],
        questions,
        log_prefix="[Benchmark-RobustnessMerge]",
    )

    results_dir = args.run_dir / "results"
    write_jsonl(results_dir / f"robustness_world_{args.world}_test_samples.jsonl", records)
    results_path = results_dir / "results.json"
    results = json.loads(results_path.read_text())
    run_robustness_cfg = results["config"].get("robustness")
    if run_robustness_cfg is not None and run_robustness_cfg != robustness_cfg:
        raise ValueError(f"shards used a different robustness config than {results_path}")
    results["config"]["robustness"] = robustness_cfg
    results.setdefault("robustness_benchmark", {})[f"world_{args.world}"] = summary
    results_path.write_text(json.dumps(results, indent=2))
    print(f"[Benchmark-RobustnessMerge] wrote world {args.world} into {results_path}")


if __name__ == "__main__":
    main()
