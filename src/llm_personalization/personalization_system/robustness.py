"""Robustness evaluation: each item is a test user whose `current_messages` is
replaced by an MC-formatted benchmark question, so `evaluate(...)` runs
unmodified; per-source accuracy is compared against a cached unpersonalized
baseline."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from llm_personalization.data.robustness_dataset import (
    RobustnessQuestion,
    format_mc_prompt,
    load_robustness_questions,
    parse_answer_letter,
)

from llm_personalization.utils.io import iter_jsonl

from .cache import CachedDataset, CachedUser

UNPERS_ROBUSTNESS_ROOT = Path("data/unpersonalized_robustness")


class RobustnessEvalDataset:
    """One item per question in `indices` (default: all). Question i is paired
    with test user `i % len(base)`, so a sharded run matches the full run."""

    def __init__(
        self,
        base: CachedDataset,
        questions: list[RobustnessQuestion],
        indices: list[int] | None = None,
    ):
        self.indices = list(range(len(questions))) if indices is None else list(indices)
        self.questions = [questions[i] for i in self.indices]
        self._users: list[CachedUser] = [
            replace(base[i % len(base)], current_messages=[{"role": "user", "content": format_mc_prompt(questions[i])}])
            for i in self.indices
        ]

    def __len__(self) -> int:
        return len(self._users)

    def __getitem__(self, i: int) -> CachedUser:
        return self._users[i]

    def __iter__(self):
        return iter(self._users)


def load_questions(robustness_cfg) -> list[RobustnessQuestion]:
    return load_robustness_questions(
        include_mmlu_pro=robustness_cfg.get("include_mmlu_pro", True),
        include_truthfulqa=robustness_cfg.get("include_truthfulqa", True),
        include_bbq=robustness_cfg.get("include_bbq", False),
        mmlu_pro_limit=robustness_cfg.get("mmlu_pro_limit", None),
        truthfulqa_limit=robustness_cfg.get("truthfulqa_limit", None),
        bbq_limit=robustness_cfg.get("bbq_limit", None),
        seed=robustness_cfg.get("seed", 42),
    )


def unpers_cache_path(gen_model_tag: str, robustness_cfg) -> Path:
    """Cache file for the unpersonalized answers of one (gen_model, question set)."""
    parts = []
    for src in ("mmlu_pro", "truthfulqa", "bbq"):
        if robustness_cfg.get(f"include_{src}", False):
            limit = robustness_cfg.get(f"{src}_limit", None)
            parts.append(f"{src}{limit if limit is not None else 'full'}")
    parts.append(f"seed{robustness_cfg.get('seed', 42)}")
    return UNPERS_ROBUSTNESS_ROOT / str(gen_model_tag) / f"{'_'.join(parts)}.jsonl"


def load_unpers_responses(path: Path, questions: list[RobustnessQuestion]) -> list[str]:
    """Cached unpers MC responses, ordered to match `questions`."""
    if not path.exists():
        raise FileNotFoundError(
            f"Unpersonalized robustness cache not found: {path}\n"
            f"Build it once per gen_model with experiments/unpersonalized_robustness."
        )
    by_id = {rec["question_id"]: rec["response"] for rec in iter_jsonl(path)}
    missing = [q.question_id for q in questions if q.question_id not in by_id]
    if missing:
        raise ValueError(
            f"Unpers cache at {path} is missing {len(missing)} question_ids "
            f"(first: {missing[:3]}). Re-run the prep job with the same robustness config."
        )
    return [by_id[q.question_id] for q in questions]


def accuracy_summary(responses: list[str], questions: list[RobustnessQuestion]) -> dict:
    """{"mmlu_pro": {correct, total, accuracy}, "truthfulqa": {...},
    "bbq": {"aggregate": {...}, <category>: {...}}}; absent sources are omitted."""
    counts: dict[str, list[int]] = {}

    def add(key: str, ok: bool) -> None:
        c = counts.setdefault(key, [0, 0])
        c[0] += int(ok)
        c[1] += 1

    for resp, q in zip(responses, questions):
        ok = parse_answer_letter(resp) == q.correct_letter
        if q.source == "bbq":
            add("bbq/aggregate", ok)
            add(f"bbq/{(q.metadata or {}).get('category', 'unknown')}", ok)
        else:
            add(q.source, ok)

    out: dict = {}
    for key in sorted(counts, key=lambda k: (k.startswith("bbq/"), k != "bbq/aggregate", k)):
        c, t = counts[key]
        entry = {"correct": c, "total": t, "accuracy": c / t}
        if key.startswith("bbq/"):
            out.setdefault("bbq", {})[key.split("/", 1)[1]] = entry
        else:
            out[key] = entry
    return out


def _flat(summary: dict) -> dict[str, dict]:
    out = {}
    for src, v in summary.items():
        if src == "bbq":
            out.update({f"bbq/{cat}": e for cat, e in v.items()})
        else:
            out[src] = v
    return out


def compare_to_unpersonalized(
    pers_responses: list[str],
    unpers_responses: list[str],
    questions: list[RobustnessQuestion],
    log_prefix: str = "[Benchmark-Robustness]",
) -> dict:
    """Grades both response sets, adds `delta_vs_unpersonalized` to every
    personalized entry, logs the table and returns
    {"unpersonalized": ..., "personalized": ...}."""
    unpers = accuracy_summary(unpers_responses, questions)
    pers = accuracy_summary(pers_responses, questions)
    unpers_flat = _flat(unpers)
    for key, entry in _flat(pers).items():
        entry["delta_vs_unpersonalized"] = entry["accuracy"] - unpers_flat[key]["accuracy"]
        print(
            f"{log_prefix}   {key}: personalized {entry['accuracy']:.4f} "
            f"vs unpersonalized {unpers_flat[key]['accuracy']:.4f} "
            f"(delta {entry['delta_vs_unpersonalized']:+.4f}, n={entry['total']})"
        )
    return {"unpersonalized": unpers, "personalized": pers}


def robustness_records(
    dataset: RobustnessEvalDataset,
    pers_responses: list[str],
    unpers_responses: list[str],
    world_idx: int,
) -> list[dict]:
    records = []
    for idx, item, q, resp, base_resp in zip(
        dataset.indices, dataset, dataset.questions, pers_responses, unpers_responses
    ):
        records.append({
            "world_idx": world_idx,
            "question_index": idx,
            "user_id": item.user_id,
            "history": item.history,
            "question_id": q.question_id,
            "source": q.source,
            "metadata": q.metadata,
            "question_text": q.question_text,
            "options": q.options,
            "option_letters": q.option_letters,
            "correct_letter": q.correct_letter,
            "personalized_response": resp,
            "personalized_letter": parse_answer_letter(resp),
            "unpersonalized_response": base_resp,
            "unpersonalized_letter": parse_answer_letter(base_resp),
        })
    return records


def shard_dir(run_dir: Path, world: int) -> Path:
    """Where run_robustness_shard.py writes the shards of one world of a run."""
    return Path(run_dir) / "results" / "robustness_shards" / f"world_{world}"


def shard_stem(shard_index: int, num_shards: int) -> str:
    return f"shard_{shard_index:02d}_of_{num_shards:02d}"


def run_robustness(
    system,
    test_dataset: CachedDataset,
    world_path: Path,
    world_idx: int,
    robustness_cfg,
    gen_model_tag: str,
    shard: tuple[int, int] | None = None,
    user_id_to_candidate_scores: dict[str, dict[tuple[str, str], float]] | None = None,
    gt_for_debug: dict[str, list[dict]] | None = None,
) -> tuple[dict, list[dict]]:
    """Robustness benchmark for one (system, world) pair. `shard=(i, n)`
    restricts it to questions i, i+n, i+2n, ... Returns (summary, records)."""
    questions = load_questions(robustness_cfg)
    indices = None if shard is None else list(range(shard[0], len(questions), shard[1]))
    dataset = RobustnessEvalDataset(test_dataset, questions, indices)
    print(f"[Benchmark-Robustness] world {world_idx}: {len(dataset)} of {len(questions)} questions")

    unpers_responses = load_unpers_responses(unpers_cache_path(gen_model_tag, robustness_cfg), dataset.questions)
    pers_responses = system.evaluate(
        dataset, world_path,
        user_id_to_candidate_scores=user_id_to_candidate_scores,
        gt_for_debug=gt_for_debug,
    )
    summary = compare_to_unpersonalized(pers_responses, unpers_responses, dataset.questions)
    return summary, robustness_records(dataset, pers_responses, unpers_responses, world_idx)
