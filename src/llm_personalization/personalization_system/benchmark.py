from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from llm_personalization.judge.judge import AttributeJudge
from llm_personalization.utils.gpu_monitor import log_gpu_usage
from llm_personalization.utils.io import write_jsonl

from .cache import CachedDataset
from .judging import WeightedAttributeJudge
from .methods.base import PersonalizationSystem
from .methods.score_utils import attribute_candidate_scores
from .preference_mapping import (
    generate_world_matrix,
    map_to_response_vector,
    response_vector_to_weighted_attributes,
    user_attributes_to_vector,
)
from .robustness import run_robustness
from .unpersonalized_cache import UnpersonalizedCache


def load_split(cfg: DictConfig, split: str) -> CachedDataset:
    return CachedDataset(
        split=split,
        conversations_path=cfg.cache[f"conversations_{split}"],
        responses_path=cfg.cache[f"responses_{split}"],
        ratings_path=cfg.cache[f"ratings_{split}"],
        history_max_len=cfg.cache.get("history_max_len", None),
        limit=cfg.cache.get(f"{split}_limit", None),
    )


def world_targets(
    cfg: DictConfig, dataset: CachedDataset, world_idx: int
) -> tuple[dict[str, list[dict]], dict[str, dict[tuple[str, str], float]]]:
    """For one world seed: each user's weighted response-attribute targets, and
    the resulting user-fit score of each cached candidate response."""
    user_attributes = list(cfg.user_attributes)
    response_attributes = list(cfg.response_attributes)
    world_matrix = generate_world_matrix(
        cfg.world_matrix_type,
        (len(user_attributes), len(response_attributes)),
        seed=world_idx,
        k=int(cfg.get("world_matrix_k", 1)),
    )
    gt: dict[str, list[dict]] = {}
    for user in dataset:
        u_vec = user_attributes_to_vector(user.gt_user_attributes, user_attributes)
        r_vec = map_to_response_vector(u_vec, world_matrix)
        gt[user.user_id] = response_vector_to_weighted_attributes(
            r_vec, response_attributes, cfg.get("num_response_attributes_per_user", None)
        )
    scores = {u.user_id: attribute_candidate_scores(u, gt[u.user_id]) for u in dataset}
    return gt, scores


def _plot_score_distribution(scores: torch.Tensor, baseline: torch.Tensor, output_dir: Path, world_idx) -> None:
    plt.figure(figsize=(10, 5))
    bins = np.arange(-10, 11, 1)
    plt.hist(scores.detach().cpu().numpy(), bins=bins, alpha=0.5, label="Personalized")
    plt.hist(baseline.detach().cpu().numpy(), bins=bins, alpha=0.5, label="Unpersonalized")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title(f"Score Distribution (World {world_idx})")
    plt.legend()
    plt.savefig(output_dir / f"score_distribution_world_{world_idx}.png")
    plt.close()


def run_benchmark(cfg: DictConfig) -> None:
    print("[Benchmark] Running benchmark...")

    world_seeds_cfg = cfg.get("world_seeds", None)
    if world_seeds_cfg is not None:
        world_seeds = [int(s) for s in world_seeds_cfg]
    else:
        world_seeds = list(range(int(cfg.num_worlds)))

    # The directory name is only for humans (all metadata is in config.yaml);
    # the seed tag keeps concurrent jobs of a sweep from colliding.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_tag = "seeds" + "_".join(map(str, world_seeds))
    run_name = f"{timestamp}_{cfg.gen_model_tag}_{cfg.method_name}_{seed_tag}_{cfg.pool_tag}"
    output_dir = Path(cfg.output_root) / run_name
    out_systems = output_dir / "trained_systems"
    out_results = output_dir / "results"
    out_systems.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)
    # The resolved config is the run's metadata: aggregation and the
    # robustness shard scripts read it instead of parsing names or logs.
    run_config = OmegaConf.to_container(cfg, resolve=True)
    run_config["world_seeds"] = world_seeds
    OmegaConf.save(OmegaConf.create(run_config), output_dir / "config.yaml")

    skip_training = cfg.get("skip_training", False)
    save_test_results = cfg.get("save_test_results", True)

    train_dataset = load_split(cfg, "train")
    val_size = int(cfg.cache.get("val_size", 0) or 0)
    val_dataset = train_dataset.split_off_val(val_size) if val_size > 0 else None
    test_dataset = load_split(cfg, "test")

    attribute_judge: AttributeJudge = hydra.utils.instantiate(cfg.attribute_judge)
    unpersonalized_cache = UnpersonalizedCache(
        responses_path=cfg.cache.unpersonalized_responses_test,
        ratings_path=cfg.cache.unpersonalized_ratings_test,
    )

    results: dict = {"timestamp": timestamp, "config": run_config}
    results_path = out_results / "results.json"

    for world_idx in world_seeds:
        print(f"[Benchmark] === world {world_idx} ===")
        train_gt, train_scores = world_targets(cfg, train_dataset, world_idx)
        val_gt, val_scores = (
            world_targets(cfg, val_dataset, world_idx) if val_dataset is not None else (None, None)
        )
        test_gt, test_scores = world_targets(cfg, test_dataset, world_idx)

        system: PersonalizationSystem = hydra.utils.instantiate(cfg.personalization_system)
        world_path = out_systems / f"world_{world_idx}"
        world_path.mkdir(parents=True, exist_ok=True)

        if not skip_training:
            log_gpu_usage("Before train")
            system.train(
                train_dataset, train_scores, world_path,
                val_dataset=val_dataset, val_user_id_to_candidate_scores=val_scores,
                gt_for_debug=train_gt, val_gt_for_debug=val_gt,
            )
            log_gpu_usage("After train")
        else:
            print("[Benchmark] skipping training")

        print("[Benchmark] generating personalized responses...")
        responses = system.evaluate(
            test_dataset, world_path,
            user_id_to_candidate_scores=test_scores, gt_for_debug=test_gt,
        )

        test_user_ids = [u.user_id for u in test_dataset]
        test_conversations = [
            u.current_messages + [{"role": "assistant", "content": r}]
            for u, r in zip(test_dataset, responses)
        ]

        # unpersonalized baseline is a pure cache lookup (no LLM/judge calls)
        print("[Benchmark] looking up unpersonalized responses + scores from cache...")
        missing = sum(1 for uid in test_user_ids if uid not in unpersonalized_cache)
        if missing:
            print(f"[Benchmark] WARNING: {missing}/{len(test_user_ids)} users missing from unpers cache")
        unpers_responses_text: list[str] = [
            (unpersonalized_cache.get(uid).response if uid in unpersonalized_cache else "")
            for uid in test_user_ids
        ]
        unpers_scores = torch.tensor(
            [unpersonalized_cache.weighted_score(uid, test_gt[uid]) for uid in test_user_ids],
            dtype=torch.float,
        )

        wj = WeightedAttributeJudge(attribute_judge, test_gt)
        log_gpu_usage("Before judge load")
        wj.load()
        log_gpu_usage("After judge load")
        print("[Benchmark] scoring personalized...")
        scores = torch.tensor(wj.judge(test_user_ids, test_conversations), dtype=torch.float)
        wj.unload()
        log_gpu_usage("After judge unload")

        valid = ~(torch.isnan(scores) | torch.isnan(unpers_scores))
        win_rate = (scores[valid] > unpers_scores[valid]).float().mean().item() if valid.any() else float("nan")
        tie_rate = (scores[valid] == unpers_scores[valid]).float().mean().item() if valid.any() else float("nan")

        print(f"[Benchmark] World {world_idx} results:")
        print(f"  - Personalized:   {scores[valid].mean():.4f} +/- {scores[valid].std():.4f}")
        print(f"  - Unpersonalized: {unpers_scores[valid].mean():.4f} +/- {unpers_scores[valid].std():.4f}")
        print(f"  - Win/Tie/Loss:   {win_rate:.4f} / {tie_rate:.4f} / {1 - win_rate - tie_rate:.4f}")

        results.setdefault("attribute_benchmark", {})[f"world_{world_idx}"] = {
            "personalized_score_mean": float(scores[valid].mean()),
            "personalized_score_std": float(scores[valid].std()),
            "unpersonalized_score_mean": float(unpers_scores[valid].mean()),
            "unpersonalized_score_std": float(unpers_scores[valid].std()),
            "win_rate": win_rate,
            "tie_rate": tie_rate,
            "loss_rate": 1 - win_rate - tie_rate,
            "n_valid": int(valid.sum()),
            "n_total": len(scores),
        }

        _plot_score_distribution(scores[valid], unpers_scores[valid], out_results, world_idx)

        if save_test_results:
            records = []
            for i, user in enumerate(test_dataset):
                records.append({
                    "user_id": user.user_id,
                    "current_messages": user.current_messages,
                    "gt_user_attributes": user.gt_user_attributes,
                    "weighted_response_targets": test_gt[user.user_id],
                    "personalized_response": responses[i],
                    "unpersonalized_response": unpers_responses_text[i],
                    "personalized_score": float(scores[i]),
                    "unpersonalized_score": float(unpers_scores[i]),
                })
            write_jsonl(out_results / f"attribute_world_{world_idx}_test_samples.jsonl", records)

        # robustness eval runs only when the config provides a `robustness` block
        robustness_cfg = cfg.get("robustness", None)
        if robustness_cfg is not None:
            print(f"[Benchmark] running robustness eval for world {world_idx}...")
            rob_summary, rob_records = run_robustness(
                system=system,
                test_dataset=test_dataset,
                world_path=world_path,
                world_idx=world_idx,
                robustness_cfg=robustness_cfg,
                gen_model_tag=cfg.gen_model_tag,
                user_id_to_candidate_scores=test_scores,
                gt_for_debug=test_gt,
            )
            results.setdefault("robustness_benchmark", {})[f"world_{world_idx}"] = rob_summary
            if save_test_results:
                write_jsonl(out_results / f"robustness_world_{world_idx}_test_samples.jsonl", rob_records)

        # rewrite after every world so a crash keeps the finished worlds
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Benchmark] results saved to {results_path}")
