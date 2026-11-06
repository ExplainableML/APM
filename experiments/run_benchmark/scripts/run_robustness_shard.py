"""Run one shard of the robustness evaluation for a trained benchmark run.

Everything except the shard spec comes from the run's saved `config.yaml`, so
the evaluated system, generator and test users always match the run. Outputs
go to <run_dir>/results/robustness_shards/world_<w>/; combine them with
merge_robustness_shards.py.

    python experiments/run_benchmark/scripts/run_robustness_shard.py \
        --run-dir data/run_benchmark/<run> --world 0 --shard-index 0 --num-shards 8 --robustness full
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from llm_personalization.personalization_system.benchmark import load_split, world_targets
from llm_personalization.personalization_system.robustness import run_robustness, shard_dir, shard_stem
from llm_personalization.utils.io import write_jsonl

ROBUSTNESS_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "robustness"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--world", type=int, default=0)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument(
        "--robustness", default=None,
        help="name of a configs/robustness/*.yaml preset; defaults to the run's own `robustness` block",
    )
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error(f"--shard-index must be in [0, {args.num_shards})")

    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    if args.robustness is not None:
        robustness_cfg = OmegaConf.load(ROBUSTNESS_CONFIG_DIR / f"{args.robustness}.yaml").robustness
    elif cfg.get("robustness") is not None:
        robustness_cfg = cfg.robustness
    else:
        parser.error("the run has no `robustness` config; pass --robustness <preset>")

    world_path = args.run_dir / "trained_systems" / f"world_{args.world}"
    if not world_path.is_dir():
        raise FileNotFoundError(f"no trained system for world {args.world}: {world_path}")

    test_dataset = load_split(cfg, "test")
    test_gt, test_scores = world_targets(cfg, test_dataset, args.world)
    system = hydra.utils.instantiate(cfg.personalization_system)
    summary, records = run_robustness(
        system=system,
        test_dataset=test_dataset,
        world_path=world_path,
        world_idx=args.world,
        robustness_cfg=robustness_cfg,
        gen_model_tag=cfg.gen_model_tag,
        shard=(args.shard_index, args.num_shards),
        user_id_to_candidate_scores=test_scores,
        gt_for_debug=test_gt,
    )

    out_dir = shard_dir(args.run_dir, args.world)
    stem = shard_stem(args.shard_index, args.num_shards)
    write_jsonl(out_dir / f"{stem}.jsonl", records)
    (out_dir / f"{stem}.json").write_text(json.dumps({
        "world": args.world,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "robustness": OmegaConf.to_container(robustness_cfg, resolve=True),
        "summary": summary,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
