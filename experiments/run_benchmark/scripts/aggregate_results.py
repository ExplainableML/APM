"""Aggregate benchmark runs into CSV tables.

Reads every <results-root>/*/results/results.json. All run metadata (method,
generator, judge, pool, world seeds) comes from the resolved config each run
saves alongside its results. If several runs cover the same
(gen_model, judge, pool, method, world), the most recent one is used.

Writes to --out-dir:
  attribute_worlds.csv      one row per (run, world): scores and win/tie/loss
  attribute_summary.csv     mean/std over worlds per (gen_model, judge, pool, method)
  robustness_worlds.csv     one row per (run, world, task): accuracies and delta
  robustness_summary.csv    mean/std over worlds per (gen_model, pool, method, task)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

GROUP = ["gen_model", "judge_model", "pool", "method"]


def load_runs(roots: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    attr_rows, rob_rows = [], []
    for root in roots:
        for path in sorted(root.glob("*/results/results.json")):
            res = json.loads(path.read_text())
            cfg = res.get("config", {})
            if "method_name" not in cfg:
                print(f"skipping {path}: no saved config (run predates config saving)")
                continue
            meta = {
                "gen_model": cfg["gen_model_tag"],
                "judge_model": cfg["judge_model_tag"],
                "pool": cfg["pool_tag"],
                "method": cfg["method_name"],
                "timestamp": res["timestamp"],
                "run_dir": str(path.parent.parent),
            }
            for world, m in res.get("attribute_benchmark", {}).items():
                attr_rows.append({
                    **meta,
                    "world": int(world.removeprefix("world_")),
                    **m,
                    "score_delta": m["personalized_score_mean"] - m["unpersonalized_score_mean"],
                })
            for world, splits in res.get("robustness_benchmark", {}).items():
                unpers = _flatten(splits["unpersonalized"])
                for task, m in _flatten(splits["personalized"]).items():
                    rob_rows.append({
                        **meta,
                        "world": int(world.removeprefix("world_")),
                        "task": task,
                        "n": m["total"],
                        "personalized_accuracy": m["accuracy"],
                        "unpersonalized_accuracy": unpers[task]["accuracy"],
                        "delta": m["delta_vs_unpersonalized"],
                    })
    return pd.DataFrame(attr_rows), pd.DataFrame(rob_rows)


def _flatten(summary: dict) -> dict[str, dict]:
    out = {}
    for src, v in summary.items():
        if src == "bbq":
            out.update({f"bbq/{cat}": e for cat, e in v.items()})
        else:
            out[src] = v
    return out


def latest_per_world(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Keep only the most recent run for each key; report what was dropped."""
    if df.empty:
        return df
    df = df.sort_values("timestamp")
    dup = df.duplicated(keys, keep="last")
    for _, row in df[dup].drop_duplicates("run_dir").iterrows():
        print(f"superseded by a newer run: {row['run_dir']}")
    return df[~dup]


def summarize(df: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby(by, dropna=False)
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out.insert(0, "n_worlds", g.size())
    return out.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, action="append", default=None,
                        help="directory containing run directories (repeatable; default data/run_benchmark)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/run_benchmark/aggregated"))
    args = parser.parse_args()
    roots = args.results_root or [Path("data/run_benchmark")]

    attr, rob = load_runs(roots)
    attr = latest_per_world(attr, GROUP + ["world"])
    rob = latest_per_world(rob, GROUP + ["world", "task"])

    attr_summary = summarize(
        attr, GROUP, ["win_rate", "tie_rate", "loss_rate", "score_delta",
                      "personalized_score_mean", "unpersonalized_score_mean"],
    )
    rob_summary = summarize(
        rob, ["gen_model", "pool", "method", "task"],
        ["personalized_accuracy", "unpersonalized_accuracy", "delta"],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in [("attribute_worlds", attr), ("attribute_summary", attr_summary),
                     ("robustness_worlds", rob), ("robustness_summary", rob_summary)]:
        df.to_csv(args.out_dir / f"{name}.csv", index=False)
    print(f"wrote {len(attr)} attribute rows and {len(rob)} robustness rows to {args.out_dir}")
    if not attr_summary.empty:
        cols = GROUP + ["n_worlds", "win_rate_mean", "win_rate_std", "score_delta_mean"]
        print(attr_summary.sort_values("win_rate_mean", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
