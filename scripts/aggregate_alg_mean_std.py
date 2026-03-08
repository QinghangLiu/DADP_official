#!/usr/bin/env python
"""
Aggregate mean/std over seeds for DV and DADP across six environments (id & ood).

Usage:
  python scripts/aggregate_alg_mean_std.py \
    --eval_dir /home/qinghang/DomainAdaptiveDiffusionPolicy/multi_seed_runs/clean_up_ckpt/eval_results_20260113_075121 \
    --out_json /home/qinghang/DomainAdaptiveDiffusionPolicy/multi_seed_runs/clean_up_ckpt/eval_results_20260113_075121/mean_std_by_alg.json

Notes:
- Algorithm mapping: filenames containing "_dv" are treated as DV; everything else is treated as DADP/base.
- Reward is computed as the average of per-task `mean_reward` values in each eval JSON.
- Success is computed as the average of per-task `mean_success_rate` values (if present; otherwise skipped).
- Statistics are computed across seeds for each (algorithm, env, eval_kind) bucket. Environment is parsed as the prefix before `_exp` in the filename (e.g., `RandomAnt-v0_exp_*` → env `RandomAnt-v0`).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate DV vs DADP mean/std across seeds")
    parser.add_argument(
        "--eval_dir",
        type=Path,
        default = "/home/pengcheng/DomainAdaptiveDiffusionPolicy/multi_seed_runs/20260301_085705/eval_results",
        help="Directory containing per-seed eval JSON files",
    )
    parser.add_argument(
        "--out_json",
        type=Path,
        default="/home/pengcheng/DomainAdaptiveDiffusionPolicy/multi_seed_runs/20260301_085705/mean_std_by_alg.json",
        help="Optional path to write aggregated summary as JSON",
    )
    return parser.parse_args()


def bucket_algorithm(name: str) -> str:
    """Infer algorithm from filename/pipeline string."""
    return "dv" if "_dv" in name else "dadp"


def extract_env_seed_kind(path: Path) -> tuple[str, int, str]:
    """Extract env (prefix before `_exp`), seed, eval_kind (id/ood) from filename."""
    # Example: RandomAnt-v0_exp_ant_28_reproduce_seed0_id.json
    m = re.search(r"^(?P<env_full>.+)_seed(?P<seed>\d+)_(?P<kind>id|ood)\.json$", path.name)
    if not m:
        raise ValueError(f"Unrecognized filename: {path.name}")
    env_full = m.group("env_full")
    env = env_full.split("_exp", 1)[0]
    seed = int(m.group("seed"))
    kind = m.group("kind")
    return env, seed, kind


def load_metrics(path: Path) -> tuple[float, float | None]:
    """Return (reward_mean_over_tasks, success_mean_over_tasks or None)."""
    with path.open("r") as f:
        data = json.load(f)

    task_results = data.get("task_results", {})
    if not task_results:
        raise ValueError(f"No task_results in {path}")

    rewards = []
    successes = []
    for task in task_results.values():
        if "mean_reward" in task:
            rewards.append(task["mean_reward"])
        if "mean_success_rate" in task:
            successes.append(task["mean_success_rate"])

    reward_mean = float(mean(rewards)) if rewards else math.nan
    success_mean = float(mean(successes)) if successes else None
    return reward_mean, success_mean


def aggregate(eval_dir: Path):
    buckets = defaultdict(lambda: {"rewards": [], "successes": [], "seeds": set()})

    for path in eval_dir.glob("*.json"):
        if path.name.startswith("aggregated_"):
            continue
        if path.name.endswith("mean_std_by_alg.json"):
            continue

        try:
            env, seed, kind = extract_env_seed_kind(path)
        except ValueError:
            continue  # skip unexpected files

        alg = bucket_algorithm(path.stem)
        reward_mean, success_mean = load_metrics(path)

        key = (alg, env, kind)
        buckets[key]["rewards"].append(reward_mean)
        buckets[key]["seeds"].add(seed)
        if success_mean is not None:
            buckets[key]["successes"].append(success_mean)

    summary = {}
    for (alg, env, kind), vals in buckets.items():
        rewards = vals["rewards"]
        successes = vals["successes"]
        seeds = sorted(vals["seeds"])

        reward_mean = mean(rewards) if rewards else math.nan
        reward_std = pstdev(rewards) if len(rewards) > 1 else 0.0
        if successes:
            success_mean = mean(successes)
            success_std = pstdev(successes) if len(successes) > 1 else 0.0
        else:
            success_mean = None
            success_std = None

        summary[f"{env}|{alg}|{kind}"] = {
            "env": env,
            "algorithm": alg,
            "eval_kind": kind,
            "num_seeds": len(seeds),
            "seeds": seeds,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "success_mean": success_mean,
            "success_std": success_std,
        }

    return summary


def main() -> None:
    args = parse_args()
    summary = aggregate(args.eval_dir)

    # Print aggregated rows first in a stable env/alg/kind order for readability
    order_alg = {"dadp": 0, "dv": 1}
    order_kind = {"id": 0, "ood": 1}

    print("Algorithm mean/std across seeds (reward, success):")
    for _, row in sorted(
        summary.items(),
        key=lambda kv: (
            kv[1]["env"],
            order_alg.get(kv[1]["algorithm"], 99),
            order_kind.get(kv[1]["eval_kind"], 99),
        ),
    ):
        reward_mean = row["reward_mean"]
        reward_std = row["reward_std"]
        success_mean = row["success_mean"]
        success_std = row["success_std"]
        success_mean_str = f"{success_mean:.3f}" if success_mean is not None else "NA"
        success_std_str = f"{success_std:.3f}" if success_std is not None else "NA"

        print(
            f"{row['env']:<24} alg={row['algorithm']:<4} kind={row['eval_kind']:<3} "
            f"seeds={row['seeds']} reward_mean={reward_mean:.3f} reward_std={reward_std:.3f} "
            f"success_mean={success_mean_str} success_std={success_std_str}"
        )

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote summary to {args.out_json}")


if __name__ == "__main__":
    main()
