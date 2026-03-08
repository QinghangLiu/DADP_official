#!/usr/bin/env python3
"""Merge task_result JSON files and compute mean rewards.

Usage:
  python scripts/merge_and_mean_rewards.py --files file1.json file2.json [--tasks 1,2,3] [--out merged_summary.json]

Behavior:
- Loads each file (expected format like train_diffusion evaluation outputs with `task_results`).
- Later files override earlier ones for the same task id.
- If `--tasks` is provided, only those task ids are included; otherwise all tasks seen are used.
- Prints mean reward and count, and (optionally) writes merged per-task data to --out.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_full_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)

def load_task_results(path: Path) -> Dict[str, Any]:
    data = load_full_json(path)
    if "task_results" not in data or not isinstance(data["task_results"], dict):
        raise ValueError(f"File {path} missing task_results")
    return data["task_results"]



def merge_full_json(files: List[Path]) -> Dict[str, Any]:
    merged_task_results: Dict[str, Any] = {}
    overall_stats: List[Dict[str, Any]] = []
    configs: List[Dict[str, Any]] = []
    for path in files:
        data = load_full_json(path)
        if "task_results" in data:
            merged_task_results.update(data["task_results"])
        if "overall_stats" in data:
            overall_stats.append(data["overall_stats"])
        if "config" in data:
            configs.append(data["config"])
    return {
        "task_results": merged_task_results,
        "all_overall_stats": overall_stats,
        "all_configs": configs,
    }


def parse_task_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    items = [tok.strip() for tok in raw.split(',') if tok.strip()]
    return [str(int(x)) for x in items]



def compute_full_stats(task_results: Dict[str, Any], task_filter: Optional[List[str]] = None) -> Dict[str, Any]:
    import numpy as np
    ids = task_filter if task_filter is not None else list(task_results.keys())
    ids = [str(tid) for tid in ids]
    # Collect all episode rewards and success rates
    overall_episode_rewards = []
    overall_success_rates = []
    episode_mean_rewards_by_task = {}
    for tid in ids:
        entry = task_results.get(str(tid))
        if not entry:
            continue
        # episode_rewards: flat list (num_episodes * num_envs)
        if "episode_rewards" in entry and entry["episode_rewards"]:
            overall_episode_rewards.extend(entry["episode_rewards"])
        # mean_success_rate: float
        if "mean_success_rate" in entry and entry["mean_success_rate"] is not None:
            overall_success_rates.append(entry["mean_success_rate"])
        # episode_mean_rewards: list (num_episodes)
        if "episode_mean_rewards" in entry and entry["episode_mean_rewards"]:
            episode_mean_rewards_by_task[tid] = entry["episode_mean_rewards"]

    sorted_task_ids = sorted(ids, key=lambda x: int(x))
    num_episodes = None
    # Try to infer num_episodes from first available task
    for tid in sorted_task_ids:
        if tid in episode_mean_rewards_by_task:
            num_episodes = len(episode_mean_rewards_by_task[tid])
            break

    def _episode_mean_series(task_ids):
        series = []
        if num_episodes is None:
            return series
        for episode_idx in range(num_episodes):
            per_task_episode_means = [
                episode_mean_rewards_by_task[tid][episode_idx]
                for tid in task_ids
                if tid in episode_mean_rewards_by_task and len(episode_mean_rewards_by_task[tid]) > episode_idx
            ]
            if per_task_episode_means:
                series.append(float(np.mean(per_task_episode_means)))
        return series

    all_task_episode_means = _episode_mean_series(sorted_task_ids)
    first_5_ids = sorted_task_ids[:5]
    last_5_ids = sorted_task_ids[-5:]
    first_5_rewards = [task_results[tid]["mean_reward"] for tid in first_5_ids if tid in task_results and "mean_reward" in task_results[tid]]
    last_5_rewards = [task_results[tid]["mean_reward"] for tid in last_5_ids if tid in task_results and "mean_reward" in task_results[tid]]
    first_5_episode_means = _episode_mean_series(first_5_ids)
    last_5_episode_means = _episode_mean_series(last_5_ids)

    stats = {
        "mean_episode_reward": float(np.mean(overall_episode_rewards)) if overall_episode_rewards else 0.0,
        "std_episode_reward": float(np.std(overall_episode_rewards)) if overall_episode_rewards else 0.0,
        "num_episode_samples": len(overall_episode_rewards),
        "num_tasks_evaluated": len(sorted_task_ids),
    }
    if overall_success_rates:
        stats["mean_success_rate"] = float(np.mean(overall_success_rates))
        stats["std_success_rate"] = float(np.std(overall_success_rates))
    if all_task_episode_means:
        stats["all_tasks_episode_std"] = float(np.std(all_task_episode_means))
    if first_5_rewards:
        stats["first_5_mean_reward"] = float(np.mean(first_5_rewards))
        stats["first_5_std_reward"] = float(np.std(first_5_episode_means)) if first_5_episode_means else 0.0
    if last_5_rewards:
        stats["last_5_mean_reward"] = float(np.mean(last_5_rewards))
        stats["last_5_std_reward"] = float(np.std(last_5_episode_means)) if last_5_episode_means else 0.0
    return stats


def main():
    parser = argparse.ArgumentParser(description="Merge evaluation JSONs and compute mean rewards.")
    default_files = [
        "results/exp_dv_ant_28_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/RandomAnt-v0/RandomAnt/82dynamics-v7/task_[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]_less_sample_step(5).json",
        "results/exp_dv_ant_28_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/RandomAnt-v0/RandomAnt/82dynamics-v7/task_[1, 2, 16, 17, 24, 25, 26, 27, 37, 38, 49, 50, 51, 52, 60, 61, 62, 63, 72, 73, 77, 78, 79, 80, 81]_less_sample_step(5).json",
    ]
    parser.add_argument("--files", nargs="+", default=default_files, help="List of JSON files to merge (later overrides earlier)")
    parser.add_argument("--tasks", type=str, default="0,1,2,7,8,9,16,17,24,25,26,27,37,38,49,50,51,52,60,61,62,63,72,73,77,78,79,80,81", help="Comma-separated task ids to include; default is all tasks present")
    parser.add_argument("--out", type=str, default=None, help="Optional output JSON path for merged per-task data and summary")
    args = parser.parse_args()

    file_paths = [Path(f).expanduser().resolve() for f in args.files]
    task_filter = parse_task_list(args.tasks)

    merged_full = merge_full_json(file_paths)
    merged = merged_full["task_results"]
    # Compute overall_stats in the same format as the original files
    overall_stats = compute_full_stats(merged, task_filter)
    # Use config from the first file if available, else empty dict
    config = merged_full["all_configs"][0] if merged_full["all_configs"] else {}

    out_data = {
        "task_results": merged,
        "overall_stats": overall_stats,
        "config": config,
    }

    # Save to user-specified path or default location
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        if file_paths:
            out_dir = file_paths[0].parent
            out_path = out_dir / "merged_summary.json"
        else:
            out_path = Path("merged_summary.json").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved merged data to {out_path}")

if __name__ == "__main__":
    main()
