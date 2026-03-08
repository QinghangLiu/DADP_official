#!/usr/bin/env python
"""Random-policy baseline runner for six DADP environments.

For each env config, the script:
- loads training task IDs from the stored args.json
- builds the corresponding Minari dataset to recover task parameters
- runs a uniform-random policy per task (Mujoco: 50 eps, Adroit: 200 eps)
- records per-task returns and aggregates per-env averages.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym  # noqa: F401 (needed for env registration side effects)
import dr_envs  # registers Random* envs
import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from customwrappers.RandomVecEnv import RandomSubprocVecEnv
from pipelines_ss.utils import is_adroit_env, set_seed

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ENV_CONFIGS: Dict[str, Path] = {
    "ant": ROOT_DIR
    / "results/exp_ant_28_reproduce_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.05_noisemixed_ddim/RandomAnt-v0/RandomAnt/82dynamics-v7/args.json",
    "halfcheetah": ROOT_DIR
    / "results/exp_halfcheetah_28_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomHalfCheetah-v0/RandomHalfCheetah/82dynamics-v7/args.json",
    "walker": ROOT_DIR
    / "results/exp_walker_28(2)_predict_mixddim_long_horizon_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomWalker2d-v0/RandomWalker2d/28dynamics-v9/args.json",
    "hopper": ROOT_DIR
    / "results/exp_hopper_28_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomHopper-v0/RandomHopper/82dynamics-v7/args.json",
    "door": ROOT_DIR
    / "results/exp_door_3_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.05_noisemixed_ddim/door-shrink-finger-medium-v0/Adroit/door_shrink_combined-v0/args.json",
    "relocate": ROOT_DIR
    / "results/exp_relocate_3_shrink_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/relocate-shrink-finger-medium-v0/Adroit/relocate_shrink_combined-v0/args.json",
}

# Explicit env IDs to use for evaluation (ensure we hit the real env names).
# Mujoco tasks use Random* ids; Adroit uses the easy version.
ENV_NAME_OVERRIDES: Dict[str, str] = {
    "ant": "RandomAnt-v0",
    "halfcheetah": "RandomHalfCheetah-v0",
    "walker": "RandomWalker2d-v0",
    "hopper": "RandomHopper-v0",
    "door": "door-shrink-finger-easy-v0",
    "relocate": "relocate-shrink-finger-easy-v0",
}


def _load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _make_env(env_name: str, seed: int) -> RandomSubprocVecEnv:
    return make_vec_env(
        env_name,
        n_envs=1,
        seed=seed,
        vec_env_cls=RandomSubprocVecEnv,
    )


def _eval_task(
    env_name: str,
    num_episodes: int,
    max_steps: int,
    seed: int,
    is_adroit: bool,
) -> Tuple[List[float], float, float]:
    env = _make_env(env_name, seed)
    try:
        returns: List[float] = []
        for ep in range(num_episodes):
            obs = env.reset()
            done = np.zeros(1, dtype=bool)
            ep_return = np.zeros(1, dtype=float)
            steps = 0
            while not bool(done[0]) and steps < max_steps:
                action = env.action_space.sample()
                action = np.expand_dims(action, axis=0)
                obs, rew, done, info = env.step(action)
                ep_return += rew
                steps += 1
            returns.append(float(ep_return.mean()))
        mean_ret = float(np.mean(returns)) if returns else 0.0
        std_ret = float(np.std(returns)) if returns else 0.0
        return returns, mean_ret, std_ret
    finally:
        env.close()


def run_random_baseline(env_key: str, cfg_path: Path, args: argparse.Namespace) -> dict:
    cfg = _load_json(cfg_path)
    task_cfg = cfg.get("task", {})
    base_env = ENV_NAME_OVERRIDES.get(env_key) or task_cfg.get("env_name") or cfg.get("env_name")
    if base_env is None:
        raise ValueError(f"env_name missing in {cfg_path}")

    max_steps = task_cfg.get("max_path_length", 1000)
    is_adroit = is_adroit_env(base_env)
    episodes_per_task = args.episodes_adroit if is_adroit else args.episodes_mujoco

    results = {}
    task_means = []
    all_returns = []

    for task_idx in range(args.tasks_per_env):
        returns, mean_ret, std_ret = _eval_task(
            base_env,
            num_episodes=episodes_per_task,
            max_steps=max_steps,
            seed=args.seed + task_idx,
            is_adroit=is_adroit,
        )
        results[task_idx] = {
            "returns": returns,
            "mean_return": mean_ret,
            "std_return": std_ret,
            "episodes": episodes_per_task,
        }
        task_means.append(mean_ret)
        all_returns.extend(returns)

    summary = {
        "env": base_env,
        "episodes_per_task": episodes_per_task,
        "num_tasks": args.tasks_per_env,
        "mean_over_tasks": float(np.mean(task_means)) if task_means else 0.0,
        "std_over_tasks": float(np.std(task_means)) if task_means else 0.0,
        "mean_over_all_returns": float(np.mean(all_returns)) if all_returns else 0.0,
        "std_over_all_returns": float(np.std(all_returns)) if all_returns else 0.0,
    }

    return {"tasks": results, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random-policy baselines across six envs")
    parser.add_argument("--episodes-mujoco", type=int, default=50, dest="episodes_mujoco")
    parser.add_argument("--episodes-adroit", type=int, default=200, dest="episodes_adroit")
    parser.add_argument("--tasks-per-env", type=int, default=1, dest="tasks_per_env")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "results/random_baseline_random_policy")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    env_summaries = {}

    for env_key, cfg_path in ENV_CONFIGS.items():
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing config for {env_key}: {cfg_path}")
        print(f"\n=== Running random baseline for {env_key} ===")
        env_result = run_random_baseline(env_key, cfg_path, args)
        env_summaries[env_key] = env_result["summary"]
        out_file = args.output_dir / f"random_{env_key}.json"
        with out_file.open("w") as f:
            json.dump(env_result, f, indent=2)
        print(f"Saved {env_key} results to {out_file}")
        print(f"Mean over tasks: {env_result['summary']['mean_over_tasks']:.4f}")

    summary_path = args.output_dir / "random_overall_summary.json"
    with summary_path.open("w") as f:
        json.dump(env_summaries, f, indent=2)
    print(f"\nOverall summary saved to {summary_path}")


if __name__ == "__main__":
    main()
