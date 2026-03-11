#!/usr/bin/env python
import argparse
import json
import os
import shutil
from argparse import Namespace
from types import SimpleNamespace

from utils.pipelines_utils import make_save_path
from train_diffusion import pipeline

def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    v_low = v.lower()
    if v_low in ("yes", "true", "t", "y", "1"):
        return True
    if v_low in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _load_json_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _dict_to_namespace(cfg: dict) -> Namespace:
    cfg = cfg.copy()
    task_cfg = cfg.pop("task", None) or {}
    ns = Namespace(**cfg)
    task_ns = SimpleNamespace(**task_cfg)
    ns.task = task_ns
    return ns


def _sanitize_name(name: str) -> str:
    return name.replace("/", "-").replace(" ", "_")


def _is_adroit_env(env_name: str) -> bool:
    env_lower = env_name.lower()
    return ("door" in env_lower) or ("relocate" in env_lower)


def _infer_mujoco_dataset(env_name: str) -> str | None:
    env_lower = env_name.lower()
    if "randomant" in env_lower:
        return "RandomAnt/28dynamics-v0"
    if "randomhalfcheetah" in env_lower:
        return "RandomHalfCheetah/28dynamics-v0"
    if "randomwalker2d" in env_lower:
        return "RandomWalker2d/28dynamics-v0"
    if "randomhopper" in env_lower:
        return "RandomHopper/28dynamics-v0"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch train/test runs from stored args.json")
    parser.add_argument("--config", required=True, help="Path to args.json template")
    parser.add_argument("--seed", type=int, default=None, help="Seed override")
    parser.add_argument("--mode", choices=["train", "test", "inference"], default=None, help="Run mode override")
    parser.add_argument("--device", default=None, help="Device override, e.g., cuda:0")
    parser.add_argument("--save_dir", default=None, help="Root save dir override (keeps subfolder layout)")
    parser.add_argument("--pipeline_suffix", default=None, help="Suffix appended to pipeline_name and name")
    parser.add_argument("--customize_task", type=str2bool, default=None, help="Force customize_task flag")
    parser.add_argument("--eval_task_mode", choices=["as-config", "id", "ood"], default="as-config", help="Eval task selection mode")
    parser.add_argument("--eval_out_dir", default=None, help="Folder to collect evaluation JSON outputs")
    parser.add_argument("--num_envs", type=int, default=None, help="Override num_envs during eval")
    parser.add_argument("--num_episodes", type=int, default=None, help="Override num_episodes during eval")
    parser.add_argument("--enable_wandb", type=str2bool, default=None, help="Override wandb toggle")
    parser.add_argument("--planner_ckpt", default="last", help="Checkpoint name for planner load")
    parser.add_argument("--policy_ckpt", default=None, help="Checkpoint name for policy load")
    parser.add_argument("--critic_ckpt", default=None, help="Checkpoint name for critic load")
    parser.add_argument("--dadp_checkpoint_path", default=None, help="Override dadp_checkpoint_path")
    parser.add_argument("--precollect_episodes", default=0, help="Override dadp_checkpoint_path")
    parser.add_argument("--condition", type=str2bool, default=None, help="Override condition flag (leave unset to keep config value)")
    parser.add_argument("--dataset", default="auto", help="Override task.dataset name (e.g., RandomWalker2d/28dynamics-v9). Use 'auto' to infer from env_name.")
    args = parser.parse_args()

    cfg = _load_json_cfg(args.config)

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.mode is not None:
        cfg["mode"] = args.mode
    if args.device is not None:
        cfg["device"] = args.device
    if args.save_dir is not None:
        cfg["save_dir"] = args.save_dir
    if args.num_envs is not None:
        cfg["num_envs"] = args.num_envs
    if args.num_episodes is not None:
        cfg["num_episodes"] = args.num_episodes
    if args.enable_wandb is not None:
        cfg["enable_wandb"] = args.enable_wandb
    if args.planner_ckpt is not None:
        cfg["planner_ckpt"] = args.planner_ckpt
    if args.policy_ckpt is not None:
        cfg["policy_ckpt"] = args.policy_ckpt
    if args.critic_ckpt is not None:
        cfg["critic_ckpt"] = args.critic_ckpt
    if args.dadp_checkpoint_path is not None:
        cfg["dadp_checkpoint_path"] = args.dadp_checkpoint_path
    if args.precollect_episodes is not None:
        cfg["precollect_episodes"] = args.precollect_episodes
    if args.condition is not None:
        cfg["condition"] = args.condition
    if args.dataset is not None:
        cfg.setdefault("task", {})
        if args.dataset == "auto":
            inferred_dataset = _infer_mujoco_dataset(cfg.get("task", {}).get("env_name", ""))
            if inferred_dataset:
                cfg["task"]["dataset"] = inferred_dataset
                cfg["dataset"] = inferred_dataset  # also set top-level for convenience
        else:
            cfg["task"]["dataset"] = args.dataset
            cfg["dataset"] = args.dataset  # also set top-level for convenience



    suffix = args.pipeline_suffix or ""
    if suffix:
        cfg["pipeline_name"] = f"{cfg.get('pipeline_name', 'exp')}{suffix}"
        cfg["name"] = f"{cfg.get('name', 'exp')}{suffix}"

    # Apply customize_task override early so eval_task_mode logic can override if needed
    if args.customize_task is not None:
        cfg["customize_task"] = args.customize_task

    is_adroit = _is_adroit_env(cfg.get("task", {}).get("env_name", ""))

    if args.eval_task_mode == "id":
        cfg["customize_task"] = False
        if is_adroit:
            cfg["training_task_ids"] = [0, 1]
            cfg["eval_task_ids"] = [0, 1]
        elif cfg.get("training_task_ids"):
            cfg["eval_task_ids"] = cfg["training_task_ids"]
    elif args.eval_task_mode == "ood":
        if is_adroit:
            cfg["customize_task"] = False
            cfg["test_task_ids"] = [2]
            cfg["eval_task_ids"] = [2]
        else:
            cfg["customize_task"] = True
            cfg["eval_task_ids"] = [1, 2, 3, 4, 5]

    ns = _dict_to_namespace(cfg)

    save_path, _ = make_save_path(ns)
    os.makedirs(save_path, exist_ok=True)

    pipeline(ns)

    if ns.mode in ("test", "inference") and args.eval_out_dir:
        results_file = os.path.join(save_path, f"task{ns.eval_task_ids}_random_embedding.json")
        if os.path.isfile(results_file):
            os.makedirs(args.eval_out_dir, exist_ok=True)
            # Treat explicit OOD runs as OOD even for Adroit (customize_task stays False there)
            eval_kind = "ood" if args.eval_task_mode == "ood" else "id"
            target_name = f"{_sanitize_name(ns.task.env_name)}_{_sanitize_name(ns.pipeline_name)}_seed{ns.seed}_{eval_kind}.json"
            shutil.copy2(results_file, os.path.join(args.eval_out_dir, target_name))
        else:
            print(f"Warning: evaluation file not found for copying: {results_file}")

if __name__ == "__main__":
    main()
