import random
import time
import uuid
import os
import json
import re
from pathlib import Path
import wandb
import wandb.sdk.data_types.video as wv
import numpy as np
import torch
from omegaconf import OmegaConf

from cleandiffuser.env.wrapper import VideoRecordingWrapper


def parse_cfg(cfg_path: str) -> OmegaConf:
    """Parses a config file and returns an OmegaConf object."""
    base = OmegaConf.load(cfg_path)
    cli = OmegaConf.from_cli()
    for k,v in cli.items():
        if v == None:
            cli[k] = True
    base.merge_with(cli)
    return base


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")


def infer_custom_task_file(env_name: str) -> str:
    """Infer default custom task JSON path for a given env name."""
    env_lower = env_name.lower()
    if 'walker' in env_lower:
        env_dir = 'walker'
    elif 'halfcheetah' in env_lower or 'cheetah' in env_lower:
        env_dir = 'hc'
    elif 'ant' in env_lower:
        env_dir = 'ant'
    elif 'hopper' in env_lower:
        env_dir = 'hopper'
    else:
        raise ValueError(f"Cannot infer custom_task_file for environment '{env_name}'")
    data_root = Path(__file__).resolve().parents[1] / 'config' / 'ood result' / env_dir / 'dadp.json'
    return str(data_root)


def load_custom_task_info(env_name: str, custom_task_file: str | None):
    if not custom_task_file:
        custom_task_file = infer_custom_task_file(env_name)
        print(f"Inferred custom_task_file: {custom_task_file}")
    if not os.path.isfile(custom_task_file):
        raise FileNotFoundError(f"custom_task_file not found: {custom_task_file}")

    with open(custom_task_file, "r") as f:
        custom_task_json = json.load(f)

    file_env = custom_task_json.get("config", {}).get("env_name")
    if file_env and file_env != env_name:
        print(
            f"Warning: env_name in custom_task_file ({file_env}) does not match args.env_name ({env_name})"
        )

    custom_task_params = {}
    custom_task_ref_scores = {}
    for key, entry in custom_task_json.get("task_results", {}).items():
        params = entry.get("task_params")
        if params is None:
            continue
        custom_task_params[int(key)] = np.asarray(params, dtype=np.float32)
        custom_task_ref_scores[int(key)] = float(entry.get("ref_score", 0.0))

    if not custom_task_params:
        raise ValueError(f"No task_params found in custom_task_file {custom_task_file}")

    return custom_task_file, custom_task_params, custom_task_ref_scores


def is_adroit_env(env_name: str) -> bool:
    env_lower = env_name.lower()
    return ("door" in env_lower) or ("relocate" in env_lower)


def resolve_adroit_env(base_env: str, task_idx: int):
    """Return env name adjusted for Adroit difficulty and the level string if applicable."""
    if not is_adroit_env(base_env):
        return base_env, None
    difficulty_map = {0: "easy", 1: "hard", 2: "medium"}
    level = difficulty_map.get(int(task_idx))
    if level is None:
        return base_env, None

    updated_env, n_subs = re.subn(r"-(easy|medium|hard)-v(\d+)", fr"-{level}-v\2", base_env, count=1)
    if n_subs == 0:
        updated_env = re.sub(r"-v(\d+)$", fr"-{level}-v\1", base_env, count=1)
    return updated_env, level

def make_save_path(args):
        # base config
    base_path = f"{args.pipeline_name}_H{args.task.planner_horizon}_Jump{args.task.stride}_History{args.task.history}"
    base_path += f"_next{args.planner_next_obs_loss_weight}"
    # guidance type
    base_path += f"_{args.guidance_type}"
    # For Planner
    base_path += f"_{args.planner_net}"
    if args.planner_net == "transformer":
        base_path += f"_d{args.planner_depth}"
        base_path += f"_width{args.planner_d_model}"
    elif args.planner_net == "unet":
        base_path += f"_width{args.unet_dim}"
    
    if not args.planner_predict_noise:
        base_path += f"_pred_x0"
    
    # pipeline_type
    base_path += f"_{args.pipeline_type}"
    base_path += f"_dp{args.use_diffusion_invdyn}"
    base_path += f"_penalty{args.terminal_penalty}"
    base_path += f"_bonus{args.full_traj_bonus}"
    base_path += f"_gamma{args.discount}"
    base_path += f"_adv{args.use_weighted_regression}"
    base_path += f"_weight{args.weight_factor}_guide{args.planner_guide_noise_scale}_noise{args.noise_type}"
    # task name
    base_path += f"/{args.task.env_name}/{args.task.dataset}/"
    
    save_path = f"{args.save_dir}/" + base_path
    video_path = "video_outputs/" + base_path
    
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    
    if os.path.exists(video_path) is False:
        os.makedirs(video_path)
    return save_path, video_path
def save_cfg(args, save_path):
      # Save args to save_path
    args_save_path = os.path.join(save_path, 'args.json')
    os.makedirs(save_path, exist_ok=True)
    args_dict = vars(args).copy()
    if "task" in args_dict:
        args_dict["task"] = vars(args.task)
    # Convert args to dictionary and handle non-serializable objects
    args_dict_to_save = {}
    for key, value in args_dict.items():
        if key == 'task':
            args_dict_to_save[key] = value  # Already converted to dict above
        elif isinstance(value, (list, tuple, dict, str, int, float, bool, type(None))):
            args_dict_to_save[key] = value
        else:
            args_dict_to_save[key] = str(value)
    
    # Save to JSON
    with open(args_save_path, 'w') as f:
        json.dump(args_dict_to_save, f, indent=4)
    
    print(f"Arguments saved to {args_save_path}")
    
    # Also save in a more readable format
    args_txt_path = os.path.join(save_path, 'args.txt')
    with open(args_txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Training Arguments\n")
        f.write("=" * 80 + "\n\n")
        for key, value in sorted(args_dict_to_save.items()):
            if key == 'task':
                f.write(f"\n{key}:\n")
                for task_key, task_value in value.items():
                    f.write(f"  {task_key}: {task_value}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"Arguments also saved to {args_txt_path}")


def precollect_trajectory(
    env_eval,
    decision_maker,
    normalizer,
    args,
    *,
    task_idx,
    num_collect,
):
    if num_collect <= 0:
        return []

    act_dim = decision_maker.act_dim
    embedding_history = getattr(decision_maker.planner, "embedding_history", decision_maker.history)
    history_len = max(decision_maker.history, embedding_history)

    print(f"Pre-collecting {num_collect} episode embeddings for task {task_idx}...")
    precollected_pool = []
    collected = 0
    while collected < num_collect:
        obs_history = []
        obs, cum_done, t = env_eval.reset(), 0.0, 0
        for _ in range(history_len):
            obs_history.append(
                torch.cat(
                    [
                        torch.tensor(
                            normalizer.normalize(obs),
                            device=args.device,
                            dtype=torch.float32,
                        ),
                        torch.zeros((args.num_envs, act_dim), device=args.device),
                    ],
                    dim=-1,
                )
            )

        episode_done = np.zeros(args.num_envs, dtype=bool)
        while not np.all(episode_done) and t < args.task.max_path_length + 1:
            current_obs = torch.tensor(
                normalizer.normalize(obs),
                device=args.device,
                dtype=torch.float32,
            )
            obs_history_tensor = torch.stack(obs_history, dim=1)
            with torch.no_grad():
                actions = decision_maker.predict(
                    obs_history=obs_history_tensor,
                    cnt_obs=current_obs,
                    num_candidates=1,
                    task_id=task_idx,
                )
            actions = actions.clamp(-1, 1)
            obs_history.append(torch.cat([current_obs, actions], dim=-1))
            if len(obs_history) > history_len:
                obs_history.pop(0)
            obs, _, done, _ = env_eval.step(actions.cpu().numpy())
            t += 1
            episode_done = done if episode_done is None else np.logical_or(episode_done, done)

        obs_history_tensor_end = torch.stack(obs_history, dim=1)
        for i in range(args.num_envs):
            precollected_pool.append(obs_history_tensor_end[i].detach())
            collected += 1
            if collected >= num_collect:
                break

    print(f"Collected {collected} embeddings for task {task_idx}.")
    return precollected_pool


def update_eval_results(
    task_results,
    overall_episode_rewards,
    overall_success_rates,
    *,
    task_idx=None,
    task=None,
    ref_score=None,
    episode_rewards=None,
    episode_success_rates=None,
    sorted_task_ids=None,
    num_episodes=None,
    eval_seed=None,
    save_path=None,
    args=None,
    task_ids_to_eval=None,
    training_ids=None,
    test_ids=None,
    finalize=False,
):
    if task_idx is not None:
        episode_rewards = np.array(episode_rewards)
        episode_mean_rewards = episode_rewards.mean(axis=1)
        episode_rewards_flat = episode_rewards.reshape(-1)
        mean_reward = float(np.mean(episode_rewards_flat))
        std_error = float(np.std(episode_rewards_flat) / np.sqrt(len(episode_rewards_flat)))
        mean_success_rate = float(np.mean(episode_success_rates)) if episode_success_rates else 0.0
        std_success_rate = (
            float(np.std(episode_success_rates) / np.sqrt(len(episode_success_rates)))
            if episode_success_rates
            else 0.0
        )

        task_results[task_idx] = {
            "task_params": task.tolist() if isinstance(task, np.ndarray) else task,
            "mean_reward": mean_reward,
            "std_error": std_error,
            "ref_score": float(ref_score[0]),
            "episode_rewards": episode_rewards_flat.tolist(),
            "episode_mean_rewards": episode_mean_rewards.tolist(),
            "episode_success_rates": episode_success_rates,
            "mean_success_rate": mean_success_rate,
            "std_success_rate": std_success_rate,
            "num_episodes": len(episode_mean_rewards),
        }

        overall_episode_rewards.extend(episode_rewards_flat.tolist())
        overall_success_rates.append(mean_success_rate)

        print("\n  RESULTS:")
        print(f"  Mean Reward: {mean_reward:.4f} ± {std_error:.4f}")
        print(f"  Mean Success Rate: {mean_success_rate:.4f} ± {std_success_rate:.4f}")
        print(f"  Reference Score: {ref_score[0]:.4f}")
        return mean_reward, std_error, mean_success_rate, std_success_rate

    if not finalize:
        return None

    if sorted_task_ids is None or num_episodes is None:
        raise ValueError("sorted_task_ids and num_episodes are required when finalize=True")

    overall_stats = {
        "mean_episode_reward": float(np.mean(overall_episode_rewards)) if overall_episode_rewards else 0.0,
        "std_episode_reward": float(np.std(overall_episode_rewards)) if overall_episode_rewards else 0.0,
        "num_episode_samples": len(overall_episode_rewards),
        "num_tasks_evaluated": len(sorted_task_ids),
    }

    if overall_success_rates:
        overall_stats["mean_success_rate"] = float(np.mean(overall_success_rates))
        overall_stats["std_success_rate"] = float(np.std(overall_success_rates))

    def _episode_mean_series(task_ids):
        series = []
        for episode_idx in range(num_episodes):
            per_task_episode_means = [
                task_results[tid]["episode_mean_rewards"][episode_idx]
                for tid in task_ids
                if tid in task_results and len(task_results[tid]["episode_mean_rewards"]) > episode_idx
            ]
            if per_task_episode_means:
                series.append(float(np.mean(per_task_episode_means)))
        return series

    all_task_episode_means = _episode_mean_series(sorted_task_ids)
    if all_task_episode_means:
        overall_stats["all_tasks_episode_std"] = float(np.std(all_task_episode_means))

    first_5_ids = sorted_task_ids[:5]
    first_5_rewards = [task_results[tid]["mean_reward"] for tid in first_5_ids if tid in task_results]
    first_5_episode_means = _episode_mean_series(first_5_ids)
    if first_5_rewards:
        overall_stats["first_5_mean_reward"] = float(np.mean(first_5_rewards))
        overall_stats["first_5_std_reward"] = (
            float(np.std(first_5_episode_means)) if first_5_episode_means else 0.0
        )

    last_5_ids = sorted_task_ids[-5:]
    last_5_rewards = [task_results[tid]["mean_reward"] for tid in last_5_ids if tid in task_results]
    last_5_episode_means = _episode_mean_series(last_5_ids)
    if last_5_rewards:
        overall_stats["last_5_mean_reward"] = float(np.mean(last_5_rewards))
        overall_stats["last_5_std_reward"] = (
            float(np.std(last_5_episode_means)) if last_5_episode_means else 0.0
        )

    print(f"\n{'='*70}")
    print("SUMMARY ACROSS EPISODES")
    print(f"{'='*70}")
    print(f"Seed Used: {eval_seed}")
    print(f"Tasks Evaluated: {len(sorted_task_ids)}")
    print(f"Episode Samples: {overall_stats['num_episode_samples']}")
    print(
        f"Mean Episode Reward: {overall_stats['mean_episode_reward']:.4f} ± {overall_stats['std_episode_reward']:.4f}"
    )
    if "all_tasks_episode_std" in overall_stats:
        print(f"Std Across Episode Means (all tasks): {overall_stats['all_tasks_episode_std']:.4f}")
    if first_5_rewards:
        print(
            f"First 5 Tasks ({first_5_ids}) Mean Reward: {overall_stats['first_5_mean_reward']:.4f} "
            f"± {overall_stats['first_5_std_reward']:.4f}"
        )
    if last_5_rewards:
        print(
            f"Last 5 Tasks ({last_5_ids}) Mean Reward: {overall_stats['last_5_mean_reward']:.4f} "
            f"± {overall_stats['last_5_std_reward']:.4f}"
        )

    if save_path is not None and args is not None:
        results_to_save = {
            "task_results": task_results,
            "overall_stats": overall_stats,
            "config": {
                "num_envs": args.num_envs,
                "num_episodes": args.num_episodes,
                "task_ids_evaluated": task_ids_to_eval,
                "training_task_ids": training_ids,
                "test_task_ids": test_ids,
                "planner_ckpt": args.planner_ckpt,
                "policy_ckpt": args.policy_ckpt,
                "env_name": args.task.env_name,
                "pipeline_type": args.pipeline_type,
                "noise_type": args.noise_type,
                "seed": eval_seed,
            },
        }

        results_file = os.path.join(save_path, f"task{args.eval_task_ids}_random_embedding.json")
        with open(results_file, "w") as f:
            json.dump(results_to_save, f, indent=2)
        print(f"Results saved to: {results_file}")
    return overall_stats
class Timer:
    def __init__(self):
        self.tik = None

    def start(self):
        self.tik = time.time()

    def stop(self):
        return time.time() - self.tik
    
    
class Logger:
    """Primary logger object. Logs in wandb."""
    def __init__(self, log_dir, cfg):
        self._log_dir = make_dir(log_dir)
        self._model_dir = make_dir(self._log_dir / 'models')
        self._video_dir = make_dir(self._log_dir / 'videos')
        self._cfg = cfg

        wandb.init(
            config=OmegaConf.to_container(cfg),
            project=cfg.project,
            group=cfg.group,
            name=cfg.exp_name,
            id=str(uuid.uuid4()),
            mode=cfg.wandb_mode,
            dir=self._log_dir
        )
        self._wandb = wandb

    def video_init(self, env, enable=False, video_id=""):
        # assert isinstance(env.env, VideoRecordingWrapper)
        if isinstance(env.env, VideoRecordingWrapper):
            video_env = env.env
        else:
            video_env = env
        if enable:
            video_env.video_recoder.stop()
            video_filename = os.path.join(self._video_dir, f"{video_id}_{wv.util.generate_id()}.mp4")
            video_env.file_path = str(video_filename)
        else:
            video_env.file_path = None
            
    def log(self, d, category):
        assert category in ['train', 'inference']
        assert 'step' in d
        print(f"[{d['step']}]", " / ".join(f"{k} {v:.2f}" for k, v in d.items()))
        with (self._log_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": d['step'], **d}) + "\n")
        _d = dict()
        for k, v in d.items():
            _d[category + "/" + k] = v
        self._wandb.log(_d, step=d['step'])
        
    def save_agent(self, agent=None, identifier='final'):
        if agent:
            fp = self._model_dir / f'model_{str(identifier)}.pt'
        agent.save(fp)
        print(f"model_{str(identifier)} saved")

    def finish(self, agent):
        try:
            self.save_agent(agent)
        except Exception as e:
            print(f"Failed to save model: {e}")
        if self._wandb:
            self._wandb.finish()


    
    


    
