import os
import minari
import torch
import random
import dr_envs
import gymnasium as gym
from omegaconf import OmegaConf
from utils.pipelines_utils import (
    set_seed,
    make_save_path,
    save_cfg,
    is_adroit_env,
    resolve_adroit_env,
    update_eval_results,
    precollect_trajectory,
    load_custom_task_info,
)
import wandb, uuid
from cleandiffuser.dataset.dataset_utils import create_splited_dataloader
from cleandiffuser.dataset.d4rl_mujoco_dataset import EmbeddingMuJoCoSeqDataset, get_task_data
from diffusionmodel import Planner, DecisionMaker
from dadp.dadp import DADP
import argparse
from customwrappers.RandomVecEnv import RandomSubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import imageio
import time
import numpy as np
from torch.utils.data import DataLoader
import json

def pipeline(args):
    args.device = args.device if torch.cuda.is_available() else "cpu"
    args_dict = vars(args).copy()
    if "task" in args_dict:
        args_dict["task"] = vars(args.task)
    if args.enable_wandb and args.mode in ["inference", "train"]:
        wandb.require("core")
        print(args)
        wandb.init(
            reinit=True,
            id=str(uuid.uuid4()),
            project=str(args.project),
            group=str(args.group),
            name=str(args.name),
            config=OmegaConf.to_container(OmegaConf.create(args_dict), resolve=True)
        )

    # Enforce deterministic behavior across Python, NumPy, and PyTorch
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    set_seed(args.seed)


    save_path, video_path = make_save_path(args)

    dataset = minari.load_dataset(f"{args.task.dataset}")

    # ---------------------- Create Dataset ----------------------
    model, metadata = DADP.load_checkpoint(args.dadp_checkpoint_path, "cpu")
    model = model.to(args.device)
    
    # Calculate max_history for padding
    embedding_history = getattr(model.training_config, "history", args.task.history)
    max_history = max(args.task.history, embedding_history)

    co_train_embeddings = bool(args.co_train_embedding and args.mode == "train")
    args.co_train_embedding = co_train_embeddings
    if co_train_embeddings:
        model.train()
    else:
        model.eval()

    predict_embedding_flag = args.predict_embedding
    if predict_embedding_flag is None:
        predict_embedding_flag = args.noise_type == "mixed_ddim"
    args.predict_embedding = bool(predict_embedding_flag)
    planner_dataset = EmbeddingMuJoCoSeqDataset(
        dataset, horizon=args.dataset_horizon, discount=args.discount, 
        stride=args.task.stride, center_mapping=(args.guidance_type!="cfg"),
        terminal_penalty=args.terminal_penalty,max_path_length=args.task.max_path_length,
        full_traj_bonus=args.full_traj_bonus,padding=max_history, 
        save_embedding_path=os.path.join(os.path.dirname(args.dadp_checkpoint_path), "embeddings_data.npz"),
    )

    obs_dim, act_dim = planner_dataset.o_dim, planner_dataset.a_dim
    training_ids = getattr(args, 'training_task_ids', None)
    test_ids = getattr(args, 'test_task_ids', None)
    if training_ids is None or test_ids is None:
        raise ValueError("training_task_ids and test_task_ids must be provided via config or CLI")

    training_planner_dataset = get_task_data(planner_dataset, training_ids)
    test_task_dataset = get_task_data(planner_dataset, test_ids)
    args.training_task_ids = training_ids
    args.test_task_ids = test_ids

    # DataLoader seeding helpers for deterministic workers
    def _seed_worker(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    dl_generator = torch.Generator()
    dl_generator.manual_seed(args.seed)

    planner_dataloader, planner_val_dataloader = create_splited_dataloader(
        training_planner_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    test_task_dataloader = DataLoader(
        test_task_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=dl_generator,
    )
    # ---------------------- Create Model Classes ----------------------
    planner_model = Planner(
        obs_dim=obs_dim,
        act_dim=act_dim,
        planner_dataset=training_planner_dataset,
        planner_horizon=args.task.planner_horizon,
        history=args.task.history,
        emb_dim=args.planner_emb_dim,
        d_model=args.planner_d_model,
        depth=args.planner_depth,
        planner_ema_rate=args.planner_ema_rate,
        planner_predict_noise=args.planner_predict_noise,
        planner_next_obs_loss_weight=args.planner_next_obs_loss_weight,
        planner_guide_noise_scale=args.planner_guide_noise_scale,
        planner_noise_type=args.noise_type,
        sample_steps=args.planner_sampling_steps,
        pipeline_type=args.pipeline_type,
        attention_mask=args.attention_mask,
        device=args.device,
        model_path=save_path,
        embedding_model=model,
        nnCondition=args.condition,
        env_type=args.task.env_name,
        train_embedding_model=co_train_embeddings,
        embedding_learning_rate=args.embedding_learning_rate,
        predict_embedding=args.predict_embedding,
    )

    # ---------------------- Training ----------------------
    if args.mode == "train":
        save_cfg(args, save_path)
        print("Starting individual model training...")
        
        # Train Planner
        planner_start_step = 0
        if args.resume_planner_ckpt is not None:
            planner_start_step = int(args.resume_planner_ckpt)
            if planner_start_step >= args.planner_diffusion_gradient_steps:
                raise ValueError(
                    f"resume_planner_ckpt ({planner_start_step}) must be < planner_diffusion_gradient_steps "
                    f"({args.planner_diffusion_gradient_steps})"
                )
            print(f"Resuming planner from checkpoint step {planner_start_step}...")

            planner_model.load(planner_start_step)

        elif args.co_train_embedding:
            planner_model.load(args.planner_ckpt)
            try:
                planner_start_step = int(args.planner_ckpt)
            except (TypeError, ValueError):
                planner_start_step = 0
        print("Training Planner...")
        planner_model.train_model(
            dataloader=planner_dataloader,
            val_dataloader=planner_val_dataloader,
            test_task_dataloader=test_task_dataloader,
            gradient_steps=args.planner_diffusion_gradient_steps,
            use_weighted_regression=bool(args.use_weighted_regression),
            weight_factor=args.weight_factor,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            evaluate_batch=100,
            enable_wandb=args.enable_wandb,
            start_step=planner_start_step
        )
        
    if args.mode in ["inference", "test"]:
        planner_model.load(args.planner_ckpt)

        decision_maker = DecisionMaker(
            obs_dim=obs_dim,
            act_dim=act_dim,
            planner_dataset=training_planner_dataset,
            planner_horizon=args.task.planner_horizon,
            history=args.task.history,
            emb_dim=args.planner_emb_dim,
            d_model=args.planner_d_model,
            depth=args.planner_depth,
            planner_ema_rate=args.planner_ema_rate,
            planner_predict_noise=args.planner_predict_noise,
            planner_next_obs_loss_weight=args.planner_next_obs_loss_weight,
            planner_guide_noise_scale=args.planner_guide_noise_scale,
            planner_sample_steps=args.planner_sampling_steps,
            predict_embedding=args.predict_embedding,
            pipeline_type=args.pipeline_type,
            attention_mask=args.attention_mask,
            device=args.device,
            model_path=save_path,
            embedding_model_path=args.dadp_checkpoint_path,
            noise_type=args.noise_type,
            nnCondition=args.condition,
        )

        decision_maker.load_ckpt(planner_ckpt=args.planner_ckpt, device=args.device)
        decision_maker.eval()

        # Prepare task list and default evaluation selection
        normalizer = planner_dataset.get_normalizer()
        task_list = planner_dataset.task_list
        custom_task_params = {}
        custom_task_ref_scores = {}
        if getattr(args, 'customize_task', False):
            (
                custom_task_file,
                custom_task_params,
                custom_task_ref_scores,
            ) = load_custom_task_info(args.task.env_name, getattr(args, 'custom_task_file', None))
            setattr(args, 'custom_task_file', custom_task_file)
            
        if hasattr(args, 'eval_task_ids') and args.eval_task_ids is not None:
            task_ids_to_eval = args.eval_task_ids
        else:
            # Default: evaluate on all tasks or specify subset
            task_ids_to_eval = list(range(len(task_list)))
            # Or specify: task_ids_to_eval = [0, 3, 6, 9]
        eval_seed = int(getattr(args, "seed", 0))

        print(f"\n{'='*70}")
        print(f"EVALUATING ON {len(task_ids_to_eval)} TASKS: {task_ids_to_eval}")
        print(f"USING SEED: {eval_seed}")
        print(f"{'='*70}\n")

        batch = next(iter(planner_val_dataloader))
        sorted_task_ids = sorted(task_ids_to_eval)

        set_seed(eval_seed)
        env_eval = None
        current_env_name = None


        task_results = {}
        overall_episode_rewards = []
        overall_success_rates = []
        precollected_pool = []

        
        for task_idx in task_ids_to_eval:

            print(f"\n{'-'*70}")
            print(f"TASK {task_idx}/{len(task_list)-1}")
            print(f"{'-'*70}")
            # Re-seed per task so stochastic env/model noise is consistent regardless of evaluation ordering


            # Choose which task to use based on argument
            if getattr(args, 'customize_task', False):
                if task_idx not in custom_task_params:
                    raise KeyError(f"Task id {task_idx} not found in custom task file {args.custom_task_file}")
                task = custom_task_params[task_idx]
                ref_score = [custom_task_ref_scores.get(task_idx, 0.0)]
                print(f"Loaded custom task from {args.custom_task_file}: {task}")
            else:
                task = task_list[task_idx]
                ref_score = planner_dataset.get_ref_score(task_idx)
                print(f"Task parameters: {task}")
                print(f"Reference score: {ref_score[0]:.2f}")

            target_env_name, adroit_level = resolve_adroit_env(args.task.env_name, task_idx)
            if adroit_level is not None:
                print(f"Adroit difficulty for task {task_idx}: {adroit_level} ({target_env_name})")

            if env_eval is None or target_env_name != current_env_name:
                if env_eval is not None:
                    env_eval.close()
                env_eval = make_vec_env(
                    target_env_name,
                    n_envs=args.num_envs,
                    seed=eval_seed,
                    vec_env_cls=RandomSubprocVecEnv
                )
                current_env_name = target_env_name

            set_seed(eval_seed + task_idx)
            env_eval.seed(eval_seed + task_idx)

            task_array = np.tile(task, (args.num_envs, 1))
            env_name_lower = target_env_name.lower()
            if not ("door" in env_name_lower or "hammer" in env_name_lower or "relocate" in env_name_lower):
                if hasattr(env_eval.unwrapped, 'set_task'):
                    env_eval.set_task(task_array)

            episode_rewards = []
            episode_success_rates = []
            # Determine max history needed for buffer
            embedding_history = getattr(decision_maker.planner, "embedding_history", args.task.history)
            history_length = max(args.task.history, embedding_history)

            if args.precollect_episodes > 0:
                precollected_pool = precollect_trajectory(
                    env_eval,
                    decision_maker,
                    normalizer,
                    args,
                    task_idx=task_idx,
                    num_collect=args.precollect_episodes,
                )
            
            frames = [] if args.plot else None
            previous_episode_history = [None] * args.num_envs

            print(f"Running {args.num_episodes} episodes...")
            for episode in range(args.num_episodes):
                obs_history = []
                obs, ep_reward, cum_done, t = env_eval.reset(), 0.0, 0.0, 0
                episode_success_flags = np.zeros(args.num_envs, dtype=bool)

                for i in range(history_length):
                    obs_history.append(torch.cat([
                        torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32),
                        torch.zeros((args.num_envs, act_dim), device=args.device)
                    ], dim=-1))

                while not np.all(cum_done) and t < args.task.max_path_length + 1:
                    current_obs = torch.tensor(
                        normalizer.normalize(obs),
                        device=args.device,
                        dtype=torch.float32
                    )

                    obs_history_tensor = torch.stack(obs_history, dim=1)

                    with torch.no_grad():
                        embedding_traj_arg = None
                        if args.precollect_episodes > 0 and precollected_pool:
                            # Sample a fresh embedding trajectory for each env at every timestep.
                            embedding_traj_arg = torch.stack(
                                [random.choice(precollected_pool) for _ in range(args.num_envs)], dim=0
                            )
                        actions = decision_maker.predict(
                            obs_history=obs_history_tensor,
                            cnt_obs=current_obs,
                            num_candidates=1,
                            task_id=task_idx,
                            embedding_traj=embedding_traj_arg,
                        )

                    actions = actions.clamp(-1, 1)
                    obs_history.append(torch.cat([current_obs, actions], dim=-1))
                    if len(obs_history) > history_length:
                        obs_history.pop(0)

                    actions_np = actions.cpu().numpy()

                    if args.plot:
                        images = env_eval.get_images()
                        if len(images) == 1:
                            frame = images[0]
                        else:
                            frame = np.concatenate(images, axis=1)
                        frames.append(frame)

                    obs, rew, done, info = env_eval.step(actions_np)

                    # Track success every step (env may not terminate on success)
                    achieved_step = np.zeros_like(episode_success_flags, dtype=bool)
                    if isinstance(info, (list, tuple)):
                        for i in range(min(len(info), args.num_envs)):
                            if isinstance(info[i], dict):
                                val = info[i].get('goal_achieved', False)
                                if isinstance(val, np.ndarray):
                                    achieved_step[i] = bool(val[i]) if val.size > i else bool(val.any())
                                else:
                                    achieved_step[i] = bool(val)
                    elif isinstance(info, dict):
                        val = info.get('goal_achieved', False)
                        if isinstance(val, np.ndarray):
                            achieved_step = achieved_step | val.astype(bool)[:len(achieved_step)]
                        else:
                            achieved_step = np.full_like(achieved_step, bool(val))
                    episode_success_flags = np.logical_or(episode_success_flags, achieved_step)

                    prev_cum_done = cum_done
                    obs_history_tensor_done = torch.stack(obs_history, dim=1)
                    newly_done = np.logical_and(np.logical_not(prev_cum_done), done)

                    if np.any(newly_done):
                        for i in np.where(newly_done)[0]:
                            previous_episode_history[i] = obs_history_tensor_done[i].detach()

                    t += 1
                    cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                    ep_reward += (rew * (1 - cum_done)) if t < args.task.max_path_length else rew

                    if episode > 15 and t % 50 == 0:
                        active_envs = int(np.sum(1 - cum_done))
                        print(f'  [Episode {episode+1}, t={t}] Active envs: {active_envs}/{args.num_envs},success rate: {episode_success_flags}', end='\r')
                    if np.any(episode_success_flags):
                        print(f'  [Episode {episode+1}, t={t}],success rate: {episode_success_flags}, reward: {np.around(ep_reward, 2)}', end='\r')

                episode_rewards.append(ep_reward)
                episode_success_rates.append(float(np.mean(episode_success_flags)))
                print(f'  Episode {episode+1}/{args.num_episodes} completed - Rewards: {np.around(ep_reward, 2)}')
                embedding_file = os.path.join(
                    save_path,
                    f"task_{task_idx}_episode_{episode+1}_embeddings_denoise_step.npy"
                )
                if args.plot and frames:
                    video_dir = f"{args.video_save_path}/{args.task.env_name}/task_{task_idx}_episode_{episode+1}_condition_dt1"
                    os.makedirs(video_dir, exist_ok=True)
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    frames_resized = [frames[i] for i in range(0, len(frames), 1)]
                    video_file = f"{video_dir}/eval_{timestamp}.gif"
                    imageio.mimsave(video_file, frames_resized, fps=30)
                    print(f"  Video saved: {video_file}")
                    frames = []  # Clear frames to save memory
                os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
                decision_maker.planner.save_embedding(embedding_file)

                obs_history_tensor_end = torch.stack(obs_history, dim=1)
                for i in range(args.num_envs):
                    if previous_episode_history[i] is None:
                        previous_episode_history[i] = obs_history_tensor_end[i].detach()

            update_eval_results(
                task_results=task_results,
                overall_episode_rewards=overall_episode_rewards,
                overall_success_rates=overall_success_rates,
                task_idx=task_idx,
                task=task,
                ref_score=ref_score,
                episode_rewards=episode_rewards,
                episode_success_rates=episode_success_rates,
            )

        if env_eval is not None:
            env_eval.close()

        update_eval_results(
            task_results=task_results,
            overall_episode_rewards=overall_episode_rewards,
            overall_success_rates=overall_success_rates,
            sorted_task_ids=sorted_task_ids,
            num_episodes=args.num_episodes,
            eval_seed=eval_seed,
            save_path=save_path,
            args=args,
            task_ids_to_eval=task_ids_to_eval,
            training_ids=training_ids,
            test_ids=test_ids,
            finalize=True,
        )
        
if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    # Task arguments
    parser.add_argument('--env_name', type=str, default="RandomHalfCheetah-v0", help='Environment name')
    parser.add_argument('--planner_horizon', type=int, default=20 ,help='Planner horizon')
    parser.add_argument('--dataset', type=str, default="RandomHalfCheetah/82dynamics-v7", help="dataset name")
    # Main arguments
    parser.add_argument('--pipeline_name', default="exp_halfcheetah_28_test", type=str, help='Pipeline name')
    parser.add_argument('--name', default="exp_halfcheetah_28", type=str, help='Name for wandb logging')
    parser.add_argument('--mode', default="train", type=str, help='Mode: train/inference/test/etc')
    default_eval_tasks = [3, 4, 5]
    parser.add_argument(
        '--eval_task_ids',
        default=default_eval_tasks,
        type=lambda s: [int(item) for item in s.split(',') if item.strip()],
        help='Comma-separated list of task IDs to evaluate (e.g., "0,3,6,9"). Defaults to curated list if omitted.'
    )
    parser.add_argument('--customize_task', default=False, help='Use a custom generated task instead of task from task list')
    parser.add_argument(
        '--custom_task_file',
        default=None,
        type=str,
        help='Path to JSON file containing custom task parameters to load when customize_task is True. If omitted, an environment-specific default is inferred.'
    )
    parser.add_argument(
        '--training_task_ids',
        default="0:25",
        type=lambda s: [int(item) for item in s.split(',') if item.strip()],
        help='Comma-separated task IDs used for training subset selection'
    )
    parser.add_argument(
        '--test_task_ids',
        default=None,
        type=lambda s: [int(item) for item in s.split(',') if item.strip()],
        help='Comma-separated task IDs reserved for validation subset selection'
    )
    parser.add_argument('--device', default="cuda:1", type=str, help='Device to use')
    parser.add_argument('--enable_wandb', default=False, type=bool, help='Enable wandb logging')

    parser.add_argument('--noise_type', default='mixed_ddim', type=str, help='The type of noise to be added to the dynamics,e.g.,env_factor_guided,embedding_guided,standard')
    parser.add_argument('--predict_embedding', default=True, type=str2bool,
                        help='Set to true to force planner embedding prediction; defaults to noise-type heuristic when omitted')
    parser.add_argument("--dadp_checkpoint_path", type=str, default="./dadp/embedding/logs/transformer/exp_halfcheetah_28/best_model.zip", help="Path to the DADP checkpoint zip file")
    parser.add_argument('--pipeline_type', default="joint", type=str, help='Pipeline type: separate/joint')
    parser.add_argument('--co_train_embedding', default=False, type=str2bool, help='Jointly update the DADP embedding encoder while training the planner')
    parser.add_argument('--embedding_learning_rate', default=None, type=float, help='Optional learning rate override for embedding co-training')
    parser.add_argument('--resume_planner_ckpt', default=None, type=str,
                        help='Planner checkpoint step (numeric, e.g., 500000) to resume training from')
    parser.add_argument('--planner_guide_noise_scale', default=0.5, type=float,
                        help='Noise scale applied during planner guidance sampling')

    parser.add_argument('--condition', default=False, type=str2bool, help='Use condition in diffusion models')

    # Optional pre-collection of embedding trajectories for inference
    parser.add_argument(
        '--precollect_episodes',
        default=0,
        type=int,
        help='If > 0, collect this many episodes before evaluation and sample from the pool as embedding context during inference.'
    )

    # Inference
    parser.add_argument('--num_envs', default=10, type=int, help='Number of environments')
    parser.add_argument('--num_episodes', default=5, type=int, help='Number of episodes')
    # Load external defaults from config/default_args.json (CLI still overrides)
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'default_args.json')
    with open(config_path, 'r') as cf:
        cfg = json.load(cf)
    parser.set_defaults(**cfg)

    args = parser.parse_args()

    # Build a task namespace
    class Task:
        pass
    task = Task()
    task.env_name = args.env_name
    task.planner_horizon = args.planner_horizon
    task.history = args.history
    task.dataset = args.dataset
    task.stride = args.stride
    task.max_path_length = args.max_path_length
    task.planner_temperature = args.planner_temperature
    task.planner_target_return = args.planner_target_return
    task.planner_w_cfg = args.planner_w_cfg
    args.task = task

    pipeline(args)