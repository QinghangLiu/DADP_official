import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import wandb
import numpy as np
import torch
import time
from model_factory import create_planner
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from dadp.dadp import DADP
from stable_baselines3.common.env_util import make_vec_env
from customwrappers.RandomVecEnv import RandomSubprocVecEnv


def _move_optimizer_to_device(optimizer, device):
    """Move optimizer state tensors to the requested device."""
    if optimizer is None:
        return
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

class DecisionMaker:
    def __init__(self, 
                  obs_dim: int,
                act_dim: int,
                planner_dataset,
                # Task configuration
                planner_horizon: int,
                history: int,
                # Planner network parameters
                emb_dim: int = 256,
                d_model: int = 256,
                depth: int = 8,
                planner_ema_rate: float = 0.995,
                planner_predict_noise: bool = True,
                planner_next_obs_loss_weight: float = 1.0,
                planner_guide_noise_scale: float = 1.0,
                planner_sample_steps: int = 20,
                predict_embedding: bool | None = None,
                # Model configuration
                pipeline_type: str = "separate",  # "joint" or "separate"
                attention_mask: bool = True,
                device: str = "cuda",
                model_path: str = "./checkpoints/",
                embedding_model_path = None,
                noise_type: str = "standard",  # "gaussian" or "vpsde" or "embedding_guided"
                nnCondition = False,
                train_embedding_model: bool = False,
                embedding_learning_rate: float = None,
                    ):
    
        if pipeline_type == "separate":
            raise ValueError("Policy/critic are disabled; use pipeline_type='joint' or 'no_prior'.")
        self.pipeline_type = pipeline_type
        self.history = history
        self.planner_horizon = planner_horizon
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.model_path = model_path
        self.device = device
        self.predict_embedding = (
            predict_embedding if predict_embedding is not None else noise_type in ("mixed_ddim", "one_step_mixed_ddim")
        )
        embedding_model, metadata = DADP.load_checkpoint(embedding_model_path, "cpu")
        embedding_model = embedding_model.to(device)
        self.embedding_model = embedding_model
        self.train_embedding_model = bool(train_embedding_model)
        self.embedding_learning_rate = embedding_learning_rate
        self.embedding_optimizer = None
        if self.train_embedding_model and not hasattr(embedding_model, "parameters"):
            raise ValueError("Embedding model is required for co-training but was not provided")
        if self.train_embedding_model:
            embedding_model.train()
            self.embedding_optimizer = getattr(embedding_model, "optimizer", None)
            if self.embedding_optimizer is None:
                lr = embedding_learning_rate
                if lr is None and hasattr(embedding_model, "training_config"):
                    lr = getattr(embedding_model.training_config, "learning_rate", None)
                lr = lr or 3e-4
                self.embedding_optimizer = torch.optim.Adam(embedding_model.parameters(), lr=lr)
            _move_optimizer_to_device(self.embedding_optimizer, device)
        else:
            embedding_model.eval()
        # Instantiate local Planner, Policy and Critic wrapper classes
        self.planner = Planner(
            obs_dim=obs_dim,
            act_dim=act_dim,
            planner_dataset=planner_dataset,
            planner_horizon=planner_horizon,
            history=history,
            emb_dim=emb_dim,
            d_model=d_model,
            depth=depth,
            planner_ema_rate=planner_ema_rate,
            planner_predict_noise=planner_predict_noise,
            planner_next_obs_loss_weight=planner_next_obs_loss_weight,
            planner_guide_noise_scale=planner_guide_noise_scale,
            planner_noise_type=noise_type,
            sample_steps=planner_sample_steps,
            pipeline_type=pipeline_type,
            attention_mask=attention_mask,
            device=device,
            model_path=model_path,
            embedding_model=embedding_model,
            nnCondition = nnCondition,
            train_embedding_model=train_embedding_model,
            embedding_learning_rate=embedding_learning_rate,
            predict_embedding=self.predict_embedding
        )

    def train(
        self,
        planner_dataloader,
        planner_val_dataloader=None,
        # Training steps
        planner_diffusion_gradient_steps: int = 10000,
        # Training configuration
        use_weighted_regression: bool = False,
        weight_factor: float = 1.0,
        # Logging and saving
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 1000,
        save_path: str = "./checkpoints/",
        enable_wandb: bool = False,
        device: str = "cuda",
    ):
        """Train planner only (policy/critic are disabled)."""
        os.makedirs(save_path, exist_ok=True)

        self.planner.train_model(
            planner_dataloader,
            val_dataloader=planner_val_dataloader,
            gradient_steps=planner_diffusion_gradient_steps,
            use_weighted_regression=use_weighted_regression,
            weight_factor=weight_factor,
            log_interval=log_interval,
            eval_interval=eval_interval,
            save_interval=save_interval,
            evaluate_batch=100,
            enable_wandb=enable_wandb,
        )

    def _loop_two_dataloaders(self, dataloader1, dataloader2):
        """Helper generator to loop over two dataloaders of potentially different lengths."""
        iter1 = iter(dataloader1)
        iter2 = iter(dataloader2)
        len1 = len(dataloader1)
        len2 = len(dataloader2)
        max_len = max(len1, len2)
        
        for _ in range(max_len):
            try:
                batch1 = next(iter1)
            except StopIteration:
                iter1 = iter(dataloader1)
                batch1 = next(iter1)
            
            try:
                batch2 = next(iter2)
            except StopIteration:
                iter2 = iter(dataloader2)
                batch2 = next(iter2)
            
            yield batch1, batch2
    def load_ckpt(self, planner_ckpt=1000000, device="cuda"):
        """Load planner checkpoint."""
        self.planner.load(planner_ckpt)

    def predict(self,
                obs_history,
                cnt_obs,
                num_candidates: int = 50,
                task_id = None,
                **kwargs
                ):
        """
        High-level inference using Planner.predict and Policy.predict.
        
        Args:
            obs_history: Historical observations
            cnt_obs: Current observation 
            num_candidates: Number of planner candidates to generate
            task_id: Optional task identifier
            
        Returns:
            actions: Predicted actions (B, act_dim)
        """
        
        device = self.device
        # Ensure inputs are tensors on correct device
        if isinstance(obs_history, list):
            obs_history = torch.stack([torch.as_tensor(x, device=device) for x in obs_history], dim=0)
        else:
            obs_history = torch.as_tensor(obs_history, device=device)
        obs_history = obs_history.to(device)
        obs_history_repeat = obs_history.unsqueeze(1).repeat(1,num_candidates, 1, 1).view(-1,obs_history.shape[1], self.obs_dim+self.act_dim)
        cnt_obs = torch.as_tensor(cnt_obs, device=device)
        
        if cnt_obs.dim() == 1:
            cnt_obs = cnt_obs.unsqueeze(0)
        
        # Generate multiple future trajectories using planner
        # Replicate current observations for multiple candidates
        cnt_obs_repeated = cnt_obs.unsqueeze(1).repeat(1, num_candidates, 1).view(-1, self.obs_dim)
        
        # Call planner.predict to get future observations
        # planner.predict(history_traj, cnt_obs, n_samples, task_id)
        if kwargs.get('embedding_traj', None) is not None:
            embedding_traj = torch.as_tensor(kwargs['embedding_traj'], device=device)
            if embedding_traj.dim() == 2:
                embedding_traj = embedding_traj.unsqueeze(0)
            embedding_traj = embedding_traj.to(device)
            future_trajs = self.planner.set_embedding_predict(
                obs_history_repeat, 
                cnt_obs_repeated, 
                embedding_traj=embedding_traj,
                n_samples=num_candidates*cnt_obs.shape[0],
                task_id=task_id
            )
        else:
            future_trajs = self.planner.predict(
                obs_history_repeat, 
                cnt_obs_repeated, 
                n_samples=num_candidates*cnt_obs.shape[0],
                task_id=task_id
            )
        
        actions = future_trajs[:, 0, self.obs_dim:]
        return actions

    def validate(self, val_dataloader, evaluate_batch=100):
        """Validate all components and return metrics."""
        n_batch = 0
        self.eval()
        val_metrics = {'planner_loss': 0.0, 'total_loss': 0}
        with torch.no_grad():
            val_iterator = iter(val_dataloader)
            for batch_idx in range(min(evaluate_batch, len(val_dataloader))):
                batch = next(val_iterator)
                planner_horizon_obs = batch["obs"]["state"].to(self.device)
                planner_horizon_action = batch["act"].to(self.device)
                planner_horizon_obs_action = torch.cat([planner_horizon_obs, planner_horizon_action], dim=-1)
                
                planner_task = (
                    batch["task_id"].to(self.device) 
                    if "task_id" in batch else None
                )
                # Sample from planner
                sampled_traj = self.planner.predict(
                    planner_horizon_obs_action[:,:self.history,:],
                    planner_horizon_obs[:,self.history,:],
                    n_samples=planner_horizon_obs.shape[0],
                    task_id=planner_task,
                )
                if self.pipeline_type != "joint":
                    planner_loss = F.mse_loss(sampled_traj, planner_horizon_obs[:,self.history+1:,:])
                else:
                    planner_loss = F.mse_loss(sampled_traj, planner_horizon_obs_action[:,self.history:,:])
                val_metrics['planner_loss'] += planner_loss.item()


                action = self.predict(
                    planner_horizon_obs_action[:,:self.history,:],
                    planner_horizon_obs[:,self.history,:],
                    num_candidates=1,
                    task_id=planner_task,
                )
                action_loss = F.mse_loss(action, planner_horizon_action[:,self.history,:])
                val_metrics['total_loss'] += action_loss.item()
                n_batch += 1
            
            # Average metrics over batches
            for key in val_metrics:
                val_metrics[key] /= max(n_batch, 1)
        return val_metrics
    def eval(self):
        """Set all components to evaluation mode."""
        self.planner.eval()


class Planner:
    def __init__(self, 
                 obs_dim: int,
                 act_dim: int,
                 planner_dataset,
                 # Task configuration
                 planner_horizon: int,
                 history: int,
                 # Network parameters
                 emb_dim: int = 256,
                 d_model: int = 256,
                 depth: int = 8,
                 planner_ema_rate: float = 0.995,
                 planner_predict_noise: bool = True,
                 planner_next_obs_loss_weight: float = 1.0,
                 planner_guide_noise_scale: float = 1.0,
                 planner_noise_type: str = "standard",  # "gaussian" or "vpsde"
                 sample_steps: int = 20,
                 # Model configuration
                 pipeline_type: str = "separate",
                 attention_mask: bool = True,
                 device: str = "cuda",
                 model_path: str = "./checkpoints/",
                 embedding_model = None,
                 nnCondition = False,
                 env_type: str = None,
                 env_quality: str = 'expert',
                 train_embedding_model: bool = False,
                 embedding_learning_rate: float = None,
                 predict_embedding: bool = False,
                 ):
        self.sample_steps = sample_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.history = history
        self.planner_horizon = planner_horizon
        self.pipeline_type = pipeline_type
        self.device = device
        self.model_path = model_path
        self.planner_noise_type = planner_noise_type
        self.predict_embedding = predict_embedding

        self.embedding_model=embedding_model
        self.train_embedding_model = bool(train_embedding_model)
        self.embedding_learning_rate = embedding_learning_rate
        self.embedding_optimizer = None
        if self.embedding_model is None and self.train_embedding_model:
            raise ValueError("Embedding model is required when co-training is enabled")
        if self.embedding_model is not None:
            if self.train_embedding_model:
                self.embedding_model.train()
                self.embedding_optimizer = getattr(self.embedding_model, "optimizer", None)
                if self.embedding_optimizer is None:
                    lr = embedding_learning_rate
                    if lr is None and hasattr(self.embedding_model, "training_config"):
                        lr = getattr(self.embedding_model.training_config, "learning_rate", None)
                    lr = lr or 3e-4
                    self.embedding_optimizer = torch.optim.Adam(self.embedding_model.parameters(), lr=lr)
                _move_optimizer_to_device(self.embedding_optimizer, device)
            else:
                self.embedding_model.eval()
        # Allow the embedding model to have a different history length than the
        # planner. Store the embedding history length and warn if it differs.
        self.embedding_history = getattr(self.embedding_model.training_config, "history", None)
        if self.embedding_history is None:
            self.embedding_history = history
        elif self.embedding_history != history:
            print(
                f"Warning: embedding history ({self.embedding_history}) differs from planner history ({history}); "
                "the embedding model will receive a trajectory of length equal to its own history when encoding."
            )
        # Create planner
        self.planner = create_planner(
            obs_dim=obs_dim,
            act_dim=act_dim,
            planner_dataset=planner_dataset,
            planner_horizon=planner_horizon,
            history=history,
            pipeline_type=pipeline_type,
            planner_emb_dim=emb_dim,
            planner_d_model=d_model,
            planner_depth=depth,
            planner_ema_rate=planner_ema_rate,
            planner_predict_noise=planner_predict_noise,
            planner_next_obs_loss_weight=planner_next_obs_loss_weight,
            planner_guide_noise_scale=planner_guide_noise_scale,
            attention_mask=attention_mask,
            device=device,
            planner_noise_type=planner_noise_type,
            nnCondition = nnCondition,
            dadp_model=self.embedding_model if planner_noise_type == 'latent_embedding_guided' else None,
            predict_embedding=predict_embedding,
        )
        self.nnCondition = nnCondition
        self.embedding_list = []
        # best-score tracking for checkpointing during training/validation
        # lower planner_loss is better; higher env return is better
        self._best_val_score = float('inf')
        self._best_env_return = float('-inf')
        self.env_type = env_type    
        self.env_quality = env_quality
        self.dataset = planner_dataset

    def evaluate_on_env(self,
                                   env_type: str,
                                   data_quality: str = 'expert',
                                   task_ids=None,
                                   num_eval_episodes: int = 1,
                                   max_path_length: int = None,
                                   history: int = None,
                                   device: str = None,
                                   model_path: str = None):
        """
        Planner-only evaluation that creates the environment using the same helper
        used in `eval_diffusion_meta_dt_style.py` and runs episodes using only
        `self.predict` to generate planner outputs. This function does not plot,
        does not validate the env, and does not depend on DecisionMaker.

        Note: this method expects the planner's `predict` to produce an output that
        contains action information (either an obs+act joint output where actions
        are in the trailing dims, or a direct action vector). If the planner
        returns only next-observations (and not actions), the method will raise
        an informative error.
        """
        # use planner defaults when arguments are not provided
        if history is None:
            history = self.history
        
        # Ensure we track enough history for embedding model if needed
        max_history = max(history, self.embedding_history)

        if device is None:
            device = self.device
        if model_path is None:
            model_path = self.model_path

        # helpers are imported at module top; use them directly

        # build a vectorized env matching `train_diffusion.py` convention
        num_envs = 10
        env_eval = make_vec_env(env_type, n_envs=num_envs, seed=None, vec_env_cls=RandomSubprocVecEnv)
        task_list = self.dataset.dataset.task_list
        normalizer = self.dataset.dataset.get_normalizer()

        # set max path length from config if not provided
        config = {}
        if max_path_length is None:
            max_path_length = config.get('max_episode_steps', 1000)

        # decide tasks
        if task_ids is None:
            task_ids = list(range(len(task_list)))

        results = {}
        env_name_lower = env_type.lower()
        if ("door" in env_name_lower or "hammer" in env_name_lower or "relocate" in env_name_lower):
            task_ids = [0]  # only one task for door/hammer/relocate envs
        for task_idx in task_ids:
            task = task_list[task_idx]
            # tile task for vectorized env
            tiled_task = np.tile(task, (num_envs, 1))
            # Only set task if env is not door or hammer
            
            if not ("door" in env_name_lower or "hammer" in env_name_lower or "relocate" in env_name_lower):
                if hasattr(env_eval, 'set_task'):
                    env_eval.set_task(tiled_task)

            episode_rewards = []
            previous_episode_history = [None] * num_envs

            for ep in range(num_eval_episodes):
                # vectorized reset
                obs = env_eval.reset()
                t = 0
                cum_done = np.zeros(num_envs, dtype=bool)
                ep_reward = np.zeros(num_envs, dtype=float)

                # Initialize history with zeros for each env
                obs_history = [
                    torch.cat([
                        torch.tensor(normalizer.normalize(obs), device=device, dtype=torch.float32),
                        torch.zeros((num_envs, self.act_dim), device=device),
                    ], dim=-1)
                    for _ in range(max_history)
                ]

                while not np.all(cum_done) and t < max_path_length + 1:
                    # Normalize current observation
                    current_obs = torch.tensor(
                        normalizer.normalize(obs),
                        device=device,
                        dtype=torch.float32
                    )

                    # Prepare obs_history tensor
                    obs_history_tensor = torch.stack(obs_history, dim=1)

                    # Predict action using planner-only predict (caller can wrap with DecisionMaker if needed)
                    with torch.no_grad():
                        actions = self.predict(
                            history_traj=obs_history_tensor,
                            cnt_obs=current_obs,
                            n_samples=num_envs,
                            task_id=task_idx
                        )[:,0,self.obs_dim:self.obs_dim+self.act_dim]


                    # actions shape: (num_envs, act_dim) or (num_envs, something)
                    actions = actions.clamp(-1, 1)

                    # Convert to numpy for vectorized step
                    actions_np = actions.cpu().numpy()

                    # Step environment (vectorized)
                    obs, rew, done, info = env_eval.step(actions_np)

                    obs_history.append(torch.cat([current_obs, actions], dim=-1))
                    if len(obs_history) > max_history:
                        obs_history.pop(0)

                    # Track newly done envs and save their history
                    prev_cum_done = cum_done.copy()
                    newly_done = np.logical_and(~prev_cum_done, done)
                    if np.any(newly_done):
                        obs_history_tensor_done = torch.stack(obs_history, dim=1)
                        for i in np.where(newly_done)[0]:
                            previous_episode_history[i] = obs_history_tensor_done[i].detach()

                    # Update episode tracking
                    t += 1
                    cum_done = np.logical_or(cum_done, done)
                    ep_reward += rew * (1 - cum_done)

                    # Capture episode-level rewards when all envs done will be handled below

                # Collect episode rewards (sum across envs for this episode run)
                episode_rewards.extend(ep_reward.tolist())

                # Ensure previous_episode_history filled
                obs_history_tensor_end = torch.stack(obs_history, dim=1)
                for i in range(num_envs):
                    if previous_episode_history[i] is None:
                        previous_episode_history[i] = obs_history_tensor_end[i].detach()

            episode_rewards = np.array(episode_rewards).reshape(-1)
            mean_reward = float(np.mean(episode_rewards)) if episode_rewards.size else 0.0
            std_error = float(np.std(episode_rewards) / np.sqrt(len(episode_rewards))) if episode_rewards.size else 0.0

            results[task_idx] = {
                'mean_reward': mean_reward,
                'std_error': std_error,
                'episode_rewards': episode_rewards.tolist(),
            }

        env_eval.close()
        return results

    def train_model(self, 
              dataloader,
              val_dataloader=None,
              test_task_dataloader=None,
              gradient_steps: int = 10000,
              use_weighted_regression: bool = False,
              weight_factor: float = 1.0,
              log_interval: int = 100,
              eval_interval: int = 1000,
              save_interval: int = 1000,
              evaluate_batch: int = 100,
              enable_wandb: bool = False,
              start_step: int = 0):
        """
        Train the planner model.
        
        Args:
            dataloader: Training dataloader
            val_dataloader: Validation dataloader
            gradient_steps: Number of gradient steps
            use_weighted_regression: Whether to use weighted regression
            weight_factor: Weight factor for weighted regression
            log_interval: Interval for logging training metrics
            eval_interval: Interval for evaluation
            save_interval: Interval for saving checkpoints
            evaluate_batch: Number of batches to evaluate on
            enable_wandb: Whether to enable wandb logging
            start_step: Global step to resume from (used for scheduler/log offsets)
        """
        start_step = int(start_step or 0)
        if start_step < 0:
            raise ValueError("start_step must be >= 0")
        if start_step >= gradient_steps:
            print(f"Planner start_step {start_step} >= target gradient_steps {gradient_steps}; nothing to train.")
            return

        # Setup learning rate scheduler and progress bar accounting for resume
        for param_group in self.planner.optimizer.param_groups:
            param_group.setdefault("initial_lr", param_group["lr"])
        last_epoch = start_step - 1 if start_step > 0 else -1
        lr_scheduler = CosineAnnealingLR(self.planner.optimizer, gradient_steps, last_epoch=last_epoch)
        remaining_steps = gradient_steps - start_step
        progress_updates = max((remaining_steps + log_interval - 1) // max(log_interval, 1), 1)
        
        # Set to training mode
        self.planner.train()
        
        # Initialize logging
        log = {"avg_planner_loss": 0.0}
        steps_accumulated = 0
        n_gradient_step = start_step
        
        pbar = tqdm(total=progress_updates, desc="Training Planner")
        
        for batch in loop_dataloader(dataloader):
            # Single training step
            planner_loss = self._train_step(batch, use_weighted_regression, weight_factor)
            log["avg_planner_loss"] += planner_loss
            steps_accumulated += 1
            lr_scheduler.step()
            
            # Evaluation
            if (n_gradient_step+1) % eval_interval == 0 and val_dataloader:

                val_metrics = self.validate(val_dataloader, evaluate_batch,enable_wandb=enable_wandb, n_gradient_step=n_gradient_step)

                if enable_wandb:
                    wandb.log(val_metrics, step=n_gradient_step + 1)
                
                
                print(f"Validation at step {n_gradient_step + 1}: {val_metrics}")
                # run test_task validation if provided
                if test_task_dataloader is not None:
                    test_metrics = self.validate(test_task_dataloader, evaluate_batch,enable_wandb=enable_wandb, n_gradient_step=n_gradient_step)
                    test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
                    if enable_wandb:
                        wandb.log(test_metrics, step=n_gradient_step + 1)
                    print(f"Test at step {n_gradient_step + 1}: {test_metrics}")

                # Optional: evaluate on-simulator environment if caller provided env config via attributes
                # We look for attributes attached to the Planner instance: eval_env_type (str), eval_data_quality (str),
                # eval_num_episodes (int). These are optional and can be set by higher-level training script.

                # run environment evaluation using planner-only flow
                print(f"Running environment evaluation on '{self.env_type}' (episodes=5)...")
                env_results = self.evaluate_on_env(self.env_type, data_quality=self.env_quality,
                                                            task_ids=[0,1,2,9,17,24],
                                                            num_eval_episodes=3,
                                                            max_path_length=1000,
                                                            history=self.history,
                                                            device=self.device,
                                                            model_path=self.model_path)
                # Aggregate env_results across tasks
                all_task_means = [v['mean_reward'] for v in env_results.values()]
                mean_return = float(np.mean(all_task_means)) if len(all_task_means) > 0 else 0.0
                print(f"Env eval mean return: {mean_return:.4f} (per-task: {all_task_means})")
                if enable_wandb:
                    wandb.log({ 'env_eval/mean_return': mean_return,
                                'env_eval/per_task': env_results,
                                'training/step': n_gradient_step + 1 }, step=n_gradient_step + 1)


                # --- save best model logic (during validation) ---
                if mean_return > getattr(self, '_best_env_return', float('-inf')):
                    self._best_env_return = mean_return
                    print(f"New best env mean return {mean_return:.4f} at step {n_gradient_step+1} - saving planner checkpoint as 'best_env'")
                    self.save('best_model')

                self.planner.train()
                    

            
            # Logging
            if (n_gradient_step + 1) % log_interval == 0:
                divisor = steps_accumulated if steps_accumulated > 0 else 1
                log["avg_planner_loss"] /= divisor
                log["gradient_steps"] = n_gradient_step + 1
                print(f"Step {n_gradient_step + 1}: {log}")
                
                if enable_wandb:
                    wandb.log(log, step=n_gradient_step + 1)
                
                if pbar.n < pbar.total:
                    pbar.update(1)
                log = {"avg_planner_loss": 0.0}
                steps_accumulated = 0
            
            # Checkpointing
            if (n_gradient_step + 1) % save_interval == 0:
                self.save(n_gradient_step + 1)
            
            n_gradient_step += 1
            if n_gradient_step >= gradient_steps:
                break
                

        if pbar.n < pbar.total:
            pbar.update(pbar.total - pbar.n)
        pbar.close()
        print(f"Planner training completed after {n_gradient_step} steps")
    
    def _train_step(self, batch, use_weighted_regression=False, weight_factor=1.0):
        """Single training step for planner."""
        planner_horizon_obs = batch["obs"]["state"].to(self.device)
        planner_horizon_action = batch["act"].to(self.device)
        planner_horizon_obs_action = torch.cat(
            [planner_horizon_obs, planner_horizon_action], dim=-1
        )

        if self.pipeline_type == "separate":
            planner_horizon_data = planner_horizon_obs_action.clone()
            planner_horizon_data[:, self.history:, self.obs_dim:] = 0
        elif self.pipeline_type == "no_prior":
            planner_horizon_data = planner_horizon_obs_action[:, self.history:, :].clone()
        elif self.pipeline_type == "joint":
            # Keep full horizon so shapes remain aligned with planner fix_mask/attention mask
            planner_horizon_data = planner_horizon_obs_action[:, self.embedding_history - self.history:, :].clone()
        else:
            raise ValueError(f"Unknown pipeline type: {self.pipeline_type}")
        
        planner_td_val = batch["val"].to(self.device)
        planner_task = (
            batch["task_id"].to(self.device) 
            if "task_id" in batch else None
        )
        planner_embedding = batch['embedding'].to(self.device) if ('embedding' in batch and not self.train_embedding_model) else None

        if planner_embedding is not None:
            #         planner_embedding = torch.nn.functional.pad(planner_embedding, (0, pad_width), mode='constant', value=0.0)
            if self.planner_noise_type == 'latent_embedding_guided':
                planner_embedding = planner_embedding.unsqueeze(1).repeat(1, self.planner_horizon * 2, 1)
            else:
                planner_embedding = planner_embedding.unsqueeze(1).repeat(1, self.planner_horizon, 1)
        else:
            if 'embedding_traj' in batch:
                embedding_traj = batch['embedding_traj'].to(self.device)
            else:
                embedding_traj = planner_horizon_obs_action[:, :self.embedding_history, :].detach()
            planner_embedding = self.get_embedding(
                embedding_traj,
                detach=not self.train_embedding_model,
            ).to(self.device)

        weighted_regression_tensor = torch.exp(
            (planner_td_val - 1) * weight_factor
        )
        condition_tensor = planner_embedding[:, 0, :] if (planner_embedding is not None and self.nnCondition) else None
        embedding_tensor = (
            planner_embedding
            if (
                planner_embedding is not None
                and self.planner_noise_type in ['embedding_guided', 'latent_embedding_guided', 'mixed_ddim', 'one_step_mixed_ddim']
            )
            else None
        )

        if self.train_embedding_model:
            self.embedding_optimizer.zero_grad(set_to_none=True)

        planner_loss = self.planner.update(
            planner_horizon_data,
            weighted_regression_tensor=weighted_regression_tensor,
            condition=condition_tensor,
            task_id=planner_task,
            embedding=embedding_tensor,
        )['loss']

        if self.train_embedding_model:
            self.embedding_optimizer.step()

        return planner_loss
    
    def validate(self, val_dataloader, evaluate_batch=100, enable_wandb=False, n_gradient_step=0):
        """Single validation step for planner."""
        self.planner.eval()
        n_batch = 0
        val_metrics = {
            'planner_loss': 0.0,
            'action_loss': 0.0,
        }
        
        max_history = max(self.history, self.embedding_history)

        with torch.no_grad():
            val_iterator = iter(val_dataloader)
            pbar = tqdm(total=min(evaluate_batch, len(val_dataloader)), desc="Validating Policy", leave=False)
            for batch_idx in range(min(evaluate_batch, len(val_dataloader))):
                
                batch = next(val_iterator)

                planner_horizon_obs = batch["obs"]["state"].to(self.device)
                planner_horizon_action = batch["act"].to(self.device)
                planner_horizon_obs_action = torch.cat([planner_horizon_obs, planner_horizon_action], dim=-1)

                planner_task = (
                    batch["task_id"].to(self.device) 
                    if "task_id" in batch else None
                )
                # Sample from planner
                sampled_traj = self.predict(
                    planner_horizon_obs_action[:,:max_history,:],
                    planner_horizon_obs[:,max_history,:],
                    n_samples=planner_horizon_obs.shape[0],
                    task_id=planner_task if self.planner_noise_type == 'env_factor_guided' else None,
                )
                if self.pipeline_type == "separate":
                    gt_length = sampled_traj.shape[1]
                    loss = F.mse_loss(sampled_traj, planner_horizon_obs[:,max_history+1:max_history+1+gt_length,:])
                    val_metrics['planner_loss'] += loss.item()
                else:
                    gt_length = sampled_traj.shape[1]
                    loss = F.mse_loss(sampled_traj, planner_horizon_obs_action[:,max_history:max_history+gt_length,:self.obs_dim+self.act_dim])
                    val_metrics['planner_loss'] += loss.item()
                    action_loss = F.mse_loss(sampled_traj[:,0,self.obs_dim:], planner_horizon_action[:,max_history,:])
                    val_metrics['action_loss'] += action_loss.item()

                n_batch += 1
                pbar.update(1)
            pbar.close()
            for key in val_metrics:
                val_metrics[key] /= max(n_batch, 1)
            
            
            return val_metrics
    def get_embedding(self, history_traj, project=False, detach: bool = True):
        """Get embedding from model."""
        grad_context = torch.enable_grad if not detach else torch.no_grad
        with grad_context():
            history_traj = history_traj[:, -self.embedding_history:, :]
            obs_history = history_traj[:, :, :self.obs_dim]
            action_history = history_traj[:, :, self.obs_dim:self.obs_dim+self.act_dim]
            embedding = self.embedding_model.dynamics.encode_history(
                obs_history,
                action_history,
            )
            if detach:
                embedding = embedding.detach()

        if self.planner_noise_type == 'latent_embedding_guided':
            embedding = embedding.unsqueeze(1).repeat(1,self.planner_horizon*2,1)
        else:
            embedding = embedding.unsqueeze(1).repeat(1,self.planner_horizon,1)
        return embedding
    
    def set_embedding_predict(self, history_traj,cnt_obs,n_samples,task_id=None,embedding_traj = None):
        """Set embedding for planner."""
        planner_prior = torch.zeros((n_samples, self.planner_horizon, self.obs_dim+self.act_dim), device=self.device)
        if self.pipeline_type == "separate":
            planner_prior[:, :self.history, :] = history_traj[:, -self.history:, self.obs_dim+self.act_dim]
            planner_prior[:, self.history, :self.obs_dim] = cnt_obs
        elif self.pipeline_type == "no_prior":
            planner_prior[:,0,:self.obs_dim] = cnt_obs
        elif self.pipeline_type == "joint":
            planner_prior[:, :self.history, :] = history_traj[:,-self.history:,:self.obs_dim+self.act_dim]
            planner_prior[:, self.history, :self.obs_dim] = cnt_obs
        else:
            raise ValueError(f"Unknown pipeline type: {self.pipeline_type}")
        planner_embedding = self.get_embedding(
                                            embedding_traj,
                                            detach=not self.train_embedding_model
                                        ).to(self.device)
        sampled_traj, _ = self.planner.sample(
                planner_prior,
                solver="ddim",
                n_samples=n_samples,
                sample_steps=self.sample_steps,
                use_ema=True,
                condition_cfg=planner_embedding[:,0,:] if self.nnCondition else None,
                w_cfg=1.0,
                temperature=1.0,
                task_id=task_id,
                embedding=planner_embedding if self.planner_noise_type in ["embedding_guided", "mixed_ddim", 'latent_embedding_guided', 'one_step_mixed_ddim'] else None
            )
        if self.pipeline_type == "no_prior":
            return sampled_traj
        elif self.pipeline_type == "separate":
            return sampled_traj[:,self.history+1:,:self.obs_dim]
        elif self.pipeline_type == "joint":
            return sampled_traj[:,self.history:,:]
        else:
            raise ValueError(f"Unknown pipeline type: {self.pipeline_type}")
 
    def predict(self,history_traj,cnt_obs,n_samples,task_id=None):
        """Predict using planner."""
        planner_prior = torch.zeros((n_samples, self.planner_horizon, self.obs_dim+self.act_dim), device=self.device)
        if self.pipeline_type == "separate":
            planner_prior[:, :self.history, :] = history_traj[:, -self.history:, self.obs_dim+self.act_dim]
            planner_prior[:, self.history, :self.obs_dim] = cnt_obs
        elif self.pipeline_type == "no_prior":
            planner_prior[:,0,:self.obs_dim] = cnt_obs
        elif self.pipeline_type == "joint":
            planner_prior[:, :self.history, :] = history_traj[:,-self.history:,:self.obs_dim+self.act_dim]
            planner_prior[:, self.history, :self.obs_dim] = cnt_obs
        else:
            raise ValueError(f"Unknown pipeline type: {self.pipeline_type}")
        planner_embedding = self.get_embedding(
                            history_traj,
                            detach=not self.train_embedding_model
                            ).to(self.device)

        sampled_traj, log = self.planner.sample(
                planner_prior,
                solver="ddim",
                n_samples=n_samples,
                sample_steps=self.sample_steps,
                use_ema=True,
                condition_cfg=planner_embedding[:,0,:] if self.nnCondition else None,
                w_cfg=1.0,
                temperature=1.0,
                task_id=task_id,
                embedding=planner_embedding if self.planner_noise_type in ['embedding_guided','latent_embedding_guided','mixed_ddim','one_step_mixed_ddim'] else None,
            )
        if self.pipeline_type == "no_prior":
            return sampled_traj
        elif self.pipeline_type == "separate":
            return sampled_traj[:,self.history+1:,:self.obs_dim]
        elif self.pipeline_type == "joint":
            return sampled_traj[:,self.history:,:]
        else:
            raise ValueError(f"Unknown pipeline type: {self.pipeline_type}")
    def save_embedding(self, path):
        """Save embeddings to path."""
        if len(self.embedding_list) == 0:
            print("No embeddings to save.")
            return
        embeddings = np.concatenate(self.embedding_list, axis=0)
        np.save(path, embeddings)
        print(f"Saved {embeddings.shape[0]} embeddings to {path}")
        self.embedding_list = []

    def save_embedding_model(self, path: str = None, metadata: dict | None = None):
        """Persist the co-trained embedding model weights from the planner side."""
        if not (self.train_embedding_model and self.embedding_model is not None):
            print("Embedding co-training disabled; nothing to save from planner.")
            return
        path = path or os.path.join(self.model_path, "planner_embedding_ckpt_latest.zip")
        metadata = metadata.copy() if metadata else {}
        metadata.setdefault("timestamp", time.time())
        save_fn = getattr(self.embedding_model, "save_checkpoint", None)
        if callable(save_fn):
            save_fn(path, metadata)
        else:
            torch.save({"state_dict": self.embedding_model.state_dict(), "metadata": metadata}, path)
            print(f"Saved planner embedding model state_dict to {path}")

    def sample(self, *args, **kwargs):
        """Sample from planner."""
        return self.planner.sample(*args, **kwargs)
    
    def train(self):
        """Set to training mode."""
        self.planner.train()
    
    def eval(self):
        """Set to evaluation mode."""
        self.planner.eval()
    
    def save(self, step):
        """Save planner checkpoint."""
        self.planner.save(os.path.join(self.model_path, "planner_ckpt_latest.pt"))
        if self.train_embedding_model:
            metadata = {"step": step}
            self.save_embedding_model(
                os.path.join(self.model_path, f"planner_embedding_ckpt_{step}.zip"),
                metadata=metadata,
            )
            self.save_embedding_model(
                os.path.join(self.model_path, "planner_embedding_ckpt_latest.zip"),
                metadata=metadata,
            )
    
    def load(self, step):
        """Load planner checkpoint."""
        self.planner.load(os.path.join(self.model_path, f"planner_ckpt_{step}.pt"))
    
    @property
    def optimizer(self):
        """Get planner optimizer."""
        return self.planner.optimizer
