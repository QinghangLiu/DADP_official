from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
class PerformanceBasedTermination(BaseCallback):
    def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=10, 
                 patience=5, min_improvement=0.02, std_threshold=0.02,model_save_path=None,
                 target_reward=None,save_freq=10000,least_train_step = 1000000):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.patience = patience
        self.min_improvement = min_improvement
        self.target_reward = target_reward
        self.least_train_step = least_train_step
        self.best_mean_reward = -np.inf
        self.evaluations_without_improvement = 0
        self.evaluation_rewards = []
        self.model_save_path = model_save_path
        self.std_threshold = std_threshold
        self.save_freq = save_freq
        self.best_converged = False
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Save model
            if self.model_save_path:
                self.model.save(self.model_save_path+f"model_{self.n_calls}.zip")
        if self.n_calls % self.eval_freq == 0:
            # Evaluate policy
            episode_rewards = []
            mean_episode_length = 0
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                episode_reward = 0
                done = False
                cum_done = 0
                mean_episode_length = 0
                while not np.all(cum_done):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _ = self.eval_env.step(action)
                    cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                    episode_reward += (reward * (1 - cum_done))
                    mean_episode_length += np.mean(1-cum_done)
                episode_rewards.append(episode_reward)
            mean_episode_length /= self.n_eval_episodes
            print(f"Evaluation over {self.n_eval_episodes} episodes: mean reward {np.mean(episode_rewards):.2f}, std {np.std(episode_rewards):.2f}, mean episode length {mean_episode_length:.2f}")
            # Log to wandb
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            self.evaluation_rewards.append(mean_reward)
            
            # Check for improvement
            if mean_reward > (1 + np.sign(self.best_mean_reward)*self.min_improvement) * self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.evaluations_without_improvement = 0
                print(f"New best reward: {mean_reward:.2f}, std: {std_reward:.2f} at step {self.n_calls}")
                #save the best model
                if self.model_save_path:
                    self.model.save(self.model_save_path+f"best_model.zip")
                self.best_converged = std_reward < self.std_threshold * abs(self.best_mean_reward)

            else:
                self.evaluations_without_improvement += 1
                print(f"No improvement for {self.evaluations_without_improvement} evaluations")
            
            # Check termination conditions
            if self.target_reward and mean_reward >= self.target_reward:
                print(f"Target reward {self.target_reward} reached!")
                return False
                
            if self.evaluations_without_improvement >= self.patience and self.best_converged and self.n_calls * self.eval_env.num_envs >= self.least_train_step:
                print(f"Early stopping: No improvement for {self.patience} evaluations")
                return False
                
        return True
    def reset(self):
        self.best_mean_reward = -np.inf
        self.evaluations_without_improvement = 0
        self.evaluation_rewards = []
        self.n_calls = 0
        self.best_converged = False
class EpisodeLengthCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Check if an episode finished
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_lengths.append(info["episode"]["l"])
        return True