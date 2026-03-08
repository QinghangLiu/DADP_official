from collections import defaultdict
import random
from typing import Dict, List, Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.utils import GaussianNormalizer, dict_apply
from tqdm import tqdm, trange
import time

# Observation functions
def add_gaussian_noise(state: np.ndarray, noise_std: float) -> np.ndarray:
    """
    Add Gaussian noise to state observations
    
    Args:
        state: Input state array of shape (..., state_dim)
        noise_std: Standard deviation of the Gaussian noise
    
    Returns:
        Noisy observation with same shape as input
    """
    noise = np.random.normal(0, noise_std, state.shape).astype(state.dtype)
    return state + noise

def mask_dimensions(state: np.ndarray, mask_dims: List[int]) -> np.ndarray:
    """
    Mask specified dimensions of state by setting them to zero
    
    Args:
        state: Input state array of shape (..., state_dim)  
        mask_dims: List of dimension indices to mask
        
    Returns:
        Masked observation with same shape as input
    """
    observation = state.copy()
    observation[..., mask_dims] = 0.0
    return observation

def return_reward_range(dataset, max_episode_steps):
    """ Return the range of episodic returns in the D4RL-MuJoCo dataset. """
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, max_episode_steps=1000):
    """ Modify the episodic return scale of the D4RL-MuJoCo dataset to be within [0, max_episode_steps]. """
    min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
    dataset["rewards"] /= max_ret - min_ret
    dataset["rewards"] *= max_episode_steps
    return dataset

def collate_fn(batch,segment_size = 8):
    i = np.random.randint(0, batch[0].observations.shape[0] - segment_size + 1)

    return {
        "id": torch.Tensor([x.id for x in batch]),
        "observations": np.array(
            [x.observations for x in batch]
        )[:,i:i+segment_size,:],
        "actions": np.array(
            [x.actions for x in batch]
        )[:,i:i+segment_size,:],
        "rewards": np.array(
            [x.rewards for x in batch]
        )[:,i:i+segment_size],
        "terminations": np.array(
            [x.terminations for x in batch]
        )[:,i:i+segment_size],
        "truncations": np.array(
            [x.truncations for x in batch]
        )[:,i:i+segment_size],
        "infos": 
            [x.infos for x in batch]
         if batch[0].infos is not None else None,
    }

class DV_D4RLMuJoCoSeqDataset(BaseDataset):
    """ **D4RL-MuJoCo Sequential Dataset**

    torch.utils.data.Dataset wrapper for D4RL-MuJoCo dataset.
    Chunk the dataset into sequences of length `horizon` without padding.
    Use GaussianNormalizer to normalize the observations as default.
    Each batch contains
    - batch["obs"]["state"], observations of shape (batch_size, horizon, o_dim)
    - batch["act"], actions of shape (batch_size, horizon, a_dim)
    - batch["rew"], rewards of shape (batch_size, horizon, 1)
    - batch["val"], Monte Carlo return of shape (batch_size, 1)

    Args:
        dataset: Dict[str, np.ndarray],
            D4RL-MuJoCo dataset. Obtained by calling `env.get_dataset()`.
        terminal_penalty: float,
            Penalty reward for early-terminal states. Default is -100.
        horizon: int,
            Length of each sequence. Default is 1.
        max_path_length: int,
            Maximum length of the episodes. Default is 1000.
        discount: float,
            Discount factor. Default is 0.99.

    Examples:
        >>> env = gym.make("halfcheetah-medium-expert-v2")
        >>> dataset = D4RLMuJoCoDataset(env.get_dataset(), horizon=32)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> batch = next(iter(dataloader))
        >>> obs = batch["obs"]["state"]  # (32, 32, 17)
        >>> act = batch["act"]           # (32, 32, 6)
        >>> rew = batch["rew"]           # (32, 32, 1)
        >>> val = batch["val"]           # (32, 1)

        >>> normalizer = dataset.get_normalizer()
        >>> obs = env.reset()[None, :]
        >>> normed_obs = normalizer.normalize(obs)
        >>> unnormed_obs = normalizer.unnormalize(normed_obs)
    """
    def __init__(
            self,
            dataset: Dict[str, np.ndarray],
            terminal_penalty: float = -100,
            horizon: int = 1,
            max_path_length: int = 1000,
            discount: float = 0.99,
            center_mapping: bool = True,
            stride: int = 1,
            full_traj_bonus: float = 100,
    ):
        super().__init__()

        observations, actions, rewards, timeouts, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["rewards"].astype(np.float32),
            dataset["timeouts"].astype(np.float32),
            dataset["terminals"].astype(np.float32))
        self.stride = stride

        self.normalizers = {
            "state": GaussianNormalizer(observations)}
        normed_observations = self.normalizers["state"].normalize(observations)

        self.horizon = horizon
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]

        n_paths = np.sum(np.logical_or(terminals, timeouts))
        self.seq_obs = np.zeros((n_paths+1, max_path_length, self.o_dim), dtype=np.float32)
        self.seq_act = np.zeros((n_paths+1, max_path_length, self.a_dim), dtype=np.float32)
        self.seq_rew = np.zeros((n_paths+1, max_path_length, 1), dtype=np.float32)
        self.seq_val = np.zeros((n_paths+1, max_path_length, 1), dtype=np.float32)
        self.indices = []

        ptr = 0
        path_idx = 0
        for i in range(timeouts.shape[0]):
            if timeouts[i] or terminals[i] or i == timeouts.shape[0] - 1:
                path_length = i - ptr + 1
                assert path_length <= max_path_length, f"current path length {path_length}"

                if terminals[i]:
                    rewards[i] = terminal_penalty if terminal_penalty is not None else rewards[i]
                    
                if path_length == max_path_length:
                    rewards[i] = rewards[i] + full_traj_bonus if full_traj_bonus is not None else rewards[i]

                self.seq_obs[path_idx, :path_length] = normed_observations[ptr:i + 1]
                self.seq_act[path_idx, :path_length] = actions[ptr:i + 1]
                self.seq_rew[path_idx, :path_length] = rewards[ptr:i + 1][:, None]

                max_start = path_length - (horizon - 1) * stride - 1
                self.indices += [(path_idx, start, start + (horizon - 1) * stride + 1) for start in range(max_start + 1)]

                ptr = i + 1
                path_idx += 1

        self.seq_val[:, -1] = self.seq_rew[:, -1]
        for i in reversed(range(max_path_length-1)):
            self.seq_val[:, i] = self.seq_rew[:, i] + discount * self.seq_val[:, i+1]
        
        print(f"max discounted return: {self.seq_val.max()}")
        print(f"min discounted return: {self.seq_val.min()}")
        
        # val \in [-1, 1]
        self.seq_val = (self.seq_val - self.seq_val.min()) / (self.seq_val.max() - self.seq_val.min())
        if center_mapping:
            self.seq_val = self.seq_val * 2 - 1
        print(f"max normed discounted return: {self.seq_val.max()}")
        print(f"min normed discounted return: {self.seq_val.min()}")

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]
        
        horizon_state = self.seq_obs[path_idx, start:end:self.stride]

        data = {
            'obs': {'state': horizon_state},
            'act': self.seq_act[path_idx, start:end:self.stride],
            'rew': self.seq_rew[path_idx, start:end:self.stride],
            'val': self.seq_val[path_idx, start],
        }

        torch_data = dict_apply(data, torch.tensor)

        return torch_data


class D4RLMuJoCoTDDataset(BaseDataset):
    """ **D4RL-MuJoCo Transition Dataset**

    torch.utils.data.Dataset wrapper for D4RL-MuJoCo dataset.
    Chunk the dataset into transitions.
    Use GaussianNormalizer to normalize the observations as default.
    Each batch contains
    - batch["obs"]["state"], observation of shape (batch_size, o_dim)
    - batch["next_obs"]["state"], next observation of shape (batch_size, o_dim)
    - batch["act"], action of shape (batch_size, a_dim)
    - batch["rew"], reward of shape (batch_size, 1)
    - batch["tml"], terminal of shape (batch_size, 1)

    Args:
        dataset: Dict[str, np.ndarray],
            D4RL-MuJoCo TD dataset. Obtained by calling `d4rl.qlearning_dataset(env)`.
        normalize_reward: bool,
            Normalize the reward. Default is False.

    Examples:
        >>> env = gym.make("halfcheetah-medium-expert-v2")
        >>> dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env))
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> batch = next(iter(dataloader))
        >>> obs = batch["obs"]["state"]  # (32, 17)
        >>> act = batch["act"]           # (32, 6)
        >>> rew = batch["rew"]           # (32, 1)
        >>> tml = batch["tml"]           # (32, 1)
        >>> next_obs = batch["next_obs"]["state"]  # (32, 17)

        >>> normalizer = dataset.get_normalizer()
        >>> obs = env.reset()[None, :]
        >>> normed_obs = normalizer.normalize(obs)
        >>> unnormed_obs = normalizer.unnormalize(normed_obs)
    """
    def __init__(self, dataset: Dict[str, np.ndarray], normalize_reward: bool = False):
        super().__init__()
        if normalize_reward:
            dataset = modify_reward(dataset, 1000)

        observations, actions, next_observations, rewards, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["next_observations"].astype(np.float32),
            dataset["rewards"].astype(np.float32),
            dataset["terminals"].astype(np.float32))

        self.normalizers = {
            "state": GaussianNormalizer(observations)}
        normed_observations = self.normalizers["state"].normalize(observations)
        normed_next_observations = self.normalizers["state"].normalize(next_observations)

        self.obs = torch.tensor(normed_observations, dtype=torch.float32)
        self.act = torch.tensor(actions, dtype=torch.float32)
        self.rew = torch.tensor(rewards, dtype=torch.float32)[:, None]
        self.tml = torch.tensor(terminals, dtype=torch.float32)[:, None]
        self.next_obs = torch.tensor(normed_next_observations, dtype=torch.float32)

        self.size = self.obs.shape[0]
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        data = {
            'obs': {
                'state': self.obs[idx], },
            'next_obs': {
                'state': self.next_obs[idx], },
            'act': self.act[idx],
            'rew': self.rew[idx],
            'tml': self.tml[idx], }

        return data
    

class RandomMuJoCoSeqDataset(BaseDataset):
    """ **D4RL-MuJoCo Sequential Dataset**

    torch.utils.data.Dataset wrapper for D4RL-MuJoCo dataset.
    Chunk the dataset into sequences of length `horizon` without padding.
    Use GaussianNormalizer to normalize the observations as default.
    Each batch contains
    - batch["obs"]["state"], observations of shape (batch_size, horizon, o_dim)
    - batch["obs"]["observation"], observations of shape (batch_size, horizon, o_dim) if observation_function provided
    - batch["act"], actions of shape (batch_size, horizon, a_dim)
    - batch["rew"], rewards of shape (batch_size, horizon, 1)
    - batch["val"], Monte Carlo return of shape (batch_size, 1)

    Args:
        dataset: Dict[str, np.ndarray],
            D4RL-MuJoCo dataset. Obtained by calling `env.get_dataset()`.
        terminal_penalty: float,
            Penalty reward for early-terminal states. Default is -100.
        horizon: int,
            Length of each sequence. Default is 1.
        max_path_length: int,
            Maximum length of the episodes. Default is 1000.
        discount: float,
            Discount factor. Default is 0.99.
        observation_function: Callable,
            Function to transform state to observation. Default is None.
        observation_kwargs: dict,
            Keyword arguments for observation_function. Default is None.

    Examples:
        >>> env = gym.make("halfcheetah-medium-expert-v2")
        >>> dataset = RandomMuJoCoSeqDataset(env.get_dataset(), horizon=32, 
        ...                                  observation_function=add_gaussian_noise,
        ...                                  observation_kwargs={'noise_std': 0.1})
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> batch = next(iter(dataloader))
        >>> obs = batch["obs"]["state"]        # (32, 32, 17)
        >>> obs_noisy = batch["obs"]["observation"]  # (32, 32, 17) - with noise
    """
    def __init__(
            self,
            dataset,
            terminal_penalty: float = 0,
            horizon: int = 1,
            max_path_length: int = 1000,
            discount: float = 0.99,
            center_mapping: bool = True,
            stride: int = 1,
            full_traj_bonus: float = 0,
            padding: int = 0,
            observation_function: Optional[Callable] = add_gaussian_noise,
            observation_kwargs: Optional[dict] = {'noise_std': 0.1},
            task_id: Optional[np.ndarray] = None,
            chunk_size: int = 100,
            episodes_per_task: int = 300,
            state_mean: Optional[np.ndarray] = None,
            state_std: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.dataset = dataset 
        self.observation_function = observation_function
        self.observation_kwargs = observation_kwargs or {}
        
        # Get metadata WITHOUT loading data
        n_episodes = dataset.total_episodes
        episode_length = dataset.total_steps // n_episodes
        
        print(f"Dataset info: {n_episodes} episodes, {episode_length} steps per episode")
        print(f"Loading in chunks of {chunk_size} episodes...")
        # Process episodes in chunks
        if task_id is not None:
            # task_id is array of task indices, e.g., [0, 2, 5]
            episode_indices = []
            for tid in task_id:
                start_ep = tid * episodes_per_task
                end_ep = min(start_ep + episodes_per_task, n_episodes)
                episode_indices.extend(range(start_ep, end_ep))
            print(f"Loading {len(episode_indices)} episodes for tasks {task_id}")
        else:
            episode_indices = list(range(n_episodes))
            print(f"Loading all {n_episodes} episodes")
        # Initialize storage lists (will concatenate later)
        all_observations = []
        all_actions = []
        all_rewards = []
        all_timeouts = []
        all_terminals = []
        all_infos = [] if hasattr(dataset[0], 'infos') and dataset[0].infos is not None else None
        


        for i in tqdm(range(0, len(episode_indices), chunk_size)):
            chunk_ep_nums = episode_indices[i:i+chunk_size]
            
            # Load only this chunk
            chunk_data = self._load_episode_chunk(
                dataset, min(chunk_ep_nums), max(chunk_ep_nums)+1, episode_length
            )
            
            # Append to storage
            all_observations.append(chunk_data["observations"])
            all_actions.append(chunk_data["actions"])
            all_rewards.append(chunk_data["rewards"])
            all_timeouts.append(chunk_data["truncations"])
            all_terminals.append(chunk_data["terminations"])
            
            if all_infos is not None:
                all_infos.extend(chunk_data["infos"])
            
            # Free memory
            del chunk_data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all chunks
        observations = np.concatenate(all_observations, axis=0).astype(np.float32)
        actions = np.concatenate(all_actions, axis=0).astype(np.float32)
        rewards = np.concatenate(all_rewards, axis=0).astype(np.float32)
        timeouts = np.concatenate(all_timeouts, axis=0).astype(np.float32)
        terminals = np.concatenate(all_terminals, axis=0).astype(np.float32)
        
        # Free intermediate lists
        del all_observations, all_actions, all_rewards, all_timeouts, all_terminals
        # Generate observations if observation_function is provided
        if self.observation_function is not None:
            projected_observations = self.observation_function(observations, **self.observation_kwargs)
        else:
            projected_observations = None
        
        #padding
        if padding > 0:
            observations = np.pad(observations,((0,0),(padding,0),(0,0)),'edge',).reshape(-1,observations.shape[-1])
            if projected_observations is not None:
                projected_observations = np.pad(projected_observations,((0,0),(padding,0),(0,0)),'edge',).reshape(-1,projected_observations.shape[-1])
            actions = np.pad(actions,((0,0),(padding,0),(0,0)),'constant',constant_values=0).reshape(-1,actions.shape[-1])
            rewards = np.pad(rewards,((0,0),(padding,0)),'constant',constant_values=0  ).reshape(-1)
            timeouts = np.pad(timeouts,((0,0),(padding,0)),'constant',constant_values=0).reshape(-1)
            terminals = np.pad(terminals,((0,0),(padding,0)),'constant',constant_values=0).reshape(-1)
        self.ref_min_scores = []
        self.ref_max_scores = []
        self.stride = stride

        # Create normalizers for both state and observation
        if state_mean is not None and state_std is not None:
            # Use provided mean and std
            print(f"Using provided normalization statistics: mean shape {state_mean.shape}, std shape {state_std.shape}")
            self.normalizers = {
                "state": GaussianNormalizer(observations)}
            # Override the computed statistics with provided ones (numpy arrays)
            self.normalizers["state"].mean = np.asarray(state_mean, dtype=np.float32)
            self.normalizers["state"].std = np.asarray(state_std, dtype=np.float32)
        else:
            # Calculate mean and std from data
            print("Calculating normalization statistics from data")
            self.normalizers = {
                "state": GaussianNormalizer(observations)}
        # print(np.min(observations[:,0]))
        # print(np.mean(observations[:,0]))
        
        if projected_observations is not None:
            self.normalizers["observation"] = GaussianNormalizer(projected_observations)
            normed_observations_obs = self.normalizers["observation"].normalize(projected_observations)
        else:
            normed_observations_obs = None
            
        normed_observations = self.normalizers["state"].normalize(observations)

        self.horizon = horizon
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]
        if task_id is not None:
            normed_observations = normed_observations
        if all_infos is not None:
            all_ref_scores = [info['ref_score'] for info in all_infos if 'ref_score' in info]
            if all_ref_scores:
                print(f"Mean ref_score: {np.mean(all_ref_scores)}")
            task_index = np.zeros((len(all_infos),episode_length+padding),dtype=np.float32)
            self.task_list = [all_infos[0]["task"]]
            ref_max_score = -np.inf
            ref_min_score = None
            self.individual_task_list = []
            for i in range(len(all_infos)):
 
                

                if np.any(all_infos[i]["task"] != self.task_list[-1]):
                    self.task_list.append(all_infos[i]["task"])
                    self.ref_max_scores.append(ref_max_score)
                    self.ref_min_scores.append(ref_min_score)
                    ref_max_score = -np.inf
                    ref_min_score = None
                if all_infos[i].get('individual_task',None) is not None:
                    self.individual_task_list.append(all_infos[i]['individual_task'])

                task_index[i] = np.ones((episode_length+padding,))*(len(self.task_list) - 1)
                all_infos[i]["task_index"] = np.ones((episode_length+padding,))*(len(self.task_list) - 1)
                ref_max_score = max(ref_max_score,all_infos[i]['ref_score'])
                ref_min_score = min(ref_min_score,all_infos[i]['ref_score']) if ref_min_score is not None else all_infos[i]['ref_score']
            self.ref_max_scores.append(ref_max_score)
            self.ref_min_scores.append(ref_min_score)
            self.task_list = np.array(self.task_list)
            self.individual_task_list = np.array(self.individual_task_list) if len(self.individual_task_list)>0 else []
            task_index = task_index.reshape(-1).astype(np.int32)
            self.task_segment = np.zeros((self.task_list.shape[0]+1,),dtype=np.int32)
        else:
            task_index = None



        n_paths = np.sum(np.logical_or(terminals, timeouts))
        self.seq_obs = np.zeros((n_paths+1, max_path_length, self.o_dim), dtype=np.float32)
        self.seq_obs_obs = np.zeros((n_paths+1, max_path_length, self.o_dim), dtype=np.float32) if normed_observations_obs is not None else None
        self.seq_act = np.zeros((n_paths+1, max_path_length, self.a_dim), dtype=np.float32)
        self.seq_rew = np.zeros((n_paths+1, max_path_length, 1), dtype=np.float32)
        self.seq_val = np.zeros((n_paths+1, max_path_length, 1), dtype=np.float32)
        self.seq_task_index = np.zeros((n_paths+1, max_path_length, 1), dtype=np.float32) if all_infos is not None else None
        self.seq_task_factor = np.zeros((n_paths+1, max_path_length, self.task_list.shape[1]), dtype=np.float32) if all_infos is not None else None
        self.indices = []

        ptr = 0
        path_idx = 0
        # for i in range(dataset.total_steps// dataset.total_episodes+padding-1,timeouts.shape[0],dataset.total_steps// dataset.total_episodes+padding):
        for i in range(timeouts.shape[0]):
            if timeouts[i] or terminals[i] or i == timeouts.shape[0] - 1:

                path_length = i - ptr + 1
                assert path_length <= max_path_length, f"current path length {path_length}"

                if terminals[i]:
                    rewards[i] = terminal_penalty if terminal_penalty is not None else rewards[i]
                    
                if path_length == max_path_length:
                    rewards[i] = rewards[i] + full_traj_bonus if full_traj_bonus is not None else rewards[i]

                self.seq_obs[path_idx, :path_length] = normed_observations[ptr:i + 1]
                if self.seq_obs_obs is not None:
                    self.seq_obs_obs[path_idx, :path_length] = normed_observations_obs[ptr:i + 1]
                self.seq_act[path_idx, :path_length] = actions[ptr:i + 1]
                self.seq_rew[path_idx, :path_length] = rewards[ptr:i + 1][:, None]

                max_start = path_length - (horizon - 1) * stride - 1
                self.indices += [(path_idx, start, start + (horizon - 1) * stride + 1) for start in range(max_start + 1)]
                if all_infos is not None:
                    self.seq_task_index[path_idx, :path_length] = task_index[ptr:i + 1][:, None]
                    if len(self.individual_task_list)>0:
                        self.seq_task_factor[path_idx, :path_length] = self.individual_task_list[path_idx][None,: ]
                    self.task_segment[task_index[i]+1] = len(self.indices)
                ptr = i + 1
                path_idx += 1

        self.seq_val[:, -1] = self.seq_rew[:, -1]
        for i in reversed(range(max_path_length-1)):
            self.seq_val[:, i] = self.seq_rew[:, i] + discount * self.seq_val[:, i+1]

        print(f"max discounted return: {self.seq_val.max()}")
        print(f"min discounted return: {self.seq_val.min()}")
        
        # val \in [-1, 1]
        self.seq_val = (self.seq_val - self.seq_val.min()) / (self.seq_val.max() - self.seq_val.min())
        # rew \in [0, 1]
        self.seq_rew = (self.seq_rew - self.seq_rew.min()) / (self.seq_rew.max() - self.seq_rew.min())
        if center_mapping:
            self.seq_val = self.seq_val * 2 - 1
        print(f"max normed discounted return: {self.seq_val.max()}")
        print(f"min normed discounted return: {self.seq_val.min()}")

    def get_normalizer(self):
        return self.normalizers["state"]
    
    def get_observation_normalizer(self):
        """Get normalizer for observations if available"""
        return self.normalizers.get("observation", None)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]
        
        horizon_state = self.seq_obs[path_idx, start:end:self.stride]
        horizon_observation = None
        
        if self.seq_obs_obs is not None:
            horizon_observation = self.seq_obs_obs[path_idx, start:end:self.stride]

        obs_dict = {'state': horizon_state}
        if horizon_observation is not None:
            obs_dict['observation'] = horizon_observation

        task_id = self.seq_task_index[path_idx, start] if self.seq_task_index is not None else None
        task_list = self.task_list[task_id.astype(int).tolist(), :] if task_id is not None else None

        data = {
            'obs': obs_dict,
            'act': self.seq_act[path_idx, start:end:self.stride],
            'rew': self.seq_rew[path_idx, start:end:self.stride],
            'val': self.seq_val[path_idx, start],
            'task_id': task_id,
            'task_list': task_list,
            'individual_task': self.seq_task_factor[path_idx, start] if self.seq_task_factor is not None else None,
        }

        torch_data = dict_apply(data, torch.tensor)

        return torch_data
    
    def get_all_task(self):
        return self.task_list
    
    def recover_environment(self, task_id: int):

        ''' recover the environment of a specific task id'''

        env = self.dataset.recover_environment()
        env.unwrapped.set_task(*self.task_list[task_id])
        return env
    
    def get_ref_score(self,task_id: int):

        ''' recover the reference score of a specific task id
            return (min_score,max_score)
        '''

        return self.ref_min_scores[task_id],self.ref_max_scores[task_id]

    def _load_episode_chunk(self, dataset, start_idx, end_idx, episode_length):
        """Load a chunk of episodes from the Minari dataset
        
        Args:
            dataset: Minari dataset object
            start_idx: Starting episode index (inclusive)
            end_idx: Ending episode index (exclusive)
            episode_length: Expected length of each episode
            
        Returns:
            Dictionary with batched episode data
        """
        chunk_size = end_idx - start_idx
        
        # Pre-allocate arrays for this chunk
        obs_dim = dataset.observation_space.shape[0]
        act_dim = dataset.action_space.shape[0]
        
        chunk_obs = np.zeros((chunk_size, episode_length, obs_dim), dtype=np.float32)
        chunk_act = np.zeros((chunk_size, episode_length, act_dim), dtype=np.float32)
        chunk_rew = np.zeros((chunk_size, episode_length), dtype=np.float32)
        chunk_term = np.zeros((chunk_size, episode_length), dtype=np.float32)
        chunk_trunc = np.zeros((chunk_size, episode_length), dtype=np.float32)
        chunk_infos = []
        
        # Load episodes one by one
        for i, ep_idx in enumerate(range(start_idx, end_idx)):
            episode = dataset[ep_idx]
            
            ep_len = len(episode.observations)
            
            # Handle episodes shorter than episode_length
            actual_len = min(ep_len, episode_length)
            
            chunk_obs[i, :actual_len] = episode.observations[:actual_len]
            chunk_act[i, :actual_len] = episode.actions[:actual_len]
            chunk_rew[i, :actual_len] = episode.rewards[:actual_len]
            chunk_term[i, :actual_len] = episode.terminations[:actual_len]
            chunk_trunc[i, :actual_len] = episode.truncations[:actual_len]
            
            if episode.infos is not None:
                chunk_infos.append(episode.infos)
        
        return {
            "observations": chunk_obs,
            "actions": chunk_act,
            "rewards": chunk_rew,
            "terminations": chunk_term,
            "truncations": chunk_trunc,
            "infos": chunk_infos if len(chunk_infos) > 0 else None,
        }

def split_dataset(dataset, split_ratio=[0.8,0.1,0.1], seed=42):
    np.random.seed(seed)
    indices = np.arange(dataset.task_list.shape[0])
    np.random.shuffle(indices)
    train_ratio = split_ratio[0] / (split_ratio[0] + split_ratio[1] + split_ratio[2])
    val_ratio = split_ratio[1] / (split_ratio[0] + split_ratio[1] + split_ratio[2])
    train_indices = []
    val_indices = []
    test_indices = []
    split1 = int(len(indices) * train_ratio)
    for i in range(len(indices)):
        if i < split1:
            train_indices += [j for j in range(dataset.task_segment[indices[i]],dataset.task_segment[indices[i]+1])]
        elif i < split1 + int(len(indices) * val_ratio):
            val_indices += [j for j in range(dataset.task_segment[indices[i]],dataset.task_segment[indices[i]+1])]
        else:
            test_indices += [j for j in range(dataset.task_segment[indices[i]],dataset.task_segment[indices[i]+1])]



    # indices = np.array([0,6,3,8,10,11,16,29])
    # for i in range(len(indices)):
    #     train_indices += [j for j in range(dataset.task_segment[indices[i]],dataset.task_segment[indices[i]+1])]


    train_set = Subset(dataset, train_indices)
    train_task_list = dataset.task_list[indices[:split1]]
    val_set = Subset(dataset, val_indices)
    val_task_list = dataset.task_list[indices[split1:split1 + int(len(indices) * val_ratio)]]
    test_set = Subset(dataset, test_indices)
    test_task_list = dataset.task_list[indices[split1 + int(len(indices) * val_ratio):]]
    return train_set, val_set, test_set


def get_task_data(dataset, task_ids: np.array):
    ''' get the dataset of specific task ids
        indices: np.array, the task ids,e.g., np.array([0,1,2])

    '''
    output_indices = []
    for i in range(len(task_ids)):
        output_indices += [j for j in range(dataset.task_segment[task_ids[i]],dataset.task_segment[task_ids[i]+1])]

    return Subset(dataset, output_indices)

class EmbeddingMuJoCoSeqDataset(RandomMuJoCoSeqDataset):
    '''This dataset is still under development. It is used to provide the embedding for each trajectory segment.
    Supports optional state_mean and state_std for normalization.'''
    def __init__(
            self,
            dataset,
            terminal_penalty: float = 0,
            horizon: int = 1,
            max_path_length: int = 1000,
            discount: float = 0.99,
            center_mapping: bool = True,
            stride: int = 1,
            full_traj_bonus: float = 0,
            padding: int = 0,
            save_embedding_path = None,
            chunk_size: int = 100,
            task_id: Optional[np.ndarray] = None,
            episodes_per_task: int = 300,
            state_mean: Optional[np.ndarray] = None,
            state_std: Optional[np.ndarray] = None,
    ):
        if save_embedding_path is not None:
            self.embedding = np.load(save_embedding_path,allow_pickle=True)['embeddings']
            print('='*50)
            print('embedding shape:',self.embedding.shape)
            print('='*50)
        else:
            raise ValueError("save_embedding_path is required for EmbeddingMuJoCoSeqDataset")
        super().__init__(dataset,terminal_penalty,horizon,max_path_length,discount,center_mapping,stride,full_traj_bonus,padding,chunk_size=chunk_size,task_id=task_id,episodes_per_task=episodes_per_task)
        self._build_task_index_lookup()
        
        # if embedding_model is None:
        #     raise ValueError("embedding_model is required for EmbeddingMuJoCoSeqDataset")
        # self.embedding_model = embedding_model  

    def _build_task_index_lookup(self):
        """Cache dataset indices grouped by task for fast same-task sampling."""
        self._task_to_indices = None
        if getattr(self, "seq_task_index", None) is None:
            return
        self._task_to_indices = defaultdict(list)
        for sample_idx, (path_idx, start, _) in enumerate(self.indices):
            task_value = self.seq_task_index[path_idx, start]
            if isinstance(task_value, np.ndarray):
                task_value = task_value.item()
            try:
                task_id = int(task_value)
            except (TypeError, ValueError):
                continue
            self._task_to_indices[task_id].append(sample_idx)

    def _sample_same_task_index(self, task_id: Optional[int], fallback_idx: int) -> int:
        """Return a dataset index from the same task, avoiding the fallback index when possible."""
        if self._task_to_indices is None or task_id is None:
            return fallback_idx
        candidates = self._task_to_indices.get(task_id)
        if not candidates:
            return fallback_idx
        if len(candidates) == 1:
            return candidates[0]
        candidate = fallback_idx
        for _ in range(5):  # a few tries to avoid returning the original index
            candidate = random.choice(candidates)
            if candidate != fallback_idx:
                break
        return candidate

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]
        
        horizon_state = self.seq_obs[path_idx, start:end:self.stride]

        task_id = None
        if getattr(self, "seq_task_index", None) is not None:
            task_value = self.seq_task_index[path_idx, start]
            if isinstance(task_value, np.ndarray):
                task_value = task_value.item()
            try:
                task_id = int(task_value)
            except (TypeError, ValueError):
                task_id = None

        embedding_sample_idx = self._sample_same_task_index(task_id, idx)
        emb_path_idx, emb_start, emb_end = self.indices[embedding_sample_idx]
        emb_state = self.seq_obs[emb_path_idx, emb_start:emb_end:self.stride]
        emb_act = self.seq_act[emb_path_idx, emb_start:emb_end:self.stride]
        emb_state_act = np.concatenate([emb_state, emb_act], axis=-1)

        data = {
            'obs': {'state': horizon_state},
            'act': self.seq_act[path_idx, start:end:self.stride],
            'rew': self.seq_rew[path_idx, start:end:self.stride],
            'val': self.seq_val[path_idx, start],
            'embedding': self.embedding[idx,:],
            'embedding_traj': emb_state_act
        }

        torch_data = dict_apply(data, torch.tensor)
        # data['embedding'] = self.embedding_model.dynamics.encode_history(data['obs']['state'][:,:self.dataset.training_config['history']],data['act'][:,:self.dataset.training_config['history']]).detach(),

        return torch_data

class PairRandomMuJoCoSeqDataset(RandomMuJoCoSeqDataset):
    """
    配对版本的RandomMuJoCoSeqDataset，初始化时只构建pair_idx_list，不提前分配paired数据。
    每次getitem时动态采样两个样本并组装为paired dict，节省内存。
    支持每个epoch后重新打乱pair_idx_list。
    Supports optional state_mean and state_std for normalization.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'task_list') and self.task_list is not None:
            self._build_task_index_sets()
            self._build_pair_idx_list()
            self._build_onehot_task_list()
        else:
            raise ValueError("PairRandomMuJoCoSeqDataset requires task information in dataset")
        self._pair_epoch_counter = 0
        self._access_counter = 0

    def _build_onehot_task_list(self):
        """构建onehot编码的task factor列表"""
        num_tasks = len(self.task_index_sets)
        self.onehot_task_list = np.eye(num_tasks, dtype=np.float32)
        

    def _build_task_index_sets(self):
        """为每个task构建索引集合（带进度条）"""
        self.task_index_sets = {}
        print("Building task index sets...")


        pbar = tqdm(total=len(self.indices), desc="Task index sets", ncols=80)
        for idx in range(len(self.indices)):
            path_idx, start, end = self.indices[idx]
            if self.seq_task_index is not None:
                task_id = self.seq_task_index[path_idx, start]
                task_id = int(task_id.item() if hasattr(task_id, 'item') else task_id)
            else:
                task_id = -1
            if task_id not in self.task_index_sets:
                self.task_index_sets[task_id] = []
            self.task_index_sets[task_id].append(idx)
            pbar.update(1)
        pbar.close()
        
        
        
        
        # 转换为numpy数组以便快速采样
        for task_id in self.task_index_sets:
            self.task_index_sets[task_id] = np.array(self.task_index_sets[task_id])
        # 过滤掉只有一个样本的task（无法配对）
        valid_task_ids = [task_id for task_id, indices in self.task_index_sets.items() if len(indices) >= 2]
        self.valid_task_ids = np.array(valid_task_ids)
        print(f"Task index sets built:")
        for task_id, indices in self.task_index_sets.items():
            print(f"  Task {task_id}: {len(indices)} samples")
        print(f"Valid tasks for pairing: {len(self.valid_task_ids)} (need >=2 samples per task)")
    
    def _get_task_id_for_idx(self, idx: int) -> int:
        """获取指定索引对应的task_id"""
        path_idx, start, end = self.indices[idx]
        
        if self.seq_task_index is not None:
            task_id = self.seq_task_index[path_idx, start]
            task_id = int(task_id.item() if hasattr(task_id, 'item') else task_id)
        else:
            task_id = 0
        
        return task_id
    
    def _get_single_np_sample(self, idx: int, ONEHOT: bool = False):
        """获取单个样本（返回np.array，不做torch转换）"""
        path_idx, start, end = self.indices[idx]
        horizon_state = self.seq_obs[path_idx, start:end:self.stride]
        horizon_observation = None
        if self.seq_obs_obs is not None:
            horizon_observation = self.seq_obs_obs[path_idx, start:end:self.stride]
        task_id = self.seq_task_index[path_idx, start] if self.seq_task_index is not None else None
        if ONEHOT and task_id is not None:
            task_list = self.onehot_task_list[task_id.astype(int).tolist(), :] if task_id is not None else None  # 保持二维形状
        else:
            task_list = self.task_list[task_id.astype(int).tolist(), :] if task_id is not None else None
        individual_task = self.seq_task_factor[path_idx, start] if self.seq_task_factor is not None else None
        return {
            'obs_state': horizon_state,
            'obs_observation': horizon_observation,
            'act': self.seq_act[path_idx, start:end:self.stride],
            'rew': self.seq_rew[path_idx, start:end:self.stride],
            'val': self.seq_val[path_idx, start],
            'task_id': task_id,
            'task_list': task_list,
            'individual_task': individual_task,
        }

    def _build_pair_idx_list(self):
        """每个task内shuffle配对，保存pair_idx_list"""
        self.pair_idx_list = []
        for task_id, task_indices in self.task_index_sets.items():
            if len(task_indices) < 2:
                continue
            indices = task_indices.copy()
            shuffled = np.random.permutation(indices)
            pairs = np.stack([indices, shuffled], axis=1)
            self.pair_idx_list.extend(pairs.tolist())
        self.num_pairs = len(self.pair_idx_list)
        print(f"PairRandomMuJoCoSeqDataset: built {self.num_pairs} pairs.")
        # 重置访问计数器
        self._access_counter = 0

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx: int):
        # 增加访问计数
        self._access_counter += 1
        
        # 当访问次数达到总数时，说明一个epoch结束，重新构建配对
        if self._access_counter >= self.num_pairs:
            self._pair_epoch_counter += 1
            print(f"[PairRandomMuJoCoSeqDataset] End of epoch {self._pair_epoch_counter}, reshuffling pairs...")
            self._build_pair_idx_list()
        
        idx1, idx2 = self.pair_idx_list[idx]
        s1 = self._get_single_np_sample(idx1)
        s2 = self._get_single_np_sample(idx2)
        def stack_or_none(a, b):
            if a is not None and b is not None:
                return np.stack([a, b], axis=0)
            else:
                return None
        obs_dict = {'state': stack_or_none(s1['obs_state'], s2['obs_state'])}
        if s1['obs_observation'] is not None and s2['obs_observation'] is not None:
            obs_dict['observation'] = stack_or_none(s1['obs_observation'], s2['obs_observation'])
        data = {
            'obs': obs_dict,
            'act': stack_or_none(s1['act'], s2['act']),
            'rew': stack_or_none(s1['rew'], s2['rew']),
            'val': stack_or_none(s1['val'], s2['val']),
            'task_id': stack_or_none(s1['task_id'], s2['task_id']),
            'task_list': stack_or_none(s1['task_list'], s2['task_list']),
            'individual_task': stack_or_none(s1['individual_task'], s2['individual_task']),
        }
        return dict_apply(data, torch.tensor)
