import os
import h5py
import numpy as np
import minari
import gymnasium as gym

def load_hdf5_episodes(hdf5_path, start_episode_id, task_vector, task_index):
    """
    Loads episodes from an HDF5 file and returns a list of Minari EpisodeBuffers.
    """
    print(f"Loading data from {hdf5_path}...")
    
    with h5py.File(hdf5_path, 'r') as f:
        observations = f['observations'][:]
        actions = f['actions'][:]
        rewards = f['rewards'][:]
        terminals = f['terminals'][:]
        
        if 'timeouts' in f:
            timeouts = f['timeouts'][:]
        else:
            timeouts = np.zeros_like(terminals)

    episode_buffers = []
    start_idx = 0
    N = len(observations)
    episode_id = start_episode_id
    
    for i in range(N):
        is_terminal = bool(terminals[i])
        is_timeout = bool(timeouts[i])
        is_last = (i == N - 1)
        
        if is_terminal or is_timeout or is_last:
            end_idx = i + 1
            
            ep_obs = observations[start_idx:end_idx]
            ep_act = actions[start_idx:end_idx]
            ep_rew = rewards[start_idx:end_idx].flatten()
            
            # Duplicate last observation for Minari format (N+1 observations)
            ep_obs_minari = np.concatenate([ep_obs, ep_obs[-1:]], axis=0)
            
            T = len(ep_obs)
            ep_term = np.zeros(T, dtype=bool)
            ep_trunc = np.zeros(T, dtype=bool)
            
            if is_terminal:
                ep_term[-1] = True
            if is_timeout:
                ep_trunc[-1] = True
            
            infos = {}
            infos['task_index'] = task_index * np.ones(T)
            infos['individual_task'] = task_vector
            infos['task'] = task_vector
            infos['ref_score'] = np.sum(ep_rew)
            infos['no_randomization_score'] = np.sum(ep_rew)

            buffer = minari.data_collector.EpisodeBuffer(
                id=episode_id,
                observations=ep_obs_minari,
                actions=ep_act,
                rewards=ep_rew,
                terminations=ep_term,
                truncations=ep_trunc,
                infos=infos
            )
            episode_buffers.append(buffer)
            
            episode_id += 1
            start_idx = end_idx
            
    return episode_buffers, episode_id

def main():
    # List of datasets to combine
    base_dir = "/home/qinghang/DomainAdaptiveDiffusionPolicy"
    files = [
        ("/home/qinghang/DomainAdaptiveDiffusionPolicy/off_dynamics_rl/dataset/adroit/relocate_shrink_finger_easy_expert.hdf5", np.array([0.0])),
        ("/home/qinghang/DomainAdaptiveDiffusionPolicy/off_dynamics_rl/dataset/adroit/relocate_shrink_finger_hard_expert.hdf5", np.array([2.0])),
        ("/home/qinghang/DomainAdaptiveDiffusionPolicy/off_dynamics_rl/dataset/adroit/relocate_shrink_finger_medium_expert.hdf5", np.array([1.0])),
        # "off_dynamics_rl/dataset/adroit/hammer_shrink_finger_easy_expert.hdf5",
        # "off_dynamics_rl/dataset/adroit/hammer_shrink_finger_hard_expert.hdf5",
        # "off_dynamics_rl/dataset/adroit/hammer_shrink_finger_medium_expert.hdf5"
    ]
    
    combined_buffers = []
    current_episode_id = 0
    
    # Load and combine all episodes
    for i, (rel_path, task_vector) in enumerate(files):
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(full_path):
            print(f"Warning: File not found: {full_path}")
            continue
            
        buffers, next_episode_id = load_hdf5_episodes(full_path, current_episode_id, task_vector, i)
        combined_buffers.extend(buffers)
        current_episode_id = next_episode_id
        print(f"Added {len(buffers)} episodes from {os.path.basename(rel_path)}")

    if not combined_buffers:
        print("No data loaded. Exiting.")
        return

    # Create a dummy environment for space definition based on the first episode
    first_obs = combined_buffers[0].observations
    first_act = combined_buffers[0].actions
    
    obs_dim = first_obs.shape[1]
    act_dim = first_act.shape[1]
    
    print(f"Inferred Observation Dim: {obs_dim}")
    print(f"Inferred Action Dim: {act_dim}")
    
    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
            self.action_space = gym.spaces.Box(-1, 1, shape=(act_dim,), dtype=np.float32)
        def reset(self, seed=None, options=None):
            return np.zeros(obs_dim, dtype=np.float32), {}
        def step(self, action):
            return np.zeros(obs_dim, dtype=np.float32), 0.0, False, False, {}

    env = DummyEnv()
    
    dataset_id = "Adroit/relocate_shrink_combined-v0"
    
    try:
        minari.delete_dataset(dataset_id)
        print(f"Deleted existing dataset {dataset_id}")
    except (KeyError, FileNotFoundError):
        pass
        
    print(f"Creating Minari dataset {dataset_id} with {len(combined_buffers)} total episodes...")
    
    dataset = minari.create_dataset_from_buffers(
        dataset_id=dataset_id,
        env=env,
        buffer=combined_buffers,
    )
    
    print(f"Successfully created dataset: {dataset_id}")

if __name__ == "__main__":
    main()
