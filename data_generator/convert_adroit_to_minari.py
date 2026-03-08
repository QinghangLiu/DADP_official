import os
import h5py
import numpy as np
import minari
import gymnasium as gym
from minari import DataCollector
from minari.storage.datasets_root_dir import get_dataset_path

def convert_hdf5_to_minari(hdf5_path, env_id, dataset_id, author="Unknown", author_email="unknown@example.com"):
    """
    Converts a D4RL-style HDF5 dataset to Minari format.
    """
    print(f"Converting {hdf5_path} to Minari dataset {dataset_id}...")
    
    with h5py.File(hdf5_path, 'r') as f:
        observations = f['observations'][:]
        actions = f['actions'][:]
        rewards = f['rewards'][:]
        terminals = f['terminals'][:]
        
        # Handle timeouts if present, otherwise infer from terminals or max steps
        if 'timeouts' in f:
            timeouts = f['timeouts'][:]
        else:
            timeouts = np.zeros_like(terminals)

    # Create the environment to get space info
    # We need to register the adroit envs first if not already done
    # Assuming adroit package is available or we can mock the spaces
    try:
        import sys
        sys.path.append(os.path.join(os.getcwd(), "off_dynamics_rl", "envs"))
        import adroit
        env = gym.make(env_id)
    except Exception as e:
        print(f"Warning: Could not create environment {env_id}: {e}")
        print("Attempting to infer spaces from data...")
        # Fallback: Create dummy env with correct spaces
        obs_dim = observations.shape[1]
        act_dim = actions.shape[1]
        
        class DummyEnv(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
                self.action_space = gym.spaces.Box(-1, 1, shape=(act_dim,), dtype=np.float32)
            def reset(self, seed=None, options=None):
                return np.zeros(obs_dim, dtype=np.float32), {}
            def step(self, action):
                return np.zeros(obs_dim, dtype=np.float32), 0.0, False, False, {}
        
        env = DummyEnv()

    # Split data into episodes
    episode_buffers = []
    
    start_idx = 0
    N = len(observations)
    
    episode_id = 0
    
    for i in range(N):
        # Check for terminal or timeout (end of episode)
        # Note: D4RL datasets often have terminals=True at the end of trajectory
        # or timeouts=True.
        # Sometimes the last step of the file is implicitly the end.
        
        is_terminal = bool(terminals[i])
        is_timeout = bool(timeouts[i])
        is_last = (i == N - 1)
        
        if is_terminal or is_timeout or is_last:
            end_idx = i + 1
            
            # Extract episode data
            ep_obs = observations[start_idx:end_idx]
            ep_act = actions[start_idx:end_idx]
            ep_rew = rewards[start_idx:end_idx].flatten()
            
            # Construct terminations and truncations
            # Length of these arrays should match the number of steps (transitions)
            # But Minari expects observations to be (T+1) if we follow the standard?
            # Actually Minari EpisodeBuffer expects:
            # observations: (T+1, ...) or (T, ...) depending on format?
            # Let's check Minari docs/examples.
            # Usually: obs (T+1), act (T), rew (T), term (T), trunc (T)
            
            # However, D4RL HDF5 usually gives obs (T), act (T), rew (T), etc.
            # The "next_observation" is usually obs[i+1].
            # So we have T steps.
            # We need to append the final observation (next_obs of last step) if available.
            # But in this flat format, obs[i+1] IS the next obs, unless it's the start of new ep.
            # For the very last step of episode, we might not have the true next_obs if it terminated.
            # But usually D4RL stores transitions.
            
            # Let's construct arrays of length T
            T = end_idx - start_idx
            
            # Minari requires observations to be length T+1 (including final observation)
            # We can try to grab the next observation from the next index if it exists and is not a new episode start
            # But here we are at the end of episode.
            # If it's terminal, the "next state" is the terminal state.
            # If we don't have it, we might have to duplicate the last one or pad.
            # Actually, for D4RL, obs[i] is the state at step t.
            # We need obs[t+1].
            
            # Let's take obs[start_idx : end_idx] as the sequence of observations.
            # This gives T observations.
            # We need one more.
            # If it's not the absolute last index of file, maybe obs[end_idx] is valid?
            # NO, obs[end_idx] is the start of NEXT episode.
            
            # So we effectively have T observations.
            # Minari EpisodeBuffer:
            # observations: The observations of the episode. Shape (N+1, *observation_space.shape).
            
            # We are missing the final observation.
            # We will duplicate the last observation as a placeholder for the final state
            # This is a common workaround when converting static datasets that dropped the final next_state.
            
            ep_obs_minari = np.concatenate([ep_obs, ep_obs[-1:]], axis=0)
            
            ep_term = np.zeros(T, dtype=bool)
            ep_trunc = np.zeros(T, dtype=bool)
            
            if is_terminal:
                ep_term[-1] = True
            if is_timeout:
                ep_trunc[-1] = True
                
            # Create EpisodeBuffer
            buffer = minari.data_collector.EpisodeBuffer(
                id=episode_id,
                observations=ep_obs_minari,
                actions=ep_act,
                rewards=ep_rew,
                terminations=ep_term,
                truncations=ep_trunc
            )
            episode_buffers.append(buffer)
            
            episode_id += 1
            start_idx = end_idx

    # Create Minari Dataset
    if dataset_id in minari.list_local_datasets():
        print(f"Dataset {dataset_id} already exists. Deleting...")
        minari.delete_dataset(dataset_id)

    dataset = minari.create_dataset_from_buffers(
        dataset_id=dataset_id,
        env=env,
        buffer=episode_buffers,
        algorithm_name="Expert",
        author=author,
        author_email=author_email,
        code_permalink="https://github.com/Farama-Foundation/Minari",
    )
    
    print(f"Successfully created Minari dataset: {dataset_id}")
    return dataset

if __name__ == "__main__":
    # Example usage
    # You can loop through your files here
    
    # Define mapping from filename to env_id and dataset_id
    # Example: door_broken_joint_easy_expert.hdf5 -> door-broken-joint-easy-v0
    
    base_path = "/home/qinghang/DomainAdaptiveDiffusionPolicy/off_dynamics_rl/dataset/adroit"
    
    # List all hdf5 files
    files = [f for f in os.listdir(base_path) if f.endswith('.hdf5')]
    
    for filename in files:
        if "expert" not in filename:
            continue
            
        # Parse filename to get env name
        # e.g. door_broken_joint_easy_expert.hdf5
        # env_id should be door-broken-joint-easy-v0 (assuming registered in adroit)
        
        name_parts = filename.replace('.hdf5', '').split('_')
        # name_parts = ['door', 'broken', 'joint', 'easy', 'expert']
        
        # Reconstruct env name. 
        # The adroit registration uses dashes.
        # Remove 'expert' from the end
        env_name_parts = name_parts[:-1]
        env_name = "-".join(env_name_parts)
        env_id = f"{env_name}-v0"
        
        dataset_id = f"{env_name}-expert-v0"
        
        file_path = os.path.join(base_path, filename)
        
        try:
            convert_hdf5_to_minari(file_path, env_id, dataset_id)
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")
