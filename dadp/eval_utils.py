import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from .dadp import DADP

BIAS = 2

def extract_embeddings_from_dataset(model, policy_dataset, batch_size=128, device="cuda"):
    """Extract embeddings from entire dataset"""
    model.eval()
    model.to(device)
    print("Extracting embeddings from entire dataset...")
    
    # Get task information
    task_list = getattr(policy_dataset, 'task_list', None)
    if task_list is not None:
        print(f"Found {len(task_list)} tasks")
    
    # Create dataloader
    full_loader = DataLoader(
        policy_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    all_embeddings = []
    all_task_ids = []
    history = model.training_config.history
    
    with torch.no_grad():
        for batch in tqdm(full_loader, desc="Processing batches"):
            # 使用与训练时相同的数据选择逻辑
            if model.training_config.use_observation and "observation" in batch["obs"]:
                states = batch["obs"]["observation"].to(device, non_blocking=True)
            else:
                states = batch["obs"]["state"].to(device, non_blocking=True)
            rewards = batch["rew"].to(device, non_blocking=True)
            actions = batch["act"].to(device, non_blocking=True)
            task_ids = batch.get("task_id", None)
            
            B, L, _ = states.shape
            
            if L < history:
                continue
            
            # Extract history
            state_history = states[:, :history, :]
            action_history = actions[:, :history, :]
            # reward_history = rewards[:, :history]
            # Get embeddings
            embeddings = model.dynamics.encode_history(state_history, action_history
                                                    #    , reward_history
                                                       )
            
            all_embeddings.append(embeddings.cpu().numpy())
            
            # Store task IDs - 修复维度处理问题
            if task_ids is not None:
                task_ids_np = task_ids.cpu().numpy()
                # 确保task_ids是正确的形状
                if task_ids_np.ndim == 0:  # 标量
                    task_ids_np = np.array([task_ids_np] * B)
                elif task_ids_np.ndim == 1 and len(task_ids_np) == 1:  # 单元素数组
                    task_ids_np = np.array([task_ids_np[0]] * B)
                elif task_ids_np.ndim > 1:  # 多维数组，压扁
                    task_ids_np = task_ids_np.flatten()
                    if len(task_ids_np) == 1:
                        task_ids_np = np.array([task_ids_np[0]] * B)
                
                # 确保长度匹配batch_size
                if len(task_ids_np) != B:
                    # 如果长度不匹配，使用第一个值填充
                    first_val = task_ids_np[0] if len(task_ids_np) > 0 else 0
                    task_ids_np = np.array([first_val] * B)
                
                # 确保是整数类型
                task_ids_np = task_ids_np.astype(int)
                all_task_ids.append(task_ids_np)
            else:
                all_task_ids.append(np.zeros(B, dtype=int))
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_task_ids = np.concatenate(all_task_ids, axis=0)
    
    # 确保task_ids是整数类型且形状正确
    all_task_ids = all_task_ids.astype(int).flatten()
    
    print(f"Extracted {len(all_embeddings)} embeddings, shape: {all_embeddings.shape}")
    print(f"Task IDs shape: {all_task_ids.shape}, unique tasks: {np.unique(all_task_ids)}")
    
    return {
        'embeddings': all_embeddings,
        'task_ids': all_task_ids,
        'task_list': task_list
    }

def save_embeddings_data(embeddings_data, save_path):
    """Save embeddings data to compressed file"""
    print(f"Saving embeddings data to {save_path}")
    np.savez_compressed(save_path, **embeddings_data)

def load_embeddings_data(save_path):
    """Load embeddings data from file"""
    print(f"Loading embeddings data from {save_path}")
    data = np.load(save_path, allow_pickle=True)
    
    return {
        'embeddings': data['embeddings'],
        'task_ids': data['task_ids'],
        'task_list': data['task_list'] if 'task_list' in data else None
    }

def extract_and_save_embeddings(model, policy_dataset, checkpoint_path, batch_size=128, device="cuda"):
    """Extract and save embeddings pipeline"""
    with torch.no_grad():
        model.eval()
        
        embeddings_data = extract_embeddings_from_dataset(
            model, policy_dataset, batch_size=batch_size, device=device
        )
        
        save_dir = os.path.dirname(checkpoint_path)
        save_path = os.path.join(save_dir, "embeddings_data.npz")
        save_embeddings_data(embeddings_data, save_path)
        
        print(f"\nExtracted {len(embeddings_data['embeddings'])} embeddings")
        return save_path

def load_dataset_from_checkpoint_metadata(checkpoint_path, dataset_name_override=None):
    """Load dataset using checkpoint metadata with optional dataset name override"""
    import json
    
    model, metadata = DADP.load_checkpoint(checkpoint_path, "cpu")
    
    # Get dataset parameters - use override if provided
    if dataset_name_override:
        dataset_name = dataset_name_override
        print(f"Using override dataset name: {dataset_name}")
    else:
        dataset_name = metadata.get('dataset_name', 'RandomWalker2d/40dynamics-v2')
        print(f"Using dataset name from metadata: {dataset_name}")
    
    training_config = model.training_config
    
    # Calculate horizon - 与训练时一致
    horizon = metadata.get('training_config').get('history')
    
    # Load normalization parameters from config.json in checkpoint directory
    state_mean = None
    state_std = None
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, 'config.json')
    
    if os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
            if 'state_mean' in config and config['state_mean'] is not None:
                state_mean = np.array(config['state_mean'], dtype=np.float32)
                print(f"Loaded state_mean shape: {state_mean.shape}")
            if 'state_std' in config and config['state_std'] is not None:
                state_std = np.array(config['state_std'], dtype=np.float32)
                print(f"Loaded state_std shape: {state_std.shape}")
    else:
        print(f"Config not found at: {config_path}")
    
    # Get observation function
    observation_function = None
    observation_kwargs = None
    if training_config.use_observation:
        obs_func_name = metadata.get('observation_function')
        obs_kwargs = metadata.get('observation_kwargs')
        
        if obs_func_name and obs_kwargs:
            from cleandiffuser.dataset.d4rl_mujoco_dataset import add_gaussian_noise, mask_dimensions
            if obs_func_name == "gaussian_noise":
                observation_function = add_gaussian_noise
                observation_kwargs = obs_kwargs
            elif obs_func_name == "mask_dimensions":
                observation_function = mask_dimensions
                observation_kwargs = obs_kwargs
    

    
    # Load dataset
    from .train_utils import get_dataset
    state_dim, action_dim, policy_dataset = get_dataset(
        dataset_name=dataset_name,
        horizon=20,
        stride=1,
        observation_function=observation_function,
        observation_kwargs=observation_kwargs,
        history=16,
        pair_dataset=False,
        state_mean=state_mean,
        state_std=state_std
    )
    
    dataset_params = {
        'dataset_name': dataset_name,
        'horizon': horizon
    }
    
    return model, metadata, state_dim, action_dim, policy_dataset, dataset_params

def load_dataset_from_checkpoint_metadata_horizon(checkpoint_path, dataset_name_override=None, horizon=21):
    """Load dataset using checkpoint metadata with optional dataset name override"""
    import json
    
    model, metadata = DADP.load_checkpoint(checkpoint_path, "cpu")
    
    # Get dataset parameters - use override if provided
    if dataset_name_override:
        dataset_name = dataset_name_override
        print(f"Using override dataset name: {dataset_name}")
    else:
        dataset_name = metadata.get('dataset_name', 'RandomWalker2d/40dynamics-v2')
        print(f"Using dataset name from metadata: {dataset_name}")
    
    training_config = model.training_config

    # Load normalization parameters from config.json in checkpoint directory
    state_mean = None
    state_std = None
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, 'config.json')
    
    if os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
            if 'state_mean' in config and config['state_mean'] is not None:
                state_mean = np.array(config['state_mean'], dtype=np.float32)
                print(f"Loaded state_mean shape: {state_mean.shape}")
            if 'state_std' in config and config['state_std'] is not None:
                state_std = np.array(config['state_std'], dtype=np.float32)
                print(f"Loaded state_std shape: {state_std.shape}")
    else:
        print(f"Config not found at: {config_path}")

    
    # Get observation function
    observation_function = None
    observation_kwargs = None
    if training_config.use_observation:
        obs_func_name = metadata.get('observation_function')
        obs_kwargs = metadata.get('observation_kwargs')
        
        if obs_func_name and obs_kwargs:
            from cleandiffuser.dataset.d4rl_mujoco_dataset import add_gaussian_noise, mask_dimensions
            if obs_func_name == "gaussian_noise":
                observation_function = add_gaussian_noise
                observation_kwargs = obs_kwargs
            elif obs_func_name == "mask_dimensions":
                observation_function = mask_dimensions
                observation_kwargs = obs_kwargs
    
    # Load dataset
    from .train_utils import get_dataset
    state_dim, action_dim, policy_dataset = get_dataset(
        dataset_name=dataset_name,
        horizon=horizon,
        stride=1,
        observation_function=observation_function,
        observation_kwargs=observation_kwargs,
        history=16,
        pair_dataset=False,
        state_mean=state_mean,
        state_std=state_std
    )
    
    dataset_params = {
        'dataset_name': dataset_name,
        'horizon': horizon
    }
    
    return model, metadata, state_dim, action_dim, policy_dataset, dataset_params

def visualize_tsne_embeddings(embeddings_data, save_dir, max_points_per_task=1000):
    """t-SNE embedding visualization and save as HTML, 每个任务最多采样 max_points_per_task 个点"""
    try:
        from sklearn.manifold import TSNE
        import plotly.express as px
    except ImportError:
        print("scikit-learn/plotly not installed, skipping t-SNE visualization.")
        return None

    X = embeddings_data['embeddings']
    task_ids = embeddings_data['task_ids']
    # 按task_ids排序
    sort_idx = np.argsort(task_ids)
    X = X[sort_idx]
    task_ids = task_ids[sort_idx]

    # 按任务分组采样
    unique_task_ids = np.unique(task_ids)
    sampled_X = []
    sampled_task_ids = []
    for tid in unique_task_ids:
        mask = (task_ids == tid)
        X_tid = X[mask]
        n = len(X_tid)
        if n > max_points_per_task:
            # idx = np.random.choice(n, max_points_per_task, replace=False)
            idx = np.linspace(0, n-1, max_points_per_task, dtype=int)
            idx = np.sort(idx)
            X_tid = X_tid[idx]
        sampled_X.append(X_tid)
        sampled_task_ids.extend([tid] * len(X_tid))
    X = np.concatenate(sampled_X, axis=0)
    task_ids = np.array(sampled_task_ids)

    tsne = TSNE(n_components=2, random_state=42, init='pca')
    X_tsne = tsne.fit_transform(X)
    fig = px.scatter(
        x=X_tsne[:, 0], y=X_tsne[:, 1], color=task_ids.astype(str),
        title="t-SNE Visualization of Embeddings (per task sampling)",
        labels={"color": "Task ID", "x": "t-SNE 1", "y": "t-SNE 2"},
        opacity=0.7
    )
    tsne_path = os.path.join(save_dir, "tsne_embeddings.html")
    fig.write_html(tsne_path)
    print(f"t-SNE visualization saved to: {tsne_path}")
    return tsne_path