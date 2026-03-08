import torch
from torch.utils.data import DataLoader, random_split
import minari
from cleandiffuser.dataset.d4rl_mujoco_dataset import RandomMuJoCoSeqDataset, get_task_data
from .dadp import EmbeddingConfig, DynamicsConfig, TrainingConfig, DADP
import random
import numpy as np
import os
import json
from datetime import datetime
import glob

BIAS = 2

### General training utility functions
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to: {seed}")

def get_dataset(dataset_name, horizon, stride=1, 
                observation_function=None, observation_kwargs=None, 
                history=16, delta_t=32):
    """
    简化的数据集获取函数，使用数据集默认值
    
    Args:
        dataset_name: 数据集名称
        horizon: 序列长度
        stride: 步长，默认1
        observation_function: 观察函数
        observation_kwargs: 观察函数参数
        history: 历史长度
        delta_t: 时间间隔
    """
    
    print(f"Loading dataset: {dataset_name}")
    
    # 根据 history 和 delta_t 自动计算需要的 padding
    required_padding = history + delta_t + BIAS
    
    print(f"Auto-calculated padding: {required_padding} (based on history={history}, delta_t={delta_t}, BIAS={BIAS})")
    
    # Load dataset - 使用简化的参数集
    dataset = minari.load_dataset(dataset_name)
    policy_dataset = RandomMuJoCoSeqDataset(
        dataset, 
        horizon=horizon, # 保留必要的horizon参数
        stride=stride,
        padding=required_padding,
        observation_function=observation_function,
        observation_kwargs=observation_kwargs,
        # 移除的参数会使用RandomMuJoCoSeqDataset的默认值：
        # discount=0.99, center_mapping=True, terminal_penalty=0, 
        # max_path_length=1000, full_traj_bonus=0
    )
    
    # Get sample to determine dimensions
    sample = policy_dataset[0]
    state_dim = sample['obs']['state'].shape[-1]
    action_dim = sample['act'].shape[-1]
    
    # 检查是否有observation数据
    has_observation = 'observation' in sample['obs']
    if has_observation:
        obs_dim = sample['obs']['observation'].shape[-1]
        print(f"Dataset includes observation data - obs_dim: {obs_dim}")
        if observation_function:
            print(f"Observation function '{observation_function.__name__}' applied")
    
    print(f"Detected state_dim: {state_dim}, action_dim: {action_dim}")
    return state_dim, action_dim, policy_dataset

def create_data_loaders_from_dataset(policy_dataset, batch_size, train_split, task_ids=None):

    print(f"Policy dataset length: {len(policy_dataset)}")
    
    # If task_ids are specified, filter the dataset to only include those tasks
    if task_ids is not None:
        # Ensure task_ids is a list or array-like
        if not isinstance(task_ids, (list, tuple, np.ndarray)):
            task_ids = [task_ids]  # Convert single value to list
        
        print(f"Filtering dataset to tasks: {task_ids}")
        policy_dataset = get_task_data(policy_dataset, np.array(task_ids))
        print(f"Filtered dataset length: {len(policy_dataset)}")
    
    # Split dataset
    train_size = int(len(policy_dataset) * train_split) 
    test_size = len(policy_dataset) - train_size
    train_dataset, test_dataset = random_split(policy_dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader

def create_configs_from_args(args, state_dim, action_dim, device, factor_dim=0):
    # Create embedding configuration - 直接使用 Transformer 参数
    embedding_config = EmbeddingConfig(
        embedding_size=args.embedding_size,
        norm_z=True,
        mask_embedding=args.mask_embedding,
        
        # Transformer parameters
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_ff=args.d_ff,
        dropout=args.dropout,
        rope_theta=getattr(args, 'rope_theta', 10000.0),
    )
    
    # Create dynamics configuration
    dynamics_config = DynamicsConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        factor_dim=factor_dim,  # 使用传入的factor_dim
        head_hidden=args.head_hidden,
        embedding_config=embedding_config,
    )
    
    # Create training configuration
    training_config = TrainingConfig(
        inverse_loss_weight=args.inverse_loss_weight,
        forward_loss_weight=args.forward_loss_weight,
        consistency_loss_weight=args.consistency_loss_weight,
        state_loss_weight=args.state_loss_weight,
        factor_loss_weight=getattr(args, 'factor_loss_weight', 0.0),
        consistency_window_size=args.consistency_window_size,
        history=args.history,
        window_size=args.window_size,
        delta_t=args.delta_t,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        device=device,
        use_observation=getattr(args, 'use_observation', False),
        detach_embedding_for_factor=getattr(args, 'detach_embedding_for_factor', False),  # 新增参数
    )
    
    return embedding_config, dynamics_config, training_config

def create_model_from_configs(dynamics_config, training_config, device):

    model = DADP(dynamics_config, training_config).to(device)
    return model

def initialize_wandb(args):
    try:
        import wandb
        wandb_logger = wandb.init(
            project=args.wandb_project,
            config=vars(args)  # Log all arguments
        )
        print(f"wandb initialized: {wandb_logger.name}")
        return wandb_logger
    except ImportError:
        print("Warning: wandb not available. Continuing without wandb logging...")
        return None
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        print("Continuing without wandb logging...")
        return None

def log_model_info(model, state_dim, action_dim, encoder_type, train_loader, test_loader, wandb_logger):
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model created with {total_params} parameters")
    print(f"Model input dimensions - State: {state_dim}, Action: {action_dim}")
    print(f"Using Transformer encoder")
    
    # Log to wandb if available
    if (wandb_logger):
        wandb_logger.log({
            "model/total_parameters": total_params,
            "model/state_dim": state_dim,
            "model/action_dim": action_dim,
            "model/encoder_type": "transformer",  # 固定为 transformer
            "dataset/train_size": len(train_loader.dataset),
            "dataset/test_size": len(test_loader.dataset),
            "dataset/total_size": len(train_loader.dataset) + len(test_loader.dataset)
        })

def finish_wandb(wandb_logger):
    if wandb_logger:
        wandb_logger.finish()

def create_log_directory(base_log_dir: str = "./dadp/embedding/logs") -> str:
    """简化版本，不需要 encoder_type 参数"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_log_dir, "transformer", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    print(f"Created log directory: {log_dir}")
    return log_dir

def save_config(args, log_dir: str, additional_info: dict = None):
    config = vars(args).copy()
    
    if additional_info:
        config.update(additional_info)
    
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Saved config to: {config_path}")



### Embedding extraction and saving/loading utilities
def extract_embeddings_from_dataset(model, policy_dataset, batch_size=128, device="cuda"):
    from tqdm import tqdm
    
    model.eval()
    model.to(device)
    print("Extracting embeddings from entire dataset...")
    
    # Get task information if available
    task_list = getattr(policy_dataset, 'task_list', None)
    if task_list is not None:
        print(f"Found {len(task_list)} tasks")
    
    # Create optimized dataloader
    full_loader = torch.utils.data.DataLoader(
        policy_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=False
    )
    
    # Collect embeddings and task IDs
    all_embeddings = []
    all_task_ids = []
    history = model.training_config.history
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(full_loader, desc="Processing batches")):
            # Move data to device
            states = batch["obs"]["state"].to(device, non_blocking=True)  # (B, L, S)
            actions = batch["act"].to(device, non_blocking=True)          # (B, L, A)
            task_ids = batch.get("task_id", None)      # (B, 1) or None
            
            B, L, S = states.shape
            
            # Skip if sequence too short
            if L < history:
                continue
            
            # Extract history directly from the sequence
            if model.training_config.delta_t:
                if L < history + 1:
                    continue
                state_history = states[:, :history, :]
                action_history = actions[:, :history, :]
            else:
                state_history = states[:, :history, :]
                action_history = actions[:, :history, :]
            
            # Extract embeddings
            embeddings = model.dynamics.encode_history(state_history, action_history)  # (B, embedding_size)
            
            # Store embeddings
            all_embeddings.append(embeddings.cpu().numpy())
            
            # Store task IDs
            if task_ids is not None:
                all_task_ids.append(task_ids.squeeze().cpu().numpy())
            else:
                all_task_ids.append(np.zeros(B, dtype=int))
    
    # Convert to numpy arrays
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_task_ids = np.concatenate(all_task_ids, axis=0)
    
    print(f"Extracted {len(all_embeddings)} embeddings, shape: {all_embeddings.shape}")
    
    # Organize episode boundaries
    episode_boundaries = get_episode_boundaries(all_task_ids, policy_dataset)
    
    embeddings_data = {
        'embeddings': all_embeddings,
        'task_ids': all_task_ids,
        'episode_boundaries': episode_boundaries,
        'task_list': task_list
    }
    
    return embeddings_data

def get_episode_boundaries(task_ids, policy_dataset):
    """Get episode boundaries based on task_id changes or dataset structure"""
    episode_boundaries = []
    
    if len(task_ids) == 0:
        return episode_boundaries
    
    # Method 1: Use dataset structure if available
    if hasattr(policy_dataset, 'indices'):
        # Group indices by path_idx (episode)
        path_indices = {}
        for i, (path_idx, start, end) in enumerate(policy_dataset.indices):
            if path_idx not in path_indices:
                path_indices[path_idx] = []
            path_indices[path_idx].append(i)
        
        current_data_idx = 0
        window_size = getattr(policy_dataset.training_config if hasattr(policy_dataset, 'training_config') else None, 'window_size', 1)
        
        for path_idx in sorted(path_indices.keys()):
            indices_in_episode = path_indices[path_idx]
            if indices_in_episode:
                start_idx = current_data_idx
                episode_length = len(indices_in_episode) * window_size
                end_idx = start_idx + episode_length
                
                if start_idx < len(task_ids):
                    episode_task_id = task_ids[start_idx]
                    episode_boundaries.append((start_idx, end_idx, int(episode_task_id)))
                    current_data_idx = end_idx
    else:
        # Method 2: Fallback - detect task_id changes
        current_task = task_ids[0]
        episode_start = 0
        
        for i in range(1, len(task_ids)):
            if task_ids[i] != current_task:
                episode_boundaries.append((episode_start, i, int(current_task)))
                episode_start = i
                current_task = task_ids[i]
        
        # Add final episode
        episode_boundaries.append((episode_start, len(task_ids), int(current_task)))
    
    return episode_boundaries

def save_embeddings_data(embeddings_data, save_path):
    """Save embeddings data to compressed file"""
    print(f"Saving embeddings data to {save_path}")
    np.savez_compressed(save_path, **embeddings_data)

def load_embeddings_data(save_path):
    """Load embeddings data from file"""
    print(f"Loading embeddings data from {save_path}")
    data = np.load(save_path, allow_pickle=True)
    
    embeddings_data = {
        'embeddings': data['embeddings'],
        'task_ids': data['task_ids'],
        'episode_boundaries': data['episode_boundaries'].tolist(),
        'task_list': data['task_list'] if 'task_list' in data else None
    }
    
    print(f"Loaded {len(embeddings_data['embeddings'])} embeddings from {len(embeddings_data['episode_boundaries'])} episodes")
    return embeddings_data

def extract_and_save_embeddings(model, policy_dataset, checkpoint_path, batch_size=128, device="cuda"):
    """Complete pipeline for extracting and saving embeddings"""
    with torch.no_grad():
        model.eval()
        
        # Extract embeddings
        embeddings_data = extract_embeddings_from_dataset(
            model, policy_dataset, batch_size=batch_size, device=device
        )
        
        # Save the embeddings data
        save_dir = os.path.dirname(checkpoint_path)
        save_path = os.path.join(save_dir, "embeddings_data.npz")
        save_embeddings_data(embeddings_data, save_path)
        
        # Print summary statistics
        embeddings = embeddings_data['embeddings']
        episode_boundaries = embeddings_data['episode_boundaries']
        
        print(f"\nSummary: {len(embeddings)} embeddings, {len(episode_boundaries)} episodes")
        
        # Task distribution
        task_counts = {}
        for _, _, task_id in episode_boundaries:
            task_counts[task_id] = task_counts.get(task_id, 0) + 1
        
        print(f"Episodes per task: {dict(sorted(task_counts.items()))}")
        
        # Episode length statistics
        episode_lengths = [end - start for start, end, _ in episode_boundaries]
        if episode_lengths:
            print(f"Episode lengths - Min: {min(episode_lengths)}, Max: {max(episode_lengths)}, Mean: {np.mean(episode_lengths):.1f}")
        
        return save_path

def load_dataset_from_checkpoint_metadata(checkpoint_path):
    """Load dataset using parameters stored in checkpoint metadata - 简化版本"""
    # Load checkpoint to get metadata
    model, metadata = DADP.load_checkpoint(checkpoint_path, "cpu")
    
    # Extract dataset parameters from metadata - 只保留必要参数
    dataset_params = {}
    for key in ['dataset_name']:  # 只保留数据集名称
        if key in metadata:
            dataset_params[key] = metadata[key]
    
    # Get training config parameters
    training_config = model.training_config
    dataset_params['horizon'] = training_config.history + 2  # BIAS=2
    
    # 从训练配置中获取 history 和 delta_t
    history = training_config.history
    delta_t = training_config.delta_t
    
    # Load dataset with simplified parameters
    state_dim, action_dim, policy_dataset = get_dataset(
        dataset_name=dataset_params.get('dataset_name', 'RandomWalker2d/40dynamics-v2'),
        horizon=dataset_params['horizon'],
        stride=1,
        history=history,
        delta_t=delta_t,
        # 其他参数使用数据集默认值
    )
    
    return model, metadata, state_dim, action_dim, policy_dataset, dataset_params

def precompute_embeddings_for_policy_training(
    embedding_model,
    policy_dataset,
    batch_size: int = 256,
    device: str = "cpu"
):
    """
    Precompute embeddings for all samples in the policy dataset
    
    Args:
        embedding_model: Pretrained DADP model
        policy_dataset: Original policy dataset
        batch_size: Batch size for embedding computation
        device: Device to use for computation
        
    Returns:
        list: List of embeddings only (much more memory efficient)
    """
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    print("Precomputing embeddings for all samples...")
    
    # Create temporary dataloader for embedding computation
    temp_loader = DataLoader(
        policy_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Important: don't shuffle to maintain order
        num_workers=0
    )
    
    # Set embedding model to eval mode
    embedding_model.eval()
    
    # Store only precomputed embeddings
    precomputed_embeddings = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(temp_loader, desc="Precomputing embeddings")):
            states = batch["obs"]["state"].to(device)
            actions = batch["act"].to(device)
            
            # Data processing - same as in PolicyTrainer
            state_history = states[:, :-2, :]      # (B_sub, history, S)
            action_history = actions[:, :-2, :]    # (B_sub, history, A)
            
            # Compute embeddings
            history_embedding = embedding_model.dynamics.encode_history(
                state_history, action_history
            )  # (B_sub, embedding_size)
            
            # Store only embeddings (much more memory efficient)
            precomputed_embeddings.extend(history_embedding.cpu())
    
    print(f"Precomputed embeddings for {len(precomputed_embeddings)} samples")
    return precomputed_embeddings


class PrecomputedEmbeddingDataset(torch.utils.data.Dataset):
    """Dataset that wraps original dataset and adds precomputed embeddings"""
    
    def __init__(self, original_dataset, precomputed_embeddings):
        self.original_dataset = original_dataset
        self.precomputed_embeddings = precomputed_embeddings
        assert len(original_dataset) == len(precomputed_embeddings), \
            f"Dataset and embeddings length mismatch: {len(original_dataset)} vs {len(precomputed_embeddings)}"
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        # Get original sample
        original_sample = self.original_dataset[idx]
        
        # Add precomputed embedding
        sample = {
            'original_data': original_sample,
            'precomputed_embedding': self.precomputed_embeddings[idx],
            'has_precomputed_embedding': True
        }
        return sample


def create_precomputed_embedding_loaders(
    embedding_model,
    policy_dataset,
    batch_size: int = 128,
    train_split: float = 0.8,
    device: str = "cpu",
    precompute_batch_size: int = 256
):
    """
    Create data loaders with precomputed embeddings
    
    Args:
        embedding_model: Pretrained DADP model
        policy_dataset: Original policy dataset
        batch_size: Batch size for training/testing
        train_split: Train/test split ratio
        device: Device for computation
        precompute_batch_size: Batch size for precomputation
        
    Returns:
        tuple: (train_loader, test_loader) with precomputed embeddings
    """
    from torch.utils.data import DataLoader, random_split
    
    # Precompute embeddings for all samples
    precomputed_embeddings = precompute_embeddings_for_policy_training(
        embedding_model=embedding_model,
        policy_dataset=policy_dataset,
        batch_size=precompute_batch_size,
        device=device
    )
    
    # Create dataset with precomputed embeddings
    precomputed_dataset = PrecomputedEmbeddingDataset(policy_dataset, precomputed_embeddings)
    
    # Split into train and test
    total_size = len(precomputed_dataset)
    train_size = int(train_split * total_size)
    test_size = total_size - train_size
    
    train_dataset, test_dataset = random_split(
        precomputed_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    print(f"Created precomputed embedding loaders:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader