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

BIAS = 2

def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")

def get_observation_function_and_kwargs(args):
    """Get observation function based on arguments"""
    if not args.observation_function or args.observation_function == "None":
        return None, None
    
    from cleandiffuser.dataset.d4rl_mujoco_dataset import add_gaussian_noise, mask_dimensions
    
    if args.observation_function == "gaussian_noise":
        return add_gaussian_noise, {"noise_std": args.observation_noise_std}
    elif args.observation_function == "mask_dimensions":
        return mask_dimensions, {"mask_dims": args.observation_mask_dims}
    
    return None, None

def get_dataset(dataset_name, horizon, stride=1, 
                observation_function=None, observation_kwargs=None, 
                history=16,
                pair_dataset=True,
                state_mean=None,
                state_std=None
                ):
    """Load dataset with specified parameters - directly creates PairRandomMuJoCoSeqDataset
    
    Args:
        state_mean: Optional pre-computed mean for normalization (np.ndarray)
        state_std: Optional pre-computed std for normalization (np.ndarray)
        If both are None, will calculate from data.
    """
    print(f"Loading dataset: {dataset_name}")
    
    dataset = minari.load_dataset(dataset_name)
    
    # Create paired dataset
    from cleandiffuser.dataset.d4rl_mujoco_dataset import PairRandomMuJoCoSeqDataset
    
    if pair_dataset:
        policy_dataset = PairRandomMuJoCoSeqDataset(
            dataset, 
            horizon=horizon,
            stride=stride,
            padding=history,
            observation_function=observation_function,
            observation_kwargs=observation_kwargs,
            max_path_length=1000+history,
            state_mean=state_mean,
            state_std=state_std
        )
    else:
        policy_dataset = RandomMuJoCoSeqDataset(
            dataset, 
            horizon=horizon,
            stride=stride,
            padding=history,
            observation_function=observation_function,
            observation_kwargs=observation_kwargs,
            max_path_length=1000+history,
            state_mean=state_mean,
            state_std=state_std
        )
    
    # Get dimensions from sample
    sample = policy_dataset[0]
    state_dim = sample['obs']['state'].shape[-1]
    action_dim = sample['act'].shape[-1]
    
    # Check observation data
    has_observation = 'observation' in sample['obs']
    if has_observation:
        obs_dim = sample['obs']['observation'].shape[-1]
        print(f"Dataset includes observation data - obs_dim: {obs_dim}")
    
    print(f"Detected state_dim: {state_dim}, action_dim: {action_dim}")
    return state_dim, action_dim, policy_dataset

def create_data_loaders_from_dataset(
    policy_dataset, 
    batch_size, 
    train_task_ids=None, 
    test_task_ids=None):
    """Create train/test data loaders from train_task_ids and test_task_ids."""
    print(f"Policy dataset length: {len(policy_dataset)}")

    if train_task_ids is None or test_task_ids is None:
        raise ValueError("You must specify both train_task_ids and test_task_ids for dataloader creation.")

    print(f"Filtering train dataset to tasks: {train_task_ids}")
    train_subset = get_task_data(policy_dataset, np.array(train_task_ids))
    print(f"Train subset length: {len(train_subset)}")
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"Filtering test dataset to tasks: {test_task_ids}")
    test_subset = get_task_data(policy_dataset, np.array(test_task_ids))
    print(f"Test subset length: {len(test_subset)}")
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader

def create_configs_from_args(args, state_dim, action_dim, device, factor_dim=0):
    """Create model configurations from arguments"""
    # Embedding config
    embedding_config = EmbeddingConfig(
        embedding_size=args.embedding_size,
        norm_z=getattr(args, 'norm_z', True),
        mask_embedding=args.mask_embedding,
        d_model=getattr(args, 'd_model', 256),
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pos_encoding_max_len=getattr(args, 'pos_encoding_max_len', 5000),
        adaptive_pooling_heads=getattr(args, 'adaptive_pooling_heads', 8),
        adaptive_pooling_dropout=getattr(args, 'adaptive_pooling_dropout', 0.1),
    )
    
    # Dynamics config
    dynamics_config = DynamicsConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        factor_dim=factor_dim,
        head_hidden=args.head_hidden,
        embedding_config=embedding_config,
    )
    
    # Training config
    training_config = TrainingConfig(
        inverse_loss_weight=args.inverse_loss_weight,
        forward_loss_weight=args.forward_loss_weight,
        intra_traj_consistency_loss_weight=getattr(args, 'intra_traj_consistency_loss_weight', 0.1),
        inter_traj_consistency_loss_weight=getattr(args, 'inter_traj_consistency_loss_weight', 0.1),
        state_loss_weight=args.state_loss_weight,
        factor_loss_weight=getattr(args, 'factor_loss_weight', 0.0),
        policy_loss_weight=getattr(args, 'policy_loss_weight', 1.0),
        cross_prediction=getattr(args, 'cross_prediction', False),
        history=args.history,
        window_size=args.window_size,
        delta_t=args.delta_t,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        device=device,
        use_observation=getattr(args, 'use_observation', False),
        detach_embedding_for_factor=getattr(args, 'detach_embedding_for_factor', False),
        detach_embedding_for_state=getattr(args, 'detach_embedding_for_state', False),
        detach_embedding_for_policy=getattr(args, 'detach_embedding_for_policy', False),
        min_visible_length=getattr(args, 'min_visible_length', 2),
    )
    
    return embedding_config, dynamics_config, training_config

def create_model_from_configs(dynamics_config, training_config, device):
    """Create DADP model from configurations"""
    return DADP(dynamics_config, training_config).to(device)

def initialize_wandb(args):
    """Initialize wandb logging"""
    try:
        import wandb
        wandb_logger = wandb.init(project=args.wandb_project, config=vars(args))
        print(f"wandb initialized: {wandb_logger.name}")
        return wandb_logger
    except (ImportError, Exception) as e:
        print(f"Warning: wandb not available: {e}")
        return None

def log_model_info(model, state_dim, action_dim, encoder_type, train_loader, test_loader, wandb_logger):
    """Log model information"""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    print(f"Input dimensions - State: {state_dim}, Action: {action_dim}")
    print(f"Using {encoder_type} encoder")

def finish_wandb(wandb_logger):
    """Finish wandb logging"""
    if wandb_logger:
        wandb_logger.finish()

def create_log_directory(base_log_dir: str = "./dadp/embedding/logs") -> str:
    """Create timestamped log directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_log_dir, "transformer", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    
    print(f"Created log directory: {log_dir}")
    return log_dir

def save_config(args, log_dir: str, additional_info: dict = None):
    """Save training configuration"""
    config = vars(args).copy()
    if additional_info:
        config.update(additional_info)
    
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Config saved to: {config_path}")
