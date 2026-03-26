import os
import torch
import numpy as np
import argparse

from dadp.train_utils import (
    get_dataset, 
    create_data_loaders_from_dataset, 
    create_configs_from_args,
    create_model_from_configs, 
    initialize_wandb, 
    log_model_info, 
    finish_wandb,
    seed_everything,
    create_log_directory,
    save_config,
    get_observation_function_and_kwargs
)

BIAS = 2

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default="walker27")

    # Task parameters
    parser.add_argument("--dataset_name", type=str, default="RandomWalker2d/28dynamics-v0")
    parser.add_argument("--pair_dataset", action="store_true", default=True)
    parser.add_argument("--train_task_ids", type=int, nargs="+",
                        default=[
                            3,  4,  5,  6,  7,  8,  9,
                            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                            20, 21, 22, 23, 24,
                            25, 26, 27,
                        ])
    parser.add_argument("--test_task_ids", type=int, nargs="+",
                        default=[0, 1, 2])
    parser.add_argument("--use_observation", action="store_true", default=False)
    parser.add_argument("--observation_function", type=str, default="mask_dimensions", choices=["gaussian_noise", "mask_dimensions", None])
    parser.add_argument("--observation_noise_std", type=float, default=0.1)
    parser.add_argument("--observation_mask_dims", type=int, nargs="+", default=[0, 1])
    
    # Normalization parameters
    parser.add_argument("--state_mean", type=float, nargs="+", default=None, help="State mean values for normalization (space-separated list)")
    parser.add_argument("--state_std", type=float, nargs="+", default=None, help="State std values for normalization (space-separated list)")
    
    # Embedding parameters
    parser.add_argument("--history", type=int, default=16)
    parser.add_argument("--min_visible_length", type=int, default=16)
    parser.add_argument("--delta_t", type=int, default=1)
    parser.add_argument("--embedding_size", type=int, default=23)
    parser.add_argument("--mask_embedding", action="store_true", default=False, help="Enable embedding masking")
    # Default: cross_prediction is True, use --no_cross_prediction to disable
    parser.add_argument("--cross_prediction", action="store_true", default=True)
    
    # Training Loss weights
    parser.add_argument("--inverse_loss_weight", type=float, default=1.0)
    parser.add_argument("--forward_loss_weight", type=float, default=1.0)
    
    # Monitoring Loss weights and options
    parser.add_argument("--state_loss_weight", type=float, default=1.0)
    # Default: detach_embedding_for_state is True, use --no_detach_embedding_for_state to disable
    parser.add_argument("--detach_embedding_for_state", action="store_true", default=True, help="Detach embedding for state loss")
    parser.add_argument("--factor_loss_weight", type=float, default=1.0)
    parser.add_argument("--onehot_factor", action="store_true", default=False, help="Use one-hot encoding for task factors")
    parser.add_argument("--detach_embedding_for_factor", action="store_true", default = True, help="Detach embedding for factor loss")
    parser.add_argument("--policy_loss_weight", type=float, default=1.0)
    # Default: detach_embedding_for_policy is True, use --no_detach_embedding_for_policy to disable
    parser.add_argument("--detach_embedding_for_policy", action="store_true", default=True, help="Detach embedding for policy loss")
    
    parser.add_argument("--intra_traj_consistency_loss_weight", type=float, default=0.0)
    parser.add_argument("--inter_traj_consistency_loss_weight", type=float, default=0.0)
    
    # Transformer parameters
    parser.add_argument("--d_model", type=int, default=256) 
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--adaptive_pooling_heads", type=int, default=8)
    parser.add_argument("--adaptive_pooling_dropout", type=float, default=0.1) 
    parser.add_argument("--pos_encoding_max_len", type=int, default=5000)
    # Default: norm_z is True, use --no_norm_z to disable
    parser.add_argument("--norm_z", action="store_true", default=True, help="Normalize embeddings") 

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--train_split", type=float, default=0.8)

    # Other parameters
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_dir", type=str, default="./dadp/embedding/logs/exp/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_checkpoint_epochs", type=int, default=10)
    
    return parser.parse_args()

def main():
    """Main training function"""

    args = parse_args()
    
    # Set random seed for reproducibility
    seed_everything(args.seed)
    
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Create log directory
    log_dir = os.path.join(args.log_dir, args.dataset_name.replace("/", "_"))
    log_dir = create_log_directory(log_dir)

    # Get observation function and kwargs
    observation_function, observation_kwargs = get_observation_function_and_kwargs(args)
    
    # Load normalization statistics if provided
    state_mean = None
    state_std = None
    if args.state_mean is not None and args.state_std is not None:
        state_mean = np.array(args.state_mean, dtype=np.float32)
        state_std = np.array(args.state_std, dtype=np.float32)
        print("Loaded normalization statistics from CLI:")
        print(f"  Mean length: {len(state_mean)}")
        print(f"  Std length: {len(state_std)}")
    elif args.state_mean is not None or args.state_std is not None:
        print("Warning: Only one of state_mean or state_std was provided; ignoring both.")
    
    # Log observation setup
    if args.use_observation:
        if observation_function is not None:
            print(f"Using observation function: {args.observation_function}")
            print(f"Observation kwargs: {observation_kwargs}")
        else:
            print("Warning: use_observation is True but no valid observation_function specified")
            print("Will use raw state data instead")
    else:
        print("Using raw state data for training")
    
    # Get dataset and dimension information
    state_dim, action_dim, policy_dataset = get_dataset(
        dataset_name=args.dataset_name,
        horizon=args.history + args.window_size + BIAS + args.delta_t - 1 - 1,
        observation_function=observation_function,
        observation_kwargs=observation_kwargs,
        history=args.history,
        state_mean=state_mean,
        state_std=state_std
    )

    # Auto-infer train_task_ids
    all_task_ids = list(range(len(policy_dataset.task_list)))
    test_task_ids = args.test_task_ids
    train_task_ids = args.train_task_ids
    if train_task_ids is None:
        train_task_ids = [i for i in all_task_ids if i not in test_task_ids]
        print(f"Auto-selected train_task_ids: {train_task_ids}")
    else:
        print(f"Using user-specified train_task_ids: {train_task_ids}")
    print(f"Using test_task_ids: {test_task_ids}")

    # Auto-detect factor_dim from dataset
    factor_dim = 0
    if args.onehot_factor:
        factor_dim = len(policy_dataset.task_index_sets)
        print(f"Using one-hot encoding for task factors, factor_dim: {factor_dim}")
    else:
        if hasattr(policy_dataset, 'task_list') and policy_dataset.task_list is not None:
            if isinstance(policy_dataset.task_list, (list, tuple, np.ndarray)):
                if len(policy_dataset.task_list) > 0:
                    first_task = policy_dataset.task_list[0]
                    if hasattr(first_task, 'shape'):
                        factor_dim = first_task.shape[0] if len(first_task.shape) > 0 else 1
                    elif isinstance(first_task, (list, tuple)):
                        factor_dim = len(first_task)
                    else:
                        factor_dim = 1
            else:
                factor_dim = 1
            
            print(f"Auto-detected factor_dim: {factor_dim} from dataset")
            print(f"Task list shape: {np.array(policy_dataset.task_list).shape}")
        else:
            print("No task_list found in dataset, factor_dim set to 0")
    
    # Save config
    additional_info = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "factor_dim": factor_dim,
        "dataset_size": len(policy_dataset),
        "device": device,
        "use_observation": args.use_observation,
        "observation_function": args.observation_function,
        "observation_kwargs": observation_kwargs,
        "auto_calculated_padding": args.history + args.delta_t + BIAS,
    }
    
    # Add task information if available
    if args.train_task_ids is not None:
        task_ids_list = args.train_task_ids if isinstance(args.train_task_ids, list) else list(args.train_task_ids)
        additional_info["selected_task_ids"] = task_ids_list
        
        if hasattr(policy_dataset, 'task_list'):
            selected_tasks = []
            for task_id in task_ids_list:
                if task_id < len(policy_dataset.task_list):
                    task = policy_dataset.task_list[task_id]
                    if hasattr(task, 'tolist'):
                        selected_tasks.append(task.tolist())
                    else:
                        selected_tasks.append(task)
            additional_info["selected_tasks"] = selected_tasks
    
    save_config(args, log_dir, additional_info)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders_from_dataset(
        policy_dataset=policy_dataset,
        batch_size=args.batch_size,
        train_task_ids=train_task_ids,
        test_task_ids=test_task_ids
    )

    # Log task information if task_ids are specified
    if args.train_task_ids is not None:
        task_ids_list = args.train_task_ids if isinstance(args.train_task_ids, list) else list(args.train_task_ids)
        print(f"Training on specific tasks: {task_ids_list}")
        
        if hasattr(policy_dataset, 'task_list'):
            selected_tasks = [policy_dataset.task_list[i] for i in task_ids_list if i < len(policy_dataset.task_list)]
            print(f"Selected task names: {selected_tasks}")
    
    # Initialize wandb
    wandb_logger = initialize_wandb(args)
    
    # Create configurations
    embedding_config, dynamics_config, training_config = create_configs_from_args(
        args, state_dim, action_dim, device, factor_dim=factor_dim
    )
    
    # Create model
    print("Creating model with Transformer encoder")
    print(f"Transformer parameters: {embedding_config.to_encoder_kwargs()}")
    
    # Log attention mask configuration
    if training_config.min_visible_length < training_config.history:
        print(f"✓ Random attention mask enabled for training")
        print(f"  - Min visible length: {training_config.min_visible_length}")
        print(f"  - Max visible length: {training_config.history} (full history)")
        print(f"  - Evaluation will use full attention (no mask)")
    else:
        print("✗ Attention mask disabled - using full attention for both training and evaluation")
    
    model = create_model_from_configs(dynamics_config, training_config, device)
    
    # Log model information
    log_model_info(model, state_dim, action_dim, "transformer", train_loader, test_loader, wandb_logger)
    
    print(f"Starting training with {args.num_epochs} epochs...")
    model.train_embedding(
        train_loader=train_loader,
        test_loader=test_loader,
        wandb=wandb_logger,
        log_dir=log_dir,
        save_checkpoint_epochs=args.save_checkpoint_epochs,
    )
    
    # Finish wandb run
    finish_wandb(wandb_logger)

    # Log paired dataset information
    print("✓ Using PairRandomMuJoCoSeqDataset by default")
    print("  - Each batch contains paired samples from the same task")
    print("  - Intra-trajectory consistency loss will be applied within windows")
    print("  - Inter-trajectory consistency loss will be applied between paired samples")
    
if __name__ == "__main__":
    main()
