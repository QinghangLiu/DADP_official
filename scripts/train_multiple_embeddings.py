#!/usr/bin/env python3
"""
Script to train embeddings on multiple environments.
Provides more flexibility than the bash script.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Environment configurations
ENV_CONFIGS = {
    "ant": {
        "dataset_name": "RandomAnt/82dynamics-v7",
        "embedding_size": 35,
        "train_task_ids": [1, 2, 3, 4, 24, 25, 26, 27, 35, 36, 37, 38, 49, 50, 51, 52, 
                          59, 60, 61, 62, 63, 77, 78, 79, 80],
        "test_task_ids": [17, 25, 26, 73, 74, 75, 76],
        "log_dir": "./dadp/embedding/logs/transformer/exp_ant_28_reproduce"
    },
    "door": {
        "dataset_name": "Adroit/door_shrink_combined-v0",
        "embedding_size": 67,
        "train_task_ids": [0, 1],
        "test_task_ids": [2],
        "log_dir": "./dadp/embedding/logs/transformer/exp_door_3"
    },
    "halfcheetah": {
        "dataset_name": "RandomHalfCheetah/82dynamics-v7",
        "embedding_size": 23,
        "train_task_ids": list(range(3, 28)),
        "test_task_ids": [0, 1, 2],
        "log_dir": "./dadp/embedding/logs/transformer/exp_halfcheetah_28"
    },
    "hopper": {
        "dataset_name": "RandomHopper/82dynamics-v7",
        "embedding_size": 14,
        "train_task_ids": list(range(3, 28)),
        "test_task_ids": [0, 1, 2],
        "log_dir": "./dadp/embedding/logs/transformer/exp_hopper_28"
    },
    "relocate": {
        "dataset_name": "Adroit/relocate_shrink_combined-v0",
        "embedding_size": 69,
        "train_task_ids": [0, 1],
        "test_task_ids": [2],
        "log_dir": "./dadp/embedding/logs/transformer/exp_relocate_3"
    },
    "walker": {
        "dataset_name": "RandomWalker2d/28dynamics-v9",
        "embedding_size": 23,
        "train_task_ids": list(range(3, 28)),
        "test_task_ids": [0, 1, 2],
        "log_dir": "./dadp/embedding/logs/transformer/exp_walker_28"
    }
}

# Common arguments for all environments
COMMON_ARGS = {
    "wandb_project": "walker27",
    "use_observation": False,
    "observation_function": "mask_dimensions",
    "observation_noise_std": 0.1,
    "observation_mask_dims": [0, 1],
    "history": 16,
    "min_visible_length": 16,
    "delta_t": 1,
    "mask_embedding": False,
    "cross_prediction": True,
    "inverse_loss_weight": 1.0,
    "forward_loss_weight": 1.0,
    "state_loss_weight": 1.0,
    "detach_embedding_for_state": True,
    "factor_loss_weight": 1.0,
    "onehot_factor": False,
    "detach_embedding_for_factor": True,
    "policy_loss_weight": 1.0,
    "detach_embedding_for_policy": True,
    "intra_traj_consistency_loss_weight": 0.0,
    "inter_traj_consistency_loss_weight": 0.0,
    "d_model": 256,
    "n_layer": 4,
    "head_hidden": 256,
    "n_head": 8,
    "d_ff": 1024,
    "dropout": 0.1,
    "adaptive_pooling_heads": 8,
    "adaptive_pooling_dropout": 0.1,
    "pos_encoding_max_len": 5000,
    "norm_z": True,
    "learning_rate": 0.0003,
    "num_epochs": 10,
    "batch_size": 128,
    "window_size": 2,
    "eval_interval": 1,
    "train_split": 0.8,
    "device": "cuda:0",
    "seed": 42,
    "save_checkpoint_epochs": 10
}


def build_command(env_name, config, cuda_device=None):
    """Build the training command for a given environment."""
    cmd = ["python", "train_embedding.py"]
    
    # Add environment-specific arguments
    cmd.extend(["--dataset_name", config["dataset_name"]])
    cmd.extend(["--embedding_size", str(config["embedding_size"])])
    cmd.extend(["--train_task_ids"] + [str(x) for x in config["train_task_ids"]])
    cmd.extend(["--test_task_ids"] + [str(x) for x in config["test_task_ids"]])
    cmd.extend(["--log_dir", config["log_dir"]])
    
    # Add common arguments
    for key, value in COMMON_ARGS.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif isinstance(value, list):
            cmd.extend([f"--{key}"] + [str(x) for x in value])
        else:
            cmd.extend([f"--{key}", str(value)])
    
    return cmd


def train_environment(env_name, cuda_device=None):
    """Train embedding for a single environment."""
    if env_name not in ENV_CONFIGS:
        print(f"Error: Unknown environment '{env_name}'")
        print(f"Available environments: {', '.join(ENV_CONFIGS.keys())}")
        return False
    
    config = ENV_CONFIGS[env_name]
    
    print("=" * 80)
    print(f"Training {env_name.upper()}")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Embedding size: {config['embedding_size']}")
    print(f"Log directory: {config['log_dir']}")
    print("=" * 80)
    
    # Set CUDA device
    env = os.environ.copy()
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    
    # Build and execute command
    cmd = build_command(env_name, config, cuda_device)
    
    try:
        result = subprocess.run(cmd, env=env, check=True)
        print(f"\nFinished training {env_name.upper()}")
        print()
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError training {env_name.upper()}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train embeddings on multiple environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available environments:
  - ant: RandomAnt/82dynamics-v7
  - door: Adroit/door_shrink_combined-v0
  - halfcheetah: RandomHalfCheetah/82dynamics-v7
  - hopper: RandomHopper/82dynamics-v7
  - relocate: Adroit/relocate_shrink_combined-v0
  - walker: RandomWalker2d/28dynamics-v9

Examples:
  # Train all environments
  python train_multiple_embeddings.py --all
  
  # Train specific environments
  python train_multiple_embeddings.py --envs ant walker hopper
  
  # Train on a different GPU
  python train_multiple_embeddings.py --all --cuda_device 0
        """
    )
    
    parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
        choices=list(ENV_CONFIGS.keys()),
        help="Specific environments to train"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all environments"
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=1,
        help="CUDA device to use (default: 1)"
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue training other envs if one fails"
    )
    
    args = parser.parse_args()
    
    # Determine which environments to train
    if args.all:
        envs_to_train = list(ENV_CONFIGS.keys())
    elif args.envs:
        envs_to_train = args.envs
    else:
        parser.print_help()
        print("\nError: Must specify either --all or --envs")
        sys.exit(1)
    
    print(f"Will train the following environments: {', '.join(envs_to_train)}")
    print(f"Using CUDA device: {args.cuda_device}")
    print()
    
    # Train each environment
    results = {}
    for env_name in envs_to_train:
        success = train_environment(env_name, args.cuda_device)
        results[env_name] = success
        
        if not success and not args.continue_on_error:
            print(f"Stopping due to error in {env_name}")
            break
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    for env_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{env_name:15s} {status}")
    print("=" * 80)
    
    # Return exit code
    all_success = all(results.values())
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
