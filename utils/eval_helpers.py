"""
Helper utilities for evaluation scripts.

This module contains small helper functions extracted from
`eval_diffusion_meta_dt_style.py` to keep the main evaluation script
clean and focused on orchestration logic.
"""
import os
import pickle
import numpy as np

from src.envs import (
    PointEnv, HalfCheetahVelEnv, HalfCheetahDirEnv,
    AntDirEnv, HopperRandParamsEnv, WalkerRandParamsWrappedEnv, ReachEnv
)


def get_env_name_mapping(env_type):
    """Map environment type to actual environment name."""
    env_name_map = {
        'walker': 'WalkerRandParams-v0',
        'hopper': 'HopperRandParams-v0',
        'cheetah_vel': 'HalfCheetahVel-v0',
        'cheetah_dir': 'HalfCheetahDir-v0',
        'ant_dir': 'AntDir-v0',
        'point_robot': 'PointRobot-v0',
        'reach': 'Reach-v0',
    }
    return env_name_map.get(env_type, env_type)


def create_env(env_type, data_quality='expert'):
    """Create single environment using Meta-DT style setup.

    This mirrors the logic previously embedded in the evaluation script.
    """
    # Load task goals
    task_file = f'./datasets/{get_env_name_mapping(env_type)}/{data_quality}/task_goals.pkl'
    
    if env_type == 'point_robot':
        env = PointEnv(max_episode_steps=20, num_tasks=20)
        env.load_all_tasks(np.load(f'./datasets/{get_env_name_mapping(env_type)}/{data_quality}/task_goals.npy'))
    elif env_type == 'cheetah_vel':
        with open(task_file, 'rb') as file:
            tasks = pickle.load(file)
        env = HalfCheetahVelEnv(tasks=tasks)
    elif env_type == 'ant_dir':
        with open(task_file, 'rb') as file:
            tasks = pickle.load(file)
        env = AntDirEnv(tasks=tasks)
    elif env_type == 'walker':
        with open(task_file, 'rb') as file:
            tasks = pickle.load(file)
        env = WalkerRandParamsWrappedEnv(tasks=tasks)
    elif env_type == 'reach':
        with open(task_file, 'rb') as fp:
            tasks = pickle.load(fp)
        env = ReachEnv(tasks=tasks)
    elif env_type == 'cheetah_dir':
        with open(task_file, 'rb') as fp:
            tasks = pickle.load(fp)
        env = HalfCheetahDirEnv(tasks=tasks)
    elif env_type == 'hopper':
        with open(task_file, 'rb') as fp:
            tasks = pickle.load(fp)
        env = HopperRandParamsEnv(tasks=tasks)
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
    
    return env


def get_dataset_name(env_type):
    """Get default dataset name for environment type."""
    dataset_map = {
        'walker': 'RandomWalker2d/50dynamics-v0',
        'hopper': 'RandomHopper/50dynamics-v0',
        'cheetah_vel': 'HalfCheetahVel/20tasks-v0',
        'cheetah_dir': 'HalfCheetahDir/2tasks-v0',
        'ant_dir': 'AntDir/50tasks-v0',
        'point_robot': 'PointRobot/20tasks-v0',
        'reach': 'Reach/20tasks-v0',
    }
    return dataset_map.get(env_type, None)


def get_default_config(env_type):
    """Get default configuration for specified environment type.

    Matches the defaults used by the original evaluation and training scripts.
    """
    default_configs = {
        'point_robot': {
            'max_episode_steps': 20,
            'planner_horizon': 10,
            'history': 4,
            'planner_emb_dim': 256,
            'planner_d_model': 256,
            'planner_depth': 6,
            'planner_sampling_steps': 20,
        },
        'cheetah_vel': {
            'max_episode_steps': 200,
            'planner_horizon': 20,
            'history': 16,
            'planner_emb_dim': 256,
            'planner_d_model': 256,
            'planner_depth': 6,
            'planner_sampling_steps': 20,
        },
        'cheetah_dir': {
            'max_episode_steps': 200,
            'planner_horizon': 20,
            'history': 16,
            'planner_emb_dim': 256,
            'planner_d_model': 256,
            'planner_depth': 6,
            'planner_sampling_steps': 20,
        },
        'ant_dir': {
            'max_episode_steps': 200,
            'planner_horizon': 20,
            'history': 16,
            'planner_emb_dim': 256,
            'planner_d_model': 256,
            'planner_depth': 6,
            'planner_sampling_steps': 20,
        },
        'walker': {
            'max_episode_steps': 200,
            'planner_horizon': 20,
            'history': 16,
            'planner_emb_dim': 256,
            'planner_d_model': 256,
            'planner_depth': 6,
            'planner_sampling_steps': 20,
        },
        'hopper': {
            'max_episode_steps': 200,
            'planner_horizon': 20,
            'history': 16,
            'planner_emb_dim': 256,
            'planner_d_model': 256,
            'planner_depth': 6,
            'planner_sampling_steps': 20,
        },
        'reach': {
            'max_episode_steps': 500,
            'planner_horizon': 20,
            'history': 16,
            'planner_emb_dim': 256,
            'planner_d_model': 256,
            'planner_depth': 6,
            'planner_sampling_steps': 20,
        },
    }
    
    if env_type not in default_configs:
        raise ValueError(f"Unknown env_type: {env_type}. Choose from {list(default_configs.keys())}")
    
    return default_configs[env_type]
