from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.diffusion import (
    ContinuousDiffusionSDE,
    EnvFactorGuidedDiffusionSDE,
    EmbeddingGuidedDiffusionSDE,
    MixedDDIM,
)
import torch, torch.nn as nn
import numpy as np

def create_planner(obs_dim: int,
                   act_dim: int,
                   planner_dataset,
                   # Task configuration
                   planner_horizon: int,
                   history: int,
                   pipeline_type: str = "separate",  # "joint" or "separate"
                   # Network architecture parameters
                   planner_emb_dim: int = 256,
                   planner_d_model: int = 256,
                   planner_depth: int = 8,
                   # Diffusion parameters
                   planner_ema_rate: float = 0.995,
                   planner_predict_noise: bool = True,
                   planner_next_obs_loss_weight: float = 1.0,
                   planner_guide_noise_scale: float = 1.0,
                   attention_mask: bool = True,
                   device: str = "cuda",
                   planner_noise_type: str = 'env_factor_guided',
                   predict_embedding: bool = False,
                   nnCondition: bool = True,
                   dadp_model = None) -> ContinuousDiffusionSDE:
    """
    Create a planner using ContinuousDiffusionSDE.

    Args:
        obs_dim: Observation dimension.
        act_dim: Action dimension.
        planner_dataset: Dataset for planner training.
        planner_horizon: Planning horizon.
        history: History length.
        pipeline_type: Type of pipeline ("joint" or "separate").
        planner_emb_dim: Embedding dimension.
        planner_d_model: Model dimension.
        planner_depth: Number of layers.
        planner_ema_rate: EMA rate for training.
        planner_predict_noise: Whether to predict noise.
        planner_next_obs_loss_weight: Loss weight for next observation.
        planner_guide_noise_scale: Guide noise scale.
        attention_mask: Whether to use an attention mask.
        device: Device to use.
        planner_noise_type: Planner noise variant (e.g., 'env_factor_guided',
            'mixed_ddim', 'one_step_mixed_ddim').
        predict_embedding: Whether the planner should predict embeddings (used by mixed_ddim).
        nnCondition: Whether to use the conditioning network.
        dadp_model: Optional DADP model handle.
    """

    planner_dim = obs_dim + act_dim if pipeline_type != "separate" else obs_dim

    # Setup attention mask for planner
    planner_attn_mask = None
    if attention_mask:
        assert history < planner_horizon, "History length must be less than planner horizon for attention mask."
        planner_attn_mask = torch.triu(torch.ones(planner_horizon, planner_horizon), diagonal=1).bool().to(device)
        planner_attn_mask[:history, :] = True
        planner_attn_mask.fill_diagonal_(False)

    nn_diffusion_planner = DiT1d(
        planner_dim,
        emb_dim=planner_emb_dim,
        d_model=planner_d_model,
        n_heads=planner_d_model // 32,
        depth=planner_depth,
        timestep_emb_type="fourier",
        dropout=0.1,
        attn_mask=planner_attn_mask
    )
    if nnCondition:
        nn_condition_planner = MLPCondition(
            in_dim=planner_dim,
            out_dim=planner_emb_dim,
            hidden_dims=[planner_emb_dim],
            act=nn.SiLU(),
            dropout=0
        )
    else:
        nn_condition_planner = None

    # Setup fix mask based on pipeline type
    if pipeline_type == "joint":
        fix_mask = torch.zeros((planner_horizon, planner_dim))
        fix_mask[:history, :] = 1.
        fix_mask[history, :obs_dim] = 1.
    elif pipeline_type == "separate":
        fix_mask = torch.zeros((planner_horizon, planner_dim))
        fix_mask[:history + 1, :] = 1.
        fix_mask[history + 1:, obs_dim:] = 1.  # only predict future states
    elif pipeline_type == "no_prior":
        fix_mask = torch.zeros((planner_horizon, planner_dim))
        fix_mask[0, :obs_dim] = 1.
    else:
        raise ValueError(f"Unknown pipeline_type: {pipeline_type}")

    # Setup loss weights
    loss_weight = torch.ones((planner_horizon, planner_dim))
    loss_weight[1] = planner_next_obs_loss_weight
    
    # Initialize planner
    print('='*50)
    print(f"Remember, you are using policy type: {planner_noise_type}")
    print('='*50)
    if planner_noise_type == 'env_factor_guided':
        planner = EnvFactorGuidedDiffusionSDE(
            nn_diffusion_planner, 
            nn_condition=nn_condition_planner,
            fix_mask=fix_mask, 
            loss_weight=loss_weight, 
            ema_rate=planner_ema_rate,
            device=device, 
            predict_noise=planner_predict_noise, 
            noise_schedule="linear",
            task=planner_dataset.task_list,
            guide_noise_scale=planner_guide_noise_scale
        )
    elif planner_noise_type == 'standard':
        planner = ContinuousDiffusionSDE(
            nn_diffusion_planner, 
            nn_condition=nn_condition_planner,
            fix_mask=fix_mask, 
            loss_weight=loss_weight, 
            ema_rate=planner_ema_rate,
            device=device, 
            predict_noise=planner_predict_noise, 
            noise_schedule="linear",
        )
    elif planner_noise_type == 'embedding_guided':
        # planner = MixedDDIM(
        planner = EmbeddingGuidedDiffusionSDE(
            
            nn_diffusion_planner, 
            nn_condition=nn_condition_planner,
            fix_mask=fix_mask, 
            loss_weight=loss_weight, 
            ema_rate=planner_ema_rate,
            device=device, 
            predict_noise=planner_predict_noise, 
            noise_schedule="linear",
            guide_noise_scale=planner_guide_noise_scale
        )
    elif planner_noise_type == 'mixed_ddim':
        planner = MixedDDIM(
            nn_diffusion_planner,
            nn_condition=nn_condition_planner,
            fix_mask=fix_mask,
            loss_weight=loss_weight,
            ema_rate=planner_ema_rate,
            device=device,
            predict_noise=planner_predict_noise,
            noise_schedule="linear",
            guide_noise_scale=planner_guide_noise_scale,
            predict_embedding=predict_embedding,
            # task=planner_dataset.task_list,
        )

    else:
        raise ValueError(f"Unknown planner_type: {planner_noise_type}")
    
    return planner
