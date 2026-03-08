import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .embedding import create_seq_to_embedding


class DynamicsHead(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        factor_dim: int = 0,
        embedding_size: int = 256,
        head_hidden: int = 512,
        encoder_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.factor_dim = factor_dim
        self.embedding_size = embedding_size
        
        # 创建 Transformer 序列编码器
        encoder_kwargs = encoder_kwargs or {}
        self.seq_encoder = create_seq_to_embedding(
            model_type="transformer",
            state_dim=state_dim,
            action_dim=action_dim,
            **encoder_kwargs
        )

        # 创建预测头 - 简化网络结构，统一使用相同的架构
        def create_head(input_dim: int, output_dim: int):
            return nn.Sequential(
                nn.Linear(input_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, output_dim),
            )

        self.inverse_head = nn.Sequential(
            *create_head(embedding_size + 2 * state_dim, head_hidden)[:-1],
            nn.Linear(head_hidden, action_dim),
            nn.Tanh(),
        )

        self.forward_head = create_head(embedding_size + state_dim + action_dim, state_dim)
        self.reward_head = create_head(embedding_size + state_dim + action_dim, 1)
        self.embedding_forward_head = create_head(embedding_size + action_dim, state_dim)
        self.policy_head = nn.Sequential(
            *create_head(embedding_size + state_dim, head_hidden)[:-1],
            nn.Linear(head_hidden, action_dim),
            nn.Tanh(),
        )
        self.state_head = create_head(embedding_size, state_dim)
        self.factor_head = create_head(embedding_size, factor_dim)

    # Encode the history to embedding
    def encode_history(self, states_hist: torch.Tensor, actions_hist: torch.Tensor, 
                      attention_mask=None, use_random_mask=False, 
                      min_visible_length=2) -> torch.Tensor:  
        # print(f"states_hist: {states_hist[0]}")
        # print(f"actions_hist: {actions_hist[0]}")      
        return self.seq_encoder(states_hist, actions_hist,
                               attention_mask=attention_mask,
                               use_random_mask=use_random_mask, 
                               min_visible_length=min_visible_length)

    # Predict the inverse dynamics: given embedding, current state, and next state, predict the current action
    def predict_action(self, embedding: torch.Tensor, current_state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        concat_input = torch.cat([embedding, current_state, next_state], dim=-1)
        return self.inverse_head(concat_input)

    def predict_reward(self, embedding: torch.Tensor, current_state: torch.Tensor, current_action: torch.Tensor) -> torch.Tensor:
        concat_input = torch.cat([embedding, current_state, current_action], dim=-1)
        return self.reward_head(concat_input)

    # Predict the forward dynamics: given embedding, current state, and current action, predict the next state
    def predict_next_state(self, embedding: torch.Tensor, current_state: torch.Tensor, current_action: torch.Tensor) -> torch.Tensor:
        concat_input = torch.cat([embedding, current_state, current_action], dim=-1)
        return self.forward_head(concat_input)

    def predict_next_state_with_embedding(self, embedding: torch.Tensor, current_action: torch.Tensor) -> torch.Tensor:
        concat_input = torch.cat([embedding, current_action], dim=-1)
        return self.embedding_forward_head(concat_input)

    # Predict the true state from embedding only (for regularization)
    def predict_state_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.state_head(embedding)

    # Predict action directly from embedding and current state
    def predict_policy_action(self, embedding: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        concat_input = torch.cat([embedding, current_state], dim=-1)
        return self.policy_head(concat_input)

    def pred_factor(self, embedding: torch.Tensor):
        return self.factor_head(embedding)
    
    
    # 简化损失函数 - 统一使用MSE
    @staticmethod
    def inverse_loss(pred_action: torch.Tensor, target_action: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_action, target_action)

    @staticmethod
    def forward_loss(pred_next_state: torch.Tensor, target_next_state: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_next_state, target_next_state)
    
    @staticmethod
    def state_loss(pred_state: torch.Tensor, target_state: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_state, target_state)
    
    @staticmethod
    def policy_loss(pred_action: torch.Tensor, target_action: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_action, target_action)
    
    @staticmethod
    def factor_loss(pred_factor: torch.Tensor, target_factor: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_factor, target_factor)

    @staticmethod
    def intra_traj_consistency_loss(embeddings: torch.Tensor) -> torch.Tensor:
        """Intra-trajectory consistency loss (original consistency loss)"""
        B, num_windows, embedding_size = embeddings.shape
        
        if num_windows < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        diff = embeddings[:, 1:, :] - embeddings[:, :-1, :]
        return F.mse_loss(diff, torch.zeros_like(diff))

    @staticmethod
    def inter_traj_consistency_loss(embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        """Inter-trajectory consistency loss: 直接计算两个embedding之间的MSE"""
        return F.mse_loss(embedding1, embedding2)
