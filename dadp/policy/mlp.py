import torch
import torch.nn as nn
from .basepolicy import BasePolicy

class MLPPolicy(BasePolicy):
    """Simple MLP policy"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embedding_dim: int,
        hidden_dims: list = [256, 256],
        dropout: float = 0.1
    ):
        super().__init__(state_dim, action_dim, embedding_dim)
        
        # Input is embedding + current state
        input_dim = embedding_dim + state_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        state_history: torch.Tensor,
        action_history: torch.Tensor,
        history_embedding: torch.Tensor, 
        current_state: torch.Tensor
        ) -> torch.Tensor:
        # Just concatenate embedding and current state
        x = torch.cat([history_embedding, current_state], dim=-1)
        return self.mlp(x)

