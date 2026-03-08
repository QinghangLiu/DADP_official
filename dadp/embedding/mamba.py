import torch
import torch.nn as nn
from .base_embedding import EmbeddingBase


from mamba_ssm import Mamba2



class MambaEmbedding(EmbeddingBase):
    """Mamba2-based sequence-to-embedding model"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embedding_size: int = 64,
        d_model: int = 256,
        n_layer: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        pooling: str = "last",
        norm_z: bool = True,
        mask_embedding: bool = False,  # 添加mask_embedding支持
        **kwargs
    ):
        # 将embedding_size作为token_size传递给base class
        super().__init__(state_dim, action_dim, token_size=embedding_size, mask_embedding=mask_embedding, **kwargs)
        
        self.pooling = pooling
        self.norm_z = norm_z
        self.d_model = d_model
        self.embedding_size = embedding_size
        
        assert pooling in {"last", "mean"}, f"Pooling '{pooling}' not supported for Mamba2. Use 'last' or 'mean'."
        
        # 重新定义input projection，将输入投影到d_model维度
        self.in_proj = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        
        # Mamba2 layers - 使用d_model作为内部维度
        self.mamba_layers = nn.ModuleList([
            Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ) for _ in range(n_layer)
        ])
        
        # Replace simple LayerNorm with MLP + LayerNorm，从d_model投影到embedding_size
        if norm_z:
            self.output_norm = nn.Sequential(
                nn.Linear(d_model, embedding_size),
                nn.LayerNorm(embedding_size)
            )
        else:
            # 即使不使用norm_z，也需要投影到正确的embedding_size
            self.output_norm = nn.Linear(d_model, embedding_size)
    
    def get_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Process tokens through Mamba2 layers and pool to get embedding
        
        Args:
            tokens: (B, L, d_model) input tokens (已经通过input_proj处理)
            
        Returns:
            embedding: (B, embedding_size) final embedding
        """
        # Process through Mamba2 layers sequentially
        h = tokens

        
        for layer in self.mamba_layers:
            h = layer(h)  # h: (B, L, d_model)


        # Pool to get final representation
        if self.pooling == "last":
            pooled = h[:, -1, :]  # Take last timestep: (B, d_model)
        elif self.pooling == "mean":
            pooled = h.mean(dim=1)  # Average over time dimension: (B, d_model)
        
        # Apply output normalization and projection to embedding_size
        embedding = self.output_norm(pooled)  # (B, embedding_size)
        
        return embedding
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model
        
        Args:
            states: (B, L, state_dim) state sequences
            actions: (B, L, action_dim) action sequences
            
        Returns:
            embedding: (B, embedding_size) final embedding
        """
        # 如果启用了mask_embedding，直接返回零向量
        if self.mask_embedding:
            batch_size = states.shape[0]
            return torch.zeros(batch_size, self.embedding_size, device=states.device, dtype=states.dtype)
        
        # 否则执行正常的forward计算
        # Concatenate states and actions
        x = torch.cat([states, actions], dim=-1)  # (B, L, state_dim + action_dim)
        
        # Project to d_model dimension
        tokens = self.in_proj(x)  # (B, L, d_model)
        
        # Get embedding
        embedding = self.get_embedding(tokens)  # (B, embedding_size)
        
        return embedding

