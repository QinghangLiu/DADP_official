import torch
import torch.nn as nn
from .base_embedding import EmbeddingBase


class LSTMEmbedding(EmbeddingBase):
    """LSTM-based sequence-to-embedding model"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_size: int = 64,
        n_layer: int = 2,
        dropout: float = 0.1,
        pooling: str = "last",
        norm_z: bool = True,
        bidirectional: bool = False,
        **kwargs
    ):
        super().__init__(state_dim, action_dim, latent_size, **kwargs)
        
        self.pooling = pooling
        self.norm_z = norm_z
        self.bidirectional = bidirectional
        
        assert pooling in {"last", "mean"}
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=latent_size,
            hidden_size=latent_size // (2 if bidirectional else 1),
            num_layers=n_layer,
            dropout=dropout if n_layer > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Output projection for bidirectional LSTM
        if bidirectional:
            self.output_proj = nn.Linear(latent_size, latent_size)
        else:
            self.output_proj = nn.Identity()
        
        # Optional output normalization
        self.output_norm = nn.LayerNorm(latent_size) if norm_z else nn.Identity()
    
    def get_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Process tokens through LSTM and pool to get embedding
        
        Args:
            tokens: (B, L, latent_size) input tokens
            
        Returns:
            embedding: (B, latent_size) final embedding
        """
        # Process through LSTM
        h, _ = self.lstm(tokens)
        h = self.output_proj(h)
        
        # Pool to get final embedding
        if self.pooling == "last":
            embedding = h[:, -1, :]  # Take last timestep
        elif self.pooling == "mean":
            embedding = h.mean(dim=1)  # Average over time dimension
        
        # Apply output normalization
        embedding = self.output_norm(embedding)
        
        return embedding


class GRUEmbedding(EmbeddingBase):
    """GRU-based sequence-to-embedding model"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_size: int = 64,
        n_layer: int = 2,
        dropout: float = 0.1,
        pooling: str = "last",
        norm_z: bool = True,
        bidirectional: bool = False,
        **kwargs
    ):
        super().__init__(state_dim, action_dim, latent_size, **kwargs)
        
        self.pooling = pooling
        self.norm_z = norm_z
        self.bidirectional = bidirectional
        
        assert pooling in {"last", "mean"}
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=latent_size,
            hidden_size=latent_size // (2 if bidirectional else 1),
            num_layers=n_layer,
            dropout=dropout if n_layer > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Output projection for bidirectional GRU
        if bidirectional:
            self.output_proj = nn.Linear(latent_size, latent_size)
        else:
            self.output_proj = nn.Identity()
        
        # Optional output normalization
        self.output_norm = nn.LayerNorm(latent_size) if norm_z else nn.Identity()
    
    def get_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Process tokens through GRU and pool to get embedding
        
        Args:
            tokens: (B, L, latent_size) input tokens
            
        Returns:
            embedding: (B, latent_size) final embedding
        """
        # Process through GRU
        h, _ = self.gru(tokens)
        h = self.output_proj(h)
        
        # Pool to get final embedding
        if self.pooling == "last":
            embedding = h[:, -1, :]  # Take last timestep
        elif self.pooling == "mean":
            embedding = h.mean(dim=1)  # Average over time dimension
        
        # Apply output normalization
        embedding = self.output_norm(embedding)
        
        return embedding


class RNNEmbedding(EmbeddingBase):
    """Simple RNN-based sequence-to-embedding model"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_size: int = 64,
        n_layer: int = 2,
        dropout: float = 0.1,
        pooling: str = "last",
        norm_z: bool = True,
        bidirectional: bool = False,
        nonlinearity: str = 'tanh',
        **kwargs
    ):
        super().__init__(state_dim, action_dim, latent_size, **kwargs)
        
        self.pooling = pooling
        self.norm_z = norm_z
        self.bidirectional = bidirectional
        
        assert pooling in {"last", "mean"}
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=latent_size,
            hidden_size=latent_size // (2 if bidirectional else 1),
            num_layers=n_layer,
            dropout=dropout if n_layer > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
            nonlinearity=nonlinearity
        )
        
        # Output projection for bidirectional RNN
        if bidirectional:
            self.output_proj = nn.Linear(latent_size, latent_size)
        else:
            self.output_proj = nn.Identity()
        
        # Optional output normalization
        self.output_norm = nn.LayerNorm(latent_size) if norm_z else nn.Identity()
    
    def get_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Process tokens through RNN and pool to get embedding
        
        Args:
            tokens: (B, L, latent_size) input tokens
            
        Returns:
            embedding: (B, latent_size) final embedding
        """
        # Process through RNN
        h, _ = self.rnn(tokens)
        h = self.output_proj(h)
        
        # Pool to get final embedding
        if self.pooling == "last":
            embedding = h[:, -1, :]  # Take last timestep
        elif self.pooling == "mean":
            embedding = h.mean(dim=1)  # Average over time dimension
        
        # Apply output normalization
        embedding = self.output_norm(embedding)
        
        return embedding
