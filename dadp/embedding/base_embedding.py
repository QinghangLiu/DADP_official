import torch
import torch.nn as nn
import random
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class EmbeddingBase(nn.Module, ABC):
    """Base class for sequence-to-embedding models"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        token_size: int = 64,
        mask_embedding: bool = False,
        **kwargs  # Allow subclasses to accept additional parameters
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.token_size = token_size
        self.mask_embedding = mask_embedding
        
        # Input projection layer - convert state+action to token_size dimension
        d_in = state_dim + action_dim
        self.in_proj = nn.Sequential(
            nn.Linear(d_in, token_size * 2),  # embedding + current_state + current_action
            nn.ReLU(),
            nn.Linear(token_size * 2, token_size * 2),
            nn.ReLU(),
            nn.Linear(token_size * 2, token_size),
        )
    
    def generate_random_attention_mask(self, batch_size: int, seq_len: int, 
                                     min_visible_length: int = 8) -> torch.Tensor:
        """Generate random attention masks for training
        
        Args:
            batch_size: Number of samples in batch
            seq_len: Total sequence length  
            min_visible_length: Minimum number of visible tokens
            
        Returns:
            Attention mask where True means "ignore this position"
        """
        masks = []
        for _ in range(batch_size):
            # Randomly choose visible length for this sample (up to full sequence)
            visible_length = random.randint(min_visible_length * 2, seq_len)
            
            # Create mask: True for positions to ignore, False for positions to attend
            mask = torch.ones(seq_len, dtype=torch.bool)
            mask[-visible_length:] = False  # Make last visible_length positions attendable
            
            masks.append(mask)
        
        return torch.stack(masks)
    
    def create_inference_mask(self, current_step: int, total_length: int) -> torch.Tensor:
        """Create attention mask for inference time
        
        Args:
            current_step: Current inference step (0-indexed)
            total_length: Total sequence length
            
        Returns:
            Attention mask for single sample
        """
        visible_length = min(current_step + 1, total_length)
        mask = torch.ones(total_length, dtype=torch.bool)
        mask[-visible_length:] = False
        return mask

    def build_tokens(self, states_hist: torch.Tensor, actions_hist: torch.Tensor) -> torch.Tensor:
        concat_input = torch.cat([states_hist, actions_hist], dim=-1)
        tokens = self.in_proj(concat_input)
        return tokens
    
    @abstractmethod
    def get_embedding(self, tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass
    
    def forward(self, states_hist: torch.Tensor, actions_hist: torch.Tensor, 
               attention_mask: Optional[torch.Tensor] = None,
               use_random_mask: bool = False, 
               min_visible_length: int = 2) -> torch.Tensor:
        """Forward pass with optional attention mask support
        
        Args:
            states_hist: Historical states [B, L, state_dim]
            actions_hist: Historical actions [B, L, action_dim]
            attention_mask: Optional attention mask [B, seq_len] 
            use_random_mask: Whether to generate random mask for training
            min_visible_length: Minimum visible length when using random mask
        """
        tokens = self.build_tokens(states_hist, actions_hist)
        
        if self.mask_embedding:
            # Return zero-initialized embedding if masking is enabled
            batch_size = tokens.shape[0]
            return torch.zeros(batch_size, self.token_size, device=tokens.device, dtype=tokens.dtype)
        else:
            # Generate random mask if requested and no mask provided
            if use_random_mask and attention_mask is None:
                batch_size, seq_len = tokens.shape[:2]
                attention_mask = self.generate_random_attention_mask(
                    batch_size, seq_len, min_visible_length
                )
                attention_mask = attention_mask.to(tokens.device)
            
            # Normal embedding computation with optional attention mask
            embedding = self.get_embedding(tokens, attention_mask)
            return embedding
