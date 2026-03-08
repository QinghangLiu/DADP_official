from .base_embedding import EmbeddingBase
from .transformer import TransformerEmbedding

def create_seq_to_embedding(model_type: str, state_dim: int, action_dim: int, **kwargs):
    """Factory function to create sequence-to-embedding models"""
    if model_type.lower() == "transformer":
        return TransformerEmbedding(
            state_dim=state_dim,
            action_dim=action_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# 导出所有主要类和函数
__all__ = [
    'EmbeddingBase',
    'TransformerEmbedding',
    'create_seq_to_embedding'
]
