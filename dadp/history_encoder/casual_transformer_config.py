from dataclasses import dataclass
from typing import Optional


@dataclass
class CausalTransformerConfig:
    """Configuration for Causal Transformer History Encoder - 简化设计"""
    
    # Model dimensions
    state_dim: int
    action_dim: int
    embedding_size: int = 256
    d_model: int = 256
    
    # Segmentation parameters - 固定设计
    total_length: int = 32
    num_segments: int = 8
    
    # Local encoder parameters
    local_n_head: int = 8
    local_d_ff: int = 1024
    local_n_layer: int = 2
    local_dropout: float = 0.1
    
    # Global transformer parameters
    global_n_head: int = 8
    global_d_ff: int = 1024
    global_n_layer: int = 4
    global_dropout: float = 0.1
    
    # Standard parameters
    norm_z: bool = True
    separate_local_encoders: bool = False  # 是否为每个segment使用独立的local encoder
    no_global_transformer: bool = False  # 新增：是否跳过global transformer
    
    def get_total_token_count(self) -> int:
        """计算总的token数量"""
        return self.num_segments
    
    def get_flattened_dimension(self) -> int:
        """计算展平后的维度"""
        return self.num_segments * self.d_model


@dataclass
class CausalDynamicsConfig:
    """Configuration for Causal Dynamics Head"""
    state_dim: int
    action_dim: int
    factor_dim: int = 0
    embedding_size: int = 256
    head_hidden: int = 512
    separate_heads: bool = True  # 是否为每个segment使用独立的head
    separate_local_encoders: bool = False  # 是否为每个segment使用独立的local encoder
    no_global_transformer: bool = False  # 新增：是否跳过global transformer
    causal_transformer_config: CausalTransformerConfig = None
    
    def __post_init__(self):
        if self.causal_transformer_config is None:
            self.causal_transformer_config = CausalTransformerConfig(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                embedding_size=self.embedding_size,
                separate_local_encoders=self.separate_local_encoders,  # 传递参数
                no_global_transformer=self.no_global_transformer  # 传递参数
            )


@dataclass
class CausalTrainingConfig:
    """Configuration for Causal Transformer training"""
    
    # Loss weights
    inverse_loss_weight: float = 1.0
    forward_loss_weight: float = 1.0
    state_loss_weight: float = 0.0
    policy_loss_weight: float = 0.0
    embedding_forward_loss_weight: float = 0.0
    factor_loss_weight: float = 0.0
    
    # 移除一致性损失权重
    # token_consistency_loss_weight: float = 0.1
    # temporal_consistency_loss_weight: float = 0.05
    
    # Training parameters
    learning_rate: float = 3e-4
    num_epochs: int = 10
    batch_size: int = 32
    eval_interval: int = 1
    
    # Sequence processing parameters - 简化为固定参数
    history: int = 32  # 固定的历史长度
    
    # Device
    device: str = "cuda"
