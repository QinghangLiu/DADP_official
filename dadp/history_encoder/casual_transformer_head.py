import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .casual_transformer_config import CausalTransformerConfig, CausalDynamicsConfig


class CausalDynamicsHead(nn.Module):
    """Dynamics head designed for causal transformer outputs with separate heads for each segment"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        factor_dim: int = 0,
        embedding_size: int = 256,
        head_hidden: int = 512,
        separate_heads: bool = True,  # 是否为每个segment使用独立的head
        separate_local_encoders: bool = False,  # 是否为每个segment使用独立的local encoder
        no_global_transformer: bool = False,  # 新增：是否跳过global transformer
        causal_transformer_config: CausalTransformerConfig = None
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.factor_dim = factor_dim
        self.embedding_size = embedding_size
        self.head_hidden = head_hidden  # 保存head_hidden
        self.separate_heads = separate_heads  # 保存separate_heads配置
        self.separate_local_encoders = separate_local_encoders  # 保存separate_local_encoders配置
        self.no_global_transformer = no_global_transformer  # 保存no_global_transformer配置
        self.causal_transformer_config = causal_transformer_config
        
        # 导入causal transformer
        from .casual_transformer import CausalTransformerHistoryEncoder
        
        # Create causal transformer history encoder
        if causal_transformer_config is None:
            causal_transformer_config = CausalTransformerConfig(
                state_dim=state_dim,
                action_dim=action_dim,
                embedding_size=embedding_size,
                separate_local_encoders=separate_local_encoders,  # 传递参数
                no_global_transformer=no_global_transformer  # 传递参数
            )
        
        self.history_encoder = CausalTransformerHistoryEncoder(
            state_dim=causal_transformer_config.state_dim,
            action_dim=causal_transformer_config.action_dim,
            embedding_size=causal_transformer_config.embedding_size,
            d_model=causal_transformer_config.d_model,
            total_length=causal_transformer_config.total_length,
            num_segments=causal_transformer_config.num_segments,
            local_n_head=causal_transformer_config.local_n_head,
            local_d_ff=causal_transformer_config.local_d_ff,
            local_n_layer=causal_transformer_config.local_n_layer,
            local_dropout=causal_transformer_config.local_dropout,
            global_n_head=causal_transformer_config.global_n_head,
            global_d_ff=causal_transformer_config.global_d_ff,
            global_n_layer=causal_transformer_config.global_n_layer,
            global_dropout=causal_transformer_config.global_dropout,
            norm_z=causal_transformer_config.norm_z,
            separate_local_encoders=causal_transformer_config.separate_local_encoders,  # 传递参数
            no_global_transformer=causal_transformer_config.no_global_transformer  # 传递参数
        )
        
        # 获取 segments 数量
        self.num_segments = causal_transformer_config.num_segments
        
        # 根据separate_heads参数决定创建独立heads还是共享heads
        if separate_heads:
            # 创建独立的 dynamics prediction heads
            self.inverse_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embedding_size + 2 * state_dim, head_hidden),
                    nn.ReLU(),
                    nn.Linear(head_hidden, head_hidden),
                    nn.ReLU(),
                    nn.Linear(head_hidden, action_dim),
                    nn.Tanh(),
                ) for _ in range(self.num_segments)
            ])

            self.forward_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embedding_size + state_dim + action_dim, head_hidden),
                    nn.ReLU(),
                    nn.Linear(head_hidden, head_hidden),
                    nn.ReLU(),
                    nn.Linear(head_hidden, state_dim),
                ) for _ in range(self.num_segments)
            ])

            self.embedding_forward_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embedding_size + action_dim, head_hidden),
                    nn.ReLU(),
                    nn.Linear(head_hidden, head_hidden),
                    nn.ReLU(),
                    nn.Linear(head_hidden, state_dim),
                ) for _ in range(self.num_segments)
            ])

            self.policy_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embedding_size + state_dim, head_hidden),
                    nn.ReLU(),
                    nn.Linear(head_hidden, head_hidden),
                    nn.ReLU(),
                    nn.Linear(head_hidden, action_dim),
                    nn.Tanh(),
                ) for _ in range(self.num_segments)
            ])

            self.state_heads = nn.ModuleList([
                nn.Linear(embedding_size, state_dim)
                for _ in range(self.num_segments)
            ])

            # Factor heads (if factor_dim > 0)
            if factor_dim > 0:
                self.factor_heads = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(embedding_size, head_hidden),
                        nn.ReLU(),
                        nn.Linear(head_hidden, head_hidden),
                        nn.ReLU(),
                        nn.Linear(head_hidden, factor_dim),
                    ) for _ in range(self.num_segments)
                ])
            else:
                self.factor_heads = None

            self._use_separate_heads = True
        else:
            # 创建共享的heads
            self._use_separate_heads = False
            self._create_shared_heads()
    
    def _create_shared_heads(self):
        """创建共享的head - 使用实例变量"""
        self.inverse_head = nn.Sequential(
            nn.Linear(self.embedding_size + 2 * self.state_dim, self.head_hidden),
            nn.ReLU(),
            nn.Linear(self.head_hidden, self.head_hidden),
            nn.ReLU(),
            nn.Linear(self.head_hidden, self.action_dim),
            nn.Tanh(),
        )

        self.forward_head = nn.Sequential(
            nn.Linear(self.embedding_size + self.state_dim + self.action_dim, self.head_hidden),
            nn.ReLU(),
            nn.Linear(self.head_hidden, self.head_hidden),
            nn.ReLU(),
            nn.Linear(self.head_hidden, self.state_dim),
        )

        self.embedding_forward_head = nn.Sequential(
            nn.Linear(self.embedding_size + self.action_dim, self.head_hidden),
            nn.ReLU(),
            nn.Linear(self.head_hidden, self.head_hidden),
            nn.ReLU(),
            nn.Linear(self.head_hidden, self.state_dim),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(self.embedding_size + self.state_dim, self.head_hidden),
            nn.ReLU(),
            nn.Linear(self.head_hidden, self.head_hidden),
            nn.ReLU(),
            nn.Linear(self.head_hidden, self.action_dim),
            nn.Tanh(),
        )

        self.state_head = nn.Linear(self.embedding_size, self.state_dim)

        # Factor head (if factor_dim > 0)
        if self.factor_dim > 0:
            self.factor_head = nn.Sequential(
                nn.Linear(self.embedding_size, self.head_hidden),
                nn.ReLU(),
                nn.Linear(self.head_hidden, self.head_hidden),
                nn.ReLU(),
                nn.Linear(self.head_hidden, self.factor_dim),
            )
        else:
            self.factor_head = None

    def encode_history(self, states_hist: torch.Tensor, actions_hist: torch.Tensor) -> torch.Tensor:
        """Encode history using causal transformer"""
        return self.history_encoder(states_hist, actions_hist)

    def predict_action(self, embedding: torch.Tensor, current_state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """
        Predict actions - 支持独立heads和共享heads两种模式
        
        Args:
            embedding: (B, num_segments, embedding_size)
            current_state: (B, num_segments, state_dim) - 已经扩展的维度 
            next_state: (B, num_segments, state_dim) - 已经扩展的维度
            
        Returns:
            pred_actions: (B, num_segments, action_dim)
        """
        B, num_segments, embedding_size = embedding.shape
        
        if self._use_separate_heads:
            # 使用独立heads
            pred_actions = []
            for i in range(num_segments):
                segment_embedding = embedding[:, i, :]  # (B, embedding_size)
                segment_current_state = current_state[:, i, :]  # (B, state_dim)
                segment_next_state = next_state[:, i, :]  # (B, state_dim)
                
                concat_input = torch.cat([segment_embedding, segment_current_state, segment_next_state], dim=-1)
                pred_action = self.inverse_heads[i](concat_input)  # (B, action_dim)
                pred_actions.append(pred_action.unsqueeze(1))  # (B, 1, action_dim)
            
            return torch.cat(pred_actions, dim=1)  # (B, num_segments, action_dim)
        else:
            # 使用共享heads
            # 展平处理，让所有segments共享同一个head
            embedding_flat = embedding.reshape(B * num_segments, embedding_size)
            current_state_flat = current_state.reshape(B * num_segments, -1)
            next_state_flat = next_state.reshape(B * num_segments, -1)
            
            concat_input = torch.cat([embedding_flat, current_state_flat, next_state_flat], dim=-1)
            pred_action_flat = self.inverse_head(concat_input)
            
            return pred_action_flat.reshape(B, num_segments, -1)

    def predict_next_state(self, embedding: torch.Tensor, current_state: torch.Tensor, current_action: torch.Tensor) -> torch.Tensor:
        """
        Predict next states - 支持独立heads和共享heads两种模式
        
        Args:
            embedding: (B, num_segments, embedding_size)
            current_state: (B, num_segments, state_dim) - 已经扩展的维度
            current_action: (B, num_segments, action_dim) - 已经扩展的维度
        """
        B, num_segments, embedding_size = embedding.shape
        
        if self._use_separate_heads:
            # 使用独立heads
            pred_next_states = []
            for i in range(num_segments):
                segment_embedding = embedding[:, i, :]  # (B, embedding_size)
                segment_current_state = current_state[:, i, :]  # (B, state_dim)
                segment_current_action = current_action[:, i, :]  # (B, action_dim)
                
                concat_input = torch.cat([segment_embedding, segment_current_state, segment_current_action], dim=-1)
                pred_next_state = self.forward_heads[i](concat_input)  # (B, state_dim)
                pred_next_states.append(pred_next_state.unsqueeze(1))  # (B, 1, state_dim)
            
            return torch.cat(pred_next_states, dim=1)  # (B, num_segments, state_dim)
        else:
            # 使用共享heads
            embedding_flat = embedding.reshape(B * num_segments, embedding_size)
            current_state_flat = current_state.reshape(B * num_segments, -1)
            current_action_flat = current_action.reshape(B * num_segments, -1)
            
            concat_input = torch.cat([embedding_flat, current_state_flat, current_action_flat], dim=-1)
            pred_next_state_flat = self.forward_head(concat_input)
            
            return pred_next_state_flat.reshape(B, num_segments, -1)

    def predict_next_state_with_embedding(self, embedding: torch.Tensor, current_action: torch.Tensor) -> torch.Tensor:
        """
        Predict next states from embedding and action - 支持独立heads和共享heads两种模式
        
        Args:
            embedding: (B, num_segments, embedding_size)
            current_action: (B, num_segments, action_dim) - 已经扩展的维度
        """
        B, num_segments, embedding_size = embedding.shape
        
        if self._use_separate_heads:
            # 使用独立heads
            pred_next_states = []
            for i in range(num_segments):
                segment_embedding = embedding[:, i, :]  # (B, embedding_size)
                segment_current_action = current_action[:, i, :]  # (B, action_dim)
                
                concat_input = torch.cat([segment_embedding, segment_current_action], dim=-1)
                pred_next_state = self.embedding_forward_heads[i](concat_input)  # (B, state_dim)
                pred_next_states.append(pred_next_state.unsqueeze(1))  # (B, 1, state_dim)
            
            return torch.cat(pred_next_states, dim=1)  # (B, num_segments, state_dim)
        else:
            # 使用共享heads
            embedding_flat = embedding.reshape(B * num_segments, embedding_size)
            current_action_flat = current_action.reshape(B * num_segments, -1)
            
            concat_input = torch.cat([embedding_flat, current_action_flat], dim=-1)
            pred_next_state_flat = self.embedding_forward_head(concat_input)
            
            return pred_next_state_flat.reshape(B, num_segments, -1)

    def predict_state_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict states from embedding - 支持独立heads和共享heads两种模式
        
        Args:
            embedding: (B, num_segments, embedding_size)
        """
        B, num_segments, embedding_size = embedding.shape
        
        if self._use_separate_heads:
            # 使用独立heads
            pred_states = []
            for i in range(num_segments):
                segment_embedding = embedding[:, i, :]  # (B, embedding_size)
                pred_state = self.state_heads[i](segment_embedding)  # (B, state_dim)
                pred_states.append(pred_state.unsqueeze(1))  # (B, 1, state_dim)
            
            return torch.cat(pred_states, dim=1)  # (B, num_segments, state_dim)
        else:
            # 使用共享heads
            embedding_flat = embedding.reshape(B * num_segments, embedding_size)
            pred_state_flat = self.state_head(embedding_flat)
            return pred_state_flat.reshape(B, num_segments, -1)

    def predict_policy_action(self, embedding: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        """
        Predict policy actions - 支持独立heads和共享heads两种模式
        
        Args:
            embedding: (B, num_segments, embedding_size)
            current_state: (B, num_segments, state_dim) - 已经扩展的维度
        """
        B, num_segments, embedding_size = embedding.shape
        
        if self._use_separate_heads:
            # 使用独立heads
            pred_actions = []
            for i in range(num_segments):
                segment_embedding = embedding[:, i, :]  # (B, embedding_size)
                segment_current_state = current_state[:, i, :]  # (B, state_dim)
                
                concat_input = torch.cat([segment_embedding, segment_current_state], dim=-1)
                pred_action = self.policy_heads[i](concat_input)  # (B, action_dim)
                pred_actions.append(pred_action.unsqueeze(1))  # (B, 1, action_dim)
            
            return torch.cat(pred_actions, dim=1)  # (B, num_segments, action_dim)
        else:
            # 使用共享heads
            embedding_flat = embedding.reshape(B * num_segments, embedding_size)
            current_state_flat = current_state.reshape(B * num_segments, -1)
            
            concat_input = torch.cat([embedding_flat, current_state_flat], dim=-1)
            pred_action_flat = self.policy_head(concat_input)
            
            return pred_action_flat.reshape(B, num_segments, -1)

    def pred_factor(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict factors - 支持独立heads和共享heads两种模式
        
        Args:
            embedding: (B, num_segments, embedding_size)
        """
        B, num_segments, _ = embedding.shape
        
        # 如果factor_dim为0，直接返回零tensor
        if self.factor_dim == 0:
            return torch.zeros(B, num_segments, self.factor_dim, device=embedding.device)
        
        if self._use_separate_heads:
            # 使用独立heads
            if self.factor_heads is not None:
                pred_factors = []
                for i in range(num_segments):
                    segment_embedding = embedding[:, i, :]  # (B, embedding_size)
                    pred_factor = self.factor_heads[i](segment_embedding)  # (B, factor_dim)
                    pred_factors.append(pred_factor.unsqueeze(1))  # (B, 1, factor_dim)
                
                return torch.cat(pred_factors, dim=1)  # (B, num_segments, factor_dim)
            else:
                return torch.zeros(B, num_segments, self.factor_dim, device=embedding.device)
        else:
            # 使用共享heads
            if self.factor_head is not None:
                embedding_flat = embedding.reshape(B * num_segments, -1)
                pred_factor_flat = self.factor_head(embedding_flat)
                return pred_factor_flat.reshape(B, num_segments, -1)
            else:
                return torch.zeros(B, num_segments, self.factor_dim, device=embedding.device)

    # Static loss functions - 简化，去掉expand逻辑
    @staticmethod
    def inverse_loss(pred_action: torch.Tensor, target_action: torch.Tensor) -> torch.Tensor:
        """Calculate inverse loss - target_action已经被外部扩展到3D"""
        B, num_segments, action_dim = pred_action.shape
        
        # 直接计算损失，不再进行expand
        segment_losses = []
        for i in range(num_segments):
            segment_loss = F.mse_loss(pred_action[:, i, :], target_action[:, i, :])
            segment_losses.append(segment_loss)
        
        return torch.stack(segment_losses)

    @staticmethod
    def forward_loss(pred_next_state: torch.Tensor, target_next_state: torch.Tensor) -> torch.Tensor:
        """Calculate forward loss - target_next_state已经被外部扩展到3D"""
        B, num_segments, state_dim = pred_next_state.shape
        
        segment_losses = []
        for i in range(num_segments):
            segment_loss = F.mse_loss(pred_next_state[:, i, :], target_next_state[:, i, :])
            segment_losses.append(segment_loss)
        
        return torch.stack(segment_losses)

    @staticmethod
    def state_loss(pred_state: torch.Tensor, target_state: torch.Tensor) -> torch.Tensor:
        """Calculate state loss - target_state已经被外部扩展到3D"""
        B, num_segments, state_dim = pred_state.shape
        
        segment_losses = []
        for i in range(num_segments):
            segment_loss = F.mse_loss(pred_state[:, i, :], target_state[:, i, :])
            segment_losses.append(segment_loss)
        
        return torch.stack(segment_losses)

    @staticmethod
    def policy_loss(pred_action: torch.Tensor, target_action: torch.Tensor) -> torch.Tensor:
        """Calculate policy loss - target_action已经被外部扩展到3D"""
        B, num_segments, action_dim = pred_action.shape
        
        segment_losses = []
        for i in range(num_segments):
            segment_loss = F.mse_loss(pred_action[:, i, :], target_action[:, i, :])
            segment_losses.append(segment_loss)
        
        return torch.stack(segment_losses)

    @staticmethod
    def factor_loss(pred_factor: torch.Tensor, target_factor: torch.Tensor) -> torch.Tensor:
        """Calculate factor loss - target_factor已经被外部扩展到3D"""
        B, num_segments, factor_dim = pred_factor.shape
        
        segment_losses = []
        for i in range(num_segments):
            segment_loss = F.mse_loss(pred_factor[:, i, :], target_factor[:, i, :])
            segment_losses.append(segment_loss)
        
        return torch.stack(segment_losses)

    @staticmethod
    def token_consistency_loss(embeddings: torch.Tensor) -> torch.Tensor:
        """Token consistency loss across segments - 只处理3D情况"""
        B, num_segments, embedding_size = embeddings.shape
        
        if num_segments < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Calculate consistency loss between adjacent segments
        consistency_losses = []
        for i in range(num_segments - 1):
            diff = embeddings[:, i+1, :] - embeddings[:, i, :]
            segment_loss = F.mse_loss(diff, torch.zeros_like(diff))
            consistency_losses.append(segment_loss)
        
        return torch.stack(consistency_losses)


def create_causal_transformer_model(
    state_dim: int,
    action_dim: int,
    factor_dim: int = 0,
    embedding_size: int = 256,
    head_hidden: int = 512,
    separate_heads: bool = True,
    separate_local_encoders: bool = False,
    no_global_transformer: bool = False,  # 新增参数
    **kwargs
) -> CausalDynamicsHead:
    """
    Factory function to create causal transformer dynamics model
    
    Args:
        state_dim: State dimension
        action_dim: Action dimension
        factor_dim: Factor dimension for task encoding
        embedding_size: Embedding dimension
        head_hidden: Hidden dimension for prediction heads
        separate_heads: Whether to use separate heads for each segment
        separate_local_encoders: Whether to use separate local encoders for each segment
        no_global_transformer: Whether to skip global transformer
        **kwargs: Additional arguments for CausalTransformerConfig
        
    Returns:
        CausalDynamicsHead model
    """
    causal_transformer_config = CausalTransformerConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        embedding_size=embedding_size,
        separate_local_encoders=separate_local_encoders,  # 传递参数
        no_global_transformer=no_global_transformer,  # 传递参数
        **kwargs
    )
    
    dynamics_config = CausalDynamicsConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        factor_dim=factor_dim,
        embedding_size=embedding_size,
        head_hidden=head_hidden,
        separate_heads=separate_heads,
        separate_local_encoders=separate_local_encoders,  # 传递参数
        no_global_transformer=no_global_transformer,  # 传递参数
        causal_transformer_config=causal_transformer_config
    )
    
    model = CausalDynamicsHead(
        state_dim=dynamics_config.state_dim,
        action_dim=dynamics_config.action_dim,
        factor_dim=dynamics_config.factor_dim,
        embedding_size=dynamics_config.embedding_size,
        head_hidden=dynamics_config.head_hidden,
        separate_heads=dynamics_config.separate_heads,
        separate_local_encoders=dynamics_config.separate_local_encoders,  # 传递参数
        no_global_transformer=dynamics_config.no_global_transformer,  # 传递参数
        causal_transformer_config=dynamics_config.causal_transformer_config
    )
    
    return model


def save_causal_model_checkpoint(
    model: CausalDynamicsHead,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    metadata: Dict[str, Any]
):
    """Save causal transformer model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metadata': metadata
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Causal model checkpoint saved: {checkpoint_path}")


def load_causal_model_checkpoint(
    checkpoint_path: str,
    state_dim: int,
    action_dim: int,
    factor_dim: int = 0,
    device: str = "cuda"
) -> tuple:
    """Load causal transformer model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model (you might want to save config in checkpoint for better restoration)
    model = create_causal_transformer_model(
        state_dim=state_dim,
        action_dim=action_dim,
        factor_dim=factor_dim
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    metadata = checkpoint.get('metadata', {})
    
    print(f"Causal model checkpoint loaded: {checkpoint_path}")
    return model, optimizer, metadata
