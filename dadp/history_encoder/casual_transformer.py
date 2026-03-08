import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalSummaryEncoder(nn.Module):
    """Local summary encoder for processing individual segments"""
    
    def __init__(
        self,
        d_model: int,
        n_head: int = 8,
        d_ff: int = 1024,
        n_layer: int = 2,
        dropout: float = 0.1,
        segment_length: int = 8  # 固定的段长度
    ):
        super().__init__()
        self.d_model = d_model
        self.segment_length = segment_length
        
        self.segment_pos_embed = nn.Parameter(torch.randn(segment_length * 2, d_model))  # 去掉 +1 for query
        nn.init.normal_(self.segment_pos_embed, mean=0, std=0.02)
        self.pos_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        # Transformer encoder layers for local processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        # 添加 adaptive pooling，遵循 transformer.py 的设计
        self.adaptive_pooling = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )
        
        # Learnable query for adaptive pooling
        self.adaptive_query = nn.Parameter(torch.randn(1, d_model))
        nn.init.normal_(self.adaptive_query, mean=0, std=0.02)
        
    def forward(self, segment_tokens: torch.Tensor) -> torch.Tensor:
        """Process a segment and return its summary token"""
        B, segment_len, d_model = segment_tokens.shape
        
        # Add fixed positional encoding
        segment_tokens = segment_tokens + self.segment_pos_embed[:segment_len].unsqueeze(0)
        segment_tokens = self.pos_dropout(segment_tokens)
        
        # Process through transformer layers
        attended = self.transformer_encoder(segment_tokens)  # (B, segment_len, d_model)
        
        # Apply adaptive pooling，遵循 transformer.py 的设计
        query = self.adaptive_query.expand(B, 1, d_model)  # (B, 1, d_model)
        
        # Use multihead attention for adaptive pooling
        summary_token, _ = self.adaptive_pooling(
            query=query,              # (B, 1, d_model)
            key=attended,             # (B, segment_len, d_model)
            value=attended            # (B, segment_len, d_model)
        )
        
        # Extract the single summary token
        summary_token = summary_token.squeeze(1)  # (B, d_model)
        summary_token = self.norm(summary_token)  # Apply layer normalization
        
        return summary_token


class GlobalCausalTransformer(nn.Module):
    """Global causal transformer for processing summary tokens"""
    
    def __init__(
        self,
        d_model: int,
        n_head: int = 8,
        d_ff: int = 1024,
        n_layer: int = 4,
        dropout: float = 0.1,
        num_segments: int = 8  # 固定的段数量
    ):
        super().__init__()
        self.d_model = d_model
        self.num_segments = num_segments
        
        # 固定的段间位置编码
        self.global_pos_embed = nn.Parameter(torch.randn(num_segments, d_model))
        nn.init.normal_(self.global_pos_embed, mean=0, std=0.02)
        self.pos_dropout = nn.Dropout(dropout)
        
        # 修改：使用 Encoder 而不是 Decoder 来实现因果掩码
        # TransformerDecoder 的因果掩码实现可能有问题
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.causal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
    def forward(self, summary_tokens: torch.Tensor) -> torch.Tensor:
        """Process summary tokens with causal attention"""
        B, num_segments, d_model = summary_tokens.shape
        
        # Add fixed global positional encoding
        tokens = summary_tokens + self.global_pos_embed[:num_segments].unsqueeze(0)
        tokens = self.pos_dropout(tokens)
        
        # 创建正确的因果掩码 - 这是关键修正
        # 在 PyTorch 的 attention 中，mask=True 表示该位置会被忽略（设为 -inf）
        # 我们要屏蔽未来位置，所以未来位置应该是 True
        causal_mask = torch.triu(
            torch.ones(num_segments, num_segments, device=tokens.device, dtype=torch.bool),
            diagonal=1  # diagonal=1 表示对角线及其以上的位置为 True（屏蔽未来）
        )
        
        # 验证掩码是否正确
        # 对于 4x4 的矩阵，应该是：
        # [[False, True,  True,  True ],
        #  [False, False, True,  True ],
        #  [False, False, False, True ],
        #  [False, False, False, False]]
        
        # 使用 TransformerEncoder 并传入因果掩码
        output_tokens = self.causal_transformer(
            src=tokens,
            mask=causal_mask  # 传入因果掩码
        )
        
        return output_tokens
    
class CausalTransformerHistoryEncoder(nn.Module):
    """Causal Transformer with hierarchical segment processing for history encoding"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embedding_size: int = 256,
        d_model: int = 256,
        
        # 固定的分段参数
        total_length: int = 32,
        num_segments: int = 8,
        
        # Local encoder parameters
        local_n_head: int = 8,
        local_d_ff: int = 1024,
        local_n_layer: int = 2,
        local_dropout: float = 0.1,
        
        # Global transformer parameters
        global_n_head: int = 8,
        global_d_ff: int = 1024,
        global_n_layer: int = 4,
        global_dropout: float = 0.1,
        
        # Standard parameters
        norm_z: bool = True,
        separate_local_encoders: bool = False,  # 是否为每个segment使用独立的local encoder
        no_global_transformer: bool = False,  # 新增：是否跳过global transformer
        **kwargs  # 忽略其他传入的参数
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.embedding_size = embedding_size
        self.norm_z = norm_z
        self.separate_local_encoders = separate_local_encoders  # 保存配置
        self.no_global_transformer = no_global_transformer  # 保存配置
        
        # 固定的分段参数
        self.total_length = total_length
        self.num_segments = num_segments
        self.segment_length = total_length // num_segments
        
        assert total_length % num_segments == 0, f"total_length ({total_length}) must be divisible by num_segments ({num_segments})"
        
        # Input projection
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        
        # 根据separate_local_encoders参数决定创建共享还是独立的local encoders
        if separate_local_encoders:
            # 为每个segment创建独立的local encoder
            self.local_encoders = nn.ModuleList([
                LocalSummaryEncoder(
                    d_model=d_model,
                    n_head=local_n_head,
                    d_ff=local_d_ff,
                    n_layer=local_n_layer,
                    dropout=local_dropout,
                    segment_length=self.segment_length
                ) for _ in range(num_segments)
            ])
            self._use_separate_local_encoders = True
        else:
            # 创建单个共享的local encoder
            self.local_encoder = LocalSummaryEncoder(
                d_model=d_model,
                n_head=local_n_head,
                d_ff=local_d_ff,
                n_layer=local_n_layer,
                dropout=local_dropout,
                segment_length=self.segment_length
            )
            self._use_separate_local_encoders = False
        
        # Global causal transformer - 只有在不跳过时才创建
        if not no_global_transformer:
            self.global_transformer = GlobalCausalTransformer(
                d_model=d_model,
                n_head=global_n_head,
                d_ff=global_d_ff,
                n_layer=global_n_layer,
                dropout=global_dropout,
                num_segments=num_segments  # 固定的段数量
            )
        else:
            self.global_transformer = None
        
        # 超级简化的输出投影 - 直接处理 (B, num_segments, d_model) -> (B, num_segments, embedding_size)
        if d_model != embedding_size:
            if norm_z:
                self.output_proj = nn.Sequential(
                    nn.Linear(d_model, embedding_size),
                    nn.LayerNorm(embedding_size)
                )
            else:
                self.output_proj = nn.Linear(d_model, embedding_size)
        else:
            self.output_proj = nn.LayerNorm(embedding_size) if norm_z else nn.Identity()


    def segment_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Segment tokens into fixed-size segments"""
        B, seq_len, d_model = tokens.shape
        
        # 固定的期望长度
        expected_len = self.total_length * 2  # state + action interleaved
        
        # 固定的重塑 - 每个段包含 segment_length*2 个tokens
        segment_token_len = self.segment_length * 2
        segmented = tokens.reshape(B, self.num_segments, segment_token_len, d_model)
        
        return segmented

    def build_tokens(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Build interleaved tokens from states and actions
        
        Args:
            states: (B, L, state_dim) state sequences
            actions: (B, L, action_dim) action sequences
            
        Returns:
            tokens: (B, 2*L, d_model) interleaved tokens
        """
        B, L, _ = states.shape
        
        # Encode states and actions separately
        state_tokens = self.state_encoder(states)    # (B, L, d_model)
        action_tokens = self.action_encoder(actions)  # (B, L, d_model)
        
        # Interleave: state_0, action_0, state_1, action_1, ...
        tokens = torch.zeros(B, 2 * L, self.d_model, device=states.device, dtype=states.dtype)
        tokens[:, 0::2, :] = state_tokens   # Even positions: states
        tokens[:, 1::2, :] = action_tokens  # Odd positions: actions
        
        return tokens
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete causal transformer
        
        Args:
            states: (B, L, state_dim) state sequences
            actions: (B, L, action_dim) action sequences
            
        Returns:
            embeddings: (B, num_segments, embedding_size) 所有segments的tokens
        """
        # Build interleaved tokens
        tokens = self.build_tokens(states, actions)  # (B, 2*L, d_model)
        
        # Segment the tokens
        segmented_tokens = self.segment_tokens(tokens)  # (B, num_segments, segment_len*2, d_model)
        
        # Process each segment through local encoder(s)
        B, num_segments, segment_len, d_model = segmented_tokens.shape
        
        if self._use_separate_local_encoders:
            # 使用独立的local encoders - 每个segment使用自己专门的encoder
            summary_tokens_list = []
            
            for i in range(num_segments):
                segment_i = segmented_tokens[:, i, :, :]  # (B, segment_len, d_model)
                summary_token_i = self.local_encoders[i](segment_i)  # (B, d_model)
                summary_tokens_list.append(summary_token_i.unsqueeze(1))  # (B, 1, d_model)
            
            summary_tokens = torch.cat(summary_tokens_list, dim=1)  # (B, num_segments, d_model)
        else:
            # 使用共享的local encoder - 所有segments共享同一个encoder
            # Reshape for batch processing: (B*num_segments, segment_len, d_model)
            flat_segments = segmented_tokens.reshape(B * num_segments, segment_len, d_model)
            
            # Get summary tokens for all segments
            summary_tokens = self.local_encoder(flat_segments)  # (B*num_segments, d_model)
            
            # Reshape back: (B, num_segments, d_model)
            summary_tokens = summary_tokens.reshape(B, num_segments, d_model)

        # 根据no_global_transformer参数决定是否使用全局transformer
        if self.no_global_transformer:
            # 跳过全局transformer，直接使用local encoder的输出
            output_tokens = summary_tokens  # (B, num_segments, d_model)
        else:
            # Process through global causal transformer
            output_tokens = self.global_transformer(summary_tokens)  # (B, num_segments, d_model)
        
        # 直接对 (B, num_segments, d_model) 应用MLP，不需要reshape！
        final_output = self.output_proj(output_tokens)  # (B, num_segments, embedding_size)
        
        return final_output