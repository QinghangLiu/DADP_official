from typing import Optional, Tuple

import torch
import torch.nn as nn
from .base_embedding import EmbeddingBase


# 模块说明：
# 该文件定义了用于将状态-动作序列编码为固定向量的 Transformer 风格 embedding。
# 主要流程：
# 1) 用 MLP 将原始 state 和 action 映射到统一的 d_model 维 token
# 2) 对交替排列的 [s0,a0,s1,a1,...] 序列添加可学习的位置编码
# 3) 将序列送入 TransformerEncoder 编码
# 4) 使用自适应注意力池化（AdaptivePooling）将序列聚合为单个向量
# 5) 线性映射并（可选）归一化得到最终 embedding


# TransformerEmbedding 功能说明（中/英双语）
#
# 中文：
# TransformerEmbedding 将输入的状态序列 `states` 和动作序列 `actions` 编码为固定维度的
# embedding 向量。处理流程如下：
# 1) 使用两个独立的 MLP（state_encoder / action_encoder）把原始的 `state_dim` 和 `action_dim`
#    映射到相同的 token 维度 `d_model`；输出为形状 (B, L, d_model) 的 state_tokens 和 action_tokens。
# 2) 按时间步交替拼接 state_tokens 与 action_tokens，得到长度为 2*L 的序列 tokens，形状为
#    (B, 2*L, d_model)，排列为 [s0, a0, s1, a1, ...]。
# 3) 给 tokens 加上可学习的位置编码（LearnablePositionalEmbedding），并应用 dropout。
# 4) 将带位置编码的 tokens 输入到 TransformerEncoder 进行上下文编码，得到 (B, 2*L, d_model) 的输出。
# 5) 使用 AdaptivePooling（基于 MultiheadAttention 的可学习 query）对序列做注意力池化，聚合为
#    (B, d_model) 的单向量表示；然后通过线性层（并可选 LayerNorm）映射到最终的 embedding_size
#    维度，得到 (B, embedding_size) 的输出。
#
# 英文：
# TransformerEmbedding encodes input state and action sequences into fixed-size embedding vectors.
# The processing steps are:
# 1) Two separate MLPs (state_encoder / action_encoder) map raw `state_dim` and `action_dim` to the
#    same token dimension `d_model`, producing state_tokens and action_tokens with shape (B, L, d_model).
# 2) Interleave state_tokens and action_tokens by time step to form a token sequence of length 2*L:
#    tokens shape is (B, 2*L, d_model), with order [s0, a0, s1, a1, ...].
# 3) Add learnable positional embeddings (LearnablePositionalEmbedding) and apply dropout.
# 4) Feed the position-augmented tokens into a TransformerEncoder to model context, producing
#    a tensor of shape (B, 2*L, d_model).
# 5) Aggregate the sequence using AdaptivePooling (a learnable-query multi-head attention pooling)
#    to get a single vector (B, d_model), then map (and optionally LayerNorm) to the final embedding
#    of shape (B, embedding_size).



class LearnablePositionalEmbedding(nn.Module):
    """
    可学习的位置编码类。

    说明：
    - 与经典的固定正弦/余弦位置编码不同，这里使用一个可学习的参数矩阵 `pos_embedding`，大小为
      `(max_seq_len, d_model)`，在训练中学习位置向量。
    - 在 forward 中会根据输入序列长度切片并加到输入特征上，形状为 `(batch_size, seq_len, d_model)`。

    参数：
    - d_model: 每个位置向量的维度（也等于 Transformer 的隐空间维度）
    - max_seq_len: 允许的最大序列长度（超过会断言失败）
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 创建并初始化可学习的位置编码参数矩阵，形状 (max_seq_len, d_model)
        # 使用正态初始化使参数有小随机值，便于训练稳定开始
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable positional embedding to input tensor

        输入：x 形状为 (batch_size, seq_len, d_model)
        返回：与 x 形状相同的张量，已添加可学习的位置编码
        """
        batch_size, seq_len, d_model = x.shape

        # 验证输入维度符合初始化时的 d_model
        assert d_model == self.d_model, f"Input d_model {d_model} != expected {self.d_model}"
        # 验证序列长度不超过最大支持长度
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} > max {self.max_seq_len}"

        # 取前 seq_len 个位置编码并扩展 batch 维度后相加
        pos_emb = self.pos_embedding[:seq_len]  # (seq_len, d_model)
        x_with_pos = x + pos_emb.unsqueeze(0)  # 广播到 (batch_size, seq_len, d_model)

        return x_with_pos


class AdaptivePooling(nn.Module):
    """
    自适应池化层（AdaptivePooling）。

    作用：使用一个可学习的 query 向量对输入序列做多头注意力（MultiheadAttention），
    将变长序列聚合为单个定长向量。支持传入 key_padding_mask 来忽略 padding 部分，并
    可选择记录 attention 权重用于可视化或分析。

    参数：
    - d_model: 输入特征维度
    - num_heads: 注意力头数
    - dropout: 注意力模块的 dropout
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, 
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.last_attention_weights = None
        self.enable_attention_recording = False
    
    def enable_recording(self, enable: bool = True):
        """控制是否记录 attention 权重。

        enable=True: 在 forward 时返回并保存最近一次 attention 权重；
        enable=False: 不保存权重以节约内存。
        """
        self.enable_attention_recording = enable
    
    def get_attention_weights(self):
        """返回最近一次保存的 attention 权重（或 None）。

        返回形状与 MultiheadAttention 输出一致，通常为 (B, num_heads, query_len, key_len)
        （具体取决于 average_attn_weights 的语义和 PyTorch 版本）。
        """
        return self.last_attention_weights
    
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, d_model = x.shape
        query = self.query.expand(B, 1, d_model)
        
        pooled, attn_weights = self.attention(
            query=query, key=x, value=x,
            key_padding_mask=key_padding_mask,  # 传递 mask
            need_weights=self.enable_attention_recording,
            average_attn_weights=True
        )
        
        if self.enable_attention_recording and attn_weights is not None:
            self.last_attention_weights = attn_weights.detach()
        
        pooled = self.norm(pooled).squeeze(1)
        return pooled


class TransformerEmbedding(EmbeddingBase):
    """
    TransformerEmbedding：将状态-动作序列编码为固定维嵌入。

    说明：
    - 输入的 `states` 和 `actions` 形状均为 (B, L, dim)
    - 先通过独立的 MLP 将 state/action 映射到相同的 `d_model` 维度
    - 将 state/action 交替排列为长度 2*L 的 token 序列，并加上可学习的位置编码
    - 使用 TransformerEncoder 对序列建模，再用 AdaptivePooling 聚合为单向量
    - 最后通过线性层（可选 LayerNorm）得到 `embedding_size` 维度的向量

    构造函数主要参数：
    - state_dim, action_dim: 原始状态/动作特征维度
    - embedding_size: 最终输出 embedding 大小
    - d_model: Transformer 隐层维度（token 尺寸）
    - n_layer: Transformer encoder 层数
    - n_head: 多头注意力头数
    - d_ff: Transformer 前馈网络隐层大小（dim_feedforward）
    - dropout: dropout 比例
    - norm_z: 是否对输出 embedding 做 LayerNorm
    - pos_encoding_max_len: 位置编码最大长度（注意在内部乘 2，因为序列包含 state/action）
    - adaptive_pooling_heads / adaptive_pooling_dropout: 自适应池化的注意力头数与 dropout
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embedding_size: int = 64,
        d_model: int = 256,
        n_layer: int = 4,
        n_head: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        norm_z: bool = True,
        mask_embedding: bool = False,
        pos_encoding_max_len: int = 5000,
        adaptive_pooling_heads: int = 8,
        adaptive_pooling_dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(state_dim, action_dim, token_size=embedding_size, 
                        mask_embedding=mask_embedding, **kwargs)
        
        self.norm_z = norm_z
        self.d_model = d_model
        self.embedding_size = embedding_size

        # Encoders
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 恢复原来的action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 使用可学习位置编码
        self.pos_embedding = LearnablePositionalEmbedding(
            d_model=d_model, max_seq_len=pos_encoding_max_len * 2
        )
        self.dropout = nn.Dropout(dropout)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        # Pooling
        self.adaptive_pooling = AdaptivePooling(
            d_model=d_model, num_heads=adaptive_pooling_heads, 
            dropout=adaptive_pooling_dropout
        )
        
        # Output
        if norm_z:
            self.output_norm = nn.Sequential(
                nn.Linear(d_model, embedding_size),
                nn.LayerNorm(embedding_size)
            )
        else:
            self.output_norm = nn.Linear(d_model, embedding_size)
        
        print(f"Using Learnable Positional Embedding (d_model={d_model}, max_len={pos_encoding_max_len * 2})")
    
    def enable_attention_recording(self, enable: bool = True):
        self.adaptive_pooling.enable_recording(enable)
    
    def get_pooling_attention_weights(self):
        return self.adaptive_pooling.get_attention_weights()
    
    
    # 恢复原来的build_tokens，移除rewards参数
    def build_tokens(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        将状态和动作编码并构造交替排列的 token 序列。

        输入：
        - states: (B, L, state_dim)
        - actions: (B, L, action_dim)

        输出 tokens: (B, 2*L, d_model)，排列为 [s0, a0, s1, a1, ...]
        """
        B, L, _ = states.shape
        state_tokens = self.state_encoder(states)
        action_tokens = self.action_encoder(actions)

        # 创建一个空的 tokens 张量并把状态/动作交替放入
        tokens = torch.zeros(B, 2*L, self.d_model, device=states.device, dtype=states.dtype)
        tokens[:, 0::2, :] = state_tokens
        tokens[:, 1::2, :] = action_tokens
        return tokens
    
    def get_embedding(self, tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get embedding with optional attention mask support
        
        Args:
            tokens: Input tokens [B, seq_len, d_model]
            attention_mask: Optional mask [B, seq_len] where True means ignore
        """
        # 验证输入维度
        batch_size, seq_len, d_model = tokens.shape
        assert d_model == self.d_model, f"Token dimension {d_model} != expected {self.d_model}"
        
        # 应用可学习位置编码
        h = self.pos_embedding(tokens)  # (batch_size, seq_len, d_model)
        assert h.shape == tokens.shape, f"Positional embedding changed shape from {tokens.shape} to {h.shape}"
        # import hashlib
        # cpu_state = torch.random.get_rng_state()
        # cpu_hash = hashlib.sha256(cpu_state.numpy().tobytes()).hexdigest()
        # if torch.cuda.is_available():
        #     cuda_state = torch.cuda.get_rng_state()
        #     cuda_hash = hashlib.sha256(cuda_state.cpu().numpy().tobytes()).hexdigest()
        # else:
        #     cuda_hash = "n/a"
        # print(f"[DADP_DEBUG_ENCODER] PRE-DROPOUT rng_cpu={cpu_hash} rng_cuda={cuda_hash}")
        
        h = self.dropout(h)
   
        # cpu_state = torch.random.get_rng_state()
        # cpu_hash = hashlib.sha256(cpu_state.numpy().tobytes()).hexdigest()
        # if torch.cuda.is_available():
        #     cuda_state = torch.cuda.get_rng_state()
        #     cuda_hash = hashlib.sha256(cuda_state.cpu().numpy().tobytes()).hexdigest()
        # else:
        #     cuda_hash = "n/a"
        # print(f"[DADP_DEBUG_ENCODER] rng_cpu={cpu_hash} rng_cuda={cuda_hash}")
        # self._debug_rng_printed = True
        # try:
        #     matmul_precision = torch.get_float32_matmul_precision()
        # except Exception:
        #     matmul_precision = "unknown"
        # print(
        #     "[DADP_DEBUG_ENCODER] "
        #     f"training={self.training} dtype={h.dtype} device={h.device} "
        #     f"matmul_precision={matmul_precision}"
        # )
        # self._debug_printed = True
        # import os 
        # import hashlib
        # max_params_env = os.getenv("DADP_DEBUG_ENCODER_WEIGHTS_MAX", "0")
        # try:
        #     max_params = int(max_params_env)
        # except Exception:
        #     max_params = 0
        # print("[DADP_DEBUG_ENCODER] encoder parameter hashes")
        # count = 0
        # for name, param in self.named_parameters():
        #     data = param.detach().cpu().numpy()
        #     data = data.tobytes(order="C")
        #     digest = hashlib.sha256(data).hexdigest()
        #     print(f"  {name}: {digest}")
        #     count += 1
        #     if max_params > 0 and count >= max_params:
        #         break
        # print(f"[DADP_DEBUG_ENCODER] param_count={count}")
        # self._debug_weights_printed = True
        # Apply transformer with optional attention mask
        h = self.transformer_encoder(h, src_key_padding_mask=attention_mask)  # (batch_size, seq_len, d_model)
        
        # Apply adaptive pooling with the same mask
        pooled = self.adaptive_pooling(h, key_padding_mask=attention_mask)  # (batch_size, d_model)
        embedding = self.output_norm(pooled)  # (batch_size, embedding_size)
        
        return embedding
    
    # 恢复原来的forward，移除rewards参数
    def forward(self, states: torch.Tensor, actions: torch.Tensor, 
               attention_mask: Optional[torch.Tensor] = None,
               use_random_mask: bool = False, 
               min_visible_length: int = 2) -> torch.Tensor:
        """Forward with attention mask support"""
        if self.mask_embedding:
            batch_size = states.shape[0]
            return torch.zeros(batch_size, self.embedding_size, 
                             device=states.device, dtype=states.dtype)
        
        tokens = self.build_tokens(states, actions)
        
        # Generate random mask if requested and no mask provided
        if use_random_mask and attention_mask is None:
            batch_size, seq_len = tokens.shape[:2]
            attention_mask = self.generate_random_attention_mask(
                batch_size, seq_len, min_visible_length
            )
            attention_mask = attention_mask.to(tokens.device)
        
        embedding = self.get_embedding(tokens, attention_mask)
        return embedding

