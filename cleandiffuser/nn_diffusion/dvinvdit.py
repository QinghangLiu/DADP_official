from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.utils import SinusoidalEmbedding
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.nn_diffusion.dit import DiTBlock, FinalLayer1d

class DVInvDiT(BaseNNDiffusion):
    def __init__(
        self,
        in_dim: int,
        act_dim: int,
        emb_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        dropout: float = 0.0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        self.in_dim, self.emb_dim = in_dim, emb_dim
        self.d_model = d_model

        self.x_proj = nn.Linear(in_dim, d_model)
        self.map_emb = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.Mish(), nn.Linear(d_model, d_model), nn.Mish())

        self.pos_emb = SinusoidalEmbedding(d_model)
        self.pos_emb_cache = None

        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, dropout,attn_mask=attn_mask) for _ in range(depth)])
        self.final_layer = FinalLayer1d(d_model, in_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize time step embedding MLP:
        nn.init.normal_(self.map_emb[0].weight, std=0.02)
        nn.init.normal_(self.map_emb[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: torch.Tensor = None):
        """
        Input:
            x:          (b, horizon, in_dim)
            noise:      (b, )
            condition:  (b, emb_dim)

        Output:
            y:          (b, horizon, in_dim)
        """
        if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != x.shape[1]:
            self.pos_emb_cache = self.pos_emb(torch.arange(x.shape[1], device=x.device))

        x = self.x_proj(x) + self.pos_emb_cache[None,]
        emb = self.map_noise(noise)

        emb = emb + condition

        emb = self.map_emb(emb)

        for block in self.blocks:
            x = block(x, emb)
        x = self.final_layer(x, emb)
        return x