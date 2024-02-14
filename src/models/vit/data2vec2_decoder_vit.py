from functools import partial

import einops
import numpy as np
import torch
from torch import nn
from kappamodules.functional.pos_embed import get_sincos_pos_embed_from_seqlens
from kappamodules.init import init_norm_as_noaffine, init_xavier_uniform_zero_bias
from kappamodules.vit import VitBlock, VitSeperateNorm

from models.base.single_model_base import SingleModelBase
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier


class Data2vec2DecoderVit(SingleModelBase):
    def __init__(
            self, 
            dim, 
            depth,
            num_attn_heads,
            eps=1e-6,
            init_weights="xavier_uniform",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = self.static_ctx["patch_size"]
        encoder_input_shape = self.static_ctx["input_shape"]
        assert len(self.patch_size) == len(encoder_input_shape) - 1
        self.dim = dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.eps = eps

        # decoder doesn't produce original image shape but encoder dim
        num_tokens, encoder_dim = self.input_shape
        self.output_shape = (num_tokens, encoder_dim)
        self.num_aux_tokens = self.static_ctx["num_aux_tokens"]

        # embed
        self.embed = nn.Linear(encoder_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))

        # fixed pos embedding
        self.seqlens = self.static_ctx["sequence_lengths"]
        pos_embed = get_sincos_pos_embed_from_seqlens(seqlens=self.seqlens, dim=self.dim)
        self.register_buffer("pos_embed", einops.rearrange(pos_embed, "... dim -> 1 (...) dim"))

        # norm ctor
        norm_ctor = nn.LayerNorm

        # blocks
        self.blocks = nn.ModuleList([
            VitBlock(
                dim=self.dim,
                num_heads=self.num_attn_heads,
                norm_ctor=norm_ctor,
                eps=eps,
                init_weights=init_weights,
            )
            for _ in range(depth)
        ])

        # decoder to patch
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.pred = nn.Linear(dim, encoder_dim)

    def model_specific_initialization(self):
        # mask token
        torch.nn.init.normal_(self.mask_token, std=.02)
        # layers
        init_xavier_uniform_zero_bias(self.embed)
        init_xavier_uniform_zero_bias(self.pred)
        # norms
        init_norm_as_noaffine(self.norm)

    @staticmethod
    def get_model_specific_param_group_modifiers():
        # ExcludeFromWdByNameModifier(name="pos_embed") -> not used because pos_embed is never learnable
        return [ExcludeFromWdByNameModifier(name="mask_token")]

    # noinspection PyMethodOverriding
    def forward(self, x, ids_restore):
        outputs = {}

        # embed tokens
        x = self.embed(x)

        # extract shapes
        bs, num_input_tokens, dim = x.shape
        _, total_num_patches = ids_restore.shape
        num_hidden_patches = total_num_patches - (num_input_tokens - self.num_aux_tokens)

        # remove aux_tokens before padding with mask tokens
        aux_tokens = x[:, :self.num_aux_tokens, :]
        x = x[:, self.num_aux_tokens:, :]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(bs, num_hidden_patches, 1)
        x = torch.cat([x, mask_tokens], dim=1)
        # unshuffle
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, dim))
        # add pos embed
        x = x + self.pos_embed

        # append aux tokens
        x = torch.cat([aux_tokens, x], dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # last layer
        x = self.norm(x)
        x = self.pred(x)

        # outputs
        outputs["main"] = x
        return outputs
