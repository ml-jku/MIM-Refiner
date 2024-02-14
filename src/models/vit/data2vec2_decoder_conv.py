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
from einops.layers.torch import Rearrange
from modules.ssl.data2vec2_conv_decoder_block import Data2vec2ConvDecoderBlock

class Data2vec2DecoderConv(SingleModelBase):
    def __init__(
            self, 
            dim, 
            depth,
            kernel_size,
            groups,
            eps=1e-6,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = self.static_ctx["patch_size"]
        encoder_input_shape = self.static_ctx["input_shape"]
        assert len(self.patch_size) == len(encoder_input_shape) - 1
        self.dim = dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.groups = groups
        self.eps = eps

        # decoder doesn't produce original image shape but encoder dim
        self.height, self.width = self.static_ctx["sequence_lengths"]
        num_tokens, encoder_dim = self.input_shape
        self.output_shape = (num_tokens, encoder_dim)
        self.num_aux_tokens = self.static_ctx["num_aux_tokens"]

        # blocks
        self.blocks = nn.ModuleList([
            Data2vec2ConvDecoderBlock(
                input_dim=encoder_dim if i == 0 else dim,
                dim=dim,
                kernel_size=kernel_size,
                groups=groups,
                eps=eps,
            )
            for i in range(depth)
        ])

        # decoder to patch
        self.pred = nn.Linear(dim, encoder_dim)

    def model_specific_initialization(self):
        self.apply(init_xavier_uniform_zero_bias)
        self.apply(init_norm_as_noaffine)

    # noinspection PyMethodOverriding
    def forward(self, x, ids_restore):
        outputs = {}

        # extract shapes
        bs, num_input_tokens, dim = x.shape
        _, total_num_patches = ids_restore.shape
        num_hidden_patches = total_num_patches - (num_input_tokens - self.num_aux_tokens)

        # remove aux_tokens
        x = x[:, self.num_aux_tokens:, :]

        # append mask tokens to sequence
        mask_tokens = torch.randn(bs, num_hidden_patches, dim, device=x.device, dtype=x.dtype)
        x = torch.cat([x, mask_tokens], dim=1)
        # unshuffle
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, dim))

        # transformer to conv format
        x = einops.rearrange(x, "bs (height width) dim -> bs dim height width", height=self.height, width=self.width)

        # apply blocks
        for blk in self.blocks:
            x = blk(x)

        # conv to transformer format
        x = einops.rearrange(x, "bs dim height width -> bs (height width) dim")

        # last layer
        x = self.pred(x)

        # outputs
        outputs["main"] = x
        return outputs
