import einops
import numpy as np
import torch

from .base.pooling_base import PoolingBase


class ToImageConcatAux(PoolingBase):
    def get_output_shape(self, input_shape):
        seqlen, dim = input_shape
        num_patches = seqlen - self.static_ctx["num_aux_tokens"]
        assert num_patches == np.prod(self.static_ctx["sequence_lengths"])
        return dim * (self.static_ctx["num_aux_tokens"] + 1), *self.static_ctx["sequence_lengths"]

    def forward(self, all_tokens, *_, **__):
        assert self.static_ctx["num_aux_tokens"] > 0
        aux_tokens = all_tokens[:, :self.static_ctx["num_aux_tokens"]]
        patch_tokens = all_tokens[:, self.static_ctx["num_aux_tokens"]:]
        # expand aux tokens to number of patch tokens
        aux_tokens = einops.repeat(
            aux_tokens,
            "bs num_aux_tokens dim -> bs num_patch_tokens (num_aux_tokens dim)",
            num_patch_tokens=patch_tokens.size(1),
        )
        patch_tokens = torch.concat([aux_tokens, patch_tokens], dim=2)
        patch_tokens = patch_tokens.reshape(patch_tokens.size(0), *self.static_ctx["sequence_lengths"], -1)
        return einops.rearrange(patch_tokens, "bs ... dim -> bs dim ...")

    def __str__(self):
        return type(self).__name__
