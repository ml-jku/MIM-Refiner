import einops
import numpy as np
from .base.pooling_base import PoolingBase


class ToImage(PoolingBase):
    def get_output_shape(self, input_shape):
        seqlen, dim = input_shape
        num_patches = seqlen - self.static_ctx["num_aux_tokens"]
        assert num_patches == np.prod(self.static_ctx["sequence_lengths"])
        return dim, *self.static_ctx["sequence_lengths"]

    def forward(self, all_tokens, *_, **__):
        patch_tokens = all_tokens[:, self.static_ctx["num_aux_tokens"]:]
        patch_tokens = patch_tokens.reshape(patch_tokens.size(0), *self.static_ctx["sequence_lengths"], -1)
        return einops.rearrange(patch_tokens, "bs ... dim -> bs dim ...")

    def __str__(self):
        return type(self).__name__
