from .base.pooling_base import PoolingBase


class ClassToken(PoolingBase):
    def get_output_shape(self, input_shape):
        _, dim = input_shape
        return dim,

    def forward(self, all_tokens, *_, **__):
        assert self.static_ctx["num_aux_tokens"] > 0
        # concatenation requires some special handling for normalization as concat of normalized vectors is not
        # normed and ViT applies norm to the pooled token, so concatenating multiple tokens requires more parameters
        # in the last layernorm of the ViT
        return all_tokens[:, :self.static_ctx["num_aux_tokens"]].flatten(start_dim=1)

    def __str__(self):
        return type(self).__name__
