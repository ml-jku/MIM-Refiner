from functools import partial

import einops
from kappamodules.init import init_truncnormal_zero_bias
from torch import nn

from models.base.single_model_base import SingleModelBase


class BatchNorm1d(SingleModelBase):
    def __init__(self, **kwargs):
        super().__init__(is_frozen=True, allow_frozen_train_mode=True, **kwargs)
        if len(self.input_shape) == 1:
            self.pre_pattern = self.post_pattern = self.post_pattern_kwargs = None
        elif len(self.input_shape) == 3:
            self.pre_pattern = "bs dim h w -> (bs h w) dim"
            self.post_pattern = "(bs h w) dim -> bs dim h w"
            self.post_pattern_kwargs = dict(h=self.input_shape[1], w=self.input_shape[2])
        else:
            raise NotImplementedError
        self.norm = nn.BatchNorm1d(self.input_shape[0], affine=False)

    def forward(self, x):
        if self.pre_pattern is not None:
            x = einops.rearrange(x, self.pre_pattern)
        x = self.norm(x)
        if self.post_pattern is not None:
            x = einops.rearrange(x, self.post_pattern, **self.post_pattern_kwargs)
        return x