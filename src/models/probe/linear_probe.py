from functools import partial

from kappamodules.init import init_truncnormal_zero_bias
from torch import nn

from models.base.single_model_base import SingleModelBase
from models.poolings import pooling_from_kwargs
from utils.factory import create


class LinearProbe(SingleModelBase):
    def __init__(self, pooling, **kwargs):
        super().__init__(**kwargs)
        self.pooling = create(pooling, pooling_from_kwargs, static_ctx=self.static_ctx)
        input_shape = self.pooling.get_output_shape(self.input_shape)
        assert len(input_shape) == 1 and len(self.output_shape) == 1
        self.probe = nn.Linear(input_shape[0], self.output_shape[0])

    def model_specific_initialization(self):
        self.apply(partial(init_truncnormal_zero_bias, std=0.01))

    def forward(self, x, apply_pooling=True):
        if apply_pooling:
            x = self.pooling(x)
        assert x.ndim == 2
        return self.probe(x)
