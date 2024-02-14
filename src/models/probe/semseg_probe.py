from functools import partial

import torch.nn.functional as F
from kappamodules.init import init_truncnormal_zero_bias
from torch import nn

from models.base.single_model_base import SingleModelBase
from models.poolings.to_image import ToImage
from models.poolings.to_image_concat_aux import ToImageConcatAux
from models.poolings.extractor_pooling import ExtractorPooling
from utils.factory import create
from models.poolings import pooling_from_kwargs

class SemsegProbe(SingleModelBase):
    def __init__(self, pooling=None, **kwargs):
        super().__init__(**kwargs)
        if pooling is None:
            self.pooling = ToImage(static_ctx=self.static_ctx)
        else:
            self.pooling = create(pooling, pooling_from_kwargs, static_ctx=self.static_ctx)
        assert (
                isinstance(self.pooling, (ToImage, ToImageConcatAux))
                or
                (
                        isinstance(self.pooling, ExtractorPooling)
                        and isinstance(self.pooling.pooling, (ToImage, ToImageConcatAux))
                )
        )
        input_shape = self.pooling.get_output_shape(self.input_shape)
        assert len(input_shape) == 3 and len(self.output_shape) == 3
        self.probe = nn.Conv2d(input_shape[0], self.output_shape[0], kernel_size=1)

    def model_specific_initialization(self):
        self.apply(partial(init_truncnormal_zero_bias, std=0.01))

    def forward(self, x, apply_pooling=True):
        assert x.ndim == 4
        if apply_pooling:
            x = self.pooling(x)
        x = self.probe(x)
        x = F.interpolate(x, size=self.output_shape[1:], mode="nearest")
        return x
