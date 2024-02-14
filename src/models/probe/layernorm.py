from functools import partial

import einops
from kappamodules.init import init_truncnormal_zero_bias
from torch import nn

from models.base.single_model_base import SingleModelBase
from models.poolings.to_image import ToImage
from models.poolings.to_image_concat_aux import ToImageConcatAux
from models.poolings.extractor_pooling import ExtractorPooling

class LayerNorm(SingleModelBase):
    def __init__(self, pooling=None, **kwargs):
        super().__init__(is_frozen=True, **kwargs)
        self.norm = nn.LayerNorm(self.input_shape[0], elementwise_affine=False)
        if isinstance(pooling, (ToImage, ToImageConcatAux)):
            self.pre_pattern = "batch dim height width -> batch height width dim"
            self.post_pattern = "batch height width dim -> batch dim height width"
        elif isinstance(pooling, ExtractorPooling):
            assert isinstance(pooling.pooling, (ToImage, ToImageConcatAux))
            self.pre_pattern = "batch dim height width -> batch height width dim"
            self.post_pattern = "batch height width dim -> batch dim height width"
        else:
            raise NotImplementedError

    def forward(self, x):
        x = einops.rearrange(x, self.pre_pattern)
        x = self.norm(x)
        x = einops.rearrange(x, self.post_pattern)
        return x
