from .base.pooling_base import PoolingBase
from utils.factory import create
from models.poolings import pooling_from_kwargs
from models.extractors import extractor_from_kwargs
from models.extractors.vit_block_extractor import VitBlockExtractor
from models.poolings.to_image import ToImage
from models.poolings.to_image_concat_aux import ToImageConcatAux

class ExtractorPooling(PoolingBase):
    def __init__(self, extractor, pooling, static_ctx):
        super().__init__(static_ctx=static_ctx)
        self.extractor = create(extractor, extractor_from_kwargs, static_ctx=self.static_ctx)
        self.pooling = create(pooling, pooling_from_kwargs, static_ctx=self.static_ctx)

    def get_output_shape(self, input_shape):
        output_shape = self.pooling.get_output_shape(input_shape)
        if isinstance(self.extractor, VitBlockExtractor):
            if self.extractor.block_indices is not None:
                if len(self.extractor.block_indices) > 1:
                    if len(output_shape) == 1:
                        output_shape = (output_shape[0] * len(self.extractor.block_indices),)
                    elif len(output_shape) == 3:
                        assert isinstance(self.pooling, (ToImage, ToImageConcatAux))
                        output_shape = (output_shape[0] * len(self.extractor.block_indices), *output_shape[1:])
                    else:
                        raise NotImplementedError
            elif self.extractor.num_last_blocks is not None:
                if len(output_shape) == 1:
                    output_shape = (output_shape[0] * self.extractor.num_last_blocks,)
                elif len(output_shape) == 3:
                    assert isinstance(self.pooling, (ToImage, ToImageConcatAux))
                    output_shape = (output_shape[0] * self.extractor.num_last_blocks, *output_shape[1:])
                else:
                    raise NotImplementedError
        return output_shape

    def forward(self, *_, **__):
        output = self.extractor.extract()
        return self.pooling(output)

    def register_hooks(self, model):
        self.extractor.register_hooks(model)

    def enable_hooks(self):
        self.extractor.enable_hooks()

    def disable_hooks(self):
        self.extractor.disable_hooks()

    def clear_extractor_outputs(self):
        self.extractor.outputs.clear()

    def __str__(self):
        return f"{type(self).__name__}(extractor={self.extractor},pooling={self.pooling})"
