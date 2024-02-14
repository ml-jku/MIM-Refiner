import torch
from torch import nn

from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from models.extractors.vit_block_extractor import VitBlockExtractor
from utils.factory import create


class MaeDecoderPerBlock(CompositeModelBase):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = create(
            encoder,
            model_from_kwargs,
            is_frozen=True,
            input_shape=self.input_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            data_container=self.data_container,
        )
        assert self.encoder.output_shape is not None
        self.decoders = nn.ModuleList(
            [
                create(
                    decoder,
                    model_from_kwargs,
                    input_shape=self.encoder.output_shape,
                    update_counter=self.update_counter,
                    path_provider=self.path_provider,
                    dynamic_ctx=self.dynamic_ctx,
                    static_ctx=self.static_ctx,
                    data_container=self.data_container,
                )
                for _ in range(self.encoder.depth)
            ],
        )
        self.extractor = VitBlockExtractor(finalizer=None)
        self.extractor.register_hooks(self.encoder)
        self.extractor.disable_hooks()

    @property
    def submodels(self):
        return dict(encoder=self.encoder, **{f"decoder{i}": self.decoders[i] for i in range(len(self.decoders))})

    # noinspection PyMethodOverriding
    def forward(self, x, mask_generator=None, idx=None):
        if mask_generator is None:
            raise NotImplementedError

        # encoder forward
        with torch.no_grad():
            with self.extractor:
                encoder_output = self.encoder(
                    x,
                    idx=idx,
                    mask_generator=mask_generator,
                )
            features = self.extractor.extract()

        # decoder forward
        predictions = [
            self.decoders[i](
                features[i],
                ids_restore=encoder_output["ids_restore"],
            )["main"]
            for i in range(len(self.decoders))
        ]

        return dict(
            mask=encoder_output["mask"],
            patch_size=self.encoder.patch_size,
            predictions=predictions,
        )
