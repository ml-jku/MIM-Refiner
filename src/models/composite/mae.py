import torch

from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from utils.factory import create


class Mae(CompositeModelBase):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = create(
            encoder,
            model_from_kwargs,
            input_shape=self.input_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            data_container=self.data_container,
        )
        assert self.encoder.output_shape is not None
        self.decoder = create(
            decoder,
            model_from_kwargs,
            input_shape=self.encoder.output_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            data_container=self.data_container,
        )

    @property
    def submodels(self):
        return dict(encoder=self.encoder, decoder=self.decoder)

    # noinspection PyMethodOverriding
    def forward(self, x, mask_generator=None, idx=None):
        outputs = {}
        if mask_generator is None:
            # no mask generator -> unmasked encoder forward pass
            assert not self.training
            _ = self.encoder(x)

        # encoder forward
        encoder_output = self.encoder(
            x,
            idx=idx,
            mask_generator=mask_generator,
        )
        outputs["mask"] = encoder_output["mask"]

        # decoder forward
        decoder_output = self.decoder(
            encoder_output["main"],
            ids_restore=encoder_output["ids_restore"],
        )
        outputs["prediction"] = decoder_output["main"]
        outputs["patch_size"] = self.encoder.patch_size
        
        return outputs
