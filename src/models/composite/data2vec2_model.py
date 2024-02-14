import torch.nn.functional as F
import einops
from collections import defaultdict
from copy import deepcopy
from functools import partial

import torch
from kappamodules.layers.drop_path import DropPath
from kappaschedules import object_to_schedule
from torch import nn

from initializers import initializer_from_kwargs
from models import model_from_kwargs, prepare_momentum_kwargs
from models.base.composite_model_base import CompositeModelBase
from models.poolings.base.handle_extractor_pooling import handle_extractor_pooling
from utils.factory import create, create_collection
from utils.model_utils import update_ema, copy_params
from utils.schedule_utils import get_value_or_default
from kappamodules.layers import ParamlessBatchNorm1d
from models.extractors.vit_block_extractor import VitBlockExtractor
from models.extractors.finalizers.stack_finalizer import StackFinalizer

class Data2vec2Model(CompositeModelBase):
    def __init__(
            self,
            encoder,
            decoder,
            num_blocks_to_average,
            target_factor=None,
            target_factor_schedule=None,
            copy_ema_on_start=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = create(
            encoder,
            model_from_kwargs,
            input_shape=self.input_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
        )
        assert self.encoder.output_shape is not None

        self.copy_ema_on_start = copy_ema_on_start
        self.target_factor = target_factor
        self.target_factor_schedule = object_to_schedule(
            target_factor_schedule,
            batch_size=self.update_counter.effective_batch_size if self.update_counter is not None else None,
            updates_per_epoch=self.update_counter.updates_per_epoch if self.update_counter is not None else None,
            start_value=target_factor,
        )
        assert self.target_factor is not None or self.target_factor_schedule is not None
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

        # initialize encoder EMA
        if self.target_factor is not None:
            assert isinstance(encoder, (dict, partial))
            momentum_encoder = prepare_momentum_kwargs(encoder)
            if isinstance(encoder, dict) and len(momentum_encoder) == 0:
                # initialize momentum_encoder via checkpoint_kwargs of encoder
                assert "initializers" in encoder and encoder["initializers"][0].get("use_checkpoint_kwargs", False)
                initializer_kwargs = deepcopy(encoder["initializers"][0])
                initializer_kwargs.pop("use_checkpoint_kwargs")
                initializer = initializer_from_kwargs(**initializer_kwargs, path_provider=self.path_provider)
                momentum_encoder = initializer.get_model_kwargs()
            self.momentum_encoder = create(
                momentum_encoder,
                model_from_kwargs,
                input_shape=self.input_shape,
                update_counter=self.update_counter,
                path_provider=self.path_provider,
                dynamic_ctx=self.dynamic_ctx,
                static_ctx=self.static_ctx,
                is_frozen=True,
                allow_frozen_train_mode=True,
            )
            # disable drop_path in momentum_encoder: momentum_encoder is kept in train mode to
            # track batchnorm stats (following MoCoV3) -> drop_path would be applied in forward pass
            assert self.momentum_encoder.is_frozen and self.momentum_encoder.training
            self.logger.info(f"disabling DropPath for momentum_encoder")
            for m in self.momentum_encoder.modules():
                if isinstance(m, DropPath):
                    m.drop_prob = 0.
        else:
            self.momentum_encoder = None

        # register extractors
        self.num_blocks_to_average = num_blocks_to_average
        self.extractor = VitBlockExtractor(
            num_last_blocks=num_blocks_to_average,
            finalizer=StackFinalizer(dim=0),
        )
        self.extractor.register_hooks(self.momentum_encoder)
        self.extractor.disable_hooks()

    @property
    def submodels(self):
        return dict(encoder=self.encoder, momentum_encoder=self.momentum_encoder, decoder=self.decoder)

    # noinspection PyMethodOverriding
    def forward(self, x, num_masks=1, mask_generator=None, idx=None):
        outputs = {}
        if mask_generator is None:
            raise NotImplementedError

        # encoder forward
        with torch.no_grad():
            with self.extractor:
                _ = self.momentum_encoder(x)
            target = self.extractor.extract()
            # instance norm (original converts to fp32 probably for numerical reasons)
            target = einops.rearrange(target, "num_blocks bs seqlen dim -> (num_blocks bs) dim seqlen")
            target = F.instance_norm(target.float())
            target = einops.rearrange(
                target,
                "(num_blocks bs) dim seqlen -> num_blocks bs seqlen dim",
                num_blocks=self.num_blocks_to_average,
            )
            # average
            target = target.float().mean(dim=0)
            # layer norm
            # (not sure why .float is not called here in original implementation,
            # either mixed precision doesnt cast it, its not needed or its a bug)
            target = F.layer_norm(target, target.shape[-1:])
            # repeat for masks
            target = einops.repeat(target, "bs seqlen dim -> (num_masks bs) seqlen dim", num_masks=num_masks)
            outputs["target"] = target

        # encoder forward
        encoder_output = self.encoder(
            einops.repeat(x, "bs c h w -> (num_masks bs) c h w", num_masks=num_masks),
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

        # add aux tokens from encoder if conv decoder is used
        missing_tokens = outputs["target"].size(1) - outputs["prediction"].size(1)
        if missing_tokens > 0:
            encoder_aux_tokens = encoder_output["main"][:, :missing_tokens]
            outputs["prediction"] = torch.concat([encoder_aux_tokens, outputs["prediction"]], dim=1)

        return outputs

    # this is typically not done
    # def model_specific_initialization(self):
    #     self.logger.info("initializing teacher_model with parameters from model")
    #     copy_params(self.encoder, self.momentum_encoder)
    #     super().model_specific_initialization()

    def after_update_step(self):
        target_factor = get_value_or_default(
            default=self.target_factor,
            schedule=self.target_factor_schedule,
            update_counter=self.update_counter,
        )
        # MoCoV3 tracks batchnorm stats from the ema model instead of copying it from the source model
        update_ema(self.encoder, self.momentum_encoder, target_factor, copy_buffers=False)