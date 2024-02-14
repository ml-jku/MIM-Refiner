from torch import nn
from kappadata.wrappers import ModeWrapper

from losses import loss_fn_from_kwargs
from models.vit.mask_generators import mask_generator_from_kwargs
from utils.factory import create
from .base.sgd_trainer import SgdTrainer


class Data2vec2Trainer(SgdTrainer):
    def __init__(self, loss_function, mask_generator, num_masks, aux_loss_weight, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = create(loss_function, loss_fn_from_kwargs, update_counter=self.update_counter)
        self.mask_generator = create(
            mask_generator,
            mask_generator_from_kwargs,
            update_counter=self.update_counter,
        )
        self.num_masks = num_masks
        self.aux_loss_weight = aux_loss_weight

    @property
    def dataset_mode(self):
        return "index x"

    def get_trainer_model(self, model):
        return self.Model(model=model, trainer=self)

    class Model(nn.Module):
        def __init__(self, model, trainer):
            super().__init__()
            self.model = model
            self.trainer = trainer

        def forward(self, batch, num_masks=None, mask_generator=None, reduction="mean"):
            outputs = {}
            # prepare data
            batch, ctx = batch
            idx = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="index", batch=batch)
            x = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="x", batch=batch)
            idx = idx.to(self.model.device, non_blocking=True)
            x = x.to(self.model.device, non_blocking=True)

            # model forward pass
            # for evaluation generators can be provided (if not provided -> no generator)
            forward_kwargs = {}
            if self.model.training:
                assert mask_generator is None
                forward_kwargs["mask_generator"] = self.trainer.mask_generator
                forward_kwargs["num_masks"] = self.trainer.num_masks
            else:
                forward_kwargs["mask_generator"] = mask_generator
                if num_masks is not None:
                    forward_kwargs["num_masks"] = num_masks
            model_outputs = self.model(x, idx=idx, **forward_kwargs)

            # calculate loss
            loss_patches, loss_aux = self.trainer.loss_fn(
                prediction=model_outputs["prediction"],
                target=model_outputs["target"],
                mask=model_outputs["mask"],
                reduction=reduction,
            )
            # weight losses
            loss = loss_patches
            if loss_aux is not None:
                loss = loss + self.trainer.aux_loss_weight * loss_aux
            losses = dict(total=loss, patch=loss_patches)
            if loss_aux is not None:
                losses["aux"] = loss_aux
            yield losses, outputs
