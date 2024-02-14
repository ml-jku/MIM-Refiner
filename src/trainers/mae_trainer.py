from torch import nn
from kappadata.wrappers import ModeWrapper

from losses import loss_fn_from_kwargs
from models.vit.mask_generators import mask_generator_from_kwargs
from utils.factory import create
from .base.sgd_trainer import SgdTrainer


class MaeTrainer(SgdTrainer):
    def __init__(self, loss_function, mask_generator, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = create(loss_function, loss_fn_from_kwargs, update_counter=self.update_counter)
        self.mask_generator = create(
            mask_generator,
            mask_generator_from_kwargs,
            update_counter=self.update_counter,
        )

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

        def forward(self, batch, mask_generator=None, reduction="mean"):
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
            else:
                forward_kwargs["mask_generator"] = mask_generator
            model_outputs = self.model(x, idx=idx, **forward_kwargs)

            # calculate loss
            loss = self.trainer.loss_fn(
                prediction=model_outputs["prediction"],
                target=x,
                mask=model_outputs["mask"],
                patch_size=model_outputs["patch_size"],
                reduction=reduction,
            )
            yield dict(total=loss, reconstruction=loss), outputs
