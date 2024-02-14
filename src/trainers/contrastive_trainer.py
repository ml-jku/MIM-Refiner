from functools import cached_property
import einops
from torch import nn
from kappadata.utils.multi_crop_utils import concat_same_shape_inputs
from kappadata.wrappers import ModeWrapper

from callbacks.online_callbacks.update_output_callback import UpdateOutputCallback
from losses import loss_fn_from_kwargs
from utils.factory import create_collection, create
from .base.sgd_trainer import SgdTrainer
from kappaschedules import object_to_schedule
from utils.schedule_utils import get_value_or_default


class ContrastiveTrainer(SgdTrainer):
    def __init__(self, loss_functions, loss_weight_schedules=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_functions = create_collection(
            loss_functions,
            loss_fn_from_kwargs,
            update_counter=self.update_counter,
        )
        self.loss_weight_schedules = {}
        if loss_weight_schedules is not None:
            for head_name, loss_weight_schedule in loss_weight_schedules.items():
                self.loss_weight_schedules[head_name] = object_to_schedule(
                    loss_weight_schedule,
                    batch_size=self.update_counter.effective_batch_size,
                    updates_per_epoch=self.update_counter.updates_per_epoch,
                )


    def get_trainer_callbacks(self, model=None):
        return [
            UpdateOutputCallback(
                patterns=["nn-", "global_alignment/", "local_alignment/"],
                verbose=False,
                every_n_updates=self.track_every_n_updates,
                every_n_samples=self.track_every_n_samples,
                **self.get_default_callback_kwargs(),
            ),
            UpdateOutputCallback(
                patterns=["nn-", "global_alignment/", "local_alignment/"],
                verbose=True,
                **self.get_default_callback_intervals(),
                **self.get_default_callback_kwargs(),
            ),
            UpdateOutputCallback(
                patterns=["target_factor/", "loss_weights/"],
                reduce="last",
                verbose=False,
                every_n_updates=self.track_every_n_updates,
                every_n_samples=self.track_every_n_samples,
                **self.get_default_callback_kwargs(),
            ),
        ]

    @property
    def output_shape(self):
        return None

    @cached_property
    def dataset_mode(self):
        try:
            self.data_container.get_dataset().getitem_confidence(0)
            return "index x class confidence"
        except AttributeError:
            return "index x class"

    def get_trainer_model(self, model):
        return self.Model(model=model, trainer=self)

    class Model(nn.Module):
        def __init__(self, model, trainer):
            super().__init__()
            self.model = model
            self.trainer = trainer

        def forward(self, batch, reduction="mean"):
            # prepare data
            batch, ctx = batch
            x = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="x", batch=batch)
            cls = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="class", batch=batch)
            idx = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="index", batch=batch)
            if ModeWrapper.has_item(mode=self.trainer.dataset_mode, item="confidence"):
                confidence = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="confidence", batch=batch)
            else:
                confidence = None
            if isinstance(x, list):
                x = [xx.to(self.model.device, non_blocking=True) for xx in x]
            else:
                assert not self.training
                x = [x.to(self.model.device, non_blocking=True)]
            cls = cls.to(self.model.device, non_blocking=True)
            idx = idx.to(self.model.device, non_blocking=True)
            if confidence is not None:
                confidence = confidence.to(self.model.device, non_blocking=True)
            x, batch_size = concat_same_shape_inputs(x)

            # MUGS augmentation not supported
            if "is_weak_global_aug" in ctx:
                raise NotImplementedError

            loss_weights = {}
            for head_name in self.trainer.loss_functions.keys():
                loss_weights[head_name] = get_value_or_default(
                    default=1.,
                    schedule=self.trainer.loss_weight_schedules.get(head_name, None),
                    update_counter=self.trainer.update_counter,
                    training=self.training,
                )

            # model forward pass
            model_outputs = self.model(x, idx=idx, cls=cls, confidence=confidence)

            # iterate over heads
            total_loss = 0
            losses = {}
            infos = {}
            for head_name in self.trainer.loss_functions.keys():
                # postprocess outputs (heads can return metrics, e.g. NN-accuracy for NNCLR)
                head_outputs = model_outputs[head_name]
                if "metrics" in head_outputs:
                    infos.update({
                        f"{key}/{head_name}": value
                        for key, value in head_outputs.pop("metrics").items()
                    })

                # convert projected/predicted to (bs, num_views, dim) for loss
                for key in head_outputs.keys():
                    # heads can return multiple "projected" values (e.g. NNCLR returns pre-swap/post-swap projected)
                    if "projected" in key:
                        head_outputs[key] = einops.rearrange(
                            head_outputs[key],
                            "(num_global_views bs) dim -> bs num_global_views dim",
                            bs=batch_size,
                        )
                    if key == "predicted":
                        head_outputs[key] = einops.rearrange(
                            head_outputs[key],
                            "(num_views bs) dim -> bs num_views dim",
                            bs=batch_size,
                        )

                # calculate losses
                head_losses, loss_infos = self.trainer.loss_functions[head_name](
                    **head_outputs,
                    reduction=reduction,
                )
                total_loss = total_loss + head_losses["total"] * loss_weights[head_name]
                losses.update({f"heads.{head_name}.{key}": value for key, value in head_losses.items()})
                infos.update({f"{key}/{head_name}": value for key, value in loss_infos.items()})
            infos.update({f"loss_weight/{key}": value for key, value in loss_weights.items()})
            yield dict(total=total_loss, **losses), infos
