from functools import partial

from callbacks.base.periodic_callback import PeriodicCallback
from utils.object_from_kwargs import objects_from_kwargs


class OfflineLossCallback(PeriodicCallback):
    def __init__(self, dataset_key, output_patterns_to_log=None, forward_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.forward_kwargs = objects_from_kwargs(forward_kwargs)
        self.output_patterns_to_log = output_patterns_to_log or []
        self.__config_id = None

    def _register_sampler_configs(self, trainer):
        self.__config_id = self._register_sampler_config_from_key(key=self.dataset_key, mode=trainer.dataset_mode)

    def _forward(self, batch, trainer_model, trainer):
        generator = trainer_model(batch=batch, reduction="none", **self.forward_kwargs)
        all_losses = {}
        all_outputs = {}
        i = 0
        while True:
            try:
                with trainer.autocast_context:
                    result = next(generator)
                if len(result) == 3:
                    losses, outputs, _ = result
                else:
                    losses, outputs = result
                cur_total_loss = losses.pop("total")
                all_losses.update({k: v.detach() for k, v in losses.items()})
                all_losses[f"total.{i}"] = cur_total_loss.detach()
                all_outputs.update(outputs)
                i += 1
            except StopIteration:
                break

        all_losses = {name: loss.cpu() for name, loss in all_losses.items()}
        outputs_to_log = {}
        for key, value in all_outputs.items():
            for pattern in self.output_patterns_to_log:
                if pattern in key:
                    outputs_to_log[key] = value.cpu()
        return all_losses, outputs_to_log

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer_model, trainer, batch_size, data_iter, **_):
        losses, outputs = self.iterate_over_dataset(
            forward_fn=partial(self._forward, trainer_model=trainer_model, trainer=trainer),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )

        # log losses
        for loss_name, loss in losses.items():
            assert loss.ndim == 1, "loss has to be calculated sample-wise to avoid errors through batch"
            # log loss
            mean_loss = loss.mean()
            self.writer.add_scalar(
                key=f"loss/{self.dataset_key}/{loss_name}",
                value=mean_loss,
                logger=self.logger,
                format_str=".5f",
            )
            # log difference to train loss
            train_loss = self.writer.log_cache.get(f"loss/online/{loss_name}/{self.to_short_interval_string()}", None)
            if train_loss is not None:
                self.writer.add_scalar(
                    key=f"lossdiff/{self.dataset_key}/{loss_name}",
                    value=mean_loss - train_loss,
                    logger=self.logger,
                    format_str=".5f",
                )
        # log outputs
        for name, output in outputs.items():
            assert output.ndim == 1, f"output has to be calculated sample-wise (name={name} shape={output.shape})"
            self.writer.add_scalar(
                key=f"{name}/{self.dataset_key}",
                value=output.float().mean(),
                logger=self.logger,
                format_str=".5f",
            )
