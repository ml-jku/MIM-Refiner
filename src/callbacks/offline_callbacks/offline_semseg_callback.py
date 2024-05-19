from functools import partial

from torchmetrics.functional.classification import multiclass_jaccard_index

from callbacks.base.periodic_callback import PeriodicCallback


class OfflineSemsegCallback(PeriodicCallback):
    def __init__(self, dataset_key, **kwargs):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.__config_id = None
        self.num_classes = None
        self.dataset_len = None

    def _register_sampler_configs(self, trainer):
        self.__config_id = self._register_sampler_config_from_key(key=self.dataset_key, mode="x semseg")

    def _before_training(self, **kwargs):
        self.num_classes = self.data_container.get_dataset("train").getdim("semseg")
        self.dataset_len = len(self.data_container.get_dataset(self.dataset_key))
        assert 2 < self.num_classes

    def _forward(self, batch, model, trainer):
        (x, target), _ = batch
        x = x.to(model.device, non_blocking=True)
        target = target.to(model.device, non_blocking=True)
        with trainer.autocast_context:
            predictions = model(x)
        mious = {
            key: multiclass_jaccard_index(
                preds=preds,
                target=target,
                num_classes=self.num_classes,
                average="micro",
                ignore_index=-1,
            ).repeat(len(x))
            for key, preds in predictions.items()
        }
        return mious

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer, batch_size, data_iter, **_):
        mious = self.iterate_over_dataset(
            forward_fn=partial(self._forward, model=model, trainer=trainer),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )

        # log
        for name, miou in mious.items():
            miou = miou.mean()
            key = f"miou/{self.dataset_key}/{name}"
            self.logger.info(f"{key}: {miou:.6f}")
            self.writer.add_scalar(key, miou)
