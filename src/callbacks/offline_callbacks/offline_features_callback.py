from functools import partial

import torch
from kappadata.wrappers import ModeWrapper

from callbacks.base.periodic_callback import PeriodicCallback
from models.extractors import extractor_from_kwargs
from models.extractors.base.forward_hook import StopForwardException
from utils.factory import create_collection
from utils.object_from_kwargs import objects_from_kwargs


class OfflineFeaturesCallback(PeriodicCallback):
    def __init__(self, dataset_key, extractors, forward_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.extractors = extractors
        self.forward_kwargs = objects_from_kwargs(forward_kwargs)
        self.__config_id = None
        self.dataset_mode = None
        self.out = self.path_provider.stage_output_path / "features"

    def _register_sampler_configs(self, trainer):
        self.dataset_mode = ModeWrapper.add_item(mode=trainer.dataset_mode, item="class")
        self.__config_id = self._register_sampler_config_from_key(key=self.dataset_key, mode=self.dataset_mode)

    def _before_training(self, model, **kwargs):
        self.out.mkdir(exist_ok=True)
        self.extractors = create_collection(self.extractors, extractor_from_kwargs, static_ctx=model.static_ctx)
        for extractor in self.extractors:
            extractor.register_hooks(model)
            extractor.disable_hooks()

    def _forward(self, batch, trainer_model, trainer):
        features = {}
        generator = trainer_model(batch=batch, **self.forward_kwargs)
        with trainer.autocast_context:
            try:
                next(generator)
            except StopForwardException:
                pass
        for extractor in self.extractors:
            features[str(extractor)] = extractor.extract().cpu()
        batch, _ = batch  # remove ctx
        target = ModeWrapper.get_item(mode=self.dataset_mode, item="class", batch=batch)
        return features, target.clone()

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer_model, trainer, batch_size, data_iter, **_):
        # setup
        for extractor in self.extractors:
            extractor.enable_hooks()

        # foward
        features, targets = self.iterate_over_dataset(
            forward_fn=partial(self._forward, trainer_model=trainer_model, trainer=trainer),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )

        # store
        targets_uri = self.out / f"{self.update_counter.cur_checkpoint}_targets.th"
        self.logger.info(f"saving targets to {targets_uri}")
        torch.save(targets, self.out / targets_uri)
        for key, value in features.items():
            features_uri = self.out / f"{self.update_counter.cur_checkpoint}_features_{key}.th"
            self.logger.info(f"saving features to {features_uri}")
            torch.save(value, features_uri)

        # cleanup
        for extractor in self.extractors:
            extractor.disable_hooks()
