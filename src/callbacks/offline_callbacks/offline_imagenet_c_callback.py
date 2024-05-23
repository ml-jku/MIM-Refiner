from collections import defaultdict
from functools import partial
from itertools import product

import numpy as np
import torch
from kappadata.common.transforms import ImagenetNoaugTransform
from kappadata.wrappers import XTransformWrapper, SubsetWrapper
from torchmetrics.functional.classification import multiclass_accuracy

from callbacks.base.periodic_callback import PeriodicCallback
from datasets.imagenet import ImageNet
from utils.kappaconfig.testrun_constants import TEST_RUN_EFFECTIVE_BATCH_SIZE


class OfflineImagenetCCallback(PeriodicCallback):
    def __init__(self, resize_size=256, center_crop_size=224, interpolation="bicubic", **kwargs):
        super().__init__(**kwargs)
        self.transform = ImagenetNoaugTransform(
            resize_size=resize_size,
            center_crop_size=center_crop_size,
            interpolation=interpolation,
        )
        self.dataset_keys = [
            f"imagenet_c_{distortion}_{level}"
            for distortion, level in product(ImageNet.IMAGENET_C_DISTORTIONS, [1, 2, 3, 4, 5])
        ]
        self.__config_ids = {}
        self.n_classes = None

    def _before_training(self, model, **kwargs):
        assert len(model.output_shape) == 1
        self.n_classes = self.data_container.get_dataset("train").getdim_class()

    def register_root_datasets(self, dataset_config_provider=None, is_mindatarun=False):
        for key in self.dataset_keys:
            if key in self.data_container.datasets:
                continue
            temp = key.replace("imagenet_c_", "")
            distortion = temp[:-2]
            level = temp[-1]
            dataset = ImageNet(
                version="imagenet_c",
                split=f"{distortion}/{level}",
                dataset_config_provider=dataset_config_provider,
            )
            dataset = XTransformWrapper(dataset=dataset, transform=ImagenetNoaugTransform())
            if is_mindatarun:
                rng = torch.Generator().manual_seed(0)
                dataset = SubsetWrapper(
                    dataset=dataset,
                    indices=torch.randperm(len(dataset), generator=rng)[:TEST_RUN_EFFECTIVE_BATCH_SIZE].tolist(),
                )
            else:
                assert len(dataset) == 50000
            self.data_container.datasets[key] = dataset

    def _register_sampler_configs(self, trainer):
        for key in self.dataset_keys:
            self.__config_ids[key] = self._register_sampler_config_from_key(key=key, mode="x class")

    @staticmethod
    def _forward(batch, model, trainer):
        (x, cls), _ = batch
        x = x.to(model.device, non_blocking=True)
        with trainer.autocast_context:
            predictions = model.classify(x)
        predictions = {name: prediction.cpu() for name, prediction in predictions.items()}
        return predictions, cls.clone()

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer, batch_size, data_iter, **_):
        all_accuracies = defaultdict(dict)
        for dataset_key in self.dataset_keys:
            # extract
            predictions, classes = self.iterate_over_dataset(
                forward_fn=partial(self._forward, model=model, trainer=trainer),
                config_id=self.__config_ids[dataset_key],
                batch_size=batch_size,
                data_iter=data_iter,
            )

            # push to GPU for accuracy calculation
            predictions = {k: v.to(model.device, non_blocking=True) for k, v in predictions.items()}
            classes = classes.to(model.device, non_blocking=True)

            # log
            for name, prediction in predictions.items():
                acc = multiclass_accuracy(
                    preds=prediction,
                    target=classes,
                    num_classes=self.n_classes,
                    average="micro",
                ).item()
                self.writer.add_scalar(f"accuracy1/{dataset_key}/{name}", acc, logger=self.logger, format_str=".4f")
                all_accuracies[name][dataset_key] = acc

        # summarize over all
        for name in all_accuracies.keys():
            acc = float(np.mean(list(all_accuracies[name].values())))
            self.writer.add_scalar(f"accuracy1/imagenet_c_overall/{name}", acc, logger=self.logger, format_str=".4f")
