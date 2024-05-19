import torch.nn.functional as F
from kappadata.wrappers import ModeWrapper
from functools import partial
from itertools import product

from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from callbacks.base.periodic_callback import PeriodicCallback
from distributed.config import get_world_size


class OfflineSegmentationCallback(PeriodicCallback):
    def __init__(self, dataset_key, mode, ignore_index=-1, mode_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.ignore_index = ignore_index
        self.mode = mode
        self.mode_kwargs = mode_kwargs or {}
        self.__config_id = None
        self.num_classes = None
        self.jaccard: MulticlassJaccardIndex = None
        self.accuracy: MulticlassAccuracy = None

    def _before_training(self, model, **_):
        dataset = self.data_container.get_dataset(self.dataset_key)
        # torchmetrics accumulation stuff doesnt work with padded distributed evaluation
        assert len(dataset) % get_world_size() == 0, \
            f"{type(self).__name__} doesnt support distributed eval with padding " \
            f"(requires len(dataset) % world_size == 0) " \
            f"len(dataset)={len(dataset)} world_size={get_world_size()}"
        self.num_classes = dataset.getdim("semseg")
        self.jaccard = {
            key: MulticlassJaccardIndex(
                num_classes=self.num_classes,
                average="macro",
                ignore_index=self.ignore_index,
            ).to(model.device)
            for key in model.heads.keys()
        }
        self.accuracy = {
            key: MulticlassAccuracy(
                num_classes=self.num_classes,
                average="macro",
                ignore_index=self.ignore_index,
            ).to(model.device)
            for key in model.heads.keys()
        }

    def _register_sampler_configs(self, trainer):
        self.__config_id = self._register_sampler_config_from_key(key=self.dataset_key, mode="x semseg")

    def _forward(self, batch, model, trainer):
        batch, _ = batch
        x = ModeWrapper.get_item(mode="x semseg", batch=batch, item="x")
        target = ModeWrapper.get_item(mode="x semseg", batch=batch, item="semseg")
        x = x.to(model.device, non_blocking=True)
        target = target.to(model.device, non_blocking=True)

        if self.mode == "slide":
            assert len(x) == 1, f"slide inference requires batch_size=1"
            batch_size = 1
            h_stride, w_stride = self.mode_kwargs["stride"]
            h_crop, w_crop = model.input_shape[1:]
            assert h_stride <= h_crop
            assert w_stride <= w_crop
            _, _, h_img, w_img = x.shape
            h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
            pred = {
                key: x.new_zeros((batch_size, self.num_classes, h_img, w_img))
                for key in model.heads.keys()
            }
            count = x.new_zeros((batch_size, 1, h_img, w_img))
            for h_idx, w_idx in product(range(h_grids), range(w_grids)):
                h_start = h_idx * h_stride
                w_start = w_idx * w_stride
                h_end = min(h_start + h_crop, h_img)
                w_end = min(w_start + w_crop, w_img)
                h_start = max(h_end - h_crop, 0)
                w_start = max(w_end - w_crop, 0)
                crop_img = x[:, :, h_start:h_end, w_start:w_end]
                # pad if image is too small
                pad_h = h_crop - crop_img.size(2)
                pad_w = w_crop - crop_img.size(3)
                crop_img = F.pad(crop_img, (0, pad_w, 0, pad_h))

                with trainer.autocast_context:
                    logits = model(crop_img)
                cutoff_h = crop_img.size(2) - pad_h
                cutoff_w = crop_img.size(3) - pad_w
                for key in logits.keys():
                    pred[key][:, :, h_start:h_end, w_start:w_end] += logits[key][:, :, :cutoff_h, :cutoff_w]
                count[:, :, h_start:h_end, w_start:w_end] += 1
            #
            assert (count == 0).sum() == 0
            pred = {key: value.div_(count) for key, value in pred.items()}
        else:
            raise NotImplementedError(f"mode {self.mode} is not implemented")

        for key, value in pred.items():
            self.jaccard[key].update(preds=value, target=target)
            self.accuracy[key].update(preds=value, target=target)

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer, batch_size, data_iter, **_):
        # setup
        for key in list(self.jaccard.keys()):
            self.jaccard[key].reset()
            self.accuracy[key].reset()

        # iterate
        self.iterate_over_dataset(
            forward_fn=partial(self._forward, model=model, trainer=trainer),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )

        for key in list(self.jaccard.keys()):
            # calculate
            miou = self.jaccard[key].compute()
            acc = self.accuracy[key].compute()
            # log
            self.writer.add_scalar(f"miou/{self.dataset_key}/{key}", miou, logger=self.logger, format_str=".6f")
            self.writer.add_scalar(f"accuracy1/{self.dataset_key}/{key}", acc, logger=self.logger, format_str=".6f")
