import os
from pathlib import Path

from kappadata.copying.image_folder import copy_imagefolder_from_global_to_local
from torchvision.datasets import ImageFolder as TVImageFolder
from torchvision.datasets.folder import default_loader

from distributed.config import barrier, is_data_rank0
from utils.num_worker_heuristic import get_fair_cpu_count
from .dataset_base import DatasetBase


class ImageFolder(DatasetBase):
    def __init__(self, global_root=None, local_root=None, **kwargs):
        super().__init__(**kwargs)
        global_root, local_root = self._get_roots(global_root, local_root, self.get_dataset_identifier())
        # get relative path (e.g. train)
        relative_path = self.get_relative_path()
        if local_root is None:
            # load data from global_root
            assert global_root is not None and global_root.exists(), f"invalid global_root '{global_root}'"
            source_root = global_root / relative_path
            assert source_root.exists(), f"invalid source_root (global) '{source_root}'"
            self.logger.info(f"data_source (global): '{source_root}'")
        else:
            # load data from local_root
            source_root = local_root / self.get_dataset_identifier() / relative_path
            if is_data_rank0():
                # copy data from global to local
                self.logger.info(f"data_source (global): '{global_root / relative_path}'")
                self.logger.info(f"data_source (local): '{source_root}'")
                copy_imagefolder_from_global_to_local(
                    global_path=global_root,
                    local_path=local_root / self.get_dataset_identifier(),
                    relative_path=relative_path,
                    # on karolina 5 was already too much for a single GPU
                    # "A worker process managed by the executor was unexpectedly terminated.
                    # This could be caused by a segmentation fault while calling the function or by an
                    # excessive memory usage causing the Operating System to kill the worker."
                    num_workers=min(10, get_fair_cpu_count()),
                    log_fn=self.logger.info,
                )
                # check folder structure
                folders = [None for f in os.listdir(source_root) if (source_root / f).is_dir()]
                self.logger.info(f"source_root '{source_root}' contains {len(folders)} folders")
            barrier()
        self.dataset = TVImageFolder(source_root, loader=self._loader)

    @property
    def _loader(self):
        return default_loader

    def get_dataset_identifier(self):
        """ returns an identifier for the dataset (used for retrieving paths from dataset_config_provider) """
        raise NotImplementedError

    def get_relative_path(self):
        """
        return the relative path to the dataset root
        - e.g. /train (ImageNet)
        - e.g. /bottle (MVTec)
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

    def getitem_x(self, idx, ctx=None):
        x, _ = self.dataset[idx]
        return x

    def getitem_class(self, idx, ctx=None):
        return self.dataset.targets[idx]

    # noinspection PyUnusedLocal
    def getitem_class_all(self, idx, ctx=None):
        return self.dataset.targets[idx]

    # noinspection PyUnusedLocal
    def getitem_fname(self, idx, ctx=None):
        return str(Path(self.dataset.samples[idx][0]).relative_to(self.dataset.root))

    def getshape_class(self):
        num_classes = len(self.dataset.classes)
        if num_classes == 2:
            num_classes = 1
        return num_classes,
