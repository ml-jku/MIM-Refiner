import os
import shutil
import zipfile

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.folder import default_loader

from distributed.config import is_data_rank0, barrier
from .base.dataset_base import DatasetBase


class Caltech101(DatasetBase):
    def __init__(self, split, global_root=None, local_root=None, **kwargs):
        super().__init__(**kwargs)
        # check args
        assert split in ["train", "test"]

        # copy dataset to local if needed
        global_root, local_root = self._get_roots(global_root, local_root, "caltech-101")
        if local_root is None:
            # load from global root
            source_root = global_root
            assert global_root.exists(), f"invalid global_root '{global_root}'"
            self.logger.info(f"data_source (global): '{global_root}'")
        else:
            # load from local root
            source_root = local_root / "caltech-101"
            if is_data_rank0():
                self.logger.info(f"data_source (global): '{global_root}'")
                self.logger.info(f"data_source (local): '{source_root}'")
                if source_root.exists():
                    assert (source_root / "101_ObjectCategories").exists()
                else:
                    # copy data from global to local (dataset is so small, bothering with zips is not worth)
                    source_root.mkdir()
                    shutil.copytree(global_root / "101_ObjectCategories", source_root / "101_ObjectCategories")
            barrier()

        self.split = split

        # VTAB splits data into 30 train samples per class and the rest are test samples
        img_root = source_root / "101_ObjectCategories"
        rng = torch.Generator().manual_seed(0)
        self.uris = []
        self.labels = []
        for i, cls in enumerate(list(sorted(os.listdir(img_root)))):
            items = list(sorted(os.listdir(img_root / cls)))
            perm = torch.randperm(len(items), generator=rng)
            if split == "train":
                indices = perm[:30]
            elif split == "test":
                indices = perm[30:]
            else:
                raise NotImplementedError
            indices = indices.sort().values
            for index in indices:
                self.uris.append(img_root / cls / items[index])
                self.labels.append(i)

        # check that all images end with .jpg
        assert all(uri.name.endswith(".jpg") for uri in self.uris)

        # check lengths
        if split == "train":
            assert len(self.uris) == 3060
        elif split == "test":
            assert len(self.uris) == 6084
        else:
            raise NotImplementedError

    def getitem_x(self, idx, ctx=None):
        return default_loader(self.uris[idx])

    # noinspection PyUnusedLocal
    def getitem_class(self, idx, ctx=None):
        return self.labels[idx]

    @staticmethod
    def getshape_class():
        return 102,

    def __len__(self):
        return len(self.uris)
