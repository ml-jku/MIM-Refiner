import os
import shutil
import zipfile

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.folder import default_loader

from distributed.config import is_data_rank0, barrier
from .base.dataset_base import DatasetBase
import scipy

class OxfordFlowers(DatasetBase):
    def __init__(self, split, global_root=None, local_root=None, **kwargs):
        super().__init__(**kwargs)
        # check args
        assert split in ["train", "test"]

        # copy dataset to local if needed
        global_root, local_root = self._get_roots(global_root, local_root, "oxford-flowers")
        if local_root is None:
            # load from global root
            source_root = global_root
            assert global_root.exists(), f"invalid global_root '{global_root}'"
            self.logger.info(f"data_source (global): '{global_root}'")
        else:
            # load from local root
            source_root = local_root / "oxford-flowers"
            if is_data_rank0():
                self.logger.info(f"data_source (global): '{global_root}'")
                self.logger.info(f"data_source (local): '{source_root}'")
                if source_root.exists():
                    assert (source_root / "jpg").exists()
                    assert (source_root / "setid.mat").exists()
                    assert (source_root / "imagelabels.mat").exists()
                else:
                    # copy data from global to local (dataset is so small, bothering with zips is not worth)
                    source_root.mkdir()
                    shutil.copytree(global_root / "jpg", source_root / "jpg")
                    shutil.copy(global_root / "setid.mat", source_root / "setid.mat")
                    shutil.copy(global_root / "imagelabels.mat", source_root / "imagelabels.mat")
            barrier()
        self.source_root = source_root
        self.split = split
        setid = scipy.io.loadmat(source_root / "setid.mat")
        if split == "train":
            self.ids = np.concatenate([setid["trnid"][0], setid["valid"][0]])
        elif split == "test":
            self.ids = setid["tstid"][0]
        else:
            raise NotImplementedError
        all_labels = scipy.io.loadmat(source_root / "imagelabels.mat")["labels"][0]
        self.labels = [all_labels[self.ids[i] - 1] - 1 for i in range(len(self.ids))]

        # check lengths
        if split == "train":
            assert len(self.ids) == 2040
        elif split == "test":
            assert len(self.ids) == 6149
        else:
            raise NotImplementedError

    def getitem_x(self, idx, ctx=None):
        return default_loader(self.source_root / "jpg" / f"image_{self.ids[idx]:05d}.jpg")

    # noinspection PyUnusedLocal
    def getitem_class(self, idx, ctx=None):
        return self.labels[idx]

    @staticmethod
    def getshape_class():
        return 102,

    def __len__(self):
        return len(self.ids)
