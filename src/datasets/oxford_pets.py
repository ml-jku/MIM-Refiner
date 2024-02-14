import os
import shutil
import zipfile

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.folder import default_loader

from distributed.config import is_data_rank0, barrier
from .base.dataset_base import DatasetBase


class OxfordPets(DatasetBase):
    def __init__(self, split, global_root=None, local_root=None, **kwargs):
        super().__init__(**kwargs)
        # check args
        assert split in ["train", "test"]

        # copy dataset to local if needed
        global_root, local_root = self._get_roots(global_root, local_root, "oxford-pets")
        if local_root is None:
            # load from global root
            source_root = global_root
            assert global_root.exists(), f"invalid global_root '{global_root}'"
            self.logger.info(f"data_source (global): '{global_root}'")
        else:
            # load from local root
            source_root = local_root / "oxford-pets"
            if is_data_rank0():
                self.logger.info(f"data_source (global): '{global_root}'")
                self.logger.info(f"data_source (local): '{source_root}'")
                if source_root.exists():
                    assert (source_root / "images").exists()
                    assert (source_root / "annotations").exists()
                else:
                    # copy data from global to local (dataset is so small, bothering with zips is not worth)
                    source_root.mkdir()
                    shutil.copytree(global_root / "images", source_root / "images")
                    shutil.copytree(global_root / "annotations", source_root / "annotations")
            barrier()
        self.source_root = source_root
        self.split = split
        if split == "train":
            annotations_fname = "trainval.txt"
        elif split == "test":
            annotations_fname = "test.txt"
        else:
            raise NotImplementedError
        with open(source_root / "annotations" / annotations_fname) as f:
            lines = f.readlines()
        self.fnames = [line.split(" ")[0] for line in lines]
        self.labels = [int(line.split(" ")[1]) - 1 for line in lines]

        # check lengths
        if split == "train":
            assert len(self.fnames) == 3680
        elif split == "test":
            assert len(self.fnames) == 3669
        else:
            raise NotImplementedError

    def getitem_x(self, idx, ctx=None):
        return default_loader(self.source_root / "images" / f"{self.fnames[idx]}.jpg")

    # noinspection PyUnusedLocal
    def getitem_class(self, idx, ctx=None):
        return self.labels[idx]

    @staticmethod
    def getshape_class():
        return 37,

    def __len__(self):
        return len(self.fnames)
