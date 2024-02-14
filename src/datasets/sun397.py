import os
import shutil
import zipfile

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.folder import default_loader

from distributed.config import is_data_rank0, barrier
from .base.dataset_base import DatasetBase


class Sun397(DatasetBase):
    def __init__(self, split, global_root=None, local_root=None, **kwargs):
        super().__init__(**kwargs)
        # check args
        assert split in ["train", "test"]

        # copy dataset to local if needed
        global_root, local_root = self._get_roots(global_root, local_root, "sun-397")
        if local_root is None:
            # load from global root
            source_root = global_root
            assert global_root.exists(), f"invalid global_root '{global_root}'"
            self.logger.info(f"data_source (global): '{global_root}'")
        else:
            # load from local root
            source_root = local_root / "sun-397"
            if is_data_rank0():
                self.logger.info(f"data_source (global): '{global_root}'")
                self.logger.info(f"data_source (local): '{source_root}'")
                if source_root.exists():
                    assert (source_root / "SUN397").exists()
                else:
                    # copy data from global to local
                    source_root.mkdir()
                    shutil.copytree(global_root / "SUN397", source_root / "SUN397")
            barrier()
        source_root = source_root / "SUN397"
        self.source_root = source_root
        self.split = split
        # get all classnames
        with open(source_root / "ClassName.txt") as f:
            lines = f.readlines()
        # VTAB makes 80/20 train/test split
        self.uris = []
        self.labels = []
        rng = torch.Generator().manual_seed(0)
        for i, line in enumerate(lines):
            cls = line.replace("\n", "")
            assert cls.startswith("/")
            cls = cls[1:]
            fnames = list(sorted(os.listdir(source_root / cls)))
            perm = torch.randperm(len(fnames), generator=rng)
            num_train_samples = int(round(len(fnames) * 0.8))
            if split == "train":
                perm = perm[:num_train_samples]
            elif split == "test":
                perm = perm[num_train_samples:]
            else:
                raise NotImplementedError
            perm = perm.sort().values
            for idx in perm:
                self.uris.append(source_root / cls / fnames[idx])
                self.labels.append(i)

        # check lengths (original paper says it should be 87003 train samples but this code gives one less)
        if split == "train":
            assert len(self.uris) == 87002, f"len(self.uris) == {len(self.uris)}"
        elif split == "test":
            assert len(self.uris) == 21752, f"len(self.uris) == {len(self.uris)}"
        else:
            raise NotImplementedError

    def getitem_x(self, idx, ctx=None):
        return default_loader(self.uris[idx])

    # noinspection PyUnusedLocal
    def getitem_class(self, idx, ctx=None):
        return self.labels[idx]

    @staticmethod
    def getshape_class():
        return 397,

    def __len__(self):
        return len(self.uris)
