import os
import shutil
import zipfile

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.folder import default_loader

from distributed.config import is_data_rank0, barrier
from .base.dataset_base import DatasetBase


class Ade20k(DatasetBase):
    def __init__(self, split, global_root=None, local_root=None, **kwargs):
        super().__init__(**kwargs)
        # check args
        assert split in ["train", "training", "val", "valid", "validation", "test"]
        if split in ["train", "training"]:
            split = "training"
        elif split in ["val", "valid", "validation", "test"]:
            split = "validation"
        else:
            raise NotImplementedError

        # copy dataset to local if needed
        global_root, local_root = self._get_roots(global_root, local_root, "ade20k")
        if local_root is None:
            # load from global root
            source_root = global_root
            assert global_root.exists(), f"invalid global_root '{global_root}'"
            self.logger.info(f"data_source (global): '{global_root}'")
        else:
            # load from local root
            source_root = local_root / "ade20k"
            if is_data_rank0():
                self.logger.info(f"data_source (global): '{global_root}'")
                self.logger.info(f"data_source (local): '{source_root}'")
                if source_root.exists():
                    assert (source_root / "images").exists()
                    assert (source_root / "annotations").exists()
                else:
                    # copy data from global to local
                    global_root_contents = os.listdir(global_root)
                    if len(global_root_contents) == 1:
                        # if global_root contains single zip file -> unzip into local root
                        with zipfile.ZipFile(global_root / global_root_contents[0]) as f:
                            f.extractall(local_root)
                    elif "annotations" in global_root_contents and "images" in global_root_contents:
                        # copy annotations/images folder
                        source_root.mkdir()
                        shutil.copytree(global_root / "images", source_root / "images")
                        shutil.copytree(global_root / "annotations", source_root / "annotations")
                    else:
                        raise NotImplementedError
            barrier()

        self.split = split
        self.img_root = source_root / "images" / split
        self.mask_root = source_root / "annotations" / split
        self.fnames = os.listdir(self.img_root)
        # images end with .jpg but annotations with .png
        assert all(fname.endswith(".jpg") for fname in self.fnames)
        self.fnames = [fname[:-len(".jpg")] for fname in self.fnames]

    def getitem_x(self, idx, ctx=None):
        return default_loader(self.img_root / f"{self.fnames[idx]}.jpg")

    # noinspection PyUnusedLocal
    def getitem_semseg(self, idx, ctx=None):
        return torch.from_numpy(np.array(Image.open(self.mask_root / f"{self.fnames[idx]}.png")).astype('int64')) - 1

    @staticmethod
    def getshape_semseg():
        return 150,

    def __len__(self):
        return len(self.fnames)
