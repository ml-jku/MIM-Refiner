import os
import shutil
import zipfile

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.folder import default_loader

from distributed.config import is_data_rank0, barrier
from .base.dataset_base import DatasetBase


class Dtd(DatasetBase):
    def __init__(self, split, global_root=None, local_root=None, **kwargs):
        super().__init__(**kwargs)
        # check args
        assert split in ["train", "test"]

        # copy dataset to local if needed
        global_root, local_root = self._get_roots(global_root, local_root, "dtd")
        if local_root is None:
            # load from global root
            source_root = global_root
            assert global_root.exists(), f"invalid global_root '{global_root}'"
            self.logger.info(f"data_source (global): '{global_root}'")
        else:
            # load from local root
            source_root = local_root / "dtd"
            if is_data_rank0():
                self.logger.info(f"data_source (global): '{global_root}'")
                self.logger.info(f"data_source (local): '{source_root}'")
                if source_root.exists():
                    assert (source_root / "images").exists()
                    assert (source_root / "labels").exists()
                else:
                    # copy data from global to local (dataset is so small, bothering with zips is not worth)
                    source_root.mkdir()
                    shutil.copytree(global_root / "images", source_root / "images")
                    shutil.copytree(global_root / "labels", source_root / "labels")
            barrier()
        self.source_root = source_root
        self.split = split
        # create mapping from classname to classid
        self.clsname_to_clsid = {
            clsname: clsid
            for clsid, clsname in enumerate(sorted(os.listdir(source_root / "images")))
        }
        assert len(self.clsname_to_clsid) == 47
        # preprocess anotations files
        if split == "train":
            annotation_fnames = ["train1.txt", "val1.txt"]
        elif split == "test":
            annotation_fnames = ["test1.txt"]
        else:
            raise NotImplementedError
        self.fnames = []
        self.labels = []
        for annotation_fname in annotation_fnames:
            with open(source_root / "labels" / annotation_fname) as f:
                lines = f.readlines()
            self.fnames += [line.replace("\n", "") for line in lines]
            self.labels += [self.clsname_to_clsid[line.split("/")[0]] for line in lines]

        # check lengths
        if split == "train":
            assert len(self.fnames) == 3760, f"len(self.fnames) == {len(self.fnames)}"
        elif split == "test":
            assert len(self.fnames) == 1880, f"len(self.fnames) == {len(self.fnames)}"
        else:
            raise NotImplementedError

    def getitem_x(self, idx, ctx=None):
        return default_loader(self.source_root / "images" / self.fnames[idx])

    # noinspection PyUnusedLocal
    def getitem_class(self, idx, ctx=None):
        return self.labels[idx]

    @staticmethod
    def getshape_class():
        return 47,

    def __len__(self):
        return len(self.fnames)
