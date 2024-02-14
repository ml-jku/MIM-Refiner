import os

from torchvision.datasets.svhn import SVHN

from .base.dataset_base import DatasetBase


class Svhn(DatasetBase):
    def __init__(self, split, global_root=None, **kwargs):
        super().__init__(**kwargs)
        global_root, _ = self._get_roots(global_root=global_root, local_root=None, dataset_identifier="svhn")
        assert split in ["train", "test"]
        # setting download when it already exists prints unnecessary message
        download = len(os.listdir(global_root)) == 0
        self.dataset = SVHN(root=global_root, download=download, split=split)

    def getitem_x(self, idx, ctx=None):
        x, _ = self.dataset[idx]
        return x

    # noinspection PyUnusedLocal
    def getitem_class(self, idx, ctx=None):
        _, y = self.dataset[idx]
        return y

    @staticmethod
    def getshape_class():
        return 10,

    def __len__(self):
        return len(self.dataset)
