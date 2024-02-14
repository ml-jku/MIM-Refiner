import os

from torchvision.datasets.cifar import CIFAR10

from .base.dataset_base import DatasetBase


class Cifar10(DatasetBase):
    def __init__(self, split, global_root=None, **kwargs):
        super().__init__(**kwargs)
        global_root, _ = self._get_roots(global_root=global_root, local_root=None, dataset_identifier="cifar10")
        assert split in ["train", "test"]
        # setting download when it already exists prints unnecessary message
        download = len(os.listdir(global_root)) == 0
        train = split == "train"
        self.dataset = CIFAR10(root=global_root, download=download, train=train)

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

    @property
    def class_names(self):
        return list(self.dataset.class_to_idx.keys())
        # return ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __len__(self):
        return len(self.dataset)
