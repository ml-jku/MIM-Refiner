import os

from torchvision.datasets.cifar import CIFAR100

from .base.dataset_base import DatasetBase


class Cifar100(DatasetBase):
    def __init__(self, split, global_root=None, **kwargs):
        super().__init__(**kwargs)
        global_root, _ = self._get_roots(global_root=global_root, local_root=None, dataset_identifier="cifar100")
        assert split in ["train", "test"]
        # setting download when it already exists prints unnecessary message
        download = len(os.listdir(global_root)) == 0
        train = split == "train"
        self.dataset = CIFAR100(root=global_root, download=download, train=train)

    def getitem_x(self, idx, ctx=None):
        x, _ = self.dataset[idx]
        return x

    # noinspection PyUnusedLocal
    def getitem_class(self, idx, ctx=None):
        _, y = self.dataset[idx]
        return y

    @staticmethod
    def getshape_class():
        return 100,

    @property
    def class_names(self):
        return list(self.dataset.class_to_idx.keys())
        # return [
        #     "apple", "aquarium_fish", "baby", "bear", "beaver",
        #     "bed", "bee", "beetle", "bicycle", "bottle",
        #     "bowl", "boy", "bridge", "bus", "butterfly",
        #     "camel", "can", "castle", "caterpillar", "cattle",
        #     "chair", "chimpanzee", "clock", "cloud", "cockroach",
        #     "couch", "crab", "crocodile", "cup", "dinosaur",
        #     "dolphin", "elephant", "flatfish", "forest", "fox",
        #     "girl", "hamster", "house", "kangaroo", "keyboard",
        #     "lamp", "lawn_mower", "leopard", "lion", "lizard",
        #     "lobster", "man", "maple_tree", "motorcycle", "mountain",
        #     "mouse", "mushroom", "oak_tree", "orange", "orchid",
        #     "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
        #     "plain", "plate", "poppy", "porcupine", "possum",
        #     "rabbit", "raccoon", "ray", "road", "rocket",
        #     "rose", "sea", "seal", "shark", "shrew",
        #     "skunk", "skyscraper", "snail", "snake", "spider",
        #     "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
        #     "tank", "telephone", "television", "tiger", "tractor",
        #     "train", "trout", "tulip", "turtle", "wardrobe",
        #     "whale", "willow_tree", "wolf", "woman", "worm",
        # ]

    def __len__(self):
        return len(self.dataset)
