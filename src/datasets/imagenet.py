from pathlib import Path

import einops
import torch
import yaml

from datasets.base.image_folder import ImageFolder


class ImageNet(ImageFolder):
    IMAGENET_C_DISTORTIONS = [
        # blur
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        # digitarl
        "contrast", "elastic_transform", "jpeg_compression", "pixelate",
        # extra
        "gaussian_blur", "saturate", "spatter", "speckle_noise",
        # noise
        "gaussian_noise", "shot_noise", "impulse_noise",
        # weather
        "frost", "snow", "fog", "brightness",
    ]

    def __init__(
            self,
            version,
            split=None,
            semi_version=None,
            **kwargs,
    ):
        self.version = version
        if version in ["imagenet_a", "imagenet_r"]:
            assert split in ["val", "test"]
        elif version == "imagenet_c":
            distortion, strength = split.split("/")
            assert distortion in ImageNet.IMAGENET_C_DISTORTIONS
            assert 1 <= int(strength) <= 5
        else:
            assert split in ["train", "val", "test"]
        if split == "test":
            split = "val"
        self.split = split
        super().__init__(**kwargs)

        # semi supervised
        self.semi_version = semi_version
        if semi_version is not None:
            raise NotImplementedError
        else:
            self.semi_labeled_uris = None

    def get_dataset_identifier(self):
        """ returns an identifier for the dataset (used for retrieving paths from dataset_config_provider) """
        return self.version

    def get_relative_path(self):
        return Path(self.split)

    def __str__(self):
        return f"{self.version}.{self.split}"

    @property
    def class_names(self):
        with open("resources/in1k_classid_to_names.yaml") as f:
            class_to_names = yaml.safe_load(f)
        idx_to_class = {v: k for k, v in self.dataset.class_to_idx.items()}
        return [class_to_names[idx_to_class[i]][0] for i in range(len(idx_to_class))]

    def getitem_class(self, idx, ctx=None):
        # NOTE: always return ints because dataloader requires consistent datatype
        # (it will check if it has a tensor and then stack it -> error if tensor and ints are mixed)
        if self.semi_labeled_uris is not None:
            path, target = self.dataset.samples[idx]
            if Path(path).name in self.semi_labeled_uris:
                return target
            return -1
        else:
            return self.dataset.targets[idx]

    def getall_class(self):
        if self.semi_labeled_uris is None:
            return self.dataset.targets
        targets = []
        for path, target in self.dataset.samples:
            if Path(path).name in self.semi_labeled_uris:
                targets.append(target)
            else:
                targets.append(-1)
        return targets

    # noinspection PyUnusedLocal
    @staticmethod
    def getitem_confidence(idx, ctx=None):
        # TODO when training with a PseudoLabelWrapper, the trainer.dataset_mode will be used for extracting features
        #   from evaluation datasets, which are not wrapped in a PseudoLabelWrapper which leads to an error
        #   this is a workarount to return a dummy value because the confidence is never used during evaluation
        return -1