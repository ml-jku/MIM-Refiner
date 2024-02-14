from pathlib import Path

from datasets.base.image_folder import ImageFolder


class Inat(ImageFolder):
    def __init__(
            self,
            version,
            split=None,
            **kwargs,
    ):
        self.version = version
        assert split in ["train", "val", "test"]
        if split == "test":
            split = "val"
        self.split = split
        super().__init__(**kwargs)

    def get_dataset_identifier(self):
        return self.version

    def get_relative_path(self):
        return Path(self.split)

    def __str__(self):
        return f"{self.version}.{self.split}"
