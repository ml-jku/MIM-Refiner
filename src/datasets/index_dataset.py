from .base.dataset_base import DatasetBase


class IndexDataset(DatasetBase):
    """
    a dataset without data (only the index is returned)
    used for an easy determinstic implementation of distributed sampling from generative models
    the index is used as seed for a generator that is then used by the generative model for sampling
    """

    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def __len__(self):
        return self.size

    def getitem_x(self, idx, ctx=None):
        raise NotImplementedError
