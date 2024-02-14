import torch


class ChunkFinalizer:
    def __init__(self, chunk_idx, num_chunks):
        super().__init__()
        self.chunk_idx = chunk_idx
        self.num_chunks = num_chunks

    def __call__(self, features):
        return torch.concat(features).chunk(self.num_chunks)[self.chunk_idx]

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{type(self).__name__}(chunk_idx={self.chunk_idx},n_chunks={self.num_chunks})"
