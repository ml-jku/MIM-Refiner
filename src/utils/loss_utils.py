import torch


def apply_reduction(tensor, reduction="mean"):
    # convert to float if .mean is not supported for dtype
    if tensor.dtype in [torch.bool, torch.long]:
        tensor = tensor.float()
    if reduction == "mean":
        return tensor.mean()
    if reduction is None or reduction == "none":
        if tensor.ndim > 1:
            return tensor.flatten(start_dim=1).squeeze(dim=1)
        return tensor
    raise NotImplementedError
