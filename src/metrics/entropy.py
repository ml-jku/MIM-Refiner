import torch
import torch.nn.functional as F


def _preds_to_dist(tensor, num_classes):
    if tensor.ndim == 1:
        # (batch_size,) -> check dtype
        assert tensor.dtype == torch.long
    elif tensor.ndim == 2:
        if tensor.size(1) == 1:
            # (batch_size, 1) -> squeeze and check dtype
            assert tensor.dtype == torch.long
            tensor = tensor.squeeze(1)
        else:
            # (batch_size, num_classes) -> argmax
            tensor = tensor.argmax(dim=1)
    else:
        raise NotImplementedError
    # count occourances of classes
    values, counts = tensor.unique(return_counts=True)
    # pad counts if not all classes occour
    if len(counts) != num_classes:
        padded_counts = torch.zeros(num_classes, dtype=counts.dtype, device=counts.device)
        padded_counts[values] = counts
        counts = padded_counts
    # convert to distribution
    dist = counts / counts.sum()
    return dist


def multiclass_entropy(y_hat, num_classes, y=None):
    y_hat_dist = _preds_to_dist(y_hat, num_classes=num_classes)
    if y is None:
        # calculate entropy
        entropy = -torch.sum(y_hat_dist * torch.log(y_hat_dist))
    else:
        # cross entropy with actual distribution
        y_dist = _preds_to_dist(y, num_classes=num_classes)
        entropy = F.cross_entropy(y_hat_dist, y_dist)
    return entropy
