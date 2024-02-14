import einops
import torch
import torch.nn.functional as F


@torch.no_grad()
def knn_predict(train_x, train_y, test_x, k=10, tau=0.07, batch_normalize=True, eps=1e-6):
    # initialize onehot vector per class (used for counting votes in classification)
    num_classes = train_y.max() + 1
    if num_classes <= 1:
        return None
    class_onehot = torch.diag(torch.ones(num_classes, device=train_x.device))

    # batchnorm features
    if batch_normalize:
        mean = train_x.mean(dim=0, keepdim=True)
        std = train_x.std(dim=0, keepdim=True) + eps
        train_x = (train_x - mean) / std
        test_x = (test_x - mean) / std

    # normalize to length 1 for cosine distance
    train_x = F.normalize(train_x, dim=1)
    test_x = F.normalize(test_x, dim=1)

    # limit k
    k = min(k, len(train_x))

    # calculate similarity
    similarities = test_x @ train_x.T
    topk_similarities, topk_indices = similarities.topk(k=k, dim=1)
    flat_topk_indices = einops.rearrange(topk_indices, "num_test knn -> (num_test knn)")
    flat_nn_labels = train_y[flat_topk_indices]
    flat_nn_onehot = class_onehot[flat_nn_labels]
    nn_onehot = einops.rearrange(flat_nn_onehot, "(num_test k) num_classes -> k num_test num_classes", k=k)

    topk_similarities = (topk_similarities / tau).exp_()
    logits = (nn_onehot * einops.rearrange(topk_similarities, "num_test knn -> knn num_test 1")).sum(dim=0)
    knn_classes = logits.argmax(dim=1)
    return knn_classes
