import logging

LOWER_IS_BETTER_KEYS = [
    "loss",
    "uniformity",
    "best_f1/FN",
    "best_f1/FP",
    "fid",
    "profiling/train_data_time",
    "profiling/train_update_time",
]
HIGHER_IS_BETTER_KEYS = [
    "accuracy",
    "nnclr_queue/accuracy",
    "knn_accuracy",
    "ap",
    "auroc",
    "auprc",
    "alignment",
    "infos/global_alignment",
    "knn_purity",
    "erank",
    "best_f1/TN",
    "best_f1/TP",
    "best_f1/f1",
    "miou",
    "silhouette",
    "pseudo_acc",
    "nn-accuracy",
    "global_alignment",
    "local_alignment",
]
NEUTRAL_KEYS = [
    "optim",
    "profiling",
    "mask_ratio",
    "freezers",
    "transform_scale",
    "ctx",
    "loss_weight",
    "gradient",
    "nnclr_queue/similarity",
    "nnclr_queue/age",
    "detach",
    "cluster_inter_dist",
    "cluster_intra_dist",
    "max_distance",
    "confidence",
    "discriminator",
    "train_len",
    "test_len",
    "loss_weight",
    "nn-age",
    "nn-similarity",
    "nn-vote-",
    "variance",
    "preds_per_class",
    "entropy",
]


def is_neutral_key(metric_key):
    for higher_is_better_key in HIGHER_IS_BETTER_KEYS:
        if metric_key.startswith(higher_is_better_key):
            return False
    for lower_is_better_key in LOWER_IS_BETTER_KEYS:
        if metric_key.startswith(lower_is_better_key):
            return False
    for neutral_key in NEUTRAL_KEYS:
        if metric_key.startswith(neutral_key):
            return True
    return False


def higher_is_better_from_metric_key(metric_key):
    for higher_is_better_key in HIGHER_IS_BETTER_KEYS:
        if metric_key.startswith(higher_is_better_key):
            return True
    for lower_is_better_key in LOWER_IS_BETTER_KEYS:
        if metric_key.startswith(lower_is_better_key):
            return False
    logging.warning(f"{metric_key} has no defined behavior for higher_is_better -> using True")
    return True
