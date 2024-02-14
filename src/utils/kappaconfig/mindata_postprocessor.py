import kappaconfig as kc

from utils.param_checking import to_2tuple
from .testrun_constants import TEST_RUN_EFFECTIVE_BATCH_SIZE, TEST_RUN_UPDATES_PER_EPOCH


class MinDataPostProcessor(kc.Processor):
    """
    hyperparams for specific properties in the dictionary and replace it such that the training duration is
    limited to a minimal configuration
    """

    def preorder_process(self, node, trace):
        if len(trace) == 0:
            return
        parent, parent_accessor = trace[-1]
        if isinstance(parent_accessor, str):
            # sampelrs
            if parent_accessor == "main_sampler_kwargs":
                if "weighted_size" in node:
                    node["weighted_size"] = TEST_RUN_EFFECTIVE_BATCH_SIZE * TEST_RUN_UPDATES_PER_EPOCH
            # datasets
            if parent_accessor == "datasets":
                for key in node.keys():
                    wrappers = []
                    # shuffle ordered dataset
                    # - ImageNet would have only class0 if not shuffled
                    # - CLEVR would require full dataset when shuffled -> disable shuffling
                    if node[key]["kind"] not in ["clevr_counts", "clevr_distances"]:
                        shuffle_wrapper = dict(
                            kind="shuffle_wrapper",
                            seed=0,
                        )
                        wrappers.append(shuffle_wrapper)
                    if node[key]["kind"] == "ade20k":
                        transforms = node[key]["sample_wrappers"][0]["transforms"]
                        for transform in transforms:
                            if "base_size" in transform:
                                transform["base_size"] = [448, 224]
                            if "size" in transform:
                                transform["size"] = 224
                    subset_wrapper = dict(
                        kind="subset_wrapper",
                        end_index=TEST_RUN_EFFECTIVE_BATCH_SIZE * TEST_RUN_UPDATES_PER_EPOCH + 1,
                    )
                    wrappers.append(subset_wrapper)
                    if "dataset_wrappers" in node[key]:
                        node[key]["dataset_wrappers"] += wrappers
                    else:
                        assert isinstance(node[key], dict), (
                            "found non-dict value inside 'datasets' node -> probably wrong template "
                            "parameter (e.g. template.version instead of template.vars.version)"
                        )
                        node[key]["dataset_wrappers"] = wrappers
            elif parent_accessor in ["effective_batch_size", "effective_labeled_batch_size"]:
                parent[parent_accessor] = min(parent[parent_accessor], TEST_RUN_EFFECTIVE_BATCH_SIZE)
            elif parent_accessor == "optim":
                # decrease lr scaling (e.g. to avoid errors when max_lr < min_lr when using a min_lr with cosine decay)
                parent[parent_accessor]["lr_scaler"] = dict(
                    kind="linear_lr_scaler",
                    divisor=TEST_RUN_EFFECTIVE_BATCH_SIZE,
                )

