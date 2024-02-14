class VitCallbackExtractors:
    def __init__(self, model_key, block_indices=None):
        self.model_key = model_key
        self.block_indices = block_indices

    def __call__(self, root):
        trainer = root["trainer"]
        if "callbacks" not in trainer:
            return
        model = root["model"]
        vit = model[self.model_key]
        if self.block_indices is None:
            if "depth" in vit:
                block_idxs = list(range(vit["depth"]))
            elif "kwargs" in vit:
                block_idxs = list(range(vit["kwargs"]["depth"]))
            else:
                block_idxs = [-1]
        else:
            block_idxs = self.block_indices
        # collect extractors
        extractors = []
        poolings = ["mean_patch"]
        if vit.get("num_cls_tokens", 1) > 0:
            poolings.append("class_token")
        for pooling in poolings:
            extractors += [
                dict(
                    kind="vit_block_extractor",
                    model_path=self.model_key,
                    pooling=dict(
                        kind=pooling,
                    ),
                    use_next_norm=False,
                    block_indices=[block_idx],
                )
                for block_idx in block_idxs
            ]
            if "target_factor" in model:
                extractors += [
                    dict(
                        kind="vit_block_extractor",
                        model_path=f"momentum_{self.model_key}",
                        pooling=dict(
                            kind=pooling,
                        ),
                        use_next_norm=False,
                        block_indices=[block_idx],
                    )
                    for block_idx in block_idxs
                ]
        extractors[-1]["raise_exception"] = True
        # replace in callbacks
        for callback in trainer["callbacks"]:
            if "extractors" in callback and callback["extractors"] == "from_processor":
                callback["extractors"] = extractors