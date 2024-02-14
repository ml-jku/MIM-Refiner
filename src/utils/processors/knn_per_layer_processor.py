from collections import defaultdict

import yaml


class KnnPerLayerProcessor:
    def __init__(self, num_blocks, has_class_token=True):
        self.num_blocks = num_blocks
        self.has_class_token = has_class_token

    def __call__(self, root):
        trainer = root["trainer"]
        # assign callbacks
        if "callbacks" in trainer:
            extractors = []
            for i in range(self.num_blocks):
                if self.has_class_token:
                    extractors += [
                        dict(
                            kind="vit_block_extractor",
                            pooling=dict(kind="class_token"),
                            use_next_norm=False,
                            block_index=i,
                        ),
                        # dict(
                        #     kind="vit_attn_extractor",
                        #     pooling=dict(kind="class_token"),
                        #     block_index=i,
                        # ),
                    ]
                extractors += [
                    dict(
                        kind="vit_block_extractor",
                        pooling=dict(kind="mean_patch"),
                        use_next_norm=False,
                        block_index=i,
                    ),
                    # dict(
                    #     kind="vit_attn_extractor",
                    #     pooling=dict(kind="mean_patch"),
                    #     block_index=i,
                    # ),
                ]
            for callback in trainer["callbacks"]:
                if "extractors" in callback and callback["extractors"] == "from_processor":
                    callback["extractors"] = extractors