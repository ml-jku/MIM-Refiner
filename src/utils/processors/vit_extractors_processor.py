from collections import defaultdict

import yaml


class VitExtractorsProcessor:
    def __init__(self, num_blocks, poolings, start_block=0):
        self.num_blocks = num_blocks
        self.poolings = poolings
        self.start_block = start_block

    def __call__(self, root):
        trainer = root["trainer"]
        if "callbacks" not in trainer:
            return

        extractors = []
        for i in range(self.start_block, self.num_blocks):
            for pooling in self.poolings:
                extractors += [
                    dict(
                        kind="vit_block_extractor",
                        pooling=dict(kind=pooling),
                        use_next_norm=False,
                        block_index=i,
                    ),
                ]
        for callback in trainer["callbacks"]:
            if "extractors" in callback and callback["extractors"] == "from_processor":
                callback["extractors"] = extractors