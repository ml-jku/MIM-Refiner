from collections import defaultdict

import yaml


class Stage3Processor:
    def __init__(self, add_projector_extractors=True):
        self.add_projector_extractors = add_projector_extractors

    @staticmethod
    def __from_scientific_str(scientific_str):
        # only values in the form 1e4 as short form for 1.0e-4 are supported
        assert "e" in scientific_str
        if "." not in scientific_str:
            scientific_str = scientific_str.replace("e", ".0e")
        if "-" not in scientific_str:
            scientific_str = scientific_str.replace("e", "e-")
        return yaml.safe_load(scientific_str)

    @staticmethod
    def __from_float_str(float_str):
        if "." not in float_str and float_str.startswith("0"):
            float_str = "0." + float_str[1:]
        else:
            assert float_str == "inf"
            float_str = ".inf"
        return yaml.safe_load(float_str)

    def __call__(self, root):
        trainer = root["trainer"]
        model = root["model"]
        heads = model["heads"]

        # assign loss functions based on the initializer name
        loss_functions = defaultdict(dict)
        for head_name, head in heads.items():
            # mugs head is composite model where the "head" submodel is e.g. a nnclr head
            if "head" in head:
                head = head["head"]
            assert len(head["initializers"]) == 1
            initializer = head["initializers"][0]
            model_name = initializer["model_name"]
            # model name is for example: contrastive_model.heads.nnclr_temp02_cls -> extract nnclr_temp02_cls
            stage2_head_name = model_name.split(".")[-1]
            if stage2_head_name == "head":
                stage2_head_name = model_name.split(".")[-2]
            # parse loss info
            split = stage2_head_name.split("_")
            for i in range(len(split)):
                # loss kind + head kind
                if split[i] in ["nnclr", "nnmugs"]:
                    if "kind" not in loss_functions[head_name]:
                        loss_functions[head_name]["kind"] = "nnclr_loss"
                    if "kind" not in head:
                        head["kind"] = "ssl.nnclr_head"
                elif split[i] == "noswap":
                    if "kind" not in loss_functions[head_name]:
                        loss_functions[head_name]["kind"] = "nnclr_noswap_loss"
                    if "kind" not in head:
                        head["kind"] = "ssl.nnclr_noswap_head"
                elif split[i] == "allswap":
                    if "kind" not in loss_functions[head_name]:
                        loss_functions[head_name]["kind"] = "nnclr_allswap_loss"
                    if "kind" not in head:
                        head["kind"] = "ssl.nnclr_allswap_head"
                # loss temperature
                elif split[i].startswith("temp"):
                    temp = self.__from_float_str(split[i][len("temp"):])
                    loss_functions[head_name]["temperature"] = temp
                elif split[i].startswith("cls"):
                    pass
                elif split[i].startswith("avg"):
                    pass
                elif split[i] == "oracle":
                    pass
                elif split[i].startswith("seed"):
                    pass
                else:
                    raise NotImplementedError(f"no action defined for '{split[i]}'")

        # assign loss functions
        assert "loss_functions" not in trainer or trainer["loss_functions"] == "from_processor"
        trainer["loss_functions"] = {key: value for key, value in loss_functions.items()}

        # assign callbacks
        if "callbacks" in trainer:
            extractors = [
                dict(
                    kind="vit_block_extractor",
                    model_path="encoder",
                    pooling=dict(kind="class_token"),
                    use_next_norm=False,
                    block_index=-1,
                ),
                dict(
                    kind="vit_block_extractor",
                    model_path="encoder",
                    pooling=dict(kind="mean_patch"),
                    use_next_norm=False,
                    block_index=-1,
                ),
                # dict(
                #     kind="vit_block_extractor",
                #     model_path="encoder",
                #     pooling=dict(kind="class_token"),
                #     use_next_norm=False,
                #     block_index=-2,
                # ),
                # dict(
                #     kind="vit_block_extractor",
                #     model_path="encoder",
                #     pooling=dict(kind="mean_patch"),
                #     use_next_norm=False,
                #     block_index=-2,
                # ),
                # dict(
                #     kind="vit_block_extractor",
                #     model_path="encoder",
                #     pooling=dict(kind="class_token"),
                #     use_next_norm=False,
                #     block_index=-3,
                # ),
                # dict(
                #     kind="vit_block_extractor",
                #     model_path="encoder",
                #     pooling=dict(kind="mean_patch"),
                #     use_next_norm=False,
                #     block_index=-3,
                # ),
                # dict(
                #     kind="vit_block_extractor",
                #     model_path="encoder",
                #     pooling=dict(kind="class_token"),
                #     use_next_norm=False,
                #     block_index=-4,
                # ),
                # dict(
                #     kind="vit_block_extractor",
                #     model_path="encoder",
                #     pooling=dict(kind="mean_patch"),
                #     use_next_norm=False,
                #     block_index=-4,
                # ),
            ]
            if self.add_projector_extractors:
                for head_name, head in heads.items():
                    if "head" in head:
                        continue
                    extractors += [
                        dict(
                            kind="generic_extractor",
                            model_path=f"heads.{head_name}.projector.0",
                        ),
                        # dict(
                        #     kind="generic_extractor",
                        #     model_path=f"heads.{head_name}.projector.2",
                        # ),
                    ]
            extractors[-1]["raise_exception"] = True
            for callback in trainer["callbacks"]:
                if "extractors" in callback and callback["extractors"] == "from_processor":
                    callback["extractors"] = extractors