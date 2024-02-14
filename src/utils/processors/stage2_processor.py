from collections import defaultdict

import yaml


class Stage2Processor:
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

        # collect attributes
        loss_functions = defaultdict(dict)
        for head_name, head in heads.items():
            split = head_name.split("_")
            for i in range(len(split)):
                # loss kind + head kind
                if split[i] == "nnclr":
                    loss_functions[head_name]["kind"] = "nnclr_loss"
                    head["kind"] = "ssl.nnclr_head"
                elif split[i] == "nnclrvote":
                    loss_functions[head_name]["kind"] = "nnclr_loss"
                    head["kind"] = "ssl.nnclr_vote_head"
                elif split[i] in ["dino", "nndino"]:
                    loss_functions[head_name]["kind"] = "dino_loss"
                    loss_functions[head_name]["student_temperature"] = 0.1
                    head["kind"] = f"ssl.{split[i]}_head"
                elif split[i] == "nnmugs":
                    loss_functions[head_name]["kind"] = "nnclr_loss"
                    head["kind"] = "ssl.mugs_head"
                    optim = head.pop("optim")
                    head["relation_model"] = dict(
                        kind="vit.mugs_relation_model",
                        kwargs=head.pop("relation_kwargs"),
                        topk=8,
                        optim=optim,
                    )
                    head["head"] = dict(
                        kind="ssl.nnclr_head",
                        kwargs=head.pop("head_kwargs"),
                        optim=optim,
                    )
                elif split[i] == "noswap":
                    loss_functions[head_name]["kind"] = "nnclr_noswap_loss"
                    head["kind"] = "ssl.nnclr_noswap_head"
                elif split[i] == "allswap":
                    loss_functions[head_name]["kind"] = "nnclr_allswap_loss"
                    head["kind"] = "ssl.nnclr_allswap_head"
                elif split[i] == "minbn":
                    loss_functions[head_name]["kind"] = "nnclr_loss"
                    head["kind"] = "ssl.nnclr_minbn_head"
                elif split[i] == "medbn":
                    loss_functions[head_name]["kind"] = "nnclr_loss"
                    head["kind"] = "ssl.nnclr_medbn_head"
                # loss temperature
                elif split[i].startswith("temp"):
                    temp = self.__from_float_str(split[i][len("temp"):])
                    loss_functions[head_name]["temperature"] = temp
                # loss teacher temperature (e.g. dino)
                elif split[i].startswith("ttemp"):
                    ttemp = self.__from_float_str(split[i][len("ttemp"):])
                    loss_functions[head_name]["teacher_temperature"] = ttemp
                # head lr
                elif split[i].startswith("lr"):
                    lr = self.__from_scientific_str(split[i][len("lr"):])
                    head["optim"]["lr"] = lr
                elif split[i] == "nonorm":
                    assert "norm_mode" not in head
                    head["norm_mode"] = None
                # oracle
                elif split[i] == "oracle":
                    if "mugs" in head["kind"]:
                        # mugs head
                        if "queue_kwargs" in head["head"]:
                            assert "guidance" not in head["head"]["queue_kwargs"]
                            head["head"]["queue_kwargs"]["guidance"] = "oracle"
                        else:
                            head["head"]["queue_kwargs"] = dict(guidance="oracle")
                    else:
                        # "normal" heads
                        if "queue_kwargs" in head:
                            assert "guidance" not in head["queue_kwargs"]
                            head["queue_kwargs"]["guidance"] = "oracle"
                        else:
                            head["queue_kwargs"] = dict(guidance="oracle")
                # head pooling (cls)
                elif split[i].startswith("cls"):
                    if split[i] == "cls":
                        head["pooling"] = dict(kind="class_token")
                    else:
                        if split[i].endswith("attn"):
                            block_idx = int(split[i][len("cls"):-len("attn")])
                            head["pooling"] = dict(
                                kind="extractor_pooling",
                                extractor=dict(
                                    kind="vit_attn_extractor",
                                    block_index=block_idx,
                                ),
                                pooling=dict(kind="class_token"),
                            )
                        else:
                            block_idx = int(split[i][len("cls"):])
                            head["pooling"] = dict(
                                kind="extractor_pooling",
                                extractor=dict(
                                    kind="vit_block_extractor",
                                    block_index=block_idx,
                                ),
                                pooling=dict(kind="class_token"),
                            )
                # head pooling (avg)
                elif split[i].startswith("avg"):
                    if split[i] == "avg":
                        head["pooling"] = dict(kind="mean_patch")
                    else:
                        if split[i].endswith("attn"):
                            block_idx = int(split[i][len("avg"):-len("attn")])
                            head["pooling"] = dict(
                                kind="extractor_pooling",
                                extractor=dict(
                                    kind="vit_attn_extractor",
                                    block_index=block_idx,
                                ),
                                pooling=dict(kind="mean_patch"),
                            )
                        else:
                            block_idx = int(split[i][len("avg"):])
                            head["pooling"] = dict(
                                kind="extractor_pooling",
                                extractor=dict(
                                    kind="vit_block_extractor",
                                    block_index=block_idx,
                                ),
                                pooling=dict(kind="mean_patch"),
                            )
                # schedule
                elif split[i] == "nosched":
                    head["optim"].pop("schedule")
                # vote
                elif split[i].startswith("votetopk"):
                    vote_topk = int(split[i][len("votetopk"):])
                    head["queue_voting_topk"] = vote_topk
                elif split[i].startswith("topk"):
                    topk = int(split[i][len("topk"):])
                    if "nnclrvote" in split:
                        head["queue_voting_topk"] = topk
                    else:
                        head["queue_topk"] = topk
                elif split[i] == "allcand":
                    head["queue_candidates_mode"] = "all"
                # keywords to ignore
                elif split[i] in ["nofreeze", "initfreeze"]:
                    pass
                elif split[i].startswith("seed"):
                    continue
                else:
                    raise NotImplementedError(f"unhandled keyword {split[i]}")

        # assign loss functions
        assert "loss_functions" not in trainer or trainer["loss_functions"] == "from_processor"
        trainer["loss_functions"] = {key: value for key, value in loss_functions.items()}

        # assign callbacks
        requires_callbacks = False
        for callback in trainer.get("callbacks", []):
            if "extractors" in callback and callback["extractors"] == "from_processor":
                requires_callbacks = True
                break
        if requires_callbacks:
            extractors = []
            # encoder extractors (if encoder is not fully frozen)
            encoder = model["encoder"]
            if "is_frozen" not in encoder:
                assert "optim" in encoder
                assert len(heads) == 1
                head_name = list(heads.keys())[0]
                if "cls" in head_name:
                    extractors += [
                        dict(
                            kind="vit_block_extractor",
                            model_path="encoder",
                            pooling=dict(kind="class_token"),
                            use_next_norm=False,
                            block_index=-1,
                        ),
                    ]
                elif "avg" in head_name:
                    extractors += [
                        dict(
                            kind="vit_block_extractor",
                            model_path="encoder",
                            pooling=dict(kind="mean_patch"),
                            use_next_norm=False,
                            block_index=-1,
                        ),
                    ]
                else:
                    raise NotImplementedError
            # head extractors
            for head_name, head in heads.items():
                if "mugs" in head_name:
                    head_name = f"{head_name}.head"
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
            for callback in trainer["callbacks"]:
                if "extractors" in callback and callback["extractors"] == "from_processor":
                    callback["extractors"] = extractors