from copy import deepcopy

import kappaconfig as kc
from itertools import product

class ProbeProcessor:
    def __init__(
            self,
            poolings=None,
            grid="dinov2",
            probe_kind="linear_probe",
            save_best_probes=True,
            num_last_blocks_grid=None,
            every_nth_block=None,
            model_path=None,
    ):
        if probe_kind == "linear_probe":
            assert isinstance(poolings, list)
        elif probe_kind == "semseg_probe":
            assert poolings is None
            poolings = ["to_image", "to_image_concat_aux"]
        else:
            raise NotImplementedError
        self.poolings = poolings
        self.grid = grid
        self.probe_kind = probe_kind
        self.save_best_probes = save_best_probes
        self.num_last_blocks_grid = num_last_blocks_grid
        self.every_nth_block = every_nth_block
        self.model_path = model_path

    @staticmethod
    def pooling_long_to_short_name(pooling):
        if pooling == "class_token":
            return "cls"
        if pooling == "mean_patch":
            return "avg"
        if pooling == "concat_class_average":
            return "concat"
        if pooling == "to_image":
            return "img"
        if pooling == "to_image_concat_aux":
            return "imgcls"
        if pooling == "identity":
            return "id"
        raise NotImplementedError(f"no shortname for pooling '{pooling}'")

    def __call__(self, root):
        model = root["model"]
        assert model["kind"] == "probe_model"
        assert model["heads"] == "from_processor"

        # load schedule
        wupcos = kc.from_file_uri("zztemplates/schedules/wupcos_percent.yaml")
        wupcos["vars"]["end_percent"].value = 0.1
        wupcos = kc.DefaultResolver().resolve(wupcos)["schedule"]

        # define template
        template = dict(
            kind=f"probe.{self.probe_kind}",
            optim=dict(
                kind="sgd",
                momentum=0.9,
                schedule=wupcos,
            )
        )
        heads = {}
        if self.grid == "debug":
            lr_grid = [0.01]
            num_last_blocks_grid = [1, 4]
        elif self.grid == "dinov2":
            lr_grid = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
            num_last_blocks_grid = [1, 4]
        elif self.grid == "dinov2-reduced":
            # for inat the full dino grid takes a lot of memory -> cut away settings that are most likely bad
            lr_grid = [0.001, 0.002, 0.005, 0.01]
            num_last_blocks_grid = [4]
        elif self.grid == "dinov2-semseg-reduced":
            # cut away settings that were never good for semseg probe
            lr_grid = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
            num_last_blocks_grid = [4]
        elif self.grid == "semseg-min":
            # cut away settings that were never good for semseg probe
            lr_grid = [0.1]
            num_last_blocks_grid = [4]
        else:
            raise NotImplementedError(f"unknown grid '{self.grid}'")
        if self.num_last_blocks_grid is not None:
            num_last_blocks_grid = self.num_last_blocks_grid
        for lr, pooling, num_last_blocks in product(lr_grid, self.poolings, num_last_blocks_grid):
            head = deepcopy(template)
            head["optim"]["lr"] = lr
            pooling_name = self.pooling_long_to_short_name(pooling)
            pooling = dict(kind=pooling)
            if num_last_blocks == 1:
                head["pooling"] = pooling
            else:
                if self.every_nth_block is not None and self.every_nth_block > 1:
                    head["pooling"] = dict(
                        kind="extractor_pooling",
                        extractor=dict(
                            kind="vit_block_extractor",
                            model_path=self.model_path,
                            add_model_path_to_repr=False,
                            block_indices=[-(1 + i * self.every_nth_block) for i in range(num_last_blocks)],
                            finalizer=dict(
                                kind="concat_finalizer",
                                dim=-1,
                            ),
                        ),
                        pooling=pooling,
                    )
                else:
                    head["pooling"] = dict(
                        kind="extractor_pooling",
                        extractor=dict(
                            kind="vit_block_extractor",
                            model_path=self.model_path,
                            add_model_path_to_repr=False,
                            num_last_blocks=num_last_blocks,
                            finalizer=dict(
                                kind="concat_finalizer",
                                dim=-1,
                            ),
                        ),
                        pooling=pooling,
                    )
            heads[f"lr{str(lr).replace('.', '')}_{pooling_name}_last{num_last_blocks}"] = head
        model["heads"] = heads

        # save best probe callbacks
        callbacks = []
        if self.save_best_probes:
            if self.probe_kind == "linear_probe":
                for name in heads.keys():
                    if "_cls_" in name:
                        norm_name = "ClassToken"
                    elif "_concat_" in name:
                        norm_name = "ConcatClassAverage"
                    elif "_avg_" in name:
                        norm_name = "MeanPatch"
                    else:
                        raise NotImplementedError
                    if "_last" in name:
                        pooling_num_last_blocks = int(name[name.index("_last") + len("_last"):].split("_")[0])
                        if pooling_num_last_blocks > 1:
                            norm_name = (
                                f"ExtractorPooling(extractor=BlockExtractor(num_last_blocks={pooling_num_last_blocks},"
                                f"pooling=None,use_next_norm=False),pooling={norm_name})"
                            )
                    callbacks.append(
                        dict(
                            kind="best_checkpoint_callback",
                            every_n_epochs=1,
                            metric_key=f"accuracy1/test/{name}",
                            model_names=[
                                f"probe_model.heads.{name}",
                                f"probe_model.norms.{norm_name}",
                            ],
                        ),
                    )
        root["trainer"]["callbacks"] += callbacks

        summarizers = []
        if self.probe_kind == "linear_probe":
            summarizers += [
                dict(
                    kind="best_metric_summary_summarizer",
                    pattern="accuracy1/test*/max",
                ),
            ]
            for pooling in self.poolings:
                pooling_name = self.pooling_long_to_short_name(pooling)
                for num_last_blocks in num_last_blocks_grid:
                    summarizers += [
                        dict(
                            kind="best_metric_summary_summarizer",
                            pattern="accuracy1/test*/max",
                            contains=[
                                f"last{num_last_blocks}",
                                pooling_name,
                            ],
                        ),
                    ]
            for num_last_blocks in num_last_blocks_grid:
                summarizers += [
                    dict(
                        kind="best_metric_summary_summarizer",
                        pattern="accuracy1/test*/max",
                        contains=f"last{num_last_blocks}"
                    ),
                ]
            for pooling in self.poolings:
                pooling_name = self.pooling_long_to_short_name(pooling)
                summarizers += [
                    dict(
                        kind="best_metric_summary_summarizer",
                        pattern="accuracy1/test*/max",
                        contains=pooling_name,
                    ),
                ]
        elif self.probe_kind == "semseg_probe":
            summarizers += [
                dict(
                    kind="best_metric_summary_summarizer",
                    pattern="miou/test/*/max",
                ),
                dict(
                    kind="best_metric_summary_summarizer",
                    pattern="miou/test/*/max",
                    contains="_img_",
                ),
                dict(
                    kind="best_metric_summary_summarizer",
                    pattern="miou/test/*/max",
                    contains="_imgcls_",
                ),
            ]
            for num_last_blocks in num_last_blocks_grid:
                summarizers += [
                    dict(
                        kind="best_metric_summary_summarizer",
                        pattern="miou/test/*/max",
                        contains=f"last{num_last_blocks}",
                    ),
                ]
        else:
            raise NotImplementedError
        assert "summary_summarizers" not in root
        root["summary_summarizers"] = summarizers
