import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--stage_id", type=str, required=True)
    return vars(parser.parse_args())


def main(stage_id):
    if os.name == "nt":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    out = Path("out") / stage_id
    out.mkdir(exist_ok=True, parents=True)

    wandb.login(host="https://api.wandb.ai/")
    api = wandb.Api()
    data = []
    run = api.run(f"jku-ssl/v4/{stage_id}")
    print(f"stage_id: {stage_id} {run.name}")
    # get head names
    head_names = list(run.config["model"]["heads"].keys())
    # sort
    head_names = (
            list(sorted([name for name in head_names if "main" in name])) +
            list(reversed(sorted([name for name in head_names if "main" not in name])))
    )

    metrics = [
        "nn-accuracy/{name}/E1",
        "nn-age/{name}/E1",
        "nn-similarity/{name}/U50",
        "global_alignment/{name}/U50",
        "local_alignment/{name}/U50",
        "loss/online/heads.{name}.total/U50",
        "knn_accuracy/k10-tau007/heads.{name}.projector.0.GenericExtractor/train_noaug_10p-test",
    ]
    # compensate bug where local_alignment was added to losses instead of infos
    try:
        next(run.scan_history(keys=[f"loss/online/heads.{head_names[0]}.local_alignment/U50"]))
        metrics = [
            k if "local_alignment" not in k else "loss/online/heads.{name}.local_alignment/U50"
            for k in metrics
        ]
    except StopIteration:
        pass
    # fetch values
    data = {}
    for metric in metrics:
        print(f"fetch metric {metric}")
        item = {}
        data[metric] = item
        for head_name in head_names:
            key = metric.format(name=head_name)
            item[head_name] = [
                kv[key]
                for kv in list(run.scan_history(keys=[key]))
            ]
    # plot
    for metric in metrics:
        plt.clf()
        for head_name in head_names:
            y = data[metric][head_name]
            plt.plot(range(len(y)), y, label=head_name)
        plt.legend()
        plt.grid()
        fname = metric.split("/")[0]
        if fname == "loss" and "local_alignment" in metric:
            fname = "local_alignment"
        plt.title(fname)
        plt.savefig(out / f"{fname}.svg")
    wandb.finish()


if __name__ == "__main__":
    main(**parse_args())
