from argparse import ArgumentParser

import numpy as np
import wandb
import pandas as pd


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("stage_ids", type=str, nargs="+")
    return vars(parser.parse_args())


def main(stage_ids):
    print(f"stage_ids: {stage_ids}")
    wandb.login(host="https://api.wandb.ai/")
    api = wandb.Api()
    data = {}
    for stage_id in stage_ids:
        run = api.run(f"jku-ssl/v4/{stage_id}")
        print(f"stage_id: {stage_id} {run.name}")
        item = dict(name=run.name)
        try:
            item["stage_id"] = run.config["model"]["initializers"]["0"].get("stage_id", "ckpt")
        except KeyError:
            item["stage_id"] = "pretrained"
        for key, value in run.summary.items():
            if "Extractor" not in key:
                continue
            if "/max" not in key:
                continue
            dataset = key.split("/")[2]
            item[dataset] = value
        data[stage_id] = item
    wandb.finish()

    #
    for stage_id in stage_ids:
        item = data[stage_id]
        to_print = [
            item["stage_id"],
            item["name"].split("/")[0].replace("in1k-", "").replace("-nnclr", ""),
            "",
            "",
            "",
        ]
        for samples_per_class in [1, 2, 5]:
            seed1 = item[f"train_{samples_per_class}pc_s1"]
            seed2 = item[f"train_{samples_per_class}pc_s2"]
            seed3 = item[f"train_{samples_per_class}pc_s3"]
            avg = np.mean([seed1, seed2, seed3])
            to_print.append(f"{avg}")
            to_print.append(f"{seed1}")
            to_print.append(f"{seed2}")
            to_print.append(f"{seed3}")
            to_print.append("")
        print("\t".join(to_print))
    print("-" * 50)
    for stage_id in stage_ids:
        item = data[stage_id]
        to_print = [
            item["stage_id"],
        ]
        for samples_per_class in [5, 2, 1]:
            seed1 = item[f"train_{samples_per_class}pc_s1"]
            seed2 = item[f"train_{samples_per_class}pc_s2"]
            seed3 = item[f"train_{samples_per_class}pc_s3"]
            avg = np.mean([seed1, seed2, seed3])
            to_print.append(f"{avg}")
        print("\t".join(to_print))
    print("fin")
    wandb.finish()


if __name__ == "__main__":
    main(**parse_args())
