import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import wandb
import platform

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--project", type=str, default="v4")
    parser.add_argument("--stage_id", type=str, required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--include_filters", type=str)
    parser.add_argument("--exclude_filters", type=str)
    return vars(parser.parse_args())


def queue_metrics_key_to_name(key):
    return key.split("/")[2]


def knn_metrics_key_to_name(key):
    if "BlockExtractor" in key:
        end_idx = 3
    else:
        end_idx = 2
    if len(key.split("/")[2].split("_")) > 6:
        key = key.replace("topk20_", "")
        key = key.replace("lr2e4_", "")
        key = key.replace("nnclr2_", "")
        key = key.replace("nonegswap_", "")
    if "semi_queue" in key:
        key = key.replace(".semi_queue", "_semiq")
    return ".".join(key.split("/")[2].split(".")[1:end_idx])


def loss_metrics_key_to_name(key):
    return key.split("/")[2]


def setup():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if platform.uname().node == "DESKTOP-QVRSC79":
        import matplotlib
        matplotlib.use('TkAgg')


def get_data(
        stage_id,
        metric,
        project="v4",
        entity="jku-ssl",
        host="https://api.wandb.ai/",
        include_filters=None,
        exclude_filters=None,
        topk=6,
        higher_is_better=True,
):
    include_filters = include_filters or []
    exclude_filters = exclude_filters or []

    wandb.login(host=host)
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{stage_id}")
    # get keys
    keys = []
    for key in run.summary.keys():
        if key.endswith("/last") or key.endswith("/last10") or key.endswith("/last50"):
            continue
        if key.endswith("/max") or key.endswith("/min"):
            continue
        if not key.startswith(metric):
            continue
        is_valid = True
        for f in include_filters:
            if f not in key:
                is_valid = False
                break
        if not is_valid:
            continue
        is_valid = True
        for f in exclude_filters:
            if f in key:
                is_valid = False
                break
        if not is_valid:
            continue
        print(key)
        keys.append(key)
    print(f"found {len(keys)} keys")

    # get values
    values = defaultdict(list)
    for row in run.scan_history(keys=keys):
        for key in keys:
            values[key].append(row[key])

    # only use the best performing runs to plot
    keys = list(values.keys())
    if topk is not None:
        topk = min(len(keys), topk)
        last_values = [v[-1] for v in values.values()]
        nan_replacement = -float("inf") if higher_is_better else float("inf")
        last_values = [nan_replacement if v == "NaN" else v for v in last_values]
        keys = [keys[i] for i in torch.tensor(last_values).topk(k=topk, largest=higher_is_better, sorted=True).indices]

    return keys, values


def main(project, stage_id, topk, include_filters, exclude_filters):
    setup()
    if include_filters is not None:
        include_filters = include_filters.split(",")
    if exclude_filters is not None:
        exclude_filters = exclude_filters.split(",")
    kwargs = dict(
        save=True,
        project=project,
        stage_id=stage_id,
        topk=topk,
        fixed_include_filters=include_filters,
        fixed_exclude_filters=exclude_filters,
    )

    _main(
        fname="queue/train_nn_acc",
        metric="nn-accuracy",
        include_filters=["U50"],
        key_to_name=queue_metrics_key_to_name,
        higher_is_better=True,
        **kwargs,
    )
    # _main(
    #     fname="queue/train_nn_acc_lib",
    #     metric="nnclr_queue/accuracy",
    #     include_filters=["U50"],
    #     key_to_name=queue_metrics_key_to_name,
    #     higher_is_better=False,
    #     **kwargs,
    # )
    # _main(
    #     fname="queue/train_nn_sim",
    #     metric="nnclr_queue/similarity",
    #     include_filters=["U50"],
    #     key_to_name=queue_metrics_key_to_name,
    #     higher_is_better=True,
    #     **kwargs,
    # )
    # _main(
    #     fname="queue/train_nn_sim_lib",
    #     metric="nnclr_queue/similarity",
    #     include_filters=["U50"],
    #     key_to_name=queue_metrics_key_to_name,
    #     higher_is_better=False,
    #     **kwargs,
    # )
    # _main(
    #     fname="queue/train_nn_age",
    #     metric="nnclr_queue/age/",
    #     include_filters=["U50"],
    #     key_to_name=queue_metrics_key_to_name,
    #     higher_is_better=True,
    #     **kwargs,
    # )
    # _main(
    #     fname="queue/test_nn_acc",
    #     metric="nnclr_queue/accuracy",
    #     include_filters=["test_aug"],
    #     key_to_name=queue_metrics_key_to_name,
    #     higher_is_better=True,
    #     **kwargs,
    # )
    # _main(
    #     fname="queue/test_nn_acc_lib",
    #     metric="nnclr_queue/accuracy",
    #     include_filters=["test_aug"],
    #     key_to_name=queue_metrics_key_to_name,
    #     higher_is_better=False,
    #     **kwargs,
    # )
    # _main(
    #     fname="queue/test_nn_sim",
    #     metric="nnclr_queue/similarity",
    #     include_filters=["test_aug"],
    #     key_to_name=queue_metrics_key_to_name,
    #     higher_is_better=True,
    #     **kwargs,
    # )
    # _main(
    #     fname="queue/test_nn_sim_lib",
    #     metric="nnclr_queue/similarity",
    #     include_filters=["test_aug"],
    #     key_to_name=queue_metrics_key_to_name,
    #     higher_is_better=False,
    #     **kwargs,
    # )
    # _main(
    #     fname="queue/test_nn_age",
    #     metric="nnclr_queue/age/",
    #     include_filters=["test_aug"],
    #     key_to_name=queue_metrics_key_to_name,
    #     higher_is_better=True,
    #     **kwargs,
    # )
    # _main(
    #     fname="queue/test_nn_age_lib",
    #     metric="nnclr_queue/age/",
    #     include_filters=["test_aug"],
    #     key_to_name=queue_metrics_key_to_name,
    #     higher_is_better=False,
    #     **kwargs,
    # )
    # projector KNN
    _main(
        fname="knn_10p_proj0",
        metric="knn_accuracy/k10-tau007",
        include_filters=["projector.0"],
        key_to_name=knn_metrics_key_to_name,
        higher_is_better=True,
        **kwargs,
    )
    _main(
        fname="knn_10p_proj0_lib",
        metric="knn_accuracy/k10-tau007",
        include_filters=["projector.0"],
        key_to_name=knn_metrics_key_to_name,
        higher_is_better=False,
        **kwargs,
    )
    # projector KNN
    _main(
        fname="knn_10p_proj2",
        metric="knn_accuracy/k10-tau007",
        include_filters=["projector.2"],
        key_to_name=knn_metrics_key_to_name,
        higher_is_better=True,
        **kwargs,
    )
    _main(
        fname="knn_10p_proj2_lib",
        metric="knn_accuracy/k10-tau007",
        include_filters=["projector.2"],
        key_to_name=knn_metrics_key_to_name,
        higher_is_better=False,
        **kwargs,
    )
    # train loss
    _main(
        fname="train_loss",
        metric="loss/online",
        include_filters=["U50"],
        key_to_name=loss_metrics_key_to_name,
        exclude_filters=["total"],
        higher_is_better=False,
        **kwargs,
    )
    # test loss
    _main(
        fname="test_loss",
        metric="loss/test",
        key_to_name=loss_metrics_key_to_name,
        include_filters=["U50"],
        higher_is_better=False,
        **kwargs,
    )
    # alignment
    _main(
        fname="global_alignment",
        metric="infos/global_alignment",
        key_to_name=loss_metrics_key_to_name,
        include_filters=["test_aug"],
        higher_is_better=True,
        **kwargs,
    )
    _main(
        fname="local_alignment",
        metric="infos/local_alignment",
        key_to_name=loss_metrics_key_to_name,
        exclude_filters=["E1", "U50"],
        higher_is_better=True,
        **kwargs,
    )


def _main(
        project,
        stage_id,
        save,
        fname,
        metric,
        key_to_name,
        fixed_include_filters=None,
        fixed_exclude_filters=None,
        include_filters=None,
        exclude_filters=None,
        higher_is_better=True,
        topk=6,
):
    if fixed_include_filters is not None:
        if include_filters is None:
            include_filters = fixed_include_filters
        else:
            include_filters += fixed_include_filters
    if fixed_exclude_filters is not None:
        if exclude_filters is None:
            exclude_filters = fixed_exclude_filters
        else:
            exclude_filters += fixed_exclude_filters
    keys, values = get_data(
        stage_id=stage_id,
        metric=metric,
        host="https://api.wandb.ai/",
        entity="jku-ssl",
        project=project,
        include_filters=include_filters,
        exclude_filters=exclude_filters,
        topk=topk,
        higher_is_better=higher_is_better,
    )

    # plot
    plt.clf()
    for key in keys:
        value = values[key]
        name = key_to_name(key)
        last_value = value[-1]
        value_str = f"{last_value:.3f}" if last_value <= 1 else f"{int(last_value)}"
        plt.plot(range(len(value)), value, label=f"{name} ({value_str})")

    # compose title
    title = fname
    if topk is not None and topk < len(values):
        title = f"{title} top{topk}/{len(values)}"
    if fixed_include_filters is not None:
        title = f"{title} incl_filters=[{','.join(fixed_include_filters)}]"
    if fixed_exclude_filters is not None:
        title = f"{title} excl_filters=[{','.join(fixed_exclude_filters)}]"
    plt.title(title)

    # format
    plt.xlabel("progress")
    plt.ylabel(metric)
    plt.grid()
    # if len(keys) <= 11:
    plt.legend()
    # plt.tight_layout()
    if save:
        dirname = stage_id
        if fixed_include_filters is not None:
            dirname += f"_incl={','.join(fixed_include_filters)}"
        if fixed_exclude_filters is not None:
            dirname += f"_excl={','.join(fixed_exclude_filters)}"
        out_dir = Path(f"out/{dirname}")
        out_fig = out_dir / f"{fname}.svg"
        out_fig.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(out_fig)
    else:
        plt.show()
    wandb.finish()


if __name__ == "__main__":
    main(**parse_args())
