import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--src", type=str, default="/data/inat18")
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    return vars(parser.parse_args())


def main(src, dst, shots, seed):
    # src
    src = Path(src).expanduser()
    assert src.exists()
    assert src.name != "train"
    src = src / "train"
    print(f"src: '{src.as_posix()}'")

    # dst
    dst = Path(dst).expanduser()
    assert not dst.exists()
    dst_train = dst / "train"
    dst_train.mkdir(parents=True)
    print(f"dst: '{dst.as_posix()}'")
    print(f"dst_train: '{dst_train.as_posix()}'")

    # check number of classes
    class_idxs = list(sorted([int(dirname) for dirname in os.listdir(src) if (src / dirname).is_dir()]))
    assert len(class_idxs) == 8142
    print(f"num_classes: {len(class_idxs)}")

    generator = torch.Generator().manual_seed(seed)
    num_samples = 0
    num_undercomplete_classes = 0
    all_rel_uris = []
    for class_idx in tqdm(class_idxs):
        fnames = list(sorted(os.listdir(src / str(class_idx))))
        if len(fnames) < shots:
            num_undercomplete_classes += 1
        perm = torch.randperm(len(fnames), generator=generator)
        fnames_fewshot = [fnames[idx] for idx in perm[:shots].tolist()]
        num_samples += len(fnames_fewshot)
        dst_train_cls = dst_train / str(class_idx)
        dst_train_cls.mkdir()
        for fname_fewshot in fnames_fewshot:
            all_rel_uris.append(f"{class_idx}/{fname_fewshot}")
            shutil.copyfile(src / str(class_idx) / fname_fewshot, dst_train_cls / fname_fewshot)
    print(f"num_samples: {num_samples}")
    print(f"num_undercomplete_classes: {num_undercomplete_classes}")
    print(f"writing uris to '{(dst / 'rel_uris.yaml').as_posix()}'")
    with open(dst / "rel_uris.yaml", "w") as f:
        yaml.safe_dump(all_rel_uris, f)


if __name__ == "__main__":
    main(**parse_args())
