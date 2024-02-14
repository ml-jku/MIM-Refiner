#   1.0G T16
#   4.2G S16
#  17.5G B16
#  61.6G L16
#  61.6G L16
#  77.8G L14
#  67.2G B16 res448
#  78.2G B8
# 127.3G H16
# 167.4G H14
# 237.8G L16 res448
# 291.4G dino-g14
# 361.5G L7
# 448.2G B4
# 473.4G G14
# 493.7G 2B
# 545.4G H16 res448
# 732.1G H14 res448
#1553.5G dino-g14 res518
#   1.6T 6.5B
from argparse import ArgumentParser

import torch
from fvcore.nn import FlopCountAnalysis

from models.vit.vit import Vit
from utils.formatting_util import short_number_str
from utils.model_utils import get_trainable_param_count

VITS = {
    "T16": dict(
        kind="vit",
        patch_size=16,
        dim=192,
        depth=12,
        num_attn_heads=3,
    ),
    "S16": dict(
        kind="vit",
        patch_size=16,
        dim=384,
        depth=12,
        num_attn_heads=6,
    ),
    "B16": dict(
        kind="vit",
        patch_size=16,
        dim=768,
        depth=12,
        num_attn_heads=12,
    ),
    "B8": dict(
        kind="vit",
        patch_size=8,
        dim=768,
        depth=12,
        num_attn_heads=12,
    ),
    "b4": dict(
        kind="vit",
        patch_size=4,
        dim=768,
        depth=12,
        num_attn_heads=12,
    ),
    "L16": dict(
        kind="vit",
        patch_size=16,
        dim=1024,
        depth=24,
        num_attn_heads=16,
    ),
    "L14": dict(
        kind="vit",
        patch_size=14,
        dim=1024,
        depth=24,
        num_attn_heads=16,
    ),
    "L7": dict(
        kind="vit",
        patch_size=7,
        dim=1024,
        depth=24,
        num_attn_heads=16,
    ),
    "H16": dict(
        kind="vit",
        patch_size=16,
        dim=1280,
        depth=32,
        num_attn_heads=16,
    ),
    "H14": dict(
        kind="vit",
        patch_size=14,
        dim=1280,
        depth=32,
        num_attn_heads=16,
    ),
    "dino-g14": dict(
        kind="vit",
        patch_size=14,
        dim=1536,
        depth=40,
        num_attn_heads=24,
    ),
    "dino-g14-res518": dict(
        kind="vit",
        patch_size=14,
        dim=1536,
        depth=40,
        num_attn_heads=24,
        resolution=518,
    ),
    "G14": dict(
        kind="vit",
        patch_size=14,
        dim=1664,
        mlp_hidden_dim=8192,
        depth=48,
        num_attn_heads=16,
    ),
    "2B": dict(
        kind="vit",
        patch_size=14,
        dim=2560,
        depth=24,
        num_attn_heads=32,
    ),
    "6.5B": dict(
        kind="vit",
        patch_size=14,
        dim=4096,
        depth=32,
        num_attn_heads=32,
    ),
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(VITS.keys()), required=True)
    return vars(parser.parse_args())


def main(model):
    model_kwargs = VITS[model]
    kind = model_kwargs.pop("kind")
    assert kind == "vit"
    res = model_kwargs.pop("resolution", 224)
    x = torch.randn(1, 3, res, res)
    model_ctor = Vit
    print(model)
    print(f"resolution: {res}")
    model = model_ctor(
        **model_kwargs,
        input_shape=x.shape[1:],
    ).eval()
    analysis = FlopCountAnalysis(model, x)

    print(f"FLOPS: {short_number_str(analysis.total())}")
    print(f"params: {short_number_str(get_trainable_param_count(model))}")
    print("------------------")
    for operator_name, flops in analysis.by_operator().items():
        print(f"{operator_name}: {short_number_str(flops)}")
    print("------------------")
    for module_name, operator_analysis in analysis.by_module_and_operator().items():
        for operator_name, flops in operator_analysis.items():
            print(f"{module_name}.{operator_name}: {short_number_str(flops)}")


if __name__ == "__main__":
    main(**parse_args())
