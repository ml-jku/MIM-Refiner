from functools import partial

import torch

from hub.postnorm_vit import PostnormVit
from hub.prenorm_vit import PrenormVit

dependencies = ["torch", "kappamodules", "einops"]

VIT_CONFIGS = dict(
    debug=dict(patch_size=16, dim=16, depth=2, num_heads=2),
    l16=dict(patch_size=16, dim=1024, depth=24, num_heads=16),
    h14=dict(patch_size=14, dim=1280, depth=32, num_heads=16),
    twob14=dict(patch_size=14, dim=2560, depth=24, num_heads=32),
)

CONFIS = {
    "debug": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["debug"],
        url=None,
    ),
    "mae_refined_l16": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["l16"],
        url="https://ml.jku.at/research/mimrefiner/download/maerefined_l16.th",
    ),
    "d2v2_refined_l16": dict(
        ctor=PostnormVit,
        ctor_kwargs=VIT_CONFIGS["l16"],
        url="https://ml.jku.at/research/mimrefiner/download/d2v2refined_l16.th",
    ),
    "mae_refined_h14": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["h14"],
        url="https://ml.jku.at/research/mimrefiner/download/maerefined_h14.th",
    ),
    "d2v2_refined_h14": dict(
        ctor=PostnormVit,
        ctor_kwargs=VIT_CONFIGS["h14"],
        url="https://ml.jku.at/research/mimrefiner/download/d2v2refined_h14.th",
    ),
    "mae_refined_twob14": dict(
        ctor=PrenormVit,
        ctor_kwargs=VIT_CONFIGS["twob14"],
        url="https://ml.jku.at/research/mimrefiner/download/maerefined_twob14.th",
    ),
}


def load_model(ctor, ctor_kwargs, url, **kwargs):
    model = ctor(**ctor_kwargs, **kwargs)
    if url is not None:
        sd = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(sd["state_dict"])
    return model


for name, config in CONFIS.items():
    globals()[name] = partial(load_model, **config)
