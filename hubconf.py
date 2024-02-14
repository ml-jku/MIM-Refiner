from functools import partial

dependencies = ["torch", "kappamodules"]

VIT_CONFIGS = dict(
    l16=dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
    h14=dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16),
    twob14=dict(patch_size=14, embed_dim=2560, depth=24, num_heads=32),
)

CONFIS = {
    "MAE-Refined-L/16": dict(
        kind="prenorm",
        vit=VIT_CONFIGS["l16"],
        url="...",
    ),
    "D2V2-Refined-L/16": dict(
        kind="postnorm",
        vit=VIT_CONFIGS["l16"],
        url="...",
    ),
    "MAE-Refined-H/14": dict(
        kind="prenorm",
        vit=VIT_CONFIGS["h14"],
        url="...",
    ),
    "D2V2-Refined-H/14": dict(
        kind="postnorm",
        vit=VIT_CONFIGS["h14"],
        url="...",
    ),
    "MAE-Refined-2B/14": dict(
        kind="prenorm",
        vit=VIT_CONFIGS["twob14"],
        url="...",
    ),
}

for model_type in AVAILABLE_MODELS:
    for model_name in AVAILABLE_MODELS[model_type]:
        globals()[f"{model_name}_{model_type}"] = partial(
            build_model, model_name, model_type
        )