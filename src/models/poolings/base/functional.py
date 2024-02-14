import einops


def to_image(all_tokens, ctx, static_ctx):
    patch_tokens = all_tokens[:, static_ctx["num_aux_tokens"]:]
    patch_height, patch_width = static_ctx["patch_size"]
    _, image_height, image_width = ctx["input_shape"] if "input_shape" in ctx else static_ctx["input_shape"]
    img = einops.rearrange(
        patch_tokens,
        "b (p q) c -> b c p q",
        p=image_height // patch_height,
        q=image_width // patch_width,
    )
    return img
