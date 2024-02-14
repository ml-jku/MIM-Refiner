from .base.param_group_modifier import ParamGroupModifier


class VitPatchEmbedLrScaleModifier(ParamGroupModifier):
    def __init__(self, scale):
        self.scale = scale

    def get_properties(self, model, name, param):
        if name.startswith("patch_embed."):
            return dict(lr_scale=self.scale)
        return {}

    def __str__(self):
        return f"{type(self).__name__}(scale={self.scale})"
