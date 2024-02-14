from .base.param_group_modifier import ParamGroupModifier


class LayerwiseLrDecayModifier(ParamGroupModifier):
    def __init__(self, decay, skip_layers=None):
        self.decay = decay
        self.skip_layers = skip_layers

    def get_properties(self, model, name, param):
        # adapted from BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        # this will split the model into len(blocks) + 2 "layers"
        # stem (patch_embed, cls_token, pos_embed) -> blocks -> last norm
        # this means that the last block will already be decayed
        if hasattr(model, "blocks"):
            num_layers = len(model.blocks) + 1
        elif hasattr(model, "model"):
            # e.g. torch_hub_model
            assert hasattr(model.model, "blocks")
            num_layers = len(model.model.blocks) + 1
            if name.startswith("model."):
                name = name[len("model."):]
        else:
            raise NotImplementedError
        scales = list(self.decay ** (num_layers - i) for i in range(num_layers))

        # shift scales if layers are skipped (first layer that is not skipped has the max learning rate)
        if self.skip_layers is not None:
            scales = scales[self.skip_layers:] + [1.] * self.skip_layers

        if (
                name.startswith("patch_embed")
                or name.startswith("cls_token")
                or name.startswith("pos_embed")
                or name.startswith("embed_norm")
                or name == "mask_token"
        ):
            return dict(lr_scale=scales[0])
        elif name.startswith("block"):
            layer = int(name.split('.')[1]) + 1
            return dict(lr_scale=scales[layer])
        elif name.startswith("norm."):
            # last norm is not scaled (i.e. original learning rate)
            return {}
        elif name.startswith("head_norm."):
            # last norm is not scaled (i.e. original learning rate)
            return {}
        elif name.startswith("head."):
            # head is not scaled (i.e. original learning rate)
            return {}
        else:
            raise NotImplementedError

    def __str__(self):
        return f"{type(self).__name__}(decay={self.decay})"
