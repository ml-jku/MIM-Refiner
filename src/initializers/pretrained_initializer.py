from pathlib import Path

import einops
import torch

from models.vit.mae_decoder import MaeDecoder
from .base.initializer_base import InitializerBase


class PretrainedInitializer(InitializerBase):
    """ initialize with weights from an external, pretrained checkpoints (e.g. original facebook MAE checkpoints) """

    def __init__(self, weights_file, root_key=None, key_mapping=None, **kwargs):
        super().__init__(**kwargs)
        self.weights_file = weights_file
        self.weights_uri = Path(self.path_provider.model_path / weights_file).expanduser()
        assert self.weights_uri.exists() and self.weights_uri.is_file(), self.weights_uri.as_posix()
        self.key_mapping = key_mapping
        self.root_key = root_key

    def _get_model_kwargs(self):
        self.logger.info(f"loading ckpt kwargs for '{self.weights_uri}'")
        kwargs = dict(kind="vit.vit")
        if "data2vec2" in self.weights_file:
            kwargs["kind"] = "vit.data2vec2_vit"
        # I-JEPA no CLS token
        if "ijepa" in self.weights_file:
            kwargs["num_cls_tokens"] = 0

        # ViT dimensions
        if "base8" in self.weights_file:
            return dict(patch_size=8, dim=768, num_attn_heads=12, depth=12, **kwargs)
        if "base16" in self.weights_file:
            return dict(patch_size=16, dim=768, num_attn_heads=12, depth=12, **kwargs)
        if "large16" in self.weights_file:
            return dict(patch_size=16, dim=1024, num_attn_heads=16, depth=24, **kwargs)
        if "huge16" in self.weights_file:
            return dict(patch_size=16, dim=1280, num_attn_heads=16, depth=32, **kwargs)
        if "huge14" in self.weights_file:
            return dict(patch_size=14, dim=1280, num_attn_heads=16, depth=32, **kwargs)
        if "twob14" in self.weights_file:
            return dict(patch_size=14, dim=2560, num_attn_heads=32, depth=24, **kwargs)

        sd = torch.load(self.weights_uri, map_location=torch.device("cpu"))
        if "ctor_kwargs" in sd:
            kwargs = sd["ctor_kwargs"]
        else:
            kwargs = {}
        self.logger.info(f"found kwargs: {kwargs}")
        return kwargs

    def init_weights(self, model):
        self.logger.info(f"loading weights from '{self.weights_uri}'")
        sd = torch.load(self.weights_uri, map_location=torch.device("cpu"))
        # unpack state_dict
        # - MLPlayground stores weights in "state_dict" field
        if "state_dict" in sd:
            sd = sd["state_dict"]
        # - MAE stores weights in "model" field
        elif "model" in sd:
            sd = sd["model"]
        # iBOT stores weights in "teacher" and "student" field -> eval teacher
        elif "teacher" in sd:
            sd = sd["teacher"]
        # - I-JEPA stores weights in "target_encoder"
        elif "target_encoder" in sd:
            sd = sd["target_encoder"]
        # select model (e.g. used when student/teacher is stored in same checkpoint)
        if self.root_key is not None:
            sd = sd[self.root_key]

        #
        if isinstance(model, MaeDecoder) and self.weights_file in [
            "mae_base16.pth", "mae_large16.pth", "mae_huge14.pth",  # MAE
            "mae_base16res448.pth", "mae_large16res448.pth",  # long sequence MAE
            "mae_base16res448e800.pth", "mae_large16res448e800.pth",  # long sequence MAE
        ]:
            for key in sd.keys():
                print(key)
            sd = {k: v for k, v in sd.items() if "decoder" in k}
        elif self.weights_file in [
            "mae_base16.pth", "mae_large16.pth", "mae_huge14.pth",  # MAE
            "mae_base16res448.pth", "mae_large16res448.pth",  # long sequence MAE
            "mae_base16res448e800.pth", "mae_large16res448e800.pth",  # long sequence MAE
        ]:
            sd = {k: v for k, v in sd.items() if "decoder" not in k and k != "mask_token"}
        elif "layergrafting" in self.weights_file:
            sd = {
                k.replace("module.momentum_encoder.", ""): v
                for k, v in sd.items()
                if k.startswith("module.momentum_encoder.") and "head" not in k
            }
        elif "ibot" in self.weights_file:
            sd = {
                k.replace("backbone.", ""): v
                for k, v in sd.items()
                if "backbone." in k
            }
        elif "mugs" in self.weights_file:
            sd = {
                k.replace("backbone.", ""): v
                for k, v in sd.items()
                if "relation_blocks" not in k and "group_head" not in k and "instance_head" not in k
            }
        elif "ijepa" in self.weights_file:
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
        elif "data2vec2" in self.weights_file:
            # finetuned checkpoints have prefix model.
            sd = {key.replace("model.", ""): value for key, value in sd.items()}
            # convert
            sd["patch_embed.proj.weight"] = sd.pop("modality_encoders.IMAGE.local_encoder.proj.weight")
            sd["patch_embed.proj.bias"] = sd.pop("modality_encoders.IMAGE.local_encoder.proj.bias")
            sd["cls_tokens.tokens"] = sd.pop("modality_encoders.IMAGE.extra_tokens")
            sd["pos_embed.embed"] = sd.pop("modality_encoders.IMAGE.fixed_positional_encoder.positions")
            sd["embed_norm.weight"] = sd.pop("modality_encoders.IMAGE.context_encoder.norm.weight")
            sd["embed_norm.bias"] = sd.pop("modality_encoders.IMAGE.context_encoder.norm.bias")
            sd = {k: v for k, v in sd.items() if "decoder" not in k}
            sd.pop("_ema", None)
        # old checkpoints flatten pos_embed
        if self.weights_file in ["maereimpl_large16.th"]:
            pos_embed = sd["pos_embed"]
            sd["pos_embed"] = pos_embed.reshape(1, 14, 14, pos_embed.size(2))
        # old decoder has different naming
        if self.weights_file == "maereimpl_large16_decoder.th":
            for key in list(sd.keys()):
                if key.startswith("embed."):
                    sd[key.replace("embed.", "first_proj.")] = sd.pop(key)
                if key.startswith("norm."):
                    sd[key.replace("norm.", "last_norm.")] = sd.pop(key)
                if key.startswith("pred."):
                    sd[key.replace("pred.", "last_proj.")] = sd.pop(key)

        # remap keys
        if self.key_mapping is not None:
            for old_key, new_key in self.key_mapping.items():
                sd[new_key] = sd.pop(old_key)

        # convert to v4 vit parameter names
        if "pos_embed" in sd:
            sd["pos_embed.embed"] = sd.pop("pos_embed")
        if "cls_token" in sd:
            sd["cls_tokens.tokens"] = sd.pop("cls_token")
        # convert pos_embed to 2d: e.g. (1, 197, 768) -> (1, 14, 14, 768)
        if "pos_embed.embed" in sd:
            pos_embed = sd["pos_embed.embed"]
            if pos_embed.ndim == 3:
                # resolution=224x224 patch_size=8
                if pos_embed.size(1) == 785:
                    pos_embed = pos_embed[:, 1:]
                    seqlen_h = seqlen_w = 28
                # resolution=224x224 patch_size=16
                elif pos_embed.size(1) == 197:
                    pos_embed = pos_embed[:, 1:]
                    seqlen_h = seqlen_w = 14
                elif pos_embed.size(1) == 196:
                    seqlen_h = seqlen_w = 14
                # resolution=224x224 patch_size=14
                elif pos_embed.size(1) == 257:
                    pos_embed = pos_embed[:, 1:]
                    seqlen_h = seqlen_w = 16
                elif pos_embed.size(1) == 256:
                    seqlen_h = seqlen_w = 16
                # resolution=448x448 patch_size=16
                elif pos_embed.size(1) == 784:
                    seqlen_h = seqlen_w = 28
                else:
                    raise NotImplementedError
                sd["pos_embed.embed"] = einops.rearrange(
                    pos_embed,
                    "1 (seqlen_h seqlen_w) dim -> 1 seqlen_h seqlen_w dim",
                    seqlen_h=seqlen_h,
                    seqlen_w=seqlen_w,
                )

        # load
        model.load_state_dict(sd)
