import torch
from torch import nn

from models.base.single_model_base import SingleModelBase
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier
from utils.factory import create


class TorchHubModel(SingleModelBase):
    def __init__(self, repo, model, mode=None, drop_path_rate=None, **kwargs):
        super().__init__(**kwargs)
        # DINOv2:
        # dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        # dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        # dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        kwargs = {}
        if drop_path_rate is not None:
            kwargs["drop_path_rate"] = drop_path_rate
        self.model = torch.hub.load(repo, model, **kwargs)
        # populate static_ctx
        if model in ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]:
            self.static_ctx["num_aux_tokens"] = 1
        self.mode = mode
        if mode is None:
            self.head = None
            patch_height, patch_width = self.model.patch_embed.patch_size
            _, height, width = self.input_shape
            assert height % patch_height == 0
            assert width % patch_width == 0
            seqlen = (height // patch_height, width // patch_width)
            self.output_shape = (seqlen[0] * seqlen[1], self.model.embed_dim)
        elif mode == "classifier":
            self.head = nn.Linear(self.model.embed_dim, self.output_shape[0])
        else:
            raise NotImplementedError

    @property
    def blocks(self):
        return self.model.blocks

    def get_model_specific_param_group_modifiers(self):
        modifiers = [
            ExcludeFromWdByNameModifier(name="model.cls_token"),
            ExcludeFromWdByNameModifier(name="model.pos_embed"),
        ]
        if self.model.register_tokens is not None:
            modifiers.append(ExcludeFromWdByNameModifier(name="model.register_tokens"))
        return modifiers

    def model_specific_initialization(self):
        if self.mode == "classifier":
            # following MAE https://github.com/facebookresearch/mae/blob/main/main_finetune.py#L257
            nn.init.trunc_normal_(self.head.weight, std=2e-5)
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.model(x, is_training=True)
        if self.mode is None:
            x = torch.concat([x["x_norm_clstoken"].unsqueeze(1), x["x_norm_patchtokens"]], dim=1)
        elif self.mode == "classifier":
            x = self.head(x["x_norm_clstoken"])
        else:
            raise NotImplementedError
        return dict(main=x)

    def classify(self, x):
        return self(x)