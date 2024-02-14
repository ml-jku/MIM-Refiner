from functools import partial

import einops
import torch
from kappamodules.functional.pos_embed import interpolate_sincos
from kappamodules.init import init_norm_as_noaffine
from kappamodules.transformer import PostnormBlock
from kappamodules.vit import VitPatchEmbed, VitPosEmbed2d, VitSeperateNorm, VitClassTokens
from torch import nn

from models.base.single_model_base import SingleModelBase
from models.poolings import pooling_from_kwargs
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier
from utils.factory import create
from utils.formatting_util import list_to_string
from utils.param_checking import to_ntuple


class Data2Vec2Vit(SingleModelBase):
    def __init__(
            self,
            patch_size,
            dim,
            depth,
            num_attn_heads,
            mlp_hidden_dim=None,
            drop_path_rate=0.,
            drop_path_decay=True,
            num_cls_tokens=1,
            pos_embed_is_learnable=False,
            use_seperate_norm=False,
            mode=None,
            pooling=None,
            init_weights="xavier_uniform",
            eps=1e-6,
            **kwargs,
    ):
        super().__init__(**kwargs)
        ndim = len(self.input_shape) - 1
        self.patch_size = to_ntuple(patch_size, n=ndim)
        self.static_ctx["patch_size"] = self.patch_size
        self.dim = dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.mode = mode
        self.eps = eps

        # initialize patch_embed
        self.patch_embed = VitPatchEmbed(
            dim=dim,
            num_channels=self.input_shape[0],
            resolution=self.input_shape[1:],
            patch_size=self.patch_size,
        )
        self.static_ctx["sequence_lengths"] = self.patch_embed.seqlens

        # pos embed
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=dim, is_learnable=pos_embed_is_learnable)
        self.logger.info(f"pos_embed.is_learnable={self.pos_embed.is_learnable}")

        # 0, 1 or more cls tokens
        self.cls_tokens = VitClassTokens(dim=dim, num_tokens=num_cls_tokens)
        self.static_ctx["num_aux_tokens"] = self.num_aux_tokens = num_cls_tokens

        # data2vec2 uses norm before first block
        self.embed_norm = nn.LayerNorm(dim, eps=eps)

        # norm ctors
        if use_seperate_norm:
            norm_ctor = partial(VitSeperateNorm, num_aux_tokens=self.num_aux_tokens)
        else:
            norm_ctor = nn.LayerNorm

        # stochastic depth
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
            self.logger.info(f"using drop_path_decay: {list_to_string(dpr)}")
        else:
            dpr = [drop_path_rate] * self.depth
            self.logger.info(f"drop_path_rate: {drop_path_rate}")

        # blocks
        self.blocks = nn.ModuleList([
            PostnormBlock(
                dim=dim,
                num_heads=num_attn_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                norm_ctor=norm_ctor,
                drop_path=dpr[i],
                eps=eps,
                init_weights=init_weights,
            )
            for i in range(self.depth)
        ])

        # initialize head
        self.pooling = create(pooling, pooling_from_kwargs, static_ctx=self.static_ctx)
        if mode is None:
            assert self.output_shape is None
            assert self.pooling is None
            self.head_norm = None
            self.head = None
            self.output_shape = (self.patch_embed.num_patches + self.num_aux_tokens, dim)
        elif mode == "classifier":
            assert self.output_shape is not None and len(self.output_shape) == 1
            assert self.pooling is not None
            # data2vec2 uses norm before first head
            self.head_norm = nn.LayerNorm(dim, eps=eps)
            head_in_dim = self.pooling.get_output_shape((self.patch_embed.num_patches + self.num_aux_tokens, dim))[0]
            self.head = nn.Linear(head_in_dim, self.output_shape[0])
        else:
            raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True):
        # LEGACY START
        if "norm.weight" in state_dict:
            state_dict["embed_norm.weight"] = state_dict.pop("norm.weight")
        if "norm.bias" in state_dict:
            state_dict["embed_norm.bias"] = state_dict.pop("norm.bias")
        # LEGACY END
        old_pos_embed = state_dict["pos_embed.embed"]
        if old_pos_embed.shape != self.pos_embed.embed.shape:
            self.logger.info(f"interpolate pos_embed: {old_pos_embed.shape} -> {self.pos_embed.embed.shape}")
            state_dict["pos_embed.embed"] = interpolate_sincos(embed=old_pos_embed, seqlens=self.pos_embed.seqlens)
        if self.mode == "classifier":
            missing_keys, unexpected_keys = super().load_state_dict(state_dict=state_dict, strict=False)
            if len(missing_keys) == 4:
                assert "head.weight" in missing_keys
                assert "head.bias" in missing_keys
                assert "head_norm.weight" in missing_keys
                assert "head_norm.bias" in missing_keys
            else:
                assert len(missing_keys) == 0, f"missing_keys={missing_keys}"
            assert len(unexpected_keys) == 0, f"unexpected_keys={unexpected_keys}"
            return missing_keys, unexpected_keys
        return super().load_state_dict(state_dict=state_dict, strict=strict)

    def model_specific_initialization(self):
        # init norm (rest is initialized from the module)
        init_norm_as_noaffine(self.embed_norm)

        # initialize head
        if self.mode == "classifier":
            # following MAE https://github.com/facebookresearch/mae/blob/main/main_finetune.py#L257
            nn.init.trunc_normal_(self.head.weight, std=2e-5)
            nn.init.zeros_(self.head.bias)

    def get_model_specific_param_group_modifiers(self):
        modifiers = []
        if self.cls_tokens.num_tokens > 0:
            modifiers.append(ExcludeFromWdByNameModifier(name="cls_tokens.tokens"))
        if self.pos_embed.is_learnable:
            modifiers.append(ExcludeFromWdByNameModifier(name="pos_embed.embed"))
        return modifiers

    def forward(self, x, mask=None, mask_generator=None, idx=None):
        outputs = {}

        # embed patches
        x = self.patch_embed(x)
        # add pos_embed
        x = self.pos_embed(x)

        if mask is not None and mask_generator is not None:
            # apply mask
            assert mask.ndim == 2
            x = einops.rearrange(x, "b ... d -> b (...) d")
            ids_shuffle = torch.argsort(mask.byte(), dim=1)
            seqlen_keep = len(mask[0]) - mask[0].sum()
            ids_keep = ids_shuffle[:, :seqlen_keep]
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.size(2)))
            # generate mask -> apply mask
            x, _, _ = mask_generator.get_mask(x, idx=idx)
        elif mask_generator is not None:
            # generate mask -> apply mask
            x, mask, ids_restore = mask_generator.get_mask(x, idx=idx)
            outputs["mask"] = mask
            outputs["ids_restore"] = ids_restore
        elif mask is not None:
            # apply mask
            assert mask.ndim == 2 and len(mask) == len(x)
            x = einops.rearrange(x, "b ... d -> b (...) d")
            ids_shuffle = torch.argsort(mask.byte(), dim=1)
            seqlen_keep = len(mask[0]) - mask[0].sum()
            ids_keep = ids_shuffle[:, :seqlen_keep]
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.size(2)))
        else:
            # no mask -> flatten to 1d
            x = einops.rearrange(x, "b ... d -> b (...) d")

        # add cls token
        x = self.cls_tokens(x)

        x = self.embed_norm(x)

        # apply blocks
        for blk in self.blocks:
            x = blk(x)

        if self.pooling is not None:
            x = self.pooling(x)
        if self.head is not None:
            x = self.head_norm(x)
            x = self.head(x)
        else:
            assert self.head_norm is None

        outputs["main"] = x
        return outputs

    def classify(self, *args, **kwargs):
        assert self.mode == "classifier"
        outputs = self.forward(*args, **kwargs)
        return dict(main=outputs["main"])
