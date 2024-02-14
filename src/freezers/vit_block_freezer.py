from kappautils.param_checking import check_at_least_one
from .base.freezer_base import FreezerBase


class VitBlockFreezer(FreezerBase):
    def __init__(
            self,
            start_index=None,
            start_percent=None,
            end_index=None,
            end_percent=None,
            indices=None,
            freeze_patch_embed=True,
            freeze_cls=True,
            freeze_pos_embed=True,
            freeze_last_norm=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        if indices is not None:
            assert start_index is None and end_index is None and start_percent is None and end_percent is None
            assert isinstance(indices, list) and all(isinstance(block_idx, int) for block_idx in indices)
        else:
            assert check_at_least_one(start_index, end_index, start_percent, end_percent)
            assert not (start_index is not None and start_percent is not None)
            assert not (end_index is not None and end_percent is not None)
            assert start_percent is None or 0. <= start_percent <= 1.
            assert end_percent is None or 0. <= end_percent <= 1.

        self.start_index = start_index
        self.start_percent = start_percent
        self.end_index = end_index
        self.end_percent = end_percent
        self.indices = indices
        self.freeze_patch_embed = freeze_patch_embed
        self.freeze_cls = freeze_cls
        self.freeze_pos_embed = freeze_pos_embed
        self.freeze_last_norm = freeze_last_norm

    def __str__(self):
        if self.indices is not None:
            block_idx_str = f"block_idxs={self.indices},"
        else:
            if self.start_index is not None:
                start_str = f"start_index={self.start_index},"
            elif self.start_percent is not None:
                start_str = f"start_percent={self.start_percent},"
            else:
                start_str = ""
            if self.end_index is not None:
                end_str = f"end_index={self.end_index},"
            elif self.end_percent is not None:
                end_str = f"end_percent={self.end_percent},"
            else:
                end_str = ""
            block_idx_str = f"{start_str}{end_str}"
        return (
            f"{type(self).__name__}({block_idx_str}"
            f"freeze_patch_embed={self.freeze_patch_embed},"
            f"freeze_cls={self.freeze_cls})"
        )

    def _update_state(self, model, requires_grad):
        # select block idxs
        if self.indices is None:
            if self.start_index is not None:
                start_index = self.start_index
            elif self.start_percent is not None:
                start_index = int(len(models.blocks) * self.start_percent)
            else:
                start_index = 0
            if self.end_index is not None:
                end_index = self.end_index
            elif self.end_percent is not None:
                end_index = int(len(model.blocks) * self.end_percent)
            else:
                end_index = len(model.blocks)
            # allow selecting ranges via negative indices
            block_indices = list(range(len(model.blocks)))[start_index:end_index]
        else:
            block_indices = self.indices

        # freeze blocks
        for idx in block_indices:
            block = model.blocks[idx]
            block.eval()
            for p in block.parameters():
                p.requires_grad = requires_grad

        # freeze other stuff
        if self.freeze_cls:
            model.cls_tokens.tokens.requires_grad = requires_grad
        if self.freeze_patch_embed:
            model.patch_embed.eval()
            for p in model.patch_embed.parameters():
                p.requires_grad = requires_grad
        if self.freeze_pos_embed:
            for p in model.pos_embed.parameters():
                p.requires_grad = requires_grad
        if self.freeze_last_norm:
            for p in model.norm.parameters():
                p.requires_grad = requires_grad