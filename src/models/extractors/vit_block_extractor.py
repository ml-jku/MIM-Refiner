from .base.extractor_base import ExtractorBase
from .base.forward_hook import ForwardHook
from kappautils.param_checking import check_at_most_one

class VitBlockExtractor(ExtractorBase):
    def __init__(self, block_index=None, block_indices=None, num_last_blocks=None, use_next_norm=False, **kwargs):
        super().__init__(**kwargs)
        assert check_at_most_one(block_index, block_indices, num_last_blocks), \
            "use either block_index or block_indices or num_last_blocks"
        if block_index is not None:
            block_indices = [block_index]
        elif block_indices is not None:
            assert isinstance(block_indices, (tuple, list))

        self.block_indices = block_indices
        self.num_last_blocks = num_last_blocks
        self.use_next_norm = use_next_norm
        # populate on register_hooks
        self._resolved_block_indices = None

    def to_string(self):
        if self.block_indices is not None:
            block_indices_str = f"block_indices=[{','.join(map(str, self.block_indices))}]"
        else:
            block_indices_str = f"num_last_blocks={self.num_last_blocks}"
        if self.pooling is not None:
            pooling_str = f"pooling={self.pooling},"
        else:
            pooling_str = ""
        return (
            f"BlockExtractor("
            f"{block_indices_str},"
            f"{pooling_str}"
            f"use_next_norm={self.use_next_norm}"
            f")"
        )

    def _get_own_outputs(self):
        return {f"block{block_idx}": self.outputs[f"block{block_idx}"] for block_idx in self._resolved_block_indices}

    def _register_hooks(self, model):
        if self.block_indices is not None:
            # make negative indices positive (for consistency in name)
            self._resolved_block_indices = [idx for idx in self.block_indices]
            for i in range(len(self._resolved_block_indices)):
                if self._resolved_block_indices[i] < 0:
                    self._resolved_block_indices[i] = len(model.blocks) + self._resolved_block_indices[i]
            # remove possible duplicates and sort
            self._resolved_block_indices = sorted(list(set(self._resolved_block_indices)))
        elif self.num_last_blocks is not None:
            self._resolved_block_indices = list(range(len(model.blocks) - self.num_last_blocks, len(model.blocks)))
        else:
            # create a hook on each block
            self._resolved_block_indices = list(range(len(model.blocks)))

        for block_idx in self._resolved_block_indices:
            hook = ForwardHook(
                outputs=self.outputs,
                output_name=f"block{block_idx}",
                raise_exception=self.raise_exception and block_idx == self._resolved_block_indices[-1],
                **self.hook_kwargs,
            )
            if self.use_next_norm:
                if block_idx == len(model.blocks) - 1:
                    # last block uses the "model.norm" as normalization
                    model.norm.register_forward_hook(hook)
                else:
                    # use the norm of the next block
                    model.blocks[block_idx + 1].norm1.register_forward_hook(hook)
            else:
                # use the unnormalized block output
                model.blocks[block_idx].register_forward_hook(hook)
            self.hooks.append(hook)
