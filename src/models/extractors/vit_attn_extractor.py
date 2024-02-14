from .base.extractor_base import ExtractorBase
from .base.forward_hook import ForwardHook


class VitAttnExtractor(ExtractorBase):
    def __init__(self, block_index=None, block_indices=None, **kwargs):
        super().__init__(**kwargs)
        if block_index is not None:
            assert block_indices is None, "use block_index or block_indices but not both"
            block_indices = [block_index]
        else:
            if block_indices is not None:
                assert block_index is None,  "use block_index or block_indices but not both"
                assert block_indices is None or isinstance(block_indices, (tuple, list))
        self.block_indices = block_indices

    def to_string(self):
        return (
            f"VitAttnExtractor("
            f"block_indices=[{','.join(map(str, self.block_indices))}],"
            f"pooling={self.pooling},"
            f")"
        )

    def _get_own_outputs(self):
        return {f"attn{block_idx}": self.outputs[f"attn{block_idx}"] for block_idx in self.block_indices}

    def _register_hooks(self, model):
        # If block_indices is None, create a hook on each block
        if self.block_indices is None:
            self.block_indices = list(range(len(model.blocks)))
        else:
            # make negative indices positive (for consistency in name)
            for i in range(len(self.block_indices)):
                if self.block_indices[i] < 0:
                    self.block_indices[i] = len(model.blocks) + self.block_indices[i]
            # remove possible duplicates and sort
            self.block_indices = sorted(list(set(self.block_indices)))

        for block_idx in self.block_indices:
            hook = ForwardHook(
                outputs=self.outputs,
                output_name=f"attn{block_idx}",
                raise_exception=self.raise_exception and block_idx == self.block_indices[-1],
                **self.hook_kwargs,
            )
            model.blocks[block_idx].attn.register_forward_hook(hook)
            self.hooks.append(hook)
