import torch
import torch.nn.functional as F

from utils.param_checking import to_2tuple
from .base.mask_generator import MaskGenerator


class RandomMaskGenerator(MaskGenerator):
    def __init__(self, mask_size=None, **kwargs):
        super().__init__(**kwargs)
        if mask_size is not None:
            self.mask_size = to_2tuple(mask_size)
            assert isinstance(self.mask_size[0], int) and self.mask_size[0] > 0
            assert isinstance(self.mask_size[1], int) and self.mask_size[1] > 0
        else:
            self.mask_size = None

    def __str__(self):
        arg_strs = [f"mask_ratio={self.mask_ratio})"]
        if self.mask_size is not None:
            arg_strs.append(f"mask_size=[{self.mask_size[0]},{self.mask_size[1]}]")
        return f"{type(self).__name__}({','.join(arg_strs)})"

    def _generate_noise(self, x, generator=None):
        bs, resolution, dim = self._get_shape(x)
        if self.mask_size is None:
            # standard random masking
            return torch.rand(bs, *resolution, device=x.device, generator=generator)
        # make sure "adjacent" patches are masked together by assigning them the same noise
        assert all(resolution[i] % self.mask_size[i] == 0 for i in range(len(resolution)))
        pooled = [resolution[i] // self.mask_size[i] for i in range(len(resolution))]
        noise = torch.rand(bs, *pooled, device=x.device, generator=generator)
        noise = F.interpolate(noise.unsqueeze(1), size=resolution, mode="nearest").squeeze(1)
        return noise
