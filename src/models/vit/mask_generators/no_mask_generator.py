import torch

from .base.mask_generator import MaskGenerator


class NoMaskGenerator(MaskGenerator):
    def __init__(self, **kwargs):
        super().__init__(mask_ratio=0., mask_ratio_schedule=None, **kwargs)

    def _generate_noise(self, x, generator=None):
        bs, resolution, dim = self._get_shape(x)
        # standard random masking (no mask due to mask_ratio == 0)
        return torch.rand(bs, *resolution, device=x.device, generator=generator)
