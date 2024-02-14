import einops
from torch import nn
import torch

class Data2vec2ConvDecoderBlock(nn.Module):
    def __init__(self, input_dim, dim, kernel_size, groups, eps=1e-6):
        super().__init__()
        assert kernel_size % 2 == 1

        self.input_dim = input_dim
        self.dim = dim
        self.kernel_size = kernel_size
        self.groups = groups
        self.eps = eps

        self.conv = nn.Conv2d(
            input_dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = einops.rearrange(x, "bs c h w -> bs h w c")
        x = self.norm(x)
        x = einops.rearrange(x, "bs h w c -> bs c h w")
        x = self.act(x)
        if self.input_dim == self.dim:
            return residual + x
        return x
