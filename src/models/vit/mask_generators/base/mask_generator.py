import numpy as np
import einops
import torch
from kappaschedules import object_to_schedule


class MaskGenerator:
    def __init__(self, mask_ratio, mask_ratio_schedule=None, seed=None, update_counter=None):
        super().__init__()
        self.update_counter = update_counter
        self.seed = seed
        self.constant_mask_ratio = mask_ratio
        self.mask_ratio_schedule = None
        assert self.update_counter is not None or mask_ratio_schedule is None
        self.mask_ratio_schedule = object_to_schedule(
            mask_ratio_schedule,
            batch_size=self.update_counter.effective_batch_size if self.update_counter is not None else None,
            updates_per_epoch=self.update_counter.updates_per_epoch if self.update_counter is not None else None,
        )

    def _get_generator(self, x, idx):
        if self.seed is None:
            return None
        assert idx is not None
        # identify generator by the index of the first sample in the batch
        # since deterministic behavior is typically coupled with sequential dataset iteration this is unique
        # otherwise the random noise has to be generated on a per-sample basis
        return torch.Generator(device=x.device).manual_seed(self.seed + idx[0].item())

    @property
    def mask_ratio(self):
        if self.mask_ratio_schedule is None:
            return self.constant_mask_ratio
        return self.constant_mask_ratio * self.mask_ratio_schedule.get_value(
            step=self.update_counter.cur_checkpoint.update,
            total_steps=self.update_counter.end_checkpoint.update,
        )

    @property
    def is_masked(self):
        return self.mask_ratio > 0

    def _generate_noise(self, x, generator=None):
        raise NotImplementedError

    @staticmethod
    def _get_shape(x):
        bs = x.shape[0]
        dim = x.shape[-1]
        resolution = x.shape[1:-1]
        return bs, resolution, dim

    def get_gradmask(self, x, idx=None):
        bs, resolution, dim = self._get_shape(x)
        seqlen = np.prod(resolution)
        seqlen_unmasked = int(seqlen * (1 - self.mask_ratio))
        seqlen_masked = seqlen - seqlen_unmasked

        # generate noise
        noise = self._generate_noise(x, generator=self._get_generator(x, idx))

        # reshape x from "image" to sequence
        x = einops.rearrange(x, "bs ... dim -> bs (...) dim")

        # sort noise for each sample
        noise = einops.rearrange(noise, "bs ... -> bs (...)")
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([bs, seqlen], device=x.device, dtype=torch.bool)
        mask[:, :seqlen_unmasked] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # prepare indices for torch.gather
        ids_shuffle = ids_shuffle.unsqueeze(-1).expand(-1, -1, dim)
        ids_keep = ids_shuffle[:, :seqlen_unmasked]
        masked_ids, unmasked_ids = ids_shuffle.split([seqlen_masked, seqlen_unmasked], dim=1)

        return ids_keep, ids_restore, masked_ids, unmasked_ids, mask

    def get_mask(self, x, idx=None):
        bs, resolution, dim = self._get_shape(x)
        seqlen = np.prod(resolution)
        seqlen_keep = int(seqlen * (1 - self.mask_ratio))

        # generate noise
        noise = self._generate_noise(x, generator=self._get_generator(x, idx))

        # reshape x from "image" to sequence
        x = einops.rearrange(x, "bs ... dim -> bs (...) dim")

        # sort noise for each sample
        noise = einops.rearrange(noise, "bs ... -> bs (...)")
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([bs, seqlen], device=x.device, dtype=torch.bool)
        mask[:, :seqlen_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # keep the first subset
        ids_keep = ids_shuffle[:, :seqlen_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, dim))
        return x_masked, mask, ids_restore
