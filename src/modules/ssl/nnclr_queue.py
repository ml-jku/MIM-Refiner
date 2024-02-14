import torch
import torch.nn.functional as F
from torch import nn

from distributed.gather import all_gather_nograd
from utils.loss_utils import apply_reduction
import kappamodules.utils.tensor_cache as tc


class NnclrQueue(nn.Module):
    def __init__(self, size, dim, topk=1, guidance="none", random_swap_p=0., random_swap_topk=None, num_classes=None):
        super().__init__()
        self.size = size
        self.dim = dim
        self.topk = topk
        self.guidance = guidance
        self.random_swap_p = random_swap_p
        self.random_swap_topk = random_swap_topk
        self.num_classes = num_classes
        # queue properties
        self.register_buffer("x", torch.randn(size, dim))
        self.register_buffer("idx", -torch.ones(size, dtype=torch.long))
        self.register_buffer("cls", -torch.ones(size, dtype=torch.long))
        self.register_buffer("age", torch.zeros(size, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    def reset_parameters(self):
        torch.nn.init.normal_(self.x)
        torch.nn.init.constant_(self.idx, -1)
        torch.nn.init.constant_(self.cls, -1)
        torch.nn.init.zeros_(self.age)
        torch.nn.init.zeros_(self.ptr)

    @torch.no_grad()
    def forward(self, x, idx, cls=None, confidence=None, reduction="mean"):
        assert x.ndim == 2
        x = x.detach()
        batch_size = len(idx)
        num_views = len(x) // batch_size

        normed_x = F.normalize(x, dim=-1)
        normed_queue_x = F.normalize(self.x, dim=-1)

        # calculate similarities
        sim = normed_x @ normed_queue_x.T
        if self.training:
            # avoid retrieving the same sample (can happen due to epoch boundaries or small datasets)
            mask = (idx[:, None] == self.idx[None, :]).repeat(num_views, 1)
            sim[mask] = -1.
        # manipulate similarities
        if self.guidance == "none":
            if self.random_swap_p > 0:
                if self.random_swap_topk is not None:
                    # simulate swapping label by swapping with the highest similarity samples from another cluster
                    # retrieve a random sample (samples from this cluster will be used as "samples with another label")
                    reference_sample_idx = torch.randint(low=0, high=self.size, size=(batch_size,), device=x.device)
                    # retrieve samples from the reference cluster
                    reference_sim = normed_queue_x[reference_sample_idx] @ normed_queue_x.T
                    _, topk_reference_idx = reference_sim.topk(self.random_swap_topk, dim=1, largest=True)
                    # create mask which samples are from the reference cluster
                    is_reference_cluster = torch.zeros(
                        size=(len(topk_reference_idx), self.size),
                        dtype=torch.bool,
                        device=sim.device,
                    ).scatter_(
                        dim=1,
                        index=topk_reference_idx,
                        src=tc.ones(size=topk_reference_idx.shape, dtype=torch.bool, device=sim.device),
                    )
                    # apply mask with probability random_swap_p
                    should_apply = torch.rand(size=(batch_size,), device=cls.device) < self.random_swap_p
                    is_reference_cluster[~should_apply] = True
                    is_reference_cluster = is_reference_cluster.repeat(num_views, 1)
                    sim[~is_reference_cluster] -= 2
                else:
                    # swap with random sample from queue
                    should_apply = torch.rand(size=(batch_size * num_views,), device=x.device) < self.random_swap_p
                    noise = torch.rand(should_apply.sum(), self.size, dtype=sim.dtype, device=sim.device)
                    sim[should_apply] = noise
        elif self.guidance == "oracle":
            assert cls is not None
            assert self.random_swap_topk is None
            if self.random_swap_p > 0.:
                # swap label for retrieval
                should_apply = torch.rand(size=(batch_size,), device=cls.device) < self.random_swap_p
                new_cls = torch.randint(low=0, high=self.num_classes, size=(batch_size,), device=cls.device)
                oracle_cls = torch.where(should_apply, new_cls, cls)
            else:
                # force retrieval of a sample from the same class
                oracle_cls = cls
            is_other_cls = (oracle_cls[:, None] != self.cls[None, :]).repeat(num_views, 1)
            sim[is_other_cls] -= 2
        elif self.guidance in ["confidence", "confidence-lookup"]:
            assert cls is not None and confidence is not None
            assert self.random_swap_topk is None
            assert self.random_swap_p == 0.
            # force retrieval of a sample from the same class according to confidence
            # i.e. if confidence is 80% -> guide nn lookup by label in 80% of the cases
            # NOTE: confidence is not defined for evaluation
            if self.training:
                should_apply_guidance = torch.bernoulli(confidence).bool()
                oracle_cls = torch.where(should_apply_guidance, cls, tc.full_like(cls, fill_value=-1))
                is_other_cls = (oracle_cls[:, None] != self.cls[None, :]).repeat(num_views, 1)
                sim[is_other_cls] -= 2
        elif self.guidance == "confidence-queue":
            assert cls is not None
            assert self.random_swap_topk is None
            assert self.random_swap_p == 0.
            # the same as oracle
            oracle_cls = cls
            is_other_cls = (oracle_cls[:, None] != self.cls[None, :]).repeat(num_views, 1)
            sim[is_other_cls] -= 2
        else:
            raise NotImplementedError

        if self.topk == float("inf"):
            # swap with any sample of the same class
            assert self.guidance
            sim[sim > -1] = 1
            sim = sim + torch.rand_like(sim)
            nn_qidx = sim.max(dim=1).indices
            nn_x = self.x[nn_qidx]
        else:
            # retrieve neighbor(s)
            _, topk_qidx = sim.topk(self.topk, dim=1, largest=True)
            if self.topk == 1:
                nn_qidx = topk_qidx.squeeze(1)
            else:
                choice = torch.randint(self.topk, size=(len(topk_qidx), 1), device=topk_qidx.device)
                nn_qidx = torch.gather(topk_qidx, dim=1, index=choice).squeeze(1)
            nn_x = self.x[nn_qidx]

        # metrics
        metrics = {
            "nn-similarity": F.cosine_similarity(x, nn_x, dim=-1).mean(dim=-1),
            "nn-age": self.age[nn_qidx],
        }
        if cls is not None:
            metrics["nn-accuracy"] = cls.repeat(num_views) == self.cls[nn_qidx]

        # shift fifo queue
        if self.training:
            x_view0 = x[:len(idx)]
            x_view0 = all_gather_nograd(x_view0)
            idx = all_gather_nograd(idx)
            if cls is not None:
                if self.guidance in ["confidence", "confidence-queue"]:
                    assert confidence is not None
                    # dont put label always in queue, labels of lower confidence samples are not put into queue often
                    should_enqueue_with_cls = torch.bernoulli(confidence).bool()
                    cls = torch.where(should_enqueue_with_cls, cls, tc.full_like(cls, fill_value=-1))
                cls = all_gather_nograd(cls)
            ptr_from = int(self.ptr)
            ptr_to = ptr_from + len(x_view0)
            overflow = ptr_to - self.size
            if overflow > 0:
                # replace end-of-queue
                self.x[ptr_from:] = x_view0[:-overflow]
                self.idx[ptr_from:] = idx[:-overflow]
                if cls is not None:
                    self.cls[ptr_from:] = cls[:-overflow]
                self.age[ptr_from:] = 0
                # replace start-of-queue
                self.x[:overflow] = x_view0[-overflow:]
                if cls is not None:
                    self.cls[:overflow] = cls[-overflow:]
                self.age[:overflow] = 0
                if idx is not None:
                    self.idx[:overflow] = idx[-overflow:]
                # update pointer
                self.ptr[0] = overflow
            else:
                self.x[ptr_from:ptr_to] = x_view0
                self.idx[ptr_from:ptr_to] = idx
                if cls is not None:
                    self.cls[ptr_from:ptr_to] = cls
                self.age[ptr_from:ptr_to] = 0
                self.ptr[0] = ptr_to % self.size
            self.age += len(x_view0)

        metrics = {k: apply_reduction(v, reduction=reduction) for k, v in metrics.items()}
        return nn_x, metrics
