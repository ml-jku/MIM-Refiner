import torch

from freezers import freezer_from_kwargs
from kappamodules.init import ALL_BATCHNORMS
from models.extractors import extractor_from_kwargs
from optimizers import optim_ctor_from_kwargs
from utils.factory import create, create_collection
from utils.model_utils import get_trainable_param_count
from .model_base import ModelBase


class SingleModelBase(ModelBase):
    def __init__(
            self,
            optim_ctor=None,
            freezers=None,
            is_frozen=False,
            update_counter=None,
            extractors=None,
            allow_frozen_train_mode=False,
            **kwargs
    ):
        super().__init__(update_counter=update_counter, **kwargs)
        self._device = torch.device("cpu")
        self.optim_ctor = create(
            optim_ctor,
            optim_ctor_from_kwargs,
            instantiate_if_ctor=False,
        )
        self.freezers = create_collection(freezers, freezer_from_kwargs, update_counter=update_counter)
        self.extractors = create_collection(
            extractors,
            extractor_from_kwargs,
            outputs=self.dynamic_ctx,
            static_ctx=self.static_ctx,
        )
        self.is_frozen = is_frozen
        self.allow_frozen_train_mode = allow_frozen_train_mode
        self._is_batch_size_dependent = None

        # check parameter combinations
        if self.is_frozen:
            if self.allow_frozen_train_mode:
                self.logger.info(
                    f"model.is_frozen=True and optim_ctor is None but this is ok "
                    f"(allow_frozen_train_mode=True -> model is probably an EMA model)"
                )
            else:
                assert self.optim_ctor is None, "model.is_frozen=True but model.optim_ctor is not None"

        # check that base methods were not overwritten
        assert type(self).before_accumulation_step == SingleModelBase.before_accumulation_step
        assert type(self).after_initializers == SingleModelBase.after_initializers

    def clear_buffers(self):
        pass

    @property
    def is_batch_size_dependent(self):
        if self._is_batch_size_dependent is None:
            for m in self.modules():
                if isinstance(m, ALL_BATCHNORMS):
                    self._is_batch_size_dependent = True
                    break
            else:
                self._is_batch_size_dependent = False
        return self._is_batch_size_dependent

    @property
    def submodels(self):
        return {self.name: self}

    def optim_step(self, grad_scaler):
        for freezer in self.freezers:
            freezer.before_optim_step(self)
        if self._optim is not None:
            self._optim.step(grad_scaler)
        # after step (e.g. for EMA)
        self.after_update_step()

    def optim_schedule_step(self):
        if self._optim is not None:
            self._optim.schedule_step()

    def optim_zero_grad(self, set_to_none=True):
        if self._optim is not None:
            self._optim.zero_grad(set_to_none)

    @property
    def device(self):
        return self._device

    def before_accumulation_step(self):
        for freezer in self.freezers:
            freezer.before_accumulation_step(self)

    @staticmethod
    def get_model_specific_param_group_modifiers():
        return []

    def register_extractor_hooks(self):
        if len(self.extractors) > 0:
            for extractor in self.extractors:
                extractor.register_hooks(self)
                extractor.enable_hooks()
        return self

    def initialize_weights(self):
        # model specific initialization
        if self.model_specific_initialization != ModelBase.model_specific_initialization:
            self.logger.info(f"{self.name} applying model specific initialization")
            self.model_specific_initialization()
        else:
            self.logger(f"{self.name} no model specific initialization")

        # freeze all parameters (and put into eval mode)
        if self.is_frozen:
            if not self.allow_frozen_train_mode:
                self.logger.info(f"{self.name} is frozen -> put in eval mode")
                self.eval()
            else:
                self.logger.info(f"{self.name} is frozen but frozen train mode is allowed -> dont put in eval mode")
            for param in self.parameters():
                param.requires_grad = False

        # freeze some parameters
        for freezer in self.freezers:
            freezer.after_weight_init(self)

        return self

    def initialize_optim(self, lr_scale_factor=None):
        if self.optim_ctor is not None:
            self.logger.info(f"{self.name} initialize optimizer")
            self._optim = self.optim_ctor(self, update_counter=self.update_counter, lr_scale_factor=lr_scale_factor)
        elif not self.is_frozen:
            if get_trainable_param_count(self) == 0:
                self.is_frozen = True
                if self.allow_frozen_train_mode:
                    self.logger.info(f"{self.name} has no trainable parameters -> freeze but keep in train mode")
                else:
                    self.logger.info(f"{self.name} has no trainable parameters -> freeze and put into eval mode")
                    self.eval()
            else:
                if not self.allow_frozen_train_mode:
                    raise RuntimeError(f"no optimizer for {self.name} and it's also not frozen")
        else:
            self.logger.info(f"{self.name} is frozen -> no optimizer to initialize")

    def apply_initializers(self):
        for initializer in self.initializers:
            initializer.init_weights(self)
            initializer.init_optim(self)
        return self

    def after_initializers(self):
        self._after_initializers()

    def _after_initializers(self):
        pass

    def train(self, mode=True):
        # avoid setting mode to train if whole network is frozen
        # this prevents the training behavior of e.g. the following components
        # - Dropout/StochasticDepth dropping during
        # - BatchNorm (in train mode the statistics are tracked)
        if self.is_frozen and mode is True:
            if self.allow_frozen_train_mode:
                return super().train(mode=mode)
            else:
                return
        return super().train(mode=mode)

    def to(self, device, *args, **kwargs):
        if isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self._device = device
        return super().to(*args, **kwargs, device=device)
