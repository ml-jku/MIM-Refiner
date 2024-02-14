import logging

from kappaschedules import object_to_schedule, PeriodicBoolSchedule

from utils.update_counter import UpdateCounter
from distributed.config import is_distributed


class FreezerBase:
    def __init__(self, update_counter: UpdateCounter = None, schedule=None, set_to_none=False):
        self.logger = logging.getLogger(type(self).__name__)
        self.update_counter = update_counter
        self.schedule = object_to_schedule(
            schedule,
            batch_size=self.update_counter.effective_batch_size if self.update_counter is not None else None,
            updates_per_epoch=self.update_counter.updates_per_epoch if self.update_counter is not None else None,
        )
        # remember current state for logging/callbacks when schedules are used
        # this should not be used in inherited classes in order to make them state less
        self.requires_grad = None

        # if set_to_none: param.requires_grad attribute is not changed
        # instead: set gradients to None before optim step
        # this resolves issues with DDP when dynamically freezing/unfreezing parameters
        # and comes at the cost of calculating the gradient before -> disable in non-DDP settings
        if not is_distributed():
            self.set_to_none = False
        self.set_to_none = set_to_none
        if set_to_none:
            assert schedule is not None, "set_to_none=True requires a schedule"

        # check if children overwrite the correct method
        assert type(self).before_accumulation_step == FreezerBase.before_accumulation_step

    def __repr__(self):
        return str(self)

    def __str__(self):
        raise NotImplementedError

    def after_weight_init(self, model):
        if self.schedule is not None:
            # state is updated before each accumulation step
            return
        self.logger.info(f"update state of {model.name}.{self} to requires_grad=False/is_frozen=True")
        self._update_state(model, requires_grad=False)

    def before_accumulation_step(self, model):
        # state was set with after_weight_init and never changes
        if self.schedule is None:
            return

        # update requires_grad of freezer
        value = self.schedule.get_value(
            step=self.update_counter.cur_checkpoint.update,
            total_steps=self.update_counter.end_checkpoint.update,
        )
        if value == 1:
            if self.requires_grad or self.requires_grad is None:
                if not isinstance(self.schedule, PeriodicBoolSchedule):
                    self.logger.info(f"update state of {model.name}.{self} to requires_grad=False/is_frozen=True")
                self.requires_grad = False
        elif value == 0:
            if not self.requires_grad or self.requires_grad is None:
                if not isinstance(self.schedule, PeriodicBoolSchedule):
                    self.logger.info(f"update state of {model.name}.{self} to requires_grad=True/is_frozen=False")
                self.requires_grad = True
        else:
            raise NotImplementedError

        # apply new state
        if self.set_to_none:
            # state is not changed because gradient is set to None before optim step
            pass
        else:
            # update requires_grad property of parameters
            self._update_state(model, requires_grad=self.requires_grad)

    def before_optim_step(self, model):
        # state was set with after_weight_init and never changes
        if self.schedule is None:
            assert not self.set_to_none
            return
        # requires_grad of parameter is set to False -> no need to set gradient to None
        if not self.set_to_none:
            return
        # parameter(s) requires_grad -> no need to set gradient to None
        if self.requires_grad:
            return
        # set gradient of parameter(s) to None
        self._set_to_none(model)

    def _update_state(self, model, requires_grad):
        raise NotImplementedError

    def _set_to_none(self, model):
        raise RuntimeError