"""
Default Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import larfm.utils.comm as comm
import weakref
from .builder import HOOKS


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.
    """

    trainer = None  # A weak reference to the trainer object.

    def modify_config(self, cfg):
        """Called after hooks are registered but before writer is built.
        Hooks can modify the config here (e.g., wandb_run_name).
        """
        pass

    def before_train(self):
        pass

    def before_epoch(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

    def after_epoch(self):
        pass

    def after_train(self):
        pass


@HOOKS.register_module()
class ModelHook(HookBase):
    def before_train(self):
        if comm.get_world_size() > 1 and hasattr(self.trainer.model, 'module') and isinstance(
            self.trainer.model.module, HookBase
        ):
            self.model = weakref.proxy(self.trainer.model.module)
        elif isinstance(self.trainer.model, HookBase):
            self.model = weakref.proxy(self.trainer.model)
        else:
            self.model = HookBase()
        self.model.trainer = self.trainer
        self.model.before_train()

    def before_epoch(self):
        self.model.before_epoch()

    def before_step(self):
        self.model.before_step()

    def after_step(self):
        self.model.after_step()

    def after_epoch(self):
        self.model.after_epoch()

    def after_train(self):
        self.model.after_train()
