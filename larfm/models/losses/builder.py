"""
Criteria Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from larfm.utils.registry import Registry

LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))

    def __call__(self, pred, target):
        if len(self.criteria) == 0:
            # loss computation occur in model
            return pred
        loss = 0
        loss_dict = {}
        has_extra_info = False
        for c in self.criteria:
            res = c(pred, target)
            if isinstance(res, tuple):
                loss_val, info = res
                loss += loss_val
                loss_dict.update(info)
                has_extra_info = True
            else:
                loss += res
        if has_extra_info:
            return loss, loss_dict
        return loss


def build_criteria(cfg):
    return Criteria(cfg)
