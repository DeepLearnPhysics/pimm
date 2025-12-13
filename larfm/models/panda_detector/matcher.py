# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(batch_dice_loss)  # type: torch.jit.ScriptModule


def batch_sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0):
    """
    Compute the focal loss for binary classification.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example (logits).
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: weighting factor in [0, 1] to balance positive/negative examples
        gamma: exponent of the modulating factor (1 - p_t)^gamma
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]
    
    prob = inputs.sigmoid()
    
    # focal loss components
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    
    focal_loss = alpha_t * focal_weight * ce_loss
    
    # compute pairwise cost matrix
    loss = torch.einsum("nc,mc->nm", focal_loss, torch.ones_like(targets))
    
    return loss / hw


batch_sigmoid_focal_loss_jit = torch.jit.script(batch_sigmoid_focal_loss)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        num_points: int = 0,
        ignore_index: int = -1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
            focal_alpha: alpha parameter for focal loss
            focal_gamma: gamma parameter for focal loss
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, (
            "all costs cant be 0"
        )

        self.num_points = num_points
        self.ignore_index = ignore_index

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching for point clouds.

        Expected structures:
            - outputs["pred_masks"]: list of length B; each Tensor has shape [num_queries, P_b]
              containing per-query mask logits over points of batch element b.
            - targets: either
                a) list of length B with dicts {"labels": LongTensor[P_b]} where labels contain
                   per-point instance ids (arbitrary integers; negatives are ignored), or
                b) a single dict {"labels": LongTensor[sum(P_b)]} and we will split by P_b
                   inferred from outputs.

        This computes a cost matrix between predicted queries (num_queries) and ground-truth
        instances (num_instances in batch element), using per-point binary masks built from labels.
        """
        pred_masks_list = outputs["pred_masks"]
        assert isinstance(pred_masks_list, (list, tuple)), "pred_masks must be a list per batch"

        bs = len(pred_masks_list)

        # normalize targets to list-of-dicts format
        if isinstance(targets, dict) and "labels" in targets:
            full_labels = targets["labels"]
            assert full_labels.dim() == 1, "targets['labels'] must be 1D over all points"
            # split by predicted per-batch point counts
            counts = [pm.shape[1] for pm in pred_masks_list]
            splits = []
            start = 0
            for c in counts:
                splits.append(full_labels[start : start + c])
                start += c
            targets_list = [{"labels": t} for t in splits]
        else:
            targets_list = targets

        indices = []

        for b in range(bs):
            out_mask_logits = pred_masks_list[b]  # [num_queries_b, P_b]
            P_b = out_mask_logits.shape[1]
            Q_b = out_mask_logits.shape[0]
            device = out_mask_logits.device

            tgt_labels = targets_list[b]["labels"].to(device)
            ignore_mask = None

            # filter ignore
            if self.ignore_index is not None:
                ignore_mask = tgt_labels != self.ignore_index
                if ignore_mask.sum() == 0:
                    indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                    continue
                tgt_labels = tgt_labels[ignore_mask]
                out_mask_logits = out_mask_logits[:, ignore_mask]
                P_b = out_mask_logits.shape[1]

            uniq_ids, inverse = torch.unique(tgt_labels, sorted=True, return_inverse=True)
            num_instances = uniq_ids.numel()

            if num_instances == 0 or Q_b == 0 or P_b == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            if self.num_points and self.num_points > 0 and P_b > self.num_points:
                sel_idx = torch.randint(0, P_b, (self.num_points,), device=device)
            else:
                sel_idx = torch.arange(P_b, device=device)

            pred_samples = out_mask_logits[:, sel_idx]

            inv_sel = inverse[sel_idx]
            targets_mat = F.one_hot(inv_sel, num_classes=num_instances).to(dtype=pred_samples.dtype).T

            with torch.amp.autocast(device_type=device.type, enabled=False):
                pred_samples_f = pred_samples.float()
                targets_mat_f = targets_mat.float()

                inputs_bce = pred_samples_f.unsqueeze(1).repeat(1, num_instances, 1)
                targets_bce = targets_mat_f.unsqueeze(0).repeat(Q_b, 1, 1)

                ce_loss = F.binary_cross_entropy_with_logits(
                    inputs_bce,
                    targets_bce,
                    reduction="none",
                )

                prob = pred_samples_f.sigmoid()
                prob_expanded = prob.unsqueeze(1).repeat(1, num_instances, 1)
                targets_expanded = targets_mat_f.unsqueeze(0).repeat(Q_b, 1, 1)

                p_t = prob_expanded * targets_expanded + (1 - prob_expanded) * (1 - targets_expanded)
                focal_weight = (1 - p_t) ** self.focal_gamma
                alpha_t = self.focal_alpha * targets_expanded + (1 - self.focal_alpha) * (1 - targets_expanded)

                focal_loss = alpha_t * focal_weight * ce_loss
                cost_mask = focal_loss.mean(dim=-1)

                inputs = pred_samples.float().sigmoid()
                targets_f = targets_mat.float()
                numerator = 2 * torch.einsum("qk,jk->qj", inputs, targets_f)
                denominator = inputs.sum(-1)[:, None] + targets_f.sum(-1)[None, :]
                cost_dice = 1 - (numerator + 1) / (denominator + 1)

            cost_class = 0.0
            if "pred_logits" in outputs and self.cost_class > 0:
                pred_logits_b = outputs["pred_logits"][b]
                C = pred_logits_b.shape[-1] - 1

                if "inst_classes" in targets_list[b]:
                    inst_class = targets_list[b]["inst_classes"].to(device)
                    if inst_class.numel() != num_instances:
                        raise ValueError(f"inst_classes length {inst_class.numel()} must equal number of GT instances {num_instances}")
                    out_prob = F.softmax(pred_logits_b[:, :C], dim=-1)
                    cost_class = -torch.log(out_prob[:, torch.clamp(inst_class, 0, C - 1)] + 1e-8)

                elif "segment" in targets_list[b]:
                    seg_b = targets_list[b]["segment"].to(device)
                    if ignore_mask is not None:
                        seg_b = seg_b[ignore_mask]
                    seg_b = seg_b[sel_idx]
                    inst_class = seg_b.new_zeros((num_instances,), dtype=torch.long)
                    for j in range(num_instances):
                        mask_j = (inv_sel == j)
                        if mask_j.any():
                            inst_class[j] = seg_b[mask_j][0]
                    out_prob = F.softmax(pred_logits_b[:, :C], dim=-1)
                    cost_class = -torch.log(out_prob[:, torch.clamp(inst_class, 0, C - 1)] + 1e-8)

            C = self.cost_mask * cost_mask + self.cost_dice * cost_dice
            if isinstance(cost_class, torch.Tensor):
                C = C + self.cost_class * cost_class

            C = C.reshape(Q_b, -1).cpu()

            match = linear_sum_assignment(C)
            indices.append(
                (
                    torch.as_tensor(match[0], dtype=torch.int64),
                    torch.as_tensor(match[1], dtype=torch.int64),
                )
            )

        return indices

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs matching for batched point clouds.

        Params:
            outputs: dict with key "pred_masks" as a list of length B, each of shape [Q, P_b].
            targets: either a list of length B with {"labels": LongTensor[P_b]}, or a dict with
                     {"labels": LongTensor[sum(P_b)]}. Labels contain per-point instance ids.

        Returns:
            list of size B with tuples (index_i, index_j): indices of selected predictions and
            corresponding target instances. len(index_i) = len(index_j) = min(Q, num_instances_b).
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
