"""
Lovasz Loss
refer https://arxiv.org/abs/1705.08790

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from typing import Optional, Dict, Union
from itertools import filterfalse
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from .builder import LOSSES

BINARY_MODE: str = "binary"
MULTICLASS_MODE: str = "multiclass"
MULTILABEL_MODE: str = "multilabel"


def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
        logits: [B, H, W] Logits at each pixel (between -infinity and +infinity)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
    """
    if per_image:
        loss = mean(
            _lovasz_hinge_flat(
                *_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore))
    return loss


def _lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss
    Args:
        logits: [P] Logits at each prediction (between -infinity and +infinity)
        labels: [P] Tensor, binary ground truth labels (0 or 1)
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def _flatten_binary_scores(scores, labels, ignore=None):
    """Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def _lovasz_softmax(
    probas,
    labels,
    classes="present",
    class_seen=None,
    class_weights=None,
    per_image=False,
    ignore=None,
    penalize_pred: bool = False,
):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [B, C, H, W] Class probabilities at each prediction (between 0 and 1).
        Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        @param labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        @param class_weights: Optional dict, tensor, or list with class weights
        @param per_image: compute the loss per image instead of per batch
        @param ignore: void class labels
    """
    if per_image:
        loss = mean(
            _lovasz_softmax_flat(
                *_flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                classes=classes,
                class_weights=class_weights,
                penalize_pred=penalize_pred,
            )
            for prob, lab in zip(probas, labels)
        )
    else:
        loss = _lovasz_softmax_flat(
            *_flatten_probas(probas, labels, ignore),
            classes=classes,
            class_seen=class_seen,
            class_weights=class_weights,
            penalize_pred=penalize_pred,
        )
    return loss


def _lovasz_softmax_flat(
    probas,
    labels,
    classes="present",
    class_seen=None,
    class_weights=None,
    penalize_pred: bool = False,
):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [P, C] Class probabilities at each prediction (between 0 and 1)
        @param labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        @param class_weights: Optional dict, tensor, or list with class weights
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_indices = []
    # precompute per-pixel predicted-class weights when penalizing predictions
    pixel_pred_weights = None
    if class_weights is not None and penalize_pred:
        with torch.no_grad():
            pred_idx = probas.argmax(dim=1)
        if isinstance(class_weights, dict):
            C = probas.size(1)
            cw_vec = torch.ones(C, dtype=probas.dtype, device=probas.device)
            for k, v in class_weights.items():
                # tolerate keys out of range silently
                if 0 <= int(k) < C:
                    cw_vec[int(k)] = torch.as_tensor(v, dtype=probas.dtype, device=probas.device)
            pixel_pred_weights = cw_vec[pred_idx]
        elif isinstance(class_weights, (torch.Tensor, list)):
            cw_vec = torch.as_tensor(class_weights, dtype=probas.dtype, device=probas.device)
            pixel_pred_weights = cw_vec[pred_idx]
    # for c in class_to_sum:
    for c in labels.unique():
        if class_seen is None:
            fg = (labels == c).type_as(probas)  # foreground for class c
            if classes == "present" and fg.sum() == 0:
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError("Sigmoid output possible only with 1 class")
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            if pixel_pred_weights is not None:
                weights_sorted = pixel_pred_weights[perm]
                losses.append(torch.dot(errors_sorted * weights_sorted, _lovasz_grad(fg_sorted)))
            else:
                losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
            class_indices.append(c)
        else:
            if c in class_seen:
                fg = (labels == c).type_as(probas)  # foreground for class c
                if classes == "present" and fg.sum() == 0:
                    continue
                if C == 1:
                    if len(classes) > 1:
                        raise ValueError("Sigmoid output possible only with 1 class")
                    class_pred = probas[:, 0]
                else:
                    class_pred = probas[:, c]
                errors = (fg - class_pred).abs()
                errors_sorted, perm = torch.sort(errors, 0, descending=True)
                perm = perm.data
                fg_sorted = fg[perm]
                if pixel_pred_weights is not None:
                    weights_sorted = pixel_pred_weights[perm]
                    losses.append(torch.dot(errors_sorted * weights_sorted, _lovasz_grad(fg_sorted)))
                else:
                    losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
                class_indices.append(c)
    
    # Apply class weights if provided.
    # When penalize_pred=True, per-pixel weights were already applied above.
    if class_weights is not None and not penalize_pred:
        if isinstance(class_weights, dict):
            weighted_losses = []
            for loss, idx in zip(losses, class_indices):
                weight = class_weights.get(idx.item(), 1.0)
                weighted_losses.append(loss * weight)
            return mean(weighted_losses)
        elif isinstance(class_weights, (torch.Tensor, list)):
            weighted_losses = []
            for loss, idx in zip(losses, class_indices):
                weight = class_weights[idx.item()]
                weighted_losses.append(loss * weight)
            return mean(weighted_losses)
    
    return mean(losses)


def _flatten_probas(probas, labels, ignore=None):
    """Flattens predictions in the batch"""
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)

    C = probas.size(1)
    probas = torch.movedim(probas, 1, -1)  # [B, C, Di, Dj, ...] -> [B, Di, Dj, ..., C]
    probas = probas.contiguous().view(-1, C)  # [P, C]

    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid]
    vlabels = labels[valid]
    return vprobas, vlabels


def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    """Nan-mean compatible with generators."""
    values = iter(values)
    if ignore_nan:
        values = filterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


@LOSSES.register_module()
class LovaszLoss(_Loss):
    def __init__(
        self,
        mode: str,
        class_seen: Optional[int] = None,
        class_weights: Optional[Union[Dict[int, float], torch.Tensor, list]] = None,
        per_image: bool = False,
        ignore_index: Optional[int] = None,
        loss_weight: float = 1.0,
        penalize_pred: bool = False,
    ):
        """Lovasz loss for segmentation task.
        It supports binary, multiclass and multilabel cases
        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            class_seen: Optional list of classes to include in loss calculation
            class_weights: Optional class weights (dict mapping class idx to weight, tensor, or list)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            per_image: If True loss computed per each image and then averaged, else computed per whole batch
            loss_weight: Global weight for the loss
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()

        self.mode = mode
        self.ignore_index = ignore_index
        self.per_image = per_image
        self.class_seen = class_seen
        self.class_weights = class_weights
        self.loss_weight = loss_weight
        self.penalize_pred = penalize_pred

    def forward(self, y_pred, y_true):
        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            loss = _lovasz_hinge(
                y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index
            )
        elif self.mode == MULTICLASS_MODE:
            y_pred = y_pred.softmax(dim=1)
            loss = _lovasz_softmax(
                y_pred,
                y_true,
                class_seen=self.class_seen,
                class_weights=self.class_weights,
                per_image=self.per_image,
                ignore=self.ignore_index,
                penalize_pred=self.penalize_pred,
            )
        else:
            raise ValueError("Wrong mode {}.".format(self.mode))
        return loss * self.loss_weight
