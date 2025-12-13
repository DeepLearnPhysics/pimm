"""
Training loss for track finding (instance segmentation) with auxiliary losses.

Computes bipartite matching between ground-truth tracks and predicted track queries,
then applies Dice loss, Focal loss, and Classification loss to matched/unmatched pairs.

Includes auxiliary losses from each decoder layer (l=0 to L).
"""

from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from larfm.models.losses.builder import LOSSES
from larfm.models.panda_detector.matcher import (
    HungarianMatcher as _HungarianMatcher,  # type: ignore
)


@LOSSES.register_module()
class InstanceSegmentationLoss(nn.Module):
    """
    Loss for track finding with auxiliary decoder outputs.
    
    Computes InstanceSegLoss at each decoder layer and sums them:
        L_total = sum_{l=0}^{L} L^{(l)}
    
    where L^{(0)} is from initial query embeddings and L^{(L)} is from final layer.
    """
    def __init__(
        self,
        cost_mask: float = 1.0,
        cost_dice: float = 1.0,
        cost_class: float = 0.0,
        num_points: int = 0,
        ignore_index: int = -1,
        loss_weight_focal: float = 1.0,
        loss_weight_dice: float = 1.0,
        cls_weight_matched: float = 2.0,
        cls_weight_noobj: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        truth_label: str = "segment",
        aux_loss_weight: float = 1.0,
        noobj_warmup_start: float = None,
        noobj_warmup_end: float = None,
        noobj_warmup_steps: int = None,
        momentum_loss_weight: float = 0.0,
    ):
        super().__init__()
        
        self.aux_loss_weight = aux_loss_weight
        
        # main loss criterion (applied to all layers)
        self.criterion = SingleLayerInstanceLoss(
            cost_mask=cost_mask,
            cost_dice=cost_dice,
            cost_class=cost_class,
            num_points=num_points,
            ignore_index=ignore_index,
            loss_weight_focal=loss_weight_focal,
            loss_weight_dice=loss_weight_dice,
            cls_weight_matched=cls_weight_matched,
            cls_weight_noobj=cls_weight_noobj,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            truth_label=truth_label,
            momentum_loss_weight=momentum_loss_weight,
        )

        # optional linear warmup schedule for no-object class weight
        self._noobj_warmup_enabled = (
            noobj_warmup_start is not None
            and noobj_warmup_end is not None
            and noobj_warmup_steps is not None
            and noobj_warmup_steps > 0
        )
        if self._noobj_warmup_enabled:
            self._noobj_start = float(noobj_warmup_start)
            self._noobj_end = float(noobj_warmup_end)
            self._noobj_steps = int(noobj_warmup_steps)
            self._noobj_step = 0
            # initialize to start
            self.criterion.cls_weight_noobj = self._noobj_start

    def update_noobj_warmup(self, step: int = None):
        """linearly update cls_weight_noobj if warmup enabled"""
        if not getattr(self, "_noobj_warmup_enabled", False):
            return
        if step is None:
            self._noobj_step = min(self._noobj_step + 1, self._noobj_steps)
        else:
            self._noobj_step = min(int(step), self._noobj_steps)
        t = 0.0 if self._noobj_steps <= 0 else min(max(self._noobj_step / float(self._noobj_steps), 0.0), 1.0)
        value = (1.0 - t) * self._noobj_start + t * self._noobj_end
        self.criterion.cls_weight_noobj = float(value)
    
    def forward(self, pred: Dict, input_dict: Dict) -> torch.Tensor:
        """
        Args:
            pred: dict with keys:
                - "pred_masks": List[Tensor] of shape (Q, P_b) per batch
                - "pred_logits_list": List[Tensor] of shape (Q, C+1) per batch  
                - "aux_outputs": optional list of auxiliary predictions (same format)
            input_dict: dict with ground truth labels
        
        Returns:
            total_loss: sum of losses from all decoder layers
        """
        # if pred is Point object with outputs attr, extract it
        if hasattr(pred, 'outputs'):
            pred = pred.outputs
        
        # final layer loss
        final_loss, components = self.criterion(pred, input_dict)
        # auxiliary losses from intermediate layers
        if "aux_outputs" in pred and pred["aux_outputs"]:
            aux_loss = pred["pred_masks"][0].new_tensor(0.0)
            for layer_idx, aux_pred in enumerate(pred["aux_outputs"]):
                aux_loss_val, aux_comp = self.criterion(aux_pred, input_dict)
                aux_loss = aux_loss + aux_loss_val
                # only log scalar losses from aux layers, not counts/statistics
                if "focal" in aux_comp:
                    components[f"aux_focal_L{layer_idx}"] = aux_comp["focal"]
                if "dice" in aux_comp:
                    components[f"aux_dice_L{layer_idx}"] = aux_comp["dice"]
                if "cls_matched" in aux_comp:
                    components[f"aux_cls_matched_L{layer_idx}"] = aux_comp[
                        "cls_matched"
                    ]
                if "cls_noobj" in aux_comp:
                    components[f"aux_cls_noobj_L{layer_idx}"] = aux_comp["cls_noobj"]
                if "momentum" in aux_comp:
                    components[f"aux_momentum_L{layer_idx}"] = aux_comp["momentum"]

            final_loss = final_loss + self.aux_loss_weight * aux_loss

        self.update_noobj_warmup()
        return final_loss, components
    
    def compute_components(self, pred: Dict, input_dict: Dict) -> Dict:
        """
        Compute and return individual loss components for logging.
        
        Returns dict with scalar metrics only:
            - focal: focal loss (unweighted) from final layer
            - dice: dice loss (unweighted) from final layer
            - cls_matched: classification loss for matched queries
            - cls_noobj: classification loss for unmatched queries (no-object)
            - num_pairs: number of matched query-instance pairs
            - queries_total: total number of queries
            - gt_instances_total: total number of GT instances
            - unmatched_queries: number of unmatched queries
            - unmatched_gt: number of unmatched GT instances
            - aux_focal_L{i}: per-auxiliary-layer focal loss (if aux_outputs present)
            - aux_dice_L{i}: per-auxiliary-layer dice loss (if aux_outputs present)
            - momentum: momentum regression loss (if pred_momentum and momentum available)
            - aux_momentum_L{i}: per-auxiliary-layer momentum loss (if aux_outputs present)
        """
        # if pred is Point object with outputs attr, extract it
        if hasattr(pred, 'outputs'):
            pred = pred.outputs
        
        # final layer components
        _, components = self.criterion(pred, input_dict)
        
        # auxiliary layer components
        if "aux_outputs" in pred and pred["aux_outputs"]:
            for layer_idx, aux_pred in enumerate(pred["aux_outputs"]):
                _, aux_comp = self.criterion(aux_pred, input_dict)
                # only log scalar losses from aux layers, not counts/statistics
                if "focal" in aux_comp:
                    components[f"aux_focal_L{layer_idx}"] = aux_comp["focal"]
                if "dice" in aux_comp:
                    components[f"aux_dice_L{layer_idx}"] = aux_comp["dice"]
                if "cls_matched" in aux_comp:
                    components[f"aux_cls_matched_L{layer_idx}"] = aux_comp["cls_matched"]
                if "cls_noobj" in aux_comp:
                    components[f"aux_cls_noobj_L{layer_idx}"] = aux_comp["cls_noobj"]
                if "momentum" in aux_comp:
                    components[f"aux_momentum_L{layer_idx}"] = aux_comp["momentum"]
        
        return components

class SingleLayerInstanceLoss(nn.Module):
    def __init__(
        self,
        cost_mask: float = 1.0,
        cost_dice: float = 1.0,
        cost_class: float = 0.0,
        num_points: int = 0,
        ignore_index: int = -1,
        loss_weight_focal: float = 1.0,
        loss_weight_dice: float = 1.0,
        cls_weight_matched: float = 2.0,
        cls_weight_noobj: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        truth_label: str = "segment",
        momentum_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight_focal = loss_weight_focal
        self.loss_weight_dice = loss_weight_dice
        self.cls_weight_matched = cls_weight_matched
        self.cls_weight_noobj = cls_weight_noobj
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.momentum_loss_weight = momentum_loss_weight

        # matcher used internally
        self.matcher = _HungarianMatcher(
            cost_class=cost_class,
            cost_mask=cost_mask,
            cost_dice=cost_dice,
            num_points=num_points,
            ignore_index=ignore_index,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )

        self.num_points = num_points
        self.truth_label = truth_label

    def _split_targets(
        self, targets: Union[torch.Tensor, List[torch.Tensor]], counts: List[int]
    ) -> List[Dict[str, torch.Tensor]]:
        if isinstance(targets, torch.Tensor):
            # accept 1D [N] or 2D column [N,1]
            if targets.dim() == 2 and targets.shape[1] == 1:
                targets = targets.squeeze(1)
            assert targets.dim() == 1
            splits: List[torch.Tensor] = []
            start = 0
            for c in counts:
                splits.append(targets[start : start + c])
                start += c
            return [{"labels": t} for t in splits]
        else:
            # ensure list elements are 1D
            out: List[Dict[str, torch.Tensor]] = []
            for t in targets:
                if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[1] == 1:
                    t = t.squeeze(1)
                out.append({"labels": t})
            return out

    def _normalize_and_split_tensor(
        self, tensor_or_list: Union[torch.Tensor, List[torch.Tensor]], counts: List[int]
    ) -> List[torch.Tensor]:
        """normalize tensor/list and split by batch counts"""
        if isinstance(tensor_or_list, torch.Tensor):
            if tensor_or_list.dim() == 2 and tensor_or_list.shape[1] == 1:
                tensor_or_list = tensor_or_list.squeeze(1)
            assert tensor_or_list.dim() == 1
            splits: List[torch.Tensor] = []
            start = 0
            for c in counts:
                splits.append(tensor_or_list[start : start + c])
                start += c
            return splits
        else:
            out: List[torch.Tensor] = []
            for t in tensor_or_list:
                if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[1] == 1:
                    t = t.squeeze(1)
                out.append(t)
            return out

    def _process_batch_data(
        self, labels_b_full: torch.Tensor, pm_b: torch.Tensor
    ) -> tuple:
        """filter ignore indices, extract unique instances"""
        if self.ignore_index is not None:
            valid_mask = labels_b_full != self.ignore_index
        else:
            valid_mask = torch.ones_like(labels_b_full, dtype=torch.bool)
        labels_b = labels_b_full[valid_mask]
        pm_b = pm_b[:, valid_mask]

        if labels_b.numel() == 0:
            return labels_b, pm_b, None, None, valid_mask

        uniq_ids, inverse = torch.unique(labels_b, sorted=True, return_inverse=True)

        return labels_b, pm_b, uniq_ids, inverse, valid_mask

    def _get_per_instance_values(
        self,
        per_point_values: torch.Tensor,
        inverse: torch.Tensor,
        num_instances: int,
        aggregation_fn: str = "first",
    ) -> torch.Tensor:
        """compute per-instance values from per-point values"""
        per_instance_values = per_point_values.new_zeros((num_instances,))
        for inst_idx in range(num_instances):
            mask = inverse == inst_idx
            if mask.any():
                if aggregation_fn == "first":
                    per_instance_values[inst_idx] = per_point_values[mask][0]
                elif aggregation_fn == "mean":
                    per_instance_values[inst_idx] = per_point_values[mask].mean()
                else:
                    raise ValueError(f"Unknown aggregation_fn: {aggregation_fn}")
        return per_instance_values

    def _get_batch_tensor(
        self,
        tensor_or_list: Union[torch.Tensor, List[torch.Tensor]],
        batch_idx: int,
        counts: List[int],
        device: torch.device,
        valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """get per-batch tensor from input_dict, apply filtering"""
        if isinstance(tensor_or_list, torch.Tensor):
            start = sum(counts[:batch_idx])
            end = start + counts[batch_idx]
            tensor_b = tensor_or_list[start:end].to(device)
        else:
            tensor_b = tensor_or_list[batch_idx].to(device)

        if tensor_b.dim() == 2 and tensor_b.shape[1] == 1:
            tensor_b = tensor_b.squeeze(1)

        if valid_mask is not None:
            tensor_b = tensor_b[valid_mask]

        return tensor_b

    def _compute_classification_loss(
        self,
        logits_b: torch.Tensor,
        seg_b: torch.Tensor,
        inverse: torch.Tensor,
        num_inst: int,
        idx_q: torch.Tensor,
        idx_gt: torch.Tensor,
        C: int,
    ) -> tuple:
        """compute classification loss for matched and unmatched queries"""
        cls_loss_b = logits_b.new_tensor(0.0)
        cls_count_b = 0

        if num_inst > 0:
            inst_class = self._get_per_instance_values(
                seg_b, inverse, num_inst, aggregation_fn="first"
            )
            inst_class = inst_class.clamp(0, C - 1)

            if idx_q.numel() > 0:
                logits_matched = logits_b[idx_q.long()]
                target_matched = inst_class[idx_gt.long()]
                cls_loss_matched = F.cross_entropy(
                    logits_matched, target_matched, reduction="mean"
                )
                cls_loss_b = cls_loss_b + self.cls_weight_matched * cls_loss_matched
                cls_count_b += 1

        Q = logits_b.shape[0]
        if Q > 0:
            mask_unmatched = torch.ones(Q, dtype=torch.bool, device=logits_b.device)
            if idx_q.numel() > 0:
                mask_unmatched[idx_q.long()] = False
            if mask_unmatched.any():
                logits_unmatched = logits_b[mask_unmatched]
                target_noobj = logits_unmatched.new_full(
                    (logits_unmatched.shape[0],), C, dtype=torch.long
                )
                cls_loss_noobj = F.cross_entropy(
                    logits_unmatched, target_noobj, reduction="mean"
                )
                cls_loss_b = cls_loss_b + self.cls_weight_noobj * cls_loss_noobj
                cls_count_b += 1

        return cls_loss_b, cls_count_b

    def _compute_classification_components(
        self,
        logits_b: torch.Tensor,
        seg_b: torch.Tensor,
        inverse: torch.Tensor,
        num_inst: int,
        idx_q: torch.Tensor,
        idx_gt: torch.Tensor,
        C: int,
        class_totals: torch.Tensor = None,
        class_matched: torch.Tensor = None,
    ) -> Dict:
        """compute classification loss components for logging"""
        ce_matched = None
        ce_noobj = None

        if num_inst > 0:
            inst_class = self._get_per_instance_values(
                seg_b, inverse, num_inst, aggregation_fn="first"
            )
            inst_class_clamped = inst_class.clamp(0, C - 1)

            if class_totals is not None:
                class_totals = class_totals + torch.bincount(
                    inst_class_clamped, minlength=C
                ).to(class_totals.dtype)

            if idx_q.numel() > 0:
                logits_matched = logits_b[idx_q.long()]
                target_matched = inst_class_clamped[idx_gt.long()]
                ce_matched = F.cross_entropy(
                    logits_matched, target_matched, reduction="mean"
                )
                if class_matched is not None:
                    class_matched = class_matched + torch.bincount(
                        target_matched, minlength=C
                    ).to(class_matched.dtype)

        Q = logits_b.shape[0]
        if Q > 0:
            mask_unmatched = torch.ones(Q, dtype=torch.bool, device=logits_b.device)
            if idx_q.numel() > 0:
                mask_unmatched[idx_q.long()] = False
            if mask_unmatched.any():
                logits_unmatched = logits_b[mask_unmatched]
                target_noobj = logits_unmatched.new_full(
                    (logits_unmatched.shape[0],), C, dtype=torch.long
                )
                ce_noobj = F.cross_entropy(
                    logits_unmatched, target_noobj, reduction="mean"
                )

        return {
            "ce_matched": ce_matched,
            "ce_noobj": ce_noobj,
            "class_totals": class_totals,
            "class_matched": class_matched,
        }

    def _compute_momentum_loss(
        self,
        mom_pred_b: torch.Tensor,
        mom_gt_per_inst: torch.Tensor,
        idx_q: torch.Tensor,
        idx_gt: torch.Tensor,
    ) -> torch.Tensor:
        """compute momentum regression loss for matched pairs"""
        mom_pred_matched = mom_pred_b[idx_q.long()]
        mom_gt_matched = mom_gt_per_inst[idx_gt.long()]
        return F.l1_loss(mom_pred_matched, mom_gt_matched, reduction="mean")

    def _compute_momentum_components(
        self,
        mom_pred_b: torch.Tensor,
        mom_gt_per_inst: torch.Tensor,
        idx_q: torch.Tensor,
        idx_gt: torch.Tensor,
    ) -> torch.Tensor:
        """compute momentum regression loss components for logging"""
        mom_pred_matched = mom_pred_b[idx_q.long()]
        mom_gt_matched = mom_gt_per_inst[idx_gt.long()]
        return F.smooth_l1_loss(mom_pred_matched, mom_gt_matched, reduction="mean")

    def forward(
        self, pred: Dict[str, List[torch.Tensor]], input_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        assert isinstance(pred, dict) and "pred_masks" in pred, (
            "pred must be a dict with key 'pred_masks'"
        )
        assert self.truth_label in input_dict, (
            f"input_dict must contain key '{self.truth_label}'"
        )
        pred_masks_list: List[torch.Tensor] = pred["pred_masks"]
        target: Union[torch.Tensor, List[torch.Tensor]] = input_dict[self.truth_label]
        counts = [pm.shape[1] for pm in pred_masks_list]
        targets_list = self._split_targets(target, counts)

        # attach semantic labels so matcher can use cost_class
        if "segment" in input_dict:
            seg_list = self._normalize_and_split_tensor(input_dict["segment"], counts)
            for b, seg_b in enumerate(seg_list):
                targets_list[b]["segment"] = seg_b

        # compute Hungarian matching
        indices = self.matcher(
            {
                "pred_masks": pred_masks_list,
                "pred_logits": pred.get("pred_logits", None),
            },
            targets_list,
        )

        # aggregate losses per batch element, then average over batch
        total_loss_focal = pred_masks_list[0].new_tensor(0.0)
        total_loss_dice = pred_masks_list[0].new_tensor(0.0)
        total_loss_cls = pred_masks_list[0].new_tensor(0.0)
        total_loss_momentum = pred_masks_list[0].new_tensor(0.0)
        num_batches_with_loss = 0
        num_batches_with_momentum = 0

        # component tracking
        total_focal = pred_masks_list[0].new_tensor(0.0)
        total_dice = pred_masks_list[0].new_tensor(0.0)
        total_pairs = 0
        total_ce_matched = pred_masks_list[0].new_tensor(0.0)
        count_ce_matched = pred_masks_list[0].new_tensor(0.0)
        total_ce_noobj = pred_masks_list[0].new_tensor(0.0)
        count_ce_noobj = pred_masks_list[0].new_tensor(0.0)
        total_momentum_loss = pred_masks_list[0].new_tensor(0.0)
        count_momentum_batches = 0
        queries_total = pred_masks_list[0].new_tensor(0.0)
        gt_instances_total = pred_masks_list[0].new_tensor(0.0)
        class_totals = None
        class_matched = None

        for b, (pm_b, tgt_b, (idx_q, idx_gt)) in enumerate(
            zip(pred_masks_list, targets_list, indices)
        ):
            queries_total = queries_total + pm_b.shape[0]

            labels_b_full = tgt_b["labels"].to(pm_b.device)
            labels_b, pm_b, uniq_ids, inverse, valid_mask = self._process_batch_data(
                labels_b_full, pm_b
            )

            if idx_q.numel() == 0 or labels_b.numel() == 0:
                continue

            gt_instances_total = gt_instances_total + uniq_ids.numel()
            num_inst = uniq_ids.numel()
            targets_mat = (
                F.one_hot(inverse, num_classes=num_inst).to(dtype=pm_b.dtype).T
            )

            idx_q = idx_q.to(pm_b.device).long()
            idx_gt = idx_gt.to(pm_b.device).long()
            pred_sel = pm_b[idx_q]  # [M, K]
            gt_sel = targets_mat[idx_gt]  # [M, K]

            num_pairs_b = pred_sel.shape[0]
            if num_pairs_b == 0:
                continue

            # focal and dice losses
            prob = pred_sel.sigmoid()
            ce_loss = F.binary_cross_entropy_with_logits(
                pred_sel, gt_sel, reduction="none"
            )
            p_t = prob * gt_sel + (1 - prob) * (1 - gt_sel)
            focal_weight = (1 - p_t) ** self.focal_gamma
            alpha_t = self.focal_alpha * gt_sel + (1 - self.focal_alpha) * (1 - gt_sel)
            focal_loss = alpha_t * focal_weight * ce_loss
            focal_loss_b = focal_loss.mean(dim=1).sum() / num_pairs_b
            total_loss_focal = total_loss_focal + focal_loss_b
            total_focal = total_focal + focal_loss.mean(dim=1).sum()

            num = 2 * (prob * gt_sel).sum(dim=1)
            den = prob.sum(dim=1) + gt_sel.sum(dim=1)
            dice = 1 - (num + 1) / (den + 1)
            dice_loss_b = dice.sum() / num_pairs_b
            total_loss_dice = total_loss_dice + dice_loss_b
            total_dice = total_dice + dice.sum()

            total_pairs += num_pairs_b
            num_batches_with_loss += 1

            # classification loss
            if "pred_logits" in pred and "segment" in input_dict:
                logits_b = pred["pred_logits"][b]  # [Q, C+1]
                C = logits_b.shape[-1] - 1
                if class_totals is None:
                    class_totals = torch.zeros(C, dtype=pm_b.dtype, device=pm_b.device)
                    class_matched = torch.zeros(C, dtype=pm_b.dtype, device=pm_b.device)

                seg_b = self._get_batch_tensor(
                    input_dict["segment"], b, counts, pm_b.device, valid_mask
                )
                cls_loss_b, cls_count_b = self._compute_classification_loss(
                    logits_b, seg_b, inverse, num_inst, idx_q, idx_gt, C
                )
                if cls_count_b > 0:
                    total_loss_cls = total_loss_cls + (cls_loss_b / cls_count_b)

                # track unweighted classification components
                cls_comp = self._compute_classification_components(
                    logits_b,
                    seg_b,
                    inverse,
                    num_inst,
                    idx_q,
                    idx_gt,
                    C,
                    class_totals,
                    class_matched,
                )
                class_totals = cls_comp["class_totals"]
                class_matched = cls_comp["class_matched"]
                if cls_comp["ce_matched"] is not None:
                    total_ce_matched = total_ce_matched + cls_comp["ce_matched"]
                    count_ce_matched = count_ce_matched + 1.0
                if cls_comp["ce_noobj"] is not None:
                    total_ce_noobj = total_ce_noobj + cls_comp["ce_noobj"]
                    count_ce_noobj = count_ce_noobj + 1.0

            # momentum regression loss
            if (
                self.momentum_loss_weight > 0
                and "pred_momentum" in pred
                and "momentum" in input_dict
            ):
                momentum_gt = input_dict["momentum"]
                mom_gt_b = self._get_batch_tensor(
                    momentum_gt, b, counts, pm_b.device, valid_mask
                )
                mom_pred_b = pred["pred_momentum"][b].to(pm_b.device)

                mom_gt_per_inst = self._get_per_instance_values(
                    mom_gt_b, inverse, uniq_ids.numel(), aggregation_fn="mean"
                )
                loss_momentum_b = self._compute_momentum_loss(
                    mom_pred_b, mom_gt_per_inst, idx_q, idx_gt
                )
                total_loss_momentum = total_loss_momentum + loss_momentum_b
                num_batches_with_momentum += 1

                # track unweighted momentum component
                loss_momentum_comp = self._compute_momentum_components(
                    mom_pred_b, mom_gt_per_inst, idx_q, idx_gt
                )
                total_momentum_loss = total_momentum_loss + loss_momentum_comp
                count_momentum_batches += 1

        if num_batches_with_momentum > 0:
            total_loss_momentum = total_loss_momentum / num_batches_with_momentum

        # average over batch
        denom = max(num_batches_with_loss, 1)
        loss_masks = self.loss_weight_focal * (
            total_loss_focal / denom
        ) + self.loss_weight_dice * (total_loss_dice / denom)
        loss_cls = total_loss_cls / denom
        loss = loss_masks + loss_cls + self.momentum_loss_weight * total_loss_momentum

        # build components dict
        denom_pairs = max(int(total_pairs), 1)
        unmatched_queries = torch.clamp(queries_total - total_pairs, min=0.0)
        unmatched_gt = torch.clamp(gt_instances_total - total_pairs, min=0.0)
        components = {
            "focal": total_focal / denom_pairs,
            "dice": total_dice / denom_pairs,
            "cls_matched": (total_ce_matched / count_ce_matched)
            if count_ce_matched > 0
            else total_focal.new_tensor(0.0),
            "cls_noobj": (total_ce_noobj / count_ce_noobj)
            if count_ce_noobj > 0
            else total_focal.new_tensor(0.0),
            "momentum": (total_momentum_loss / count_momentum_batches)
            if count_momentum_batches > 0
            else total_focal.new_tensor(0.0),
            "num_pairs": total_pairs,
            "queries_total": queries_total,
            "gt_instances_total": gt_instances_total,
            "unmatched_queries": unmatched_queries,
            "unmatched_gt": unmatched_gt,
            "num_cls_matched": count_ce_matched,
            "num_cls_noobj": count_ce_noobj,
        }

        return loss, components