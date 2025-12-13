"""
Chamfer Distance Loss for Point Cloud Reconstruction

Implements bidirectional chamfer distance for MAE-style reconstruction.
Also includes matched feature loss using nearest-neighbor correspondences.
"""

import torch
import torch.nn as nn
from .builder import LOSSES
import torch.nn.functional as F

def chamfer_distance_fixed_k(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute chamfer distance when pred has fixed K points per patch.
    Also returns nearest neighbor indices for matched feature loss.
    
    Args:
        pred: (B, K, 3) predicted points
        target: (B, K, 3) target points (padded if necessary)
        target_mask: (B, K) bool mask, True for valid target points
    
    Returns:
        loss: scalar chamfer distance
        pred_to_target_idx: (B, K) index of nearest target for each pred
        target_to_pred_idx: (B, K) index of nearest pred for each target
    """
    B, K, _ = pred.shape
    
    # (B, K, K) pairwise squared distances via ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩
    # avoids materializing (B, K, K, 3) intermediate tensor
    pred_sq = (pred ** 2).sum(dim=-1, keepdim=True)  # (B, K, 1)
    target_sq = (target ** 2).sum(dim=-1, keepdim=True)  # (B, K, 1)
    cross = torch.bmm(pred, target.transpose(1, 2))  # (B, K, K)
    dist_sq = pred_sq + target_sq.transpose(1, 2) - 2 * cross  # (B, K, K)
    dist_sq = dist_sq.clamp(min=0)  # numerical stability
    
    if target_mask is not None:
        # mask out invalid target points with large distance
        invalid_mask = ~target_mask.unsqueeze(1)  # (B, 1, K)
        dist_sq_masked = dist_sq.masked_fill(invalid_mask, float('inf'))
    else:
        dist_sq_masked = dist_sq
    
    # pred -> target: for each pred point, find nearest valid target
    min_p2t, pred_to_target_idx = dist_sq_masked.min(dim=2)  # (B, K)
    
    # target -> pred: for each target point, find nearest pred
    min_t2p, target_to_pred_idx = dist_sq.min(dim=1)  # (B, K)
    
    if target_mask is not None:
        valid_counts = target_mask.float().sum(dim=1).clamp(min=1)  # (B,)
        loss_p2t = min_p2t.mean(dim=1)  # average over K predictions
        loss_t2p = (min_t2p * target_mask.float()).sum(dim=1) / valid_counts
    else:
        loss_p2t = min_p2t.mean(dim=1)
        loss_t2p = min_t2p.mean(dim=1)
    
    loss = (loss_p2t + loss_t2p).mean()
    
    return loss, pred_to_target_idx, target_to_pred_idx


def matched_feature_loss(
    pred_feat: torch.Tensor,
    target_feat: torch.Tensor,
    pred_to_target_idx: torch.Tensor,
    target_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute feature reconstruction loss using nearest-neighbor matches from chamfer.
    
    For each predicted point, compare its predicted features to the features
    of its nearest target point (found via chamfer distance).
    
    Args:
        pred_feat: (B, K, D) predicted features
        target_feat: (B, K, D) target features
        pred_to_target_idx: (B, K) index of nearest target for each pred
        target_mask: (B, K) bool mask for valid targets
    
    Returns:
        loss: scalar MSE loss for matched features
    """
    B, K, D = pred_feat.shape
    
    # gather target features at matched indices
    # pred_to_target_idx: (B, K) -> expand to (B, K, D)
    idx_expanded = pred_to_target_idx.unsqueeze(-1).expand(-1, -1, D)
    matched_target_feat = torch.gather(target_feat, dim=1, index=idx_expanded)
    
    # Huber between predicted and matched target features
    feat_diff = F.smooth_l1_loss(pred_feat, matched_target_feat, reduction='none')
    
    if target_mask is not None:
        # check if matched target is valid
        matched_valid = torch.gather(target_mask, dim=1, index=pred_to_target_idx)
        feat_loss = (feat_diff.mean(dim=-1) * matched_valid.float()).sum() / matched_valid.float().sum().clamp(min=1)
    else:
        feat_loss = feat_diff.mean()
    
    return feat_loss


@LOSSES.register_module()
class ChamferLoss(nn.Module):
    """
    Chamfer distance loss for point cloud coordinate reconstruction,
    with optional matched feature (energy) loss.
    
    Chamfer distance measures geometric similarity between point sets.
    Feature loss uses nearest-neighbor matching from chamfer to compare
    scalar features (like energy) at matched point pairs.
    """
    
    def __init__(
        self,
        loss_weight: float = 1.0,
        coord_weight: float = 1.0,
        feat_weight: float = 0.1,
    ):
        """
        Args:
            loss_weight: overall loss weight
            coord_weight: weight for coordinate chamfer distance
            feat_weight: weight for matched feature MSE loss
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.coord_weight = coord_weight
        self.feat_weight = feat_weight
    
    def forward(
        self,
        pred_coord: torch.Tensor,
        target_coord: torch.Tensor,
        pred_feat: torch.Tensor = None,
        target_feat: torch.Tensor = None,
        target_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            pred_coord: (B, K, 3) predicted coordinates
            target_coord: (B, K, 3) target coordinates
            pred_feat: (B, K, D) predicted features (e.g., energy)
            target_feat: (B, K, D) target features
            target_mask: (B, K) bool mask for valid target points
        
        Returns:
            total_loss: weighted sum of chamfer and feature losses
        """
        # chamfer distance for coordinates
        coord_loss, pred_to_target_idx, _ = chamfer_distance_fixed_k(
            pred_coord, target_coord, target_mask
        )
        
        # matched feature loss (e.g., energy)
        if pred_feat is not None and target_feat is not None and self.feat_weight > 0:
            feat_loss = matched_feature_loss(
                pred_feat, target_feat, pred_to_target_idx, target_mask
            )
        else:
            feat_loss = 0.0
        
        total_loss = self.coord_weight * coord_loss + self.feat_weight * feat_loss
        return {"loss": self.loss_weight * total_loss, "coord_loss": coord_loss, "feat_loss": feat_loss}