# modified from https://github.com/facebookresearch/pytorch3d/blob/7a3c0cbc9d7b0e70ef39b7f3c35e9ce2b7376f32/pytorch3d/loss/chamfer.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import pointops
from larfm.models.utils import offset2bincount


def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: Union[str, None]
) -> None:
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"] or None.
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction is not None and point_reduction not in ["mean", "sum", "max"]:
        raise ValueError(
            'point_reduction must be one of ["mean", "sum", "max"] or None'
        )
    if point_reduction is None and batch_reduction is not None:
        raise ValueError("Batch reduction must be None if point_reduction is None")

def _chamfer_distance_single_direction(
    x,
    y,
    x_offsets,
    y_offsets,
    x_normals,
    y_normals,
    weights,
    point_reduction: Union[str, None],
    norm: int,
):
    return_normals = x_normals is not None and y_normals is not None

    # inputs are always flattened (n, D) format with batched offsets
    if x.ndim != 2:
        raise ValueError(f"Expected x to be 2D (n, D), got {x.ndim}D")
    if y.ndim != 2:
        raise ValueError(f"Expected y to be 2D (m, D), got {y.ndim}D")
    
    n_x, D = x.shape
    if y.shape[1] != D:
        raise ValueError(f"y feature dimension {y.shape[1]} != x feature dimension {D}")
    
    # batch size from offsets
    N = x_offsets.shape[0]
    if y_offsets.shape[0] != N:
        raise ValueError(f"y_offsets batch size {y_offsets.shape[0]} != x_offsets batch size {N}")
    
    x_flat = x.contiguous()
    y_flat = y.contiguous()
    x_lengths = offset2bincount(x_offsets)

    if weights is not None:
        if weights.size(0) != N:
            raise ValueError(f"weights must be of shape ({N},), got {weights.shape}")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            return (torch.zeros(N, device=x.device), torch.zeros(n_x, device=x.device, dtype=torch.long))

    # pointops only supports 3D coordinates
    if D != 3:
        raise ValueError(f"pointops requires D=3, but got D={D}")
    # Note: pointops.knn_query uses L2 internally to find nearest neighbors,
    # but we compute the final distance using the specified norm
    if norm not in (1, 2):
        raise ValueError(f"norm must be 1 or 2, got {norm}")

    # use pointops for knn query: query points x, reference points y
    # pointops.knn_query returns indices and distances, but distances don't have gradients
    # We need to compute distances manually from indices to allow gradients to flow
    x_idx_flat, _ = pointops.knn_query(
        1, y_flat.float(), y_offsets.int(), x_flat.float(), x_offsets.int()
    )  # x_idx_flat: (n_x, 1), _ is distance without gradients

    # squeeze the K dimension
    x_idx_flat = x_idx_flat.squeeze(-1)  # (n_x,)
    
    # x_idx_flat contains global indices into y_flat (the flattened target array)
    x_idx = x_idx_flat  # (n_x,) - global indices
    
    # Compute distances manually to allow gradients to flow through coordinates
    # Gather the nearest neighbor points from y_flat using the indices
    nearest_y = y_flat[x_idx.long()]  # (n_x, 3)
    # Compute distances using the specified norm
    diff = x_flat - nearest_y  # (n_x, 3)
    if norm == 1:
        dist_flat = torch.sum(torch.abs(diff), dim=1)  # L1 norm: sum of absolute differences
    elif norm == 2:
        dist_flat = torch.sum(diff ** 2, dim=1)  # L2 norm squared: sum of squared differences
    else:
        raise ValueError(f"Unsupported norm: {norm}")

    # create batch indices for x points
    x_batch = torch.zeros(n_x, device=x.device, dtype=torch.long)
    for i in range(N):
        start = x_offsets[i-1] if i > 0 else 0
        end = x_offsets[i]
        x_batch[start:end] = i

    if weights is not None:
        cham_x = dist_flat * weights[x_batch]
    else:
        cham_x = dist_flat

    if point_reduction == "max":
        assert not return_normals
        # group by batch and take max
        cham_x = torch_scatter.segment_coo(cham_x, x_batch, reduce="max")  # (N,)
    elif point_reduction is not None:
        # group by batch and sum
        cham_x = torch_scatter.segment_coo(cham_x, x_batch, reduce="sum")  # (N,)
        if point_reduction == "mean":
            x_lengths_clamped = x_lengths.clamp(min=1)
            cham_x /= x_lengths_clamped

    cham_dist = cham_x
    return cham_dist, x_idx


def _apply_batch_reduction(
    cham_x, weights, batch_reduction: Union[str, None]
):
    if batch_reduction is None:
        return cham_x
    # batch_reduction == "sum"
    N = cham_x.shape[0]
    cham_x = cham_x.sum()
    if batch_reduction == "mean":
        if weights is None:
            div = max(N, 1)
        elif weights.sum() == 0.0:
            div = 1
        else:
            div = weights.sum()
        cham_x /= div
    return cham_x


def chamfer_distance(
    x,
    y,
    x_offsets=None,
    y_offsets=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: Union[str, None] = "mean",
    norm: int = 2,
    single_directional: bool = False,
    abs_cosine: bool = True,
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (n, D) or (N, P1, D) or a Pointclouds object representing
            a batch of point clouds. If flattened (n, D), x_offsets must be provided.
        y: FloatTensor of shape (m, D) or (N, P2, D) or a Pointclouds object representing
            a batch of point clouds. If flattened (m, D), y_offsets must be provided.
        x_offsets: Optional LongTensor of shape (batch_size,) giving cumulative offsets
            for point cloud x. Required if x is flattened, otherwise computed from shape.
        y_offsets: Optional LongTensor of shape (batch_size,) giving cumulative offsets
            for point cloud y. Required if y is flattened, otherwise computed from shape.
        x_normals: Optional FloatTensor of shape (n, 3) or (N, P1, 3).
        y_normals: Optional FloatTensor of shape (m, 3) or (N, P2, 3).
        weights: Optional FloatTensor of shape (batch_size,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum", "max"] or None. Using "max" leads to the
            Hausdorff distance.
        norm: int indicates the norm used for the distance. Only supports 2 for L2 (required by pointops).
        single_directional: If False (default), loss comes from both the distance between
            each point in x and its nearest neighbor in y and each point in y and its nearest
            neighbor in x. If True, loss is the distance between each point in x and its
            nearest neighbor in y.
        abs_cosine: If False, loss_normals is from one minus the cosine similarity.
            If True (default), loss_normals is from one minus the absolute value of the
            cosine similarity, which means that exactly opposite normals are considered
            equivalent to exactly matching normals, i.e. sign does not matter.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y. If point_reduction is None, a 2-element
          tuple of Tensors containing forward and backward loss terms shaped (N, P1)
          and (N, P2) (if single_directional is False) or a Tensor containing loss
          terms shaped (N, P1) (if single_directional is True) is returned.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None. If point_reduction is None, a 2-element
          tuple of Tensors containing forward and backward loss terms shaped (N, P1)
          and (N, P2) (if single_directional is False) or a Tensor containing loss
          terms shaped (N, P1) (if single_directional is True) is returned.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    if point_reduction == "max" and (x_normals is not None or y_normals is not None):
        raise ValueError('Normals must be None if point_reduction is "max"')

    cham_x, x_idx = _chamfer_distance_single_direction(
        x,
        y,
        x_offsets,
        y_offsets,
        x_normals,
        y_normals,
        weights,
        point_reduction,
        norm,
    )
    if single_directional:
        loss = cham_x
    else:
        cham_y, y_idx = _chamfer_distance_single_direction(
            y,
            x,
            y_offsets,
            x_offsets,
            y_normals,
            x_normals,
            weights,
            point_reduction,
            norm,
        )
        if point_reduction == "max":
            loss = torch.maximum(cham_x, cham_y)
        elif point_reduction is not None:
            loss = cham_x + cham_y
        else:
            loss = (cham_x, cham_y)
    return _apply_batch_reduction(loss, weights, batch_reduction), x_idx


class ChamferLoss(nn.Module):
    """
    Chamfer distance loss for point cloud coordinate reconstruction,
    with optional matched feature (energy) loss.
    
    Uses pointops for efficient KNN computation with offsets.
    """
    
    def __init__(
        self,
        loss_weight: float = 1.0,
        coord_weight: float = 1.0,
        feat_weight: float = 0.1,
        norm=2,
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
        self.norm = norm
    
    def forward(
        self,
        pred_coord: torch.Tensor,
        target_coord: torch.Tensor,
        pred_feat: torch.Tensor,
        target_feat: torch.Tensor,
        pred_offset: torch.Tensor,
        target_offset: torch.Tensor,
    ) -> dict:
        """
        Compute reconstruction loss using chamfer distance with offsets.
        
        Args:
            pred_coord: (N, 3) or (B, K, 3) predicted coordinates
            target_coord: (M, 3) or (B, K, 3) target coordinates
            pred_feat: (N, D) or (B, K, D) predicted features (e.g., energy)
            target_feat: (M, D) or (B, K, D) target features
            pred_offset: (B,) cumulative offsets for pred_coord (required if flattened)
            target_offset: (B,) cumulative offsets for target_coord (required if flattened)
        
        Returns:
            dict with 'loss', 'coord_loss', 'feat_loss'
        """
        # compute chamfer distance
        coord_loss, pred_to_target_idx = chamfer_distance(
            pred_coord,
            target_coord,
            x_offsets=pred_offset,
            y_offsets=target_offset,
            batch_reduction="mean",
            point_reduction="mean",
            norm=self.norm,
        )
        
        # matched feature loss if provided
        if self.feat_weight > 0:
            matched_target_feat = target_feat[pred_to_target_idx]
            feat_loss = F.smooth_l1_loss(pred_feat, matched_target_feat, reduction='mean')
        else:
            feat_loss = torch.tensor(0.0, device=coord_loss.device)
        
        total_loss = self.coord_weight * coord_loss + self.feat_weight * feat_loss
        return {
            "loss": self.loss_weight * total_loss,
            "coord_loss": coord_loss,
            "feat_loss": feat_loss
        }
