import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

try:
    from mmcv.ops import diff_iou_rotated_2d
    HAS_MMCV = True
except ImportError:
    HAS_MMCV = False
    raise ImportError("mmcv.ops.diff_iou_rotated_2d is required for CODrone training. "
                     "Please install mmcv-full: pip install mmcv-full")

__all__ = ['oriented_box_iou', 'RotatedIoULoss', 'box_cxcywh_to_xyxyxyxy', 'box_xyxyxyxy_to_cxcywh']


def norm_angle(angle, angle_range='le90'):
    return (angle + np.pi / 2) % np.pi - np.pi / 2


def box_cxcywh_to_xyxyxyxy(boxes):
    cx, cy, w, h, angle = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3], boxes[..., 4]
    
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    
    # Half dimensions
    w_half = w * 0.5
    h_half = h * 0.5
    
    # Four corners relative to center (le90)
    corners_x = torch.stack([
        -w_half, w_half, w_half, -w_half
    ], dim=-1)
    corners_y = torch.stack([
        -h_half, -h_half, h_half, h_half
    ], dim=-1)
    
    # Rotate
    rotated_x = corners_x * cos_a.unsqueeze(-1) - corners_y * sin_a.unsqueeze(-1)
    rotated_y = corners_x * sin_a.unsqueeze(-1) + corners_y * cos_a.unsqueeze(-1)
    
    # Translate
    rotated_x += cx.unsqueeze(-1)
    rotated_y += cy.unsqueeze(-1)
    
    # Interleave as (x1,y1,x2,y2,x3,y3,x4,y4)
    polygons = torch.stack([rotated_x, rotated_y], dim=-1).flatten(start_dim=-2)
    
    return polygons


def box_xyxyxyxy_to_cxcywh(polygons):
    # Reshape to get corner points
    corners = polygons.reshape(*polygons.shape[:-1], 4, 2)  # (..., 4, 2)
    
    # Calculate center
    cx = corners[..., :, 0].mean(dim=-1)
    cy = corners[..., :, 1].mean(dim=-1)
    
    # Width/height from first two edges
    edge1 = corners[..., 1, :] - corners[..., 0, :]  # vector: p1 -> p2
    edge2 = corners[..., 3, :] - corners[..., 0, :]  # vector: p1 -> p4
    
    w = torch.norm(edge1, dim=-1)
    h = torch.norm(edge2, dim=-1)
    
    # Angle from first edge (le90)
    angle = torch.atan2(edge1[..., 1], edge1[..., 0])
    # No extra normalization here
    
    return torch.stack([cx, cy, w, h, angle], dim=-1)


def line_intersect(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    # Check if lines are parallel
    if torch.abs(denom) < 1e-8:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    
    return torch.stack([px, py], dim=-1)


def polygon_area(vertices):
    n = vertices.shape[-2]
    x = vertices[..., 0]
    y = vertices[..., 1]
    
    # Shoelace formula
    area = 0.5 * torch.abs(
        torch.sum(x * torch.roll(y, -1, dims=-1) - torch.roll(x, -1, dims=-1) * y, dim=-1)
    )
    return area


def oriented_box_iou(boxes1, boxes2, mode='iou'):
    # Store original shapes
    shape1 = boxes1.shape
    shape2 = boxes2.shape
    
    # Handle empty input cases
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        if shape1[:-1] == shape2[:-1]:
            output_shape = shape1[:-1]
        else:
            output_shape = shape1[:-1] + shape2[:-1]
        return torch.zeros(output_shape, device=boxes1.device, dtype=boxes1.dtype)
    
    boxes1_norm = boxes1 
    boxes2_norm = boxes2  
    
    # Flatten to 2D for mmcv processing
    boxes1_flat = boxes1_norm.reshape(-1, 5)  # (N1, 5)
    boxes2_flat = boxes2_norm.reshape(-1, 5)  # (N2, 5)
    
    # Add batch dimension for mmcv (expects 3D input)
    boxes1_batch = boxes1_flat.unsqueeze(0)  # (1, N1, 5)
    boxes2_batch = boxes2_flat.unsqueeze(0)  # (1, N2, 5)
    
    # Compute IoU via mmcv
    ious_matrix = diff_iou_rotated_2d(boxes1_batch, boxes2_batch)  # (1, N1, N2)
    ious_matrix = ious_matrix.squeeze(0)  # (N1, N2)
    
    # Clamp to valid range
    ious_matrix = torch.clamp(ious_matrix, 0.0, 1.0)
    
    # Handle different output cases
    if shape1[:-1] == shape2[:-1]:
        # Element-wise: use diagonal
        if ious_matrix.dim() == 2 and ious_matrix.size(0) == ious_matrix.size(1):
            ious = torch.diag(ious_matrix)
        elif ious_matrix.dim() == 2:
            # Different sizes: min diagonal
            min_size = min(ious_matrix.size(0), ious_matrix.size(1))
            if min_size > 0:
                ious = torch.diag(ious_matrix[:min_size, :min_size])
            else:
                ious = torch.empty(0, device=ious_matrix.device, dtype=ious_matrix.dtype)
        else:
            ious = ious_matrix
        
        # Reshape back
        target_shape = shape1[:-1]
        if ious.numel() == torch.prod(torch.tensor(target_shape)):
            ious = ious.reshape(target_shape)
    else:
        # Pairwise IoU
        ious = ious_matrix
        target_shape = shape1[:-1] + shape2[:-1]
        if ious.numel() == torch.prod(torch.tensor(target_shape)):
            ious = ious.reshape(target_shape)
    
    return ious


class RotatedIoULoss(nn.Module):
    
    def __init__(self, mode='log', eps=1e-6, reduction='mean', loss_weight=1.0):
        super().__init__()
        assert mode in ['linear', 'square', 'log']
        self.mode = mode
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        pred_normalized = pred    
        target_normalized = target  
        
        # Calculate IoU using precise mmcv implementation
        ious = oriented_box_iou(pred_normalized, target_normalized)
        ious = torch.clamp(ious, min=self.eps)
        
        # Calculate loss based on mode
        if self.mode == 'linear':
            loss = 1 - ious
        elif self.mode == 'square':
            loss = 1 - ious ** 2
        elif self.mode == 'log':
            loss = -torch.log(ious)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")
        
        # Apply weights
        if weight is not None:
            if weight.dim() != loss.dim():
                weight = weight.unsqueeze(-1) if weight.dim() + 1 == loss.dim() else weight
            loss = loss * weight
        
        # Apply reduction
        if reduction == 'mean':
            if avg_factor is not None:
                loss = loss.sum() / avg_factor
            else:
                loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        # else 'none': return loss as is
        
        return loss * self.loss_weight 