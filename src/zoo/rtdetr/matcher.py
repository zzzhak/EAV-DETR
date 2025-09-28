import torch
import torch.nn.functional as F 

from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .oriented_ops import oriented_box_iou

from src.core import register


@register
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0, spatial_loss_weight=1.0, angle_loss_weight=1.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        # Weights for spatial/angle terms
        self.spatial_loss_weight = spatial_loss_weight
        self.angle_loss_weight = angle_loss_weight

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    @staticmethod
    def angle_difference(pred_angle, target_angle):
        """Angle diff in [-pi/2, pi/2), handling periodicity."""
        diff = pred_angle - target_angle
        # Use remainder to handle wrap-around
        diff = torch.remainder(diff + torch.pi/2, torch.pi) - torch.pi/2
        return torch.abs(diff)

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Match predictions and targets for oriented boxes.

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 5] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 5] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 5]

        # Concat target labels/boxes; handle empty targets
        non_empty_targets = [v for v in targets if len(v["labels"]) > 0]
        
        if len(non_empty_targets) == 0:
            # All-empty batch: keep device/dtype
            device = outputs["pred_logits"].device
            tgt_ids = torch.empty(0, dtype=torch.int64, device=device)
            tgt_bbox = torch.empty(0, 5, dtype=torch.float32, device=device)
        else:
            tgt_ids = torch.cat([v["labels"] for v in non_empty_targets])
            tgt_bbox = torch.cat([v["boxes"] for v in non_empty_targets])

        # Ensure 5D rotated box format
        if tgt_bbox.shape[0] > 0:
            assert out_bbox.shape[-1] == 5 and tgt_bbox.shape[-1] == 5, \
                f"CODrone requires 5D boxes, got pred: {out_bbox.shape[-1]}, target: {tgt_bbox.shape[-1]}"

        # No targets: build empty cost matrix
        if tgt_bbox.shape[0] == 0:
            device = out_bbox.device
            C = torch.zeros((bs, num_queries, 0), device=device).cpu()
        else:
            # Compute cost matrix
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            if self.use_focal_loss:
                out_prob = out_prob[:, tgt_ids]
                neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class        
            else:
                cost_class = -out_prob[:, tgt_ids]

            # L1 cost for (cx, cy, w, h)
            cost_spatial = torch.cdist(out_bbox[:, :4], tgt_bbox[:, :4], p=1)

            # Angle cost (pairwise)
            pred_angle = out_bbox[:, 4].unsqueeze(1)
            target_angle = tgt_bbox[:, 4].unsqueeze(0)
            
            # Periodicity-aware angle error
            cost_angle_raw = self.angle_difference(pred_angle, target_angle)
            
            # Normalize angle cost
            cost_angle_normalized = cost_angle_raw * (2.0 / torch.pi)
            
            # Combine spatial and angle terms
            cost_bbox = (self.spatial_loss_weight * cost_spatial + 
                         self.angle_loss_weight * cost_angle_normalized)

            # Pairwise rotated IoU as giou term
            out_bbox_exp = out_bbox.unsqueeze(1).expand(-1, tgt_bbox.shape[0], -1)  # (N, M, 5)
            tgt_bbox_exp = tgt_bbox.unsqueeze(0).expand(out_bbox.shape[0], -1, -1)  # (N, M, 5)
                
            # Compute pairwise rotated IoU
            ious = oriented_box_iou(out_bbox_exp, tgt_bbox_exp)
            cost_giou = -ious
            
            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
