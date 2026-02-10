import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
from torch.nn.functional import interpolate

# from torchvision.ops import box_convert, generalized_box_iou
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .oriented_ops import oriented_box_iou, RotatedIoULoss

from src.misc.dist import get_world_size, is_dist_available_and_initialized
from src.misc.nested_tensor import nested_tensor_from_tensor_list
from src.core import register


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor for rare class (default: 0.25).
        gamma: (optional) Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def dice_loss(inputs, targets, num_boxes):
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
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes



@register
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, matcher, weight_dict, losses, alpha=0.75, gamma=2.0, eos_coef=1e-4, num_classes=12, 
                 spatial_loss_weight=1.0, angle_loss_weight=1.0, encoder_aux_loss_weight=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            spatial_loss_weight: weight for spatial dimensions (cx, cy, w, h) in bbox loss
            angle_loss_weight: weight for angle dimension in bbox loss
            encoder_aux_loss_weight: weight for encoder auxiliary prediction loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma
        
        # Loss weights for spatial dims and angle
        self.spatial_loss_weight = spatial_loss_weight
        self.angle_loss_weight = angle_loss_weight
        
        # Encoder auxiliary loss weight
        self.encoder_aux_loss_weight = encoder_aux_loss_weight
        
        # Oriented IoU loss
        self.oriented_iou_loss = RotatedIoULoss(mode='log', loss_weight=1.0)

    def _generate_gaussian_heatmap_targets(self, targets, feat_height, feat_width, device):
        """
        Build Gaussian heatmap targets for encoder auxiliary predictions.
        Returns [B, C, H, W].
        """
        batch_size = len(targets)
        
        # Heatmap tensor [B, C, H, W]
        heatmap_targets = torch.zeros(batch_size, self.num_classes, feat_height, feat_width, 
                                    dtype=torch.float32, device=device)
        
        for batch_idx, target in enumerate(targets):
            boxes = target['boxes']  # [N, 5] normalized OBB (cx, cy, w, h, angle)
            labels = target['labels']  # [N]
            
            if len(boxes) == 0:
                continue
                
            for obj_idx in range(len(boxes)):
                norm_cx, norm_cy, norm_w, norm_h, angle = boxes[obj_idx]
                class_id = labels[obj_idx].item()
                
                if class_id < 0 or class_id >= self.num_classes:
                    print(f"Warning: invalid class_id {class_id}, skip")
                    continue
                
                # Convert to feature map coords
                ix = int(torch.floor(norm_cx * feat_width).item())
                iy = int(torch.floor(norm_cy * feat_height).item())
                
                # Clamp to valid range
                ix = max(0, min(ix, feat_width - 1))
                iy = max(0, min(iy, feat_height - 1))
                
                # Size on feature map scale
                w_feat = norm_w * feat_width
                h_feat = norm_h * feat_height
                
                # Radius from log of geometric mean size
                base_size = torch.sqrt(w_feat * h_feat)
                C = 2.0
                radius = int(C * torch.log(1 + base_size))
                
                # Minimum radius for tiny objects
                if radius < 1:
                    radius = 1
                
                # Sigma from 3-sigma rule
                sigma = (2 * radius + 1) / 6.0
                
                if radius == 0:
                    # Fallback: single peak
                    heatmap_targets[batch_idx, class_id, iy, ix] = 1.0
                    continue
                
                # Gaussian kernel
                gaussian_kernel = self._generate_gaussian_kernel(radius, sigma, device)
                
                # Placement bounds
                y_min = max(0, iy - radius)
                y_max = min(feat_height, iy + radius + 1)
                x_min = max(0, ix - radius)
                x_max = min(feat_width, ix + radius + 1)
                
                # Kernel crop
                ky_min = max(0, radius - iy)
                ky_max = ky_min + (y_max - y_min)
                kx_min = max(0, radius - ix)
                kx_max = kx_min + (x_max - x_min)
                
                # Apply kernel
                kernel_region = gaussian_kernel[ky_min:ky_max, kx_min:kx_max]
                
                # Current target region
                target_region = heatmap_targets[batch_idx, class_id, y_min:y_max, x_min:x_max]
                
                # Max merge for overlaps
                heatmap_targets[batch_idx, class_id, y_min:y_max, x_min:x_max] = torch.max(
                    target_region, kernel_region
                )
        
        return heatmap_targets
    
    def _generate_gaussian_kernel(self, radius, sigma, device):
        """
        Create a 2D Gaussian kernel of size (2*radius+1).
        """
        size = 2 * radius + 1
        x = torch.arange(size, dtype=torch.float32, device=device)
        y = torch.arange(size, dtype=torch.float32, device=device)
        
        # Meshgrid
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        
        # Distance to center
        center = radius
        dist_sq = (x_grid - center) ** 2 + (y_grid - center) ** 2
        
        # Gaussian
        gaussian_kernel = torch.exp(-dist_sq / (2 * sigma ** 2))
        
        return gaussian_kernel
    
    def loss_encoder_aux_pred(self, outputs, targets):
        """
        Focal loss for encoder auxiliary predictions.
        """
        if 'encoder_aux_pred' not in outputs:
            return {'loss_encoder_aux': torch.tensor(0.0, device=next(iter(outputs.values())).device)}
        
        aux_pred = outputs['encoder_aux_pred']  # [B, C, H, W]
        _, num_classes, feat_height, feat_width = aux_pred.shape
        
        # Class count check
        if num_classes != self.num_classes:
            print(f"Warning: aux_pred classes {num_classes} != expected {self.num_classes}")
            return {'loss_encoder_aux': torch.tensor(0.0, device=aux_pred.device)}
        
        # Build heatmap targets
        heatmap_targets = self._generate_gaussian_heatmap_targets(targets, feat_height, feat_width, aux_pred.device)
        
        # Sigmoid focal loss
        aux_pred_flat = aux_pred.view(-1)  # [B*C*H*W]
        heatmap_targets_flat = heatmap_targets.view(-1)  # [B*C*H*W]
        
        # Positive count for normalization
        num_pos = torch.clamp(heatmap_targets_flat.sum(), min=1.0)
        
        # Focal loss
        prob = torch.sigmoid(aux_pred_flat).detach()
        ce_loss = F.binary_cross_entropy_with_logits(aux_pred_flat, heatmap_targets_flat, reduction='none')
        p_t = prob * heatmap_targets_flat + (1 - prob) * (1 - heatmap_targets_flat)
        alpha_t = self.alpha * heatmap_targets_flat + (1 - self.alpha) * (1 - heatmap_targets_flat)
        focal_loss = alpha_t * ce_loss * ((1 - p_t) ** self.gamma)
        
        # Normalize
        loss_encoder_aux = focal_loss.sum() / num_pos
        
        return {'loss_encoder_aux': loss_encoder_aux}

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_bce': loss}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        # ce_loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction="none")
        # prob = F.sigmoid(src_logits) # TODO .detach()
        # p_t = prob * target + (1 - prob) * (1 - target)
        # alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        # loss = alpha_t * ce_loss * ((1 - p_t) ** self.gamma)
        # loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target.float(), self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # Decoder angles are normalized to [-π/2, π/2)
        assert src_boxes.shape[-1] == 5 and target_boxes.shape[-1] == 5, \
            f"CODrone requires 5D boxes, got pred: {src_boxes.shape[-1]}, target: {target_boxes.shape[-1]}"
        
        # Oriented IoU for quality
        ious = oriented_box_iou(src_boxes, target_boxes)
        ious = ious.detach()
        
        # NaN/Inf guard
        if torch.isnan(ious).any() or torch.isinf(ious).any():
            print("Warning: NaN/Inf in IoU.")
            print(f"src range: [{src_boxes.min():.6f}, {src_boxes.max():.6f}]")
            print(f"tgt range: [{target_boxes.min():.6f}, {target_boxes.max():.6f}]")
            print(f"NaN: {torch.isnan(ious).sum().item()}, Inf: {torch.isinf(ious).sum().item()}")
            # Clamp to continue training
            ious = torch.clamp(torch.nan_to_num(ious, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=torch.float32)
        target_score_o[idx] = ious.to(dtype=torch.float32)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        # Clamp for numerical stability
        pred_score = torch.clamp(pred_score, min=1e-7, max=1-1e-7)
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
        # Ensure dtype consistency
        src_logits_f32 = src_logits.float()
        target_score_f32 = target_score.float()
        weight_f32 = weight.float()
        
        loss = F.binary_cross_entropy_with_logits(src_logits_f32, target_score_f32, weight=weight_f32, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """BBox losses for oriented detection with 5D boxes (cx, cy, w, h, angle)."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # Ensure 5D format
        assert src_boxes.shape[-1] == 5 and target_boxes.shape[-1] == 5, \
            f"CODrone requires 5D boxes, got pred: {src_boxes.shape[-1]}, target: {target_boxes.shape[-1]}"

        losses = {}

        # L1 on spatial dims (cx, cy, w, h) in [0, 1]
        spatial_l1 = F.l1_loss(src_boxes[..., :4], target_boxes[..., :4], reduction='none')
        spatial_loss = spatial_l1.sum() / num_boxes

        # Angle with periodic wrap
        def angle_difference(pred_angle, target_angle):
            """Angle diff in [-π/2, π/2) with periodicity"""
            diff = pred_angle - target_angle
            # Use remainder for periodicity
            diff = torch.remainder(diff + torch.pi/2, torch.pi) - torch.pi/2
            return torch.abs(diff)
        
        angle_errors = angle_difference(src_boxes[..., 4], target_boxes[..., 4])
        # Normalize angle loss to [0, 1]
        angle_loss_normalized = (angle_errors * (2.0 / torch.pi)).sum() / num_boxes
        
        # Weighted sum
        losses['loss_bbox'] = (self.spatial_loss_weight * spatial_loss + 
                              self.angle_loss_weight * angle_loss_normalized)

        # Oriented IoU loss
        loss_oriented_iou = self.oriented_iou_loss(src_boxes, target_boxes, reduction_override='none')
        losses['loss_oriented_iou'] = loss_oriented_iou.sum() / num_boxes
        
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,

            'bce': self.loss_labels_bce,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
            
            
        # Handle empty-annotation batches
        num_boxes = num_boxes / get_world_size()
        
        # If all-empty batch, use batch_size for normalization
        if num_boxes.item() == 0:
            # Keep contribution per sample stable
            num_boxes = torch.as_tensor([len(targets)], dtype=torch.float, device=num_boxes.device)
            print(f"Warning: all-empty batch. Use batch_size ({len(targets)}) for normalization.")
        
        num_boxes = num_boxes.item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # 🆕 计算编码器辅助预测损失
        if 'encoder_aux_pred' in outputs:
            encoder_aux_loss_dict = self.loss_encoder_aux_pred(outputs, targets)
            # 应用权重系数
            encoder_aux_loss_dict = {k: v * self.encoder_aux_loss_weight for k, v in encoder_aux_loss_dict.items()}
            losses.update(encoder_aux_loss_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        '''get_cdn_matched_indices
        '''
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res