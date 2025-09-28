import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
from src.core import register

__all__ = ['RTDETRPostProcessor']

@register
class RTDETRPostProcessor(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category']
    
    def __init__(self, num_classes=12, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    def forward(self, outputs, model_input_sizes):
        """
        Post-process model outputs.
        Args:
            outputs: {'pred_logits', 'pred_boxes'}
            model_input_sizes: [B, 2] as [h, w] after preprocessing
        Returns:
            List of dicts with labels, boxes, scores. For 5D (cx, cy, w, h, angle),
            angle stays normalized; spatial dims are mapped to pixels.
        """
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

        # Check if we have oriented detection (5 parameters) or regular detection (4 parameters)
        is_oriented = boxes.shape[-1] == 5
        
        if is_oriented:
            # Oriented: keep 5D pixel format
            bbox_pred = self._process_5dim_boxes(boxes, model_input_sizes)
        else:
            # Regular detection: use original processing for 4D horizontal boxes
            bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
            bbox_pred *= model_input_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            boxes = bbox_pred
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        # For onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # Optional category remap
        if self.remap_mscoco_category:
            from ...data.coco import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        
        return results
    
    def _process_5dim_boxes(self, boxes, model_input_sizes):
        """
        Convert normalized 5D boxes to pixel 5D boxes.
        Args:
            boxes: [B, Q, 5] (cx, cy, w, h, angle) normalized
            model_input_sizes: [B, 2] as [h, w]
        Returns:
            [B, Q, 5]
        """
        bbox_pred = boxes.clone()
        
        # model_input_sizes format is [h, w]
        model_h = model_input_sizes[:, 0].unsqueeze(1)  
        model_w = model_input_sizes[:, 1].unsqueeze(1) 
        
        bbox_pred[:, :, 0] *= model_w  # cx
        bbox_pred[:, :, 1] *= model_h  # cy
        bbox_pred[:, :, 2] *= model_w  # w
        bbox_pred[:, :, 3] *= model_h  # h
        # angle unchanged
        
        return bbox_pred

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 

    @property
    def iou_types(self, ):
        return ('bbox', )