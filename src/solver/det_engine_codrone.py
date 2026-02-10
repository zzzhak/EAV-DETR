import math
import os
import sys
import pathlib
from typing import Iterable

import numpy as np
import torch
import torch.amp 

# CODrone evaluator
from src.data.codrone.codrone_eval import CODroneEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Non-finite loss {}, exit.".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_oriented(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors,
                      data_loader, base_ds, device, output_dir):

    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # Oriented evaluator with 5D boxes
    iou_types = getattr(postprocessors, 'iou_types', ['bbox'])
    
    # Get class names robustly
    if hasattr(base_ds, 'dataset') and hasattr(base_ds.dataset, 'CLASSES'):
        # CODroneDetection.CLASSES
        class_names = list(base_ds.dataset.CLASSES)
    elif hasattr(base_ds, 'CLASSES'):
        class_names = list(base_ds.CLASSES)
    elif hasattr(base_ds, 'category_names'):
        class_names = base_ds.category_names
    elif hasattr(base_ds, 'class_names'):
        class_names = base_ds.class_names
    else:
        # Default classes (12)
        class_names = [
            'car', 'truck', 'traffic-sign', 'people', 'motor', 'bicycle',
            'traffic-light', 'tricycle', 'bridge', 'bus', 'boat', 'ship'
        ]
    
    oriented_evaluator = CODroneEvaluator(
        class_names=class_names,
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        maxDets=100  
    )
    
    # Postprocessor outputs 5D rotated boxes
    print("Postprocessor uses 5D rotated boxes")
    
    print(f"Oriented evaluator ready (5D), classes: {len(class_names)}")

    # Collect annotations (normalized 5D)
    batch_annotations = []
    current_img_id = 0

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        # Use model input size for denormalization (coordinate consistency)
        model_input_sizes = torch.stack([t["size"] for t in targets], dim=0)        
        results = postprocessors(outputs, model_input_sizes)

        res = {}
        current_batch_annotations = []
        
        for target, output in zip(targets, results):
            img_id = current_img_id
            current_img_id += 1
            
            # Denormalize from normalized GT boxes
            gt_boxes = target['boxes']  # [N, 5] 
            gt_labels = target['labels']
            
            # Use model input size for pixel conversion
            model_h, model_w = target['size']
            
            # 转换到CPU
            if hasattr(gt_boxes, 'cpu'):
                gt_boxes = gt_boxes.cpu()
            if hasattr(gt_labels, 'cpu'):
                gt_labels = gt_labels.cpu()
            if hasattr(model_h, 'cpu'):
                model_h = model_h.cpu()
            if hasattr(model_w, 'cpu'):
                model_w = model_w.cpu()
                
            # Convert to pixel coordinates
            if len(gt_boxes) > 0 and gt_boxes.shape[-1] == 5:
                gt_boxes_pixel = gt_boxes.clone()
                # Spatial denorm with input size; angle unchanged
                gt_boxes_pixel[:, 0] *= model_w  # cx
                gt_boxes_pixel[:, 1] *= model_h  # cy  
                gt_boxes_pixel[:, 2] *= model_w  # w
                gt_boxes_pixel[:, 3] *= model_h  # h
                # angle kept
                print(f"Sample {current_img_id-1}: GT={len(gt_boxes)} Pred={len(output.get('boxes', []))}")
            else:
                print(f"Warn: invalid GT shape {gt_boxes.shape if len(gt_boxes) > 0 else 'empty'}, expect 5D")
                gt_boxes_pixel = torch.empty((0, 5))
                gt_labels = torch.empty((0,))
            
            annotation = {
                'boxes': gt_boxes_pixel, 
                'labels': gt_labels
            }
            current_batch_annotations.append(annotation)
            
            # Predictions (5D)
            pred_boxes = output.get('boxes', torch.empty((0, 5)))
            pred_scores = output.get('scores', torch.empty((0,)))
            pred_labels = output.get('labels', torch.empty((0,)))

            if hasattr(pred_boxes, 'cpu'):
                pred_boxes = pred_boxes.cpu().numpy()
            if hasattr(pred_scores, 'cpu'):
                pred_scores = pred_scores.cpu().numpy()
            if hasattr(pred_labels, 'cpu'):
                pred_labels = pred_labels.cpu().numpy()
            
            # Ensure 5D predictions
            if len(pred_boxes) > 0 and pred_boxes.shape[-1] != 5:
                print(f"Warn: pred dim {pred_boxes.shape[-1]}, expect 5")
                pred_boxes = np.empty((0, 5))
                pred_scores = np.empty((0,))
                pred_labels = np.empty((0,))
            
            # 构建预测结果字典
            prediction = {
                'boxes': pred_boxes,
                'scores': pred_scores, 
                'labels': pred_labels
            }
            res[img_id] = prediction

        batch_annotations.extend(current_batch_annotations)
        
        # Batch update predictions
        if oriented_evaluator is not None:
            oriented_evaluator.update(res)

    # Add GT annotations to evaluator
    if oriented_evaluator is not None and len(batch_annotations) > 0:
        oriented_evaluator.add_annotations(batch_annotations)

    print(f"Collected: {len(oriented_evaluator.predictions)} preds, {len(batch_annotations)} gts")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    if oriented_evaluator is not None:
        oriented_evaluator.synchronize_between_processes()

    # Accumulate -> summarize
    if oriented_evaluator is not None:
        print("Accumulate...")
        oriented_evaluator.accumulate()
        print("Summarize...")
        eval_results = oriented_evaluator.summarize()
    
    stats = {}
    if oriented_evaluator is not None:
        if 'bbox' in iou_types and hasattr(oriented_evaluator, 'stats'):
            stats['oriented_eval_bbox'] = oriented_evaluator.stats.tolist()
            stats['coco_eval_bbox'] = oriented_evaluator.stats.tolist()
        
        # Oriented-specific metrics
        if hasattr(oriented_evaluator, 'mean_ap'):
            stats['oriented_mAP'] = oriented_evaluator.mean_ap
            stats['mAP'] = oriented_evaluator.mean_ap 
            
        print(f"Oriented IoU eval done. mAP: {stats.get('oriented_mAP', 0.0):.4f}")
            
    return stats, oriented_evaluator

# Keep the original evaluate function for compatibility
@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, 
             data_loader, base_ds, device, output_dir):
    """
    Original evaluation for COCO datasets.
    """
    from src.data import CocoEvaluator
    
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        # Use model input size for denormalization
        model_input_sizes = torch.stack([t["size"] for t in targets], dim=0)        
        results = postprocessors(outputs, model_input_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    
    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    return stats, coco_evaluator