import torch
import numpy as np
from typing import List, Dict, Any, Optional
import contextlib
import os
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
 
try:
    from mmcv.ops import box_iou_rotated
    MMCV_AVAILABLE = True
    print("[INFO] Using MMCV rotated IoU (5D)")
except ImportError:
    MMCV_AVAILABLE = False
    print("[ERROR] MMCV not available for rotated IoU")
    raise ImportError("MMCV is required for rotated IoU calculation. Please install mmcv-full.")
 
def rotated_iou_mmcv_5dim(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute rotated IoU in 5D format (cx,cy,w,h,angle)."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    boxes1_tensor = torch.from_numpy(boxes1.astype(np.float32))
    boxes2_tensor = torch.from_numpy(boxes2.astype(np.float32))
    ious = box_iou_rotated(boxes1_tensor, boxes2_tensor)
    return np.clip(ious.numpy(), 0.0, 1.0)

def _evaluate_single_class_worker(args):
    """Worker: AP for one class at one IoU threshold."""
    evaluator_data, class_id, iou_threshold = args
    predictions = evaluator_data['predictions']
    annotations = evaluator_data['annotations']
    maxDets = evaluator_data['maxDets']

    all_tps, all_fps, all_scores = [], [], []
    total_gt = 0
    common_ids = set(predictions.keys()) & set(annotations.keys())

    for img_id in common_ids:
        img_predictions = _get_image_predictions_worker(predictions, img_id, class_id, maxDets)
        img_annotations = _get_image_annotations_worker(annotations, img_id, class_id)
        if len(img_predictions) == 0:
            total_gt += len(img_annotations)
            continue
        if len(img_annotations) == 0:
            for pred in img_predictions:
                all_tps.append(0)
                all_fps.append(1)
                all_scores.append(pred['score'])
            continue
        img_predictions.sort(key=lambda x: x['score'], reverse=True)
        img_tp, img_fp, img_scores = _evaluate_image_predictions_worker(
            img_predictions, img_annotations, iou_threshold
        )
        all_tps.extend(img_tp)
        all_fps.extend(img_fp)
        all_scores.extend(img_scores)
        total_gt += len(img_annotations)

    if len(all_tps) == 0 or total_gt == 0:
        return 0.0

    sorted_indices = np.argsort(all_scores)[::-1]
    tp = np.array(all_tps)[sorted_indices]
    fp = np.array(all_fps)[sorted_indices]
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recall = tp_cumsum / (total_gt + 1e-8)
    ap = _compute_ap_worker(precision, recall)
    return ap

def _get_image_predictions_worker(predictions, img_id: int, class_id: int, maxDets: int) -> List[Dict]:
    """Worker: predictions for one image and class."""
    if img_id not in predictions:
        return []
    pred = predictions[img_id]
    if len(pred['labels']) == 0:
        return []
    class_mask = pred['labels'] == class_id
    if class_mask.sum() == 0:
        return []
    class_boxes = pred['boxes'][class_mask]
    class_scores = pred['scores'][class_mask]
    sorted_indices = np.argsort(class_scores)[::-1]
    if len(sorted_indices) > maxDets:
        sorted_indices = sorted_indices[:maxDets]
    class_boxes = class_boxes[sorted_indices]
    class_scores = class_scores[sorted_indices]
    return [{'box': box, 'score': float(score)} for box, score in zip(class_boxes, class_scores)]

def _get_image_annotations_worker(annotations, img_id: int, class_id: int) -> List[Dict]:
    """Worker: annotations for one image and class."""
    if img_id not in annotations:
        return []
    ann = annotations[img_id]
    if len(ann['labels']) == 0:
        return []
    gt_class_mask = ann['labels'] == class_id
    if gt_class_mask.sum() == 0:
        return []
    gt_boxes = ann['boxes'][gt_class_mask]
    return [{'box': box} for box in gt_boxes]

def _evaluate_image_predictions_worker(predictions: List[Dict], annotations: List[Dict], 
                                     iou_threshold: float) -> tuple:
    """Worker: evaluate predictions of one image."""
    num_preds = len(predictions)
    num_gts = len(annotations)
    if num_preds == 0:
        return [], [], []
    if num_gts == 0:
        tp_list = [0] * num_preds
        fp_list = [1] * num_preds
        score_list = [pred['score'] for pred in predictions]
        return tp_list, fp_list, score_list
    gt_boxes = np.array([ann['box'] for ann in annotations])
    gt_matched = [False] * num_gts
    tp_list, fp_list, score_list = [], [], []
    for pred in predictions:
        pred_box = pred['box'].reshape(1, -1)
        score = pred['score']
        ious = rotated_iou_mmcv_5dim(pred_box, gt_boxes)
        best_gt_idx = np.argmax(ious[0])
        max_iou = float(np.clip(ious[0, best_gt_idx], 0.0, 1.0))
        if max_iou >= iou_threshold and not gt_matched[best_gt_idx]:
            tp_list.append(1)
            fp_list.append(0)
            gt_matched[best_gt_idx] = True
        else:
            tp_list.append(0)
            fp_list.append(1)
        score_list.append(score)
    return tp_list, fp_list, score_list

def _compute_ap_worker(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute AP from precision/recall arrays."""
    if len(precision) == 0 or len(recall) == 0:
        return 0.0
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
 
class CODroneEvaluator:
    
    def __init__(self, class_names: List[str], iou_thresholds: Optional[List[float]] = None, 
                 maxDets: int = 100, num_workers: Optional[int] = None):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.maxDets = maxDets
        if num_workers is None:
            self.num_workers = min(mp.cpu_count(), len(class_names), 12)
        else:
            self.num_workers = max(1, min(num_workers, mp.cpu_count()))
        self.predictions = {}
        self.annotations = {}
        self.img_ids = []
        self.eval_results = []
        self.stats = None
        self.mean_ap = 0.0
        print(f"[INFO] Evaluator init: classes={self.num_classes}, workers={self.num_workers}")
    
    def update(self, predictions: Dict[int, Dict[str, Any]]):
        """Ingest predictions: {image_id: {labels, boxes, scores}} (boxes in 5D pixels)."""
        for img_id, pred in predictions.items():
            if img_id not in self.img_ids:
                self.img_ids.append(img_id)
            if 'boxes' in pred and len(pred['boxes']) > 0:
                boxes = pred['boxes']
                if hasattr(boxes, 'cpu'):
                    boxes = boxes.cpu()
                if hasattr(boxes, 'numpy'):
                    boxes = boxes.numpy()
                labels = pred['labels']
                if hasattr(labels, 'cpu'):
                    labels = labels.cpu()
                if hasattr(labels, 'numpy'):
                    labels = labels.numpy()
                scores = pred['scores']
                if hasattr(scores, 'cpu'):
                    scores = scores.cpu()
                if hasattr(scores, 'numpy'):
                    scores = scores.numpy()
                if boxes.shape[-1] != 5:
                    print(f"[WARN] Expect 5D boxes, got {boxes.shape[-1]}; trying convert...")
                    if boxes.shape[-1] == 8:
                        boxes = self._convert_8points_to_5param(boxes)
                    else:
                        boxes = np.empty((0, 5))
                        labels = np.empty((0,))
                        scores = np.empty((0,))
                self.predictions[img_id] = {
                    'boxes': boxes,
                    'labels': labels,
                    'scores': scores
                }
            else:
                self.predictions[img_id] = {
                    'boxes': np.empty((0, 5)),
                    'labels': np.empty((0,)),
                    'scores': np.empty((0,))
                }
    
    def add_annotations(self, annotations: List[Dict[str, Any]]):
        """Ingest ground-truth (5D pixel boxes)."""
        for img_idx, ann in enumerate(annotations):
            img_id = img_idx
            if 'boxes' in ann and len(ann['boxes']) > 0:
                boxes = ann['boxes']
                if hasattr(boxes, 'cpu'):
                    boxes = boxes.cpu()
                if hasattr(boxes, 'numpy'):
                    boxes = boxes.numpy()
                labels = ann['labels']
                if hasattr(labels, 'cpu'):
                    labels = labels.cpu()
                if hasattr(labels, 'numpy'):
                    labels = labels.numpy()
                if boxes.shape[-1] == 5:
                    boxes_5dim = boxes
                elif boxes.shape[-1] == 8:
                    boxes_5dim = self._convert_8points_to_5param(boxes)
                else:
                    print(f"[WARN] Unknown GT format: {boxes.shape[-1]}D")
                    boxes_5dim = np.empty((0, 5))
                    labels = np.empty((0,))
                self.annotations[img_id] = {
                    'boxes': boxes_5dim,
                    'labels': labels
                }
            else:
                self.annotations[img_id] = {
                    'boxes': np.empty((0, 5)),
                    'labels': np.empty((0,))
                }
    
    def _convert_8points_to_5param(self, boxes_8pt: np.ndarray) -> np.ndarray:
        """Convert 8-point polygon to [cx,cy,w,h,angle]."""
        boxes_5param = []
        for box_8pt in boxes_8pt:
            try:
                corners = box_8pt.reshape(4, 2)
                cx = np.mean(corners[:, 0])
                cy = np.mean(corners[:, 1])
                edge1 = corners[1] - corners[0]
                edge2 = corners[2] - corners[1]
                w = np.linalg.norm(edge1)
                h = np.linalg.norm(edge2)
                angle = np.arctan2(edge1[1], edge1[0])
                if h > w:
                    w, h = h, w
                    angle += np.pi / 2
                boxes_5param.append([cx, cy, w, h, angle])
            except Exception:
                boxes_5param.append([0, 0, 1, 1, 0])
        return np.array(boxes_5param, dtype=np.float32)
    
    def synchronize_between_processes(self):
        """No-op for single process."""
        pass
    
    def accumulate(self):
        """Prepare for summarize; basic sanity prints."""
        print(f"[INFO] Accumulate: preds={len(self.predictions)}, gts={len(self.annotations)}")
        if len(self.predictions) == 0 or len(self.annotations) == 0:
            print("[WARN] Empty predictions or annotations")
            return
        common_ids = set(self.predictions.keys()) & set(self.annotations.keys())
        if len(common_ids) == 0:
            print("[WARN] No matching image ids between preds and gts")
            return
        print(f"[INFO] Evaluate {len(common_ids)} samples (workers={self.num_workers})")
    
    def summarize(self):
        """Run evaluation and compute metrics."""
        print("[INFO] Start evaluation...")
        if len(self.predictions) == 0 or len(self.annotations) == 0:
            print("[WARN] No data to evaluate")
            self.stats = np.zeros(12)
            self.mean_ap = 0.0
            return {}
        results = self._evaluate_multiprocess()
        
        # 设置COCO兼容的stats
        self.stats = np.array([
            results.get('mAP', 0.0),
            results.get('mAP@0.50', 0.0), 
            results.get('mAP@0.75', 0.0),
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ])
        self.mean_ap = results.get('mAP', 0.0)
        self._print_evaluation_results(results)
        return results
    
    def _evaluate_multiprocess(self) -> Dict[str, float]:
        """Full evaluation with multiprocessing if enabled."""
        results = {}
        evaluator_data = {
            'predictions': self.predictions,
            'annotations': self.annotations, 
            'maxDets': self.maxDets
        }
        print(f"[INFO] Compute AP with {self.num_workers} workers")
        if self.num_workers == 1:
            print("[INFO] Fallback to single process")
            all_aps = []
            class_aps = {name: [] for name in self.class_names}
            for iou_thresh in self.iou_thresholds:
                thresh_aps = []
                for class_id, class_name in enumerate(self.class_names):
                    ap = self._evaluate_class(class_id, iou_thresh)
                    thresh_aps.append(ap)
                    class_aps[class_name].append(ap)
                    results[f'{class_name}_AP@{iou_thresh:.2f}'] = ap
                all_aps.append(thresh_aps)
                results[f'mAP@{iou_thresh:.2f}'] = np.mean(thresh_aps)
        else:
            try:
                all_aps = []
                class_aps = {name: [] for name in self.class_names}
                for iou_thresh in self.iou_thresholds:
                    tasks = [(evaluator_data, class_id, iou_thresh) for class_id in range(self.num_classes)]
                    with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                        thresh_aps = list(executor.map(_evaluate_single_class_worker, tasks))
                    for class_id, (class_name, ap) in enumerate(zip(self.class_names, thresh_aps)):
                        class_aps[class_name].append(ap)
                        results[f'{class_name}_AP@{iou_thresh:.2f}'] = ap
                    all_aps.append(thresh_aps)
                    results[f'mAP@{iou_thresh:.2f}'] = np.mean(thresh_aps)
            except Exception as e:
                print(f"[WARN] Multiprocess failed, fallback: {e}")
                return self._evaluate()
        for class_name in self.class_names:
            results[f'{class_name}_mAP'] = np.mean(class_aps[class_name])
        results['mAP'] = np.mean(all_aps)
        results['mAP@0.50'] = results.get('mAP@0.50', 0.0)
        results['mAP@0.75'] = results.get('mAP@0.75', 0.0)
        return results
    
    def _evaluate(self) -> Dict[str, float]:
        """Single-process evaluation (fallback)."""
        results = {}
        all_aps = []
        class_aps = {name: [] for name in self.class_names}
        for iou_thresh in self.iou_thresholds:
            thresh_aps = []
            for class_id, class_name in enumerate(self.class_names):
                ap = self._evaluate_class(class_id, iou_thresh)
                thresh_aps.append(ap)
                class_aps[class_name].append(ap)
                results[f'{class_name}_AP@{iou_thresh:.2f}'] = ap
            all_aps.append(thresh_aps)
            results[f'mAP@{iou_thresh:.2f}'] = np.mean(thresh_aps)
        for class_name in self.class_names:
            results[f'{class_name}_mAP'] = np.mean(class_aps[class_name])
        results['mAP'] = np.mean(all_aps)
        results['mAP@0.50'] = results.get('mAP@0.50', 0.0)
        results['mAP@0.75'] = results.get('mAP@0.75', 0.0)
        return results
    
    def _evaluate_class(self, class_id: int, iou_threshold: float) -> float:
        """AP for one class at a given IoU (image-wise evaluation)."""
        all_tps, all_fps, all_scores = [], [], []
        total_gt = 0
        common_ids = set(self.predictions.keys()) & set(self.annotations.keys())
        for img_id in common_ids:
            img_predictions = self._get_image_predictions(img_id, class_id)
            img_annotations = self._get_image_annotations(img_id, class_id)
            if len(img_predictions) == 0:
                total_gt += len(img_annotations)
                continue
            if len(img_annotations) == 0:
                for pred in img_predictions:
                    all_tps.append(0)
                    all_fps.append(1)
                    all_scores.append(pred['score'])
                continue
            img_predictions.sort(key=lambda x: x['score'], reverse=True)
            img_tp, img_fp, img_scores = self._evaluate_image_predictions(
                img_predictions, img_annotations, iou_threshold
            )
            all_tps.extend(img_tp)
            all_fps.extend(img_fp)
            all_scores.extend(img_scores)
            total_gt += len(img_annotations)
        if len(all_tps) == 0 or total_gt == 0:
            return 0.0
        sorted_indices = np.argsort(all_scores)[::-1]
        tp = np.array(all_tps)[sorted_indices]
        fp = np.array(all_fps)[sorted_indices]
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recall = tp_cumsum / (total_gt + 1e-8)
        ap = self._compute_ap(precision, recall)
        return ap
    
    def _get_image_predictions(self, img_id: int, class_id: int) -> List[Dict]:
        """Predictions of one image/class (apply per-class maxDets)."""
        if img_id not in self.predictions:
            return []
        pred = self.predictions[img_id]
        if len(pred['labels']) == 0:
            return []
        class_mask = pred['labels'] == class_id
        if class_mask.sum() == 0:
            return []
        class_boxes = pred['boxes'][class_mask]
        class_scores = pred['scores'][class_mask]
        sorted_indices = np.argsort(class_scores)[::-1]
        if len(sorted_indices) > self.maxDets:
            sorted_indices = sorted_indices[:self.maxDets]
        class_boxes = class_boxes[sorted_indices]
        class_scores = class_scores[sorted_indices]
        return [{'box': box, 'score': float(score)} for box, score in zip(class_boxes, class_scores)]
    
    def _get_image_annotations(self, img_id: int, class_id: int) -> List[Dict]:
        """Ground-truth of one image/class."""
        if img_id not in self.annotations:
            return []
        ann = self.annotations[img_id]
        if len(ann['labels']) == 0:
            return []
        gt_class_mask = ann['labels'] == class_id
        if gt_class_mask.sum() == 0:
            return []
        gt_boxes = ann['boxes'][gt_class_mask]
        return [{'box': box} for box in gt_boxes]
    
    def _evaluate_image_predictions(self, predictions: List[Dict], annotations: List[Dict], 
                                  iou_threshold: float) -> tuple:
        """Evaluate predictions of one image (preds sorted by score)."""
        num_preds = len(predictions)
        num_gts = len(annotations)
        if num_preds == 0:
            return [], [], []
        if num_gts == 0:
            tp_list = [0] * num_preds
            fp_list = [1] * num_preds
            score_list = [pred['score'] for pred in predictions]
            return tp_list, fp_list, score_list
        gt_boxes = np.array([ann['box'] for ann in annotations])
        gt_matched = [False] * num_gts
        tp_list, fp_list, score_list = [], [], []
        for pred in predictions:
            pred_box = pred['box'].reshape(1, -1)
            score = pred['score']
            ious = rotated_iou_mmcv_5dim(pred_box, gt_boxes)
            best_gt_idx = np.argmax(ious[0])
            max_iou = float(np.clip(ious[0, best_gt_idx], 0.0, 1.0))
            if max_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                tp_list.append(1)
                fp_list.append(0)
                gt_matched[best_gt_idx] = True
            else:
                tp_list.append(0)
                fp_list.append(1)
            score_list.append(score)
        return tp_list, fp_list, score_list
    
    def _compute_ap(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """Compute AP (full PR area)."""
        if len(precision) == 0 or len(recall) == 0:
            return 0.0
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
    
    def _print_evaluation_results(self, results: Dict[str, float]):
        """Print summary metrics."""
        print("="*48)
        print("CODrone rotated IoU (5D) evaluation")
        print("="*48)
        print(f"workers: {self.num_workers}")
        print(f"mAP: {results['mAP']:.4f}")
        print(f"mAP@0.50: {results.get('mAP@0.50', 0.0):.4f}")
        print(f"mAP@0.75: {results.get('mAP@0.75', 0.0):.4f}")
        print("per-class mAP:")
        for class_name in self.class_names:
            class_map = results.get(f'{class_name}_mAP', 0.0)
            print(f"  {class_name}: {class_map:.4f}")
        print("="*48)