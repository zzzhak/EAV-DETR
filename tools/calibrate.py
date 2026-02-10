import os
import sys
import json
import re
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Any, Optional
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core import YAMLConfig
from src.zoo.eavdetr.oriented_ops import oriented_box_iou
import src.misc.dist as dist

# Import geometry utilities
try:
    from geometry_utils import convert_to_corners, calculate_expansion_margin
    print("Geometry utils loaded successfully")
except ImportError as e:
    raise ImportError(f"Failed to import geometry utils: {e}")


def parse_flight_condition(filename: str) -> str:
    """
    Parse flight condition from CODrone dataset filename
    
    Args:
        filename: e.g., "chenhuachengpark_day_30m_30c_frame_1650__1__0___0"
    Returns:
        condition: e.g., "30m_30deg" or "unknown"
    """
    try:
        # Match pattern: XXm_YYc (e.g., 30m_30c, 60m_90c)
        pattern = r'(\d+)m_(\d+)c'
        match = re.search(pattern, filename)
        
        if match:
            height = match.group(1)  # Extract height
            angle = match.group(2)   # Extract angle
            
            # Convert from "c" format to "deg" format
            return f"{height}m_{angle}deg"
        else:
            print(f"Warning: Cannot parse flight condition from: {filename}")
            return "unknown"
            
    except Exception as e:
        print(f"Warning: Error parsing flight condition: {e}, filename: {filename}")
        return "unknown"


def get_all_flight_conditions() -> List[str]:
    """
    Get all predefined flight conditions for CODrone dataset
    
    Returns:
        conditions: List of all conditions
    """
    return [
        "30m_30deg",   # 30m height, 30 degree
        "30m_90deg",   # 30m height, 90 degree
        "60m_30deg",   # 60m height, 30 degree
        "60m_90deg",   # 60m height, 90 degree
        "100m_30deg",  # 100m height, 30 degree
        "100m_90deg",  # 100m height, 90 degree
        "unknown"      # unknown condition
    ]


class ConformalCalibrator:
    """Conformal Prediction Calibrator"""
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
        """
        Initialize calibrator
        
        Args:
            config_path: Configuration file path
            checkpoint_path: Model checkpoint path
            device: Inference device
        """
        self.device = device
        print(f"Loading config: {config_path}")
        self.cfg = YAMLConfig(config_path)
        
        # Load model
        print("Building model...")
        self.model = self.cfg.model.to(device)
        
        # Load weights
        if checkpoint_path:
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'ema' in checkpoint:
                state = checkpoint['ema']['module']
                print("Loading EMA weights for calibration")
            elif 'model' in checkpoint:
                state = checkpoint['model']
                print("Loading model weights for calibration")
            else:
                state = checkpoint
                print("Loading direct weights for calibration")
            
            self.model.load_state_dict(state)
        
        # Set to evaluation mode (no post-processing)
        self.model.eval()
        
        # Load calibration data (using validation set)
        print("Loading calibration dataset...")
        self.dataloader = self.cfg.val_dataloader
        
        print("Conformal calibrator initialized successfully")
        print(f"   Device: {device}")
        print(f"   Calibration batches: {len(self.dataloader)}")
        print(f"   Dataset type: {type(self.dataloader.dataset).__name__}")
        
    def build_cost_matrix(self, pred_boxes: torch.Tensor, pred_scores: torch.Tensor, 
                         gt_boxes: torch.Tensor, score_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build cost matrix for Hungarian matching
        
        Args:
            pred_boxes: Predicted boxes [N, 5] normalized (cx, cy, w, h, angle)
            pred_scores: Prediction scores [N]
            gt_boxes: Ground truth boxes [M, 5] normalized (cx, cy, w, h, angle)
            score_threshold: Confidence threshold
            
        Returns:
            cost_matrix: Cost matrix [N_filtered, M]
            high_conf_mask: High confidence mask [N]
        """
        # Filter high confidence predictions
        high_conf_mask = pred_scores > score_threshold
        if high_conf_mask.sum() == 0:
            # No high confidence predictions
            empty_cost = torch.empty(0, len(gt_boxes), device=pred_boxes.device)
            return empty_cost, high_conf_mask
        
        filtered_pred_boxes = pred_boxes[high_conf_mask]  # [N_filtered, 5]
        
        if len(gt_boxes) == 0:
            # No GT boxes
            cost_matrix = torch.ones(len(filtered_pred_boxes), 0, device=pred_boxes.device)
            return cost_matrix, high_conf_mask
        
        # Calculate oriented IoU matrix
        pred_expanded = filtered_pred_boxes.unsqueeze(1).expand(-1, len(gt_boxes), -1)  # [N_filtered, M, 5]
        gt_expanded = gt_boxes.unsqueeze(0).expand(len(filtered_pred_boxes), -1, -1)    # [N_filtered, M, 5]
        
        # Calculate oriented IoU
        ious = oriented_box_iou(pred_expanded, gt_expanded)  # [N_filtered, M]
        
        # Convert to cost matrix (cost = 1 - IoU)
        cost_matrix = 1.0 - ious
        
        return cost_matrix, high_conf_mask
    
    def hungarian_matching(self, cost_matrix: torch.Tensor, 
                          iou_threshold: float = 0.5) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Run Hungarian algorithm for optimal matching
        
        Args:
            cost_matrix: Cost matrix [N, M]
            iou_threshold: IoU filtering threshold
            
        Returns:
            matched_pairs: List of matched pairs [(pred_idx, gt_idx), ...]
            matched_ious: Corresponding IoU values
        """
        if cost_matrix.size(0) == 0 or cost_matrix.size(1) == 0:
            return [], []
        
        # Convert to numpy for Hungarian algorithm
        cost_matrix_np = cost_matrix.cpu().numpy()
        
        # Run Hungarian algorithm
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix_np)
        
        # Calculate IoU values and filter by threshold
        matched_pairs = []
        matched_ious = []
        
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            iou_value = 1.0 - cost_matrix_np[pred_idx, gt_idx]  # Convert back to IoU
            
            # Only keep matches above threshold
            if iou_value > iou_threshold:
                matched_pairs.append((pred_idx, gt_idx))
                matched_ious.append(iou_value)
        
        return matched_pairs, matched_ious
    
    def process_single_image(self, outputs: Dict, targets: Dict, 
                           img_size: Tuple[int, int]) -> List[float]:
        """
        Process single image for prediction-target matching
        
        Args:
            outputs: Model raw output {'pred_logits', 'pred_boxes'}
            targets: Ground truth annotations {'boxes', 'labels', 'size', ...}
            img_size: Image size (H, W)
            
        Returns:
            margins: Non-conformity scores for this image
        """
        pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
        pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 5]
        
        gt_boxes = targets['boxes']
        gt_labels = targets['labels']   # [num_gts]
        
        pred_probs = F.sigmoid(pred_logits)  # [num_queries, num_classes] 
        pred_scores = pred_probs.max(dim=-1)[0]
        
        cost_matrix, high_conf_mask = self.build_cost_matrix(
            pred_boxes, pred_scores, gt_boxes, score_threshold=0.5)
        
        matched_pairs, matched_ious = self.hungarian_matching(
            cost_matrix, iou_threshold=0.5)
        
        if len(matched_pairs) == 0:
            return []
        
        high_conf_indices = torch.where(high_conf_mask)[0]
        margins = []
        
        img_h, img_w = img_size
        
        for pred_idx, gt_idx in matched_pairs:
            global_pred_idx = high_conf_indices[pred_idx].item()
            
            pred_box_norm = pred_boxes[global_pred_idx]
            gt_box_norm = gt_boxes[gt_idx]
            
            pred_box_pixel = pred_box_norm.clone()
            pred_box_pixel[0] *= img_w  # cx
            pred_box_pixel[1] *= img_h  # cy  
            pred_box_pixel[2] *= img_w  # w
            pred_box_pixel[3] *= img_h  # h
            
            gt_box_pixel = gt_box_norm.clone()
            gt_box_pixel[0] *= img_w   # cx
            gt_box_pixel[1] *= img_h   # cy
            gt_box_pixel[2] *= img_w   # w  
            gt_box_pixel[3] *= img_h   # h
            
            try:
                pred_corners = convert_to_corners(pred_box_pixel.cpu().numpy())
                gt_corners = convert_to_corners(gt_box_pixel.cpu().numpy())
                
                margin = calculate_expansion_margin(pred_corners, gt_corners)
                
                margins.append(margin)
                
            except Exception as e:
                raise RuntimeError(f"Geometry calculation failed (pred_idx={global_pred_idx}, gt_idx={gt_idx}): {e}")
        
        return margins
    
    def calibrate(self, output_path: str = './conformal_params.json', 
                 alpha: float = 0.1) -> Dict[str, float]:
        """
        Execute conformal prediction calibration with flight condition grouping
        
        Args:
            output_path: Parameter save path
            alpha: Risk level (e.g., 0.1 means 90% coverage)
            
        Returns:
            calibration_params: Calibration parameters dictionary
        """
        print("Starting conformal calibration...")
        
        conditions = get_all_flight_conditions()
        
        conditional_margins = {cond: [] for cond in conditions}
        condition_stats = {cond: {'samples': 0, 'matches': 0} for cond in conditions}
        
        processed_samples = 0
        total_matches = 0
        
        with torch.no_grad():
            for batch_idx, (samples, targets) in enumerate(self.dataloader):
                samples = samples.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                outputs = self.model(samples)
                
                if not isinstance(outputs, dict) or 'pred_logits' not in outputs or 'pred_boxes' not in outputs:
                    continue
                
                batch_size = samples.shape[0]
                for i in range(batch_size):
                    single_output = {
                        'pred_logits': outputs['pred_logits'][i:i+1],
                        'pred_boxes': outputs['pred_boxes'][i:i+1]
                    }
                    single_target = targets[i]
                    
                    filename = single_target.get('filename', 'unknown_file')
                    current_condition = parse_flight_condition(filename)
                    
                    if 'size' in single_target:
                        img_size = tuple(single_target['size'].cpu().numpy())  # (H, W)
                    else:
                        img_size = (1024, 1024)
                    
                    try:
                        margins = self.process_single_image(single_output, single_target, img_size)
                        
                        if current_condition in conditional_margins:
                            conditional_margins[current_condition].extend(margins)
                            condition_stats[current_condition]['matches'] += len(margins)
                        else:
                            conditional_margins['unknown'].extend(margins)
                            condition_stats['unknown']['matches'] += len(margins)
                        
                        processed_samples += 1
                        total_matches += len(margins)
                        if current_condition in condition_stats:
                            condition_stats[current_condition]['samples'] += 1
                        else:
                            condition_stats['unknown']['samples'] += 1
                        
                        
                    except Exception as e:
                        raise RuntimeError(f"Sample {processed_samples+1} processing failed: {e}")
                
        
        total_collected_margins = sum(len(margins) for margins in conditional_margins.values())
        if total_collected_margins == 0:
            raise RuntimeError("No non-conformity scores collected")
        
        
        
        conditions_results = {}
        for condition in conditions:
            margins_list = conditional_margins[condition]
            
            if len(margins_list) == 0:
                print(f"   {condition}: No data, skipping")
                conditions_results[condition] = {
                    'q_alpha': None,
                    'total_matches': 0,
                    'margin_stats': None,
                    'warning': 'No data available for this condition'
                }
                continue
            
            margins_array = np.array(margins_list)
            q_alpha = np.quantile(margins_array, 1 - alpha)
            
            margin_stats = {
                'mean': float(np.mean(margins_array)),
                'std': float(np.std(margins_array)),
                'min': float(np.min(margins_array)),
                'max': float(np.max(margins_array)),
                'median': float(np.median(margins_array)),
                'q25': float(np.quantile(margins_array, 0.25)),
                'q75': float(np.quantile(margins_array, 0.75)),
                'q90': float(np.quantile(margins_array, 0.90)),
                'q95': float(np.quantile(margins_array, 0.95)),
                'q99': float(np.quantile(margins_array, 0.99))
            }
            
            conditions_results[condition] = {
                'q_alpha': float(q_alpha),
                'total_matches': len(margins_list),
                'total_samples': condition_stats[condition]['samples'],
                'matches_per_sample': len(margins_list) / max(condition_stats[condition]['samples'], 1),
                'margin_stats': margin_stats
            }
            
        
        all_margins_combined = []
        for condition, margins_list in conditional_margins.items():
            all_margins_combined.extend(margins_list)
        
        global_stats = None
        global_q_alpha = None
        
        if len(all_margins_combined) > 0:
            all_margins_array = np.array(all_margins_combined)
            global_q_alpha = np.quantile(all_margins_array, 1 - alpha)
            
            global_stats = {
                'q_alpha': float(global_q_alpha),
                'total_matches': len(all_margins_combined),
                'total_samples': processed_samples,
                'matches_per_sample': len(all_margins_combined) / max(processed_samples, 1),
                'margin_stats': {
                    'mean': float(np.mean(all_margins_array)),
                    'std': float(np.std(all_margins_array)),
                    'min': float(np.min(all_margins_array)),
                    'max': float(np.max(all_margins_array)),
                    'median': float(np.median(all_margins_array)),
                    'q25': float(np.quantile(all_margins_array, 0.25)),
                    'q75': float(np.quantile(all_margins_array, 0.75)),
                    'q90': float(np.quantile(all_margins_array, 0.90)),
                    'q95': float(np.quantile(all_margins_array, 0.95)),
                    'q99': float(np.quantile(all_margins_array, 0.99))
                }
            }
            
        else:
            print("   Warning: No global data available")

        calibration_params = {
            'alpha': alpha,
            'coverage_target': 1 - alpha,
            'calibration_mode': 'conditional',
            'conditions': conditions_results,
            'global': global_stats,
            'summary': {
                'total_samples': processed_samples,
                'total_matches': total_collected_margins,
                'conditions_with_data': len([c for c in conditions_results.values() if c['q_alpha'] is not None]),
                'conditions_without_data': len([c for c in conditions_results.values() if c['q_alpha'] is None])
            },
            'calibration_info': {
                'config_file': getattr(self.cfg, 'config_path', 'unknown'),
                'device': self.device,
                'score_threshold': 0.5,
                'iou_threshold': 0.5,
                'flight_conditions': conditions
            }
        }
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(calibration_params, f, indent=2, ensure_ascii=False)
        
        print("\nCalibration completed")
        
        
        print(f"\nCalibration parameters saved to: {output_path}")
        
        return calibration_params


def main():
    parser = argparse.ArgumentParser(description='Conformal prediction calibration script')
    parser.add_argument('-c', '--config', 
                        default='configs/eavdetr/r50vd_codrone.yml',
                        help='Configuration file path')
    parser.add_argument('-r', '--checkpoint', required=True,
                        help='Model checkpoint path (required)')
    parser.add_argument('--device', default='cuda:0',
                        help='Inference device (default: cuda:0)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Risk level (default: 0.1 for 90%% coverage)')
    parser.add_argument('--output', default='./conformal_params.json',
                        help='Output parameter file path (default: ./conformal_params.json)')
    
    args = parser.parse_args()
    
    # Check required files
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    # Validate alpha range
    if not 0 < args.alpha < 1:
        print(f"Error: alpha must be in range (0,1), current: {args.alpha}")
        return
    
    print("="*80)
    print("CODrone Oriented Object Detection - Conformal Prediction Calibration")
    print("="*80)
    print(f"Config file: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Risk level: {args.alpha} (target coverage: {1-args.alpha:.1%})")
    print(f"Output file: {args.output}")
    print("-"*80)
    
    # Initialize calibrator
    calibrator = ConformalCalibrator(args.config, args.checkpoint, args.device)
    
    # Execute calibration
    calibration_params = calibrator.calibrate(args.output, args.alpha)
    
    if calibration_params:
        print("\n" + "="*80)
        print("Grouped conformal prediction calibration completed!")
        print(f"Target coverage: {calibration_params['coverage_target']:.1%}")
        print("Calibration mode: grouped by flight conditions")
        print(f"Valid conditions: {calibration_params['summary']['conditions_with_data']}/{len(get_all_flight_conditions())-1}")  # -1 because unknown doesn't count
        print(f"Parameter file: {args.output}")
        
        # Show global parameters
        if 'global' in calibration_params and calibration_params['global']:
            global_stats = calibration_params['global']
            print(f"\nGlobal parameters:")
            print(f"   Global q_α: {global_stats['q_alpha']:.6f} (all conditions combined)")
            print(f"   Global matches: {global_stats['total_matches']}")
        
        # Show key parameters for each condition
        print(f"\nKey parameters q_α by condition:")
        for condition, result in calibration_params['conditions'].items():
            if result['q_alpha'] is not None:
                print(f"   {condition}: {result['q_alpha']:.6f} ({result['total_matches']} matches)")
            elif condition != 'unknown':  # Don't show 'unknown' condition without data
                print(f"   {condition}: no data")
        
        print("="*80)
    else:
        raise RuntimeError("Calibration returned empty result")


if __name__ == '__main__':
    main()