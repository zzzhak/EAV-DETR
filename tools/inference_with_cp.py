import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Any, Optional
import argparse
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core import YAMLConfig
from src.zoo.eavdetr.oriented_ops import oriented_box_iou
import src.misc.dist as dist

try:
    from geometry_utils import convert_to_corners, expand_obb, is_point_inside_obb
    print("Geometry utils loaded successfully")
except ImportError as e:
    raise ImportError(f"Failed to import geometry utils: {e}")

try:
    from calibrate import parse_flight_condition, get_all_flight_conditions
    print("Condition parsing functions loaded successfully")
except ImportError as e:
    raise ImportError(f"Failed to import condition parsing functions: {e}")


class ConformalPredictor:
    """Conformal Prediction Inferencer"""
    
    def __init__(self, config_path: str, checkpoint_path: str, 
                 conformal_params_path: str, device: str = 'cuda:0'):
        self.device = device
        print(f"Loading config: {config_path}")
        self.cfg = YAMLConfig(config_path)
        
        print("Building model...")
        self.model = self.cfg.model.to(device)
        
        if checkpoint_path:
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'ema' in checkpoint:
                state = checkpoint['ema']['module']
                print("Loading EMA weights for inference")
            elif 'model' in checkpoint:
                state = checkpoint['model']
                print("Loading model weights for inference")
            else:
                state = checkpoint
                print("Loading direct weights for inference")
            
            self.model.load_state_dict(state)
        
        self.model.eval()
        
        print(f"Loading conditional conformal parameters: {conformal_params_path}")
        self.conformal_params = self.load_conformal_params(conformal_params_path)
        self.alpha = self.conformal_params['alpha']
        self.target_coverage = self.conformal_params['coverage_target']
        
        self.conditional_q_alphas = {}
        conditions = self.conformal_params['conditions']
        
        valid_conditions = []
        invalid_conditions = []
        
        for condition, result in conditions.items():
            if result.get('q_alpha') is not None:
                self.conditional_q_alphas[condition] = result['q_alpha']
                valid_conditions.append(condition)
            else:
                invalid_conditions.append(condition)
        
        if not self.conditional_q_alphas:
            raise ValueError("No conditions have valid q_alpha values for conditional conformal prediction.\n"
                           "Ensure calibration process generated valid q_alpha parameters for at least one condition.")
        
        print(f"Conditional conformal prediction parameter status:")
        print(f"   Valid conditions ({len(valid_conditions)}):")
        for condition in valid_conditions:
            q_alpha = self.conditional_q_alphas[condition]
            print(f"      {condition}: q_α = {q_alpha:.6f}")
        
        if invalid_conditions:
            print(f"   Invalid conditions ({len(invalid_conditions)}):")
            for condition in invalid_conditions:
                print(f"      {condition}: no q_alpha data")
        
        print(f"   Warning: Strict mode - testing only uses valid conditions, stops on invalid conditions")
        
        print("Loading test dataset...")
        if hasattr(self.cfg, 'test_dataloader') and self.cfg.test_dataloader is not None:
            self.dataloader = self.cfg.test_dataloader
            print("Using test_dataloader")
        else:
            raise RuntimeError("test_dataloader configuration not found. Conformal prediction inference requires dedicated test set.\n"
                             "Add test_dataloader configuration to config file or ensure test data loads correctly.")
        
        self.total_matched_boxes = 0
        self.covered_boxes = 0
        
        print("Conditional conformal prediction inferencer initialized successfully")
        print(f"   Device: {device}")
        print(f"   Test batches: {len(self.dataloader)}")
        print(f"   Target coverage: {self.target_coverage:.1%}")
        print(f"   Mode: strict conditional conformal prediction (no fallback)")
        
    def load_conformal_params(self, params_path: str) -> Dict[str, Any]:
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Conformal parameters file not found: {params_path}")
        
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        required_keys = ['alpha', 'coverage_target', 'calibration_mode']
        for key in required_keys:
            if key not in params:
                raise ValueError(f"Conformal parameters file missing required key: {key}")
        
        if params.get('calibration_mode') != 'conditional':
            raise ValueError(f"Parameter file not in conditional mode, current mode: {params.get('calibration_mode')}")
        
        if 'conditions' not in params:
            raise ValueError("Conditional conformal parameters file missing 'conditions' key")
        
        
        return params
    
    def build_cost_matrix(self, pred_boxes: torch.Tensor, pred_scores: torch.Tensor, 
                         gt_boxes: torch.Tensor, score_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        high_conf_mask = pred_scores > score_threshold
        if high_conf_mask.sum() == 0:
            empty_cost = torch.empty(0, len(gt_boxes), device=pred_boxes.device)
            return empty_cost, high_conf_mask
        
        filtered_pred_boxes = pred_boxes[high_conf_mask]  # [N_filtered, 5]
        
        if len(gt_boxes) == 0:
            cost_matrix = torch.ones(len(filtered_pred_boxes), 0, device=pred_boxes.device)
            return cost_matrix, high_conf_mask
        
        pred_expanded = filtered_pred_boxes.unsqueeze(1).expand(-1, len(gt_boxes), -1)  # [N_filtered, M, 5]
        gt_expanded = gt_boxes.unsqueeze(0).expand(len(filtered_pred_boxes), -1, -1)    # [N_filtered, M, 5]
        
        ious = oriented_box_iou(pred_expanded, gt_expanded)  # [N_filtered, M]
        
        cost_matrix = 1.0 - ious
        
        return cost_matrix, high_conf_mask
    
    def hungarian_matching(self, cost_matrix: torch.Tensor, 
                          iou_threshold: float = 0.5) -> Tuple[List[Tuple[int, int]], List[float]]:
        if cost_matrix.size(0) == 0 or cost_matrix.size(1) == 0:
            return [], []
        
        cost_matrix_np = cost_matrix.cpu().numpy()
        
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix_np)
        
        matched_pairs = []
        matched_ious = []
        
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            iou_value = 1.0 - cost_matrix_np[pred_idx, gt_idx]  # Convert back to IoU
            
            if iou_value > iou_threshold:
                matched_pairs.append((pred_idx, gt_idx))
                matched_ious.append(iou_value)
        
        return matched_pairs, matched_ious
    
    def process_single_image(self, outputs: Dict, targets: Dict, 
                           img_size: Tuple[int, int], q_alpha_for_this_image: float) -> Tuple[int, int]:
        pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
        pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 5]
        
        gt_boxes = targets['boxes']     # [num_gts, 5] normalized coordinates
        gt_labels = targets['labels']   # [num_gts]
        
        pred_probs = F.sigmoid(pred_logits)  # [num_queries, num_classes] 
        pred_scores = pred_probs.max(dim=-1)[0]  # [num_queries] max class confidence
        
        cost_matrix, high_conf_mask = self.build_cost_matrix(
            pred_boxes, pred_scores, gt_boxes, score_threshold=0.5)
        
        matched_pairs, matched_ious = self.hungarian_matching(
            cost_matrix, iou_threshold=0.5)
        
        if len(matched_pairs) == 0:
            return 0, 0
        
        high_conf_indices = torch.where(high_conf_mask)[0]
        covered_count = 0
        
        img_h, img_w = img_size
        
        for pred_idx, gt_idx in matched_pairs:
            global_pred_idx = high_conf_indices[pred_idx].item()
            
            pred_box_norm = pred_boxes[global_pred_idx]  # [5] normalized coordinates
            gt_box_norm = gt_boxes[gt_idx]               # [5] normalized coordinates
            
            pred_box_pixel = pred_box_norm.clone()
            pred_box_pixel[0] *= img_w
            pred_box_pixel[1] *= img_h
            pred_box_pixel[2] *= img_w
            pred_box_pixel[3] *= img_h
            
            gt_box_pixel = gt_box_norm.clone()
            gt_box_pixel[0] *= img_w
            gt_box_pixel[1] *= img_h
            gt_box_pixel[2] *= img_w
            gt_box_pixel[3] *= img_h
            
            try:
                pred_corners = convert_to_corners(pred_box_pixel.cpu().numpy())
                
                conformal_corners = expand_obb(pred_corners, q_alpha_for_this_image)
                
                gt_corners = convert_to_corners(gt_box_pixel.cpu().numpy())
                
                is_fully_covered = True
                for point in gt_corners:
                    if not is_point_inside_obb(point, conformal_corners):
                        is_fully_covered = False
                        break
                
                if is_fully_covered:
                    covered_count += 1
                
            except Exception as e:
                raise RuntimeError(f"Conformal prediction processing failed (pred_idx={global_pred_idx}, gt_idx={gt_idx}): {e}")
        
        return len(matched_pairs), covered_count
    
    def evaluate(self) -> Dict[str, Any]:
        print("Starting conformal prediction evaluation...")
        print(f"   Target coverage: {self.target_coverage:.1%}")
        print(f"   Dataset: test set")
        print(f"   Mode: strict conditional conformal prediction (dynamic q_α)")
        
        all_conditions = get_all_flight_conditions()
        self.conditional_stats = {cond: {'matched': 0, 'covered': 0} for cond in all_conditions}
        
        self.total_matched_boxes = 0
        self.covered_boxes = 0
        
        processed_samples = 0
        
        with torch.no_grad():
            for batch_idx, (samples, targets) in enumerate(self.dataloader):
                samples = samples.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                outputs = self.model(samples)
                
                if not isinstance(outputs, dict) or 'pred_logits' not in outputs or 'pred_boxes' not in outputs:
                    raise RuntimeError(f"Abnormal model output format, conformal prediction inference aborted.\n"
                                     f"Expected output contains 'pred_logits' and 'pred_boxes', actual output: "
                                     f"{outputs.keys() if isinstance(outputs, dict) else type(outputs)}\n"
                                     f"Please check if model configuration and weight files are correct")
                
                batch_size = samples.shape[0]
                for i in range(batch_size):
                    single_output = {
                        'pred_logits': outputs['pred_logits'][i:i+1],
                        'pred_boxes': outputs['pred_boxes'][i:i+1]
                    }
                    single_target = targets[i]
                    
                    filename = single_target.get('filename', 'unknown_file')
                    current_condition = parse_flight_condition(filename)
                    
                    if current_condition in self.conditional_q_alphas:
                        q_alpha_for_this_image = self.conditional_q_alphas[current_condition]
                        alpha_source = f"condition {current_condition}"
                    else:
                        available_conditions = list(self.conditional_q_alphas.keys())
                        raise RuntimeError(f"Test sample flight condition '{current_condition}' has no corresponding q_alpha value\n"
                                         f"   Sample file: {filename}\n"
                                         f"   Sample number: {processed_samples + 1}\n"
                                         f"   Available conditions: {available_conditions}\n"
                                         f"   Ensure this condition contains sufficient data in calibration phase, or check filename parsing correctness\n"
                                         f"   Strict mode provides no fallback, must use accurate condition matching")
                    
                    if 'size' in single_target:
                        img_size = tuple(single_target['size'].cpu().numpy())  # (H, W)
                    else:
                        img_size = (1024, 1024)
                        print(f"Warning: Image size not found, using default: {img_size}")
                    
                    try:
                        matched_count, covered_count = self.process_single_image(
                            single_output, single_target, img_size, q_alpha_for_this_image)
                        
                        self.conditional_stats[current_condition]['matched'] += matched_count
                        self.conditional_stats[current_condition]['covered'] += covered_count
                        
                        self.total_matched_boxes += matched_count
                        self.covered_boxes += covered_count
                        processed_samples += 1
                        
                        
                        
                    except Exception as e:
                        raise RuntimeError(f"Sample {processed_samples+1} processing failed: {e}")
                
                if (batch_idx + 1) % 10 == 0:
                    current_coverage = self.covered_boxes / max(self.total_matched_boxes, 1)
                    print(f"   Progress: {batch_idx+1}/{len(self.dataloader)} batches, "
                          f"cumulative coverage: {current_coverage:.1%}")
        
        if self.total_matched_boxes == 0:
            raise RuntimeError("No targets matched, conformal prediction evaluation failed.\n"
                              "Possible causes:\n"
                              "1. Model prediction confidence too low (<0.5)\n"
                              "2. Matching IoU too low (<0.5)\n"
                              "3. Test set annotation format error")
        
        print(f"\nCalculating coverage rates by condition...")
        conditional_coverage = {}
        overall_success = True
        
        for condition, stats in self.conditional_stats.items():
            matched = stats['matched']
            covered = stats['covered']
            
            if matched > 0:
                coverage_rate = covered / matched
                is_condition_success = coverage_rate >= self.target_coverage
                
                conditional_coverage[condition] = {
                    'matched': matched,
                    'covered': covered,
                    'coverage_rate': coverage_rate,
                    'target_coverage': self.target_coverage,
                    'is_success': is_condition_success,
                    'q_alpha_used': self.conditional_q_alphas.get(condition, None)
                }
                
                if not is_condition_success:
                    overall_success = False
                
                print(f"   {condition}: {covered}/{matched} = {coverage_rate:.1%} {'PASS' if is_condition_success else 'FAIL'}")
            else:
                conditional_coverage[condition] = {
                    'matched': 0,
                    'covered': 0,
                    'coverage_rate': None,
                    'target_coverage': self.target_coverage,
                    'is_success': None,
                    'q_alpha_used': self.conditional_q_alphas.get(condition, None)
                }
                print(f"   {condition}: no data")
        
        observed_coverage = self.covered_boxes / self.total_matched_boxes
        global_success = observed_coverage >= self.target_coverage
        
        print(f"\nGlobal coverage: {self.covered_boxes}/{self.total_matched_boxes} = {observed_coverage:.1%} {'PASS' if global_success else 'FAIL'}")
        
        
        results = {
            'target_coverage': self.target_coverage,
            'observed_coverage': observed_coverage,
            'total_matched_boxes': self.total_matched_boxes,
            'covered_boxes': self.covered_boxes,
            'is_success': global_success,
            'alpha': self.alpha,
            'processed_samples': processed_samples,
            
            'conditional_coverage': conditional_coverage,
            'overall_conditional_success': overall_success,
            'conditions_tested': len([c for c in conditional_coverage.values() if c['matched'] > 0]),
            'conditions_passed': len([c for c in conditional_coverage.values() if c['is_success'] is True]),
            
            'conformal_params_summary': {
                'calibration_mode': 'conditional_strict',
                'available_conditions': list(self.conditional_q_alphas.keys()),
                'strict_mode': True,
                'fallback_enabled': False
            }
        }
        
        return results
    
    
    def print_results(self, results: Dict[str, Any]):
        print("\n" + "="*80)
        print("CONDITIONAL CONFORMAL PREDICTION EVALUATION RESULTS")
        print("="*80)
        
        print("GLOBAL SUMMARY:")
        print(f"   Target Confidence (1 - alpha): {results['target_coverage']:.1%}")
        print(f"   Total True Positives Matched: {results['total_matched_boxes']}")
        print(f"   Total True Positives Covered: {results['covered_boxes']}")
        print(f"   Global Observed Coverage: {results['observed_coverage']:.1%} ({results['covered_boxes']} / {results['total_matched_boxes']})")
        
        if results['is_success']:
            print(f"   Global Result: SUCCESS. Global coverage ({results['observed_coverage']:.1%}) >= target ({results['target_coverage']:.1%})")
        else:
            print(f"   Global Result: FAILURE. Global coverage ({results['observed_coverage']:.1%}) < target ({results['target_coverage']:.1%})")
        
        print(f"\nCONDITIONAL RESULTS BY FLIGHT CONDITION:")
        print(f"   Conditions Tested: {results['conditions_tested']}")
        print(f"   Conditions Passed: {results['conditions_passed']}")
        print(f"   Overall Conditional Success: {'YES' if results['overall_conditional_success'] else 'NO'}")
        print()
        
        conditional_coverage = results['conditional_coverage']
        conditions_with_data = [(cond, stats) for cond, stats in conditional_coverage.items() if stats['matched'] > 0]
        conditions_without_data = [(cond, stats) for cond, stats in conditional_coverage.items() if stats['matched'] == 0]
        
        if conditions_with_data:
            print("   Conditions with Test Data:")
            for condition, stats in conditions_with_data:
                coverage_rate = stats['coverage_rate']
                target = stats['target_coverage']
                matched = stats['matched']
                covered = stats['covered']
                q_alpha = stats['q_alpha_used']
                success_icon = "PASS" if stats['is_success'] else "FAIL"
                
                print(f"     {condition}:")
                print(f"       Coverage: {covered}/{matched} = {coverage_rate:.1%} (target: {target:.1%}) {success_icon}")
                print(f"       q_α used: {q_alpha:.6f}")
        
        if conditions_without_data:
            print("\n   Conditions without Test Data:")
            for condition, stats in conditions_without_data:
                q_alpha = stats['q_alpha_used']
                if q_alpha is not None:
                    print(f"     {condition}: No test samples (q_α available: {q_alpha:.6f})")
                else:
                    print(f"     {condition}: No test samples (q_α unavailable: null)")
        
        params_summary = results.get('conformal_params_summary', {})
        print(f"\nCONFORMAL PARAMETERS:")
        print(f"   Calibration Mode: {params_summary.get('calibration_mode', 'unknown')}")
        print(f"   Available Conditions: {len(params_summary.get('available_conditions', []))}")
        print(f"   Strict Mode: {'Enabled' if params_summary.get('strict_mode', False) else 'Disabled'}")
        print(f"   Fallback: {'Disabled' if not params_summary.get('fallback_enabled', True) else 'Enabled'}")
        print(f"   Risk Level (α): {results['alpha']}")
        print(f"   Processed Samples: {results['processed_samples']}")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Conformal prediction inference and evaluation script')
    parser.add_argument('-c', '--config', 
                        default='configs/eavdetr/r50vd_codrone.yml',
                        help='Configuration file path')
    parser.add_argument('-r', '--checkpoint', required=True,
                        help='Model checkpoint path (required)')
    parser.add_argument('-p', '--conformal-params', required=True,
                        help='Conformal parameters file path (required)')
    parser.add_argument('--device', default='cuda:0',
                        help='Inference device (default: cuda:0)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    if not os.path.exists(args.conformal_params):
        raise FileNotFoundError(f"Conformal parameters file not found: {args.conformal_params}")
    
    print("="*80)
    print("CODrone Oriented Object Detection - Conformal Prediction Inference and Evaluation")
    print("="*80)
    print(f"Config file: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Conformal parameters: {args.conformal_params}")
    print(f"Device: {args.device}")
    print("-"*80)
    
    try:
        predictor = ConformalPredictor(
            args.config, args.checkpoint, args.conformal_params, args.device)
    except Exception as e:
        raise RuntimeError(f"Initialization failed: {e}")
    
    try:
        results = predictor.evaluate()
        
        predictor.print_results(results)
        
        results_path = './evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {results_path}")
        
    except Exception as e:
        raise RuntimeError(f"Evaluation failed: {e}")


if __name__ == '__main__':
    main()