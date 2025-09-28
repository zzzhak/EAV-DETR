import numpy as np
import math
from typing import Tuple, List, Union, Optional
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union


def convert_to_corners(obb: np.ndarray) -> np.ndarray:
    cx, cy, w, h, angle = obb

    half_w = w / 2.0
    half_h = h / 2.0
    
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    local_corners = np.array([
        [-half_w, -half_h],
        [+half_w, -half_h],
        [+half_w, +half_h],
        [-half_w, +half_h]
    ], dtype=np.float32)
    
    rotated_corners = np.zeros_like(local_corners, dtype=np.float32)
    for i, (lx, ly) in enumerate(local_corners):
        rx = cos_angle * lx - sin_angle * ly
        ry = sin_angle * lx + cos_angle * ly
        rotated_corners[i] = [cx + rx, cy + ry]
    
    return rotated_corners.astype(np.float32)


def get_obb_lines(obb_corners: np.ndarray) -> List[Tuple[float, float, float]]:
    center = np.mean(obb_corners, axis=0)
    lines = []
    num_corners = len(obb_corners)
    for i in range(num_corners):
        p1 = obb_corners[i]
        p2 = obb_corners[(i + 1) % num_corners]
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1  
        C = -A * x1 - B * y1
        
        midpoint = (p1 + p2) / 2.0
        norm_length = np.sqrt(A * A + B * B)
        if norm_length > 1e-10:
            test_offset = 0.1
            test_point = midpoint + test_offset * np.array([A, B]) / norm_length
            dist_test = np.linalg.norm(test_point - center)
            dist_mid = np.linalg.norm(midpoint - center)
            if dist_test < dist_mid:
                A, B, C = -A, -B, -C
        
        lines.append((float(A), float(B), float(C)))
    
    return lines


def signed_distance_point_to_line(point: Union[np.ndarray, Tuple[float, float]], 
                                 line_params: Tuple[float, float, float]) -> float:
    if isinstance(point, np.ndarray):
        px, py = point[0], point[1]
    else:
        px, py = point
    
    A, B, C = line_params
    
    denominator = np.sqrt(A * A + B * B)
    if denominator < 1e-10:
        return 0.0
    distance = (A * px + B * py + C) / denominator
    
    return float(distance)


def is_point_inside_obb(point: Union[np.ndarray, Tuple[float, float]], 
                       obb_corners: np.ndarray) -> bool:
    try:
        if isinstance(point, np.ndarray):
            px, py = float(point[0]), float(point[1])
        else:
            px, py = float(point[0]), float(point[1])
        
        lines = get_obb_lines(obb_corners)
        for A, B, C in lines:
            distance = signed_distance_point_to_line((px, py), (A, B, C))
            if distance > 1e-3:
                return False
        
        return True
        
    except Exception as e:
        raise RuntimeError(f"Point containment test failed: {e}")


def is_obb_inside_obb(inner_obb_corners: np.ndarray, 
                     outer_obb_corners: np.ndarray) -> bool:
    for corner in inner_obb_corners:
        if not is_point_inside_obb(corner, outer_obb_corners):
            return False
    
    return True


def calculate_expansion_margin(predicted_obb_corners: np.ndarray, 
                             ground_truth_obb_corners: np.ndarray) -> float:
    if is_obb_inside_obb(ground_truth_obb_corners, predicted_obb_corners):
        return 0.0
    pred_lines = get_obb_lines(predicted_obb_corners)
    max_required_margin = 0.0
    for gt_corner in ground_truth_obb_corners:
        distances = []
        for line_params in pred_lines:
            dist = signed_distance_point_to_line(gt_corner, line_params)
            distances.append(dist)
        required_margin_for_this_corner = np.max(distances)
        max_required_margin = max(max_required_margin, required_margin_for_this_corner)
    
    return max(0.0, float(max_required_margin))


def expand_obb(obb_corners: np.ndarray, margin: float) -> np.ndarray:

    if margin <= 0:
        return obb_corners.copy()
    
    try:
        lines = get_obb_lines(obb_corners)
        expanded_lines = []
        for A, B, C in lines:
            norm_length = np.sqrt(A * A + B * B)
            if norm_length < 1e-10:
                raise RuntimeError(f"Normal vector too small: {norm_length}")
            C_new = C - margin * norm_length
            expanded_lines.append((A, B, C_new))
        expanded_corners = np.zeros_like(obb_corners)
        num_lines = len(expanded_lines)
        
        for i in range(num_lines):
            line1 = expanded_lines[(i - 1) % num_lines]
            line2 = expanded_lines[i]
            A1, B1, C1 = line1
            A2, B2, C2 = line2
            det = A1 * B2 - A2 * B1
            if abs(det) < 1e-10:
                prev_idx = (i - 1) % num_lines
                raise RuntimeError(f"Lines {prev_idx} and {i} are parallel, det={det}")
            x = (B1 * C2 - B2 * C1) / det
            y = (A2 * C1 - A1 * C2) / det
            expanded_corners[i] = [x, y]
        
        return expanded_corners.astype(np.float32)
        
    except Exception as e:
        raise RuntimeError(f"OBB expansion failed: {e}")


def visualize_obb_expansion(original_corners: np.ndarray, 
                          expanded_corners: np.ndarray, 
                          gt_corners: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Draw original box
        orig_polygon = patches.Polygon(original_corners, linewidth=2, 
                                     edgecolor='blue', facecolor='none', 
                                     label='Original Prediction')
        ax.add_patch(orig_polygon)
        
        # Draw expanded box
        exp_polygon = patches.Polygon(expanded_corners, linewidth=2, 
                                    edgecolor='green', facecolor='none',
                                    linestyle='--', label='Expanded Prediction')
        ax.add_patch(exp_polygon)
        
        # Draw ground truth box if provided
        if gt_corners is not None:
            gt_polygon = patches.Polygon(gt_corners, linewidth=2, 
                                       edgecolor='red', facecolor='none',
                                       label='Ground Truth')
            ax.add_patch(gt_polygon)
        
        # Set axes
        all_corners = [original_corners, expanded_corners]
        if gt_corners is not None:
            all_corners.append(gt_corners)
        
        all_points = np.vstack(all_corners)
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        
        margin_x = (x_max - x_min) * 0.1
        margin_y = (y_max - y_min) * 0.1
        
        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_min - margin_y, y_max + margin_y)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('OBB Expansion Visualization')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
        
    except (ImportError, Exception):
        pass


def test_geometry_utils():
    obb = np.array([100.0, 50.0, 80.0, 40.0, np.pi/4])
    corners = convert_to_corners(obb)
    assert corners.shape == (4, 2), "Invalid corners shape"
    
    lines = get_obb_lines(corners)
    assert len(lines) == 4, "Should have 4 lines"
    
    test_point = np.array([100.0, 50.0])
    dist = signed_distance_point_to_line(test_point, lines[0])
    
    gt_obb = np.array([105.0, 55.0, 70.0, 30.0, np.pi/6])
    gt_corners = convert_to_corners(gt_obb)
    margin = calculate_expansion_margin(corners, gt_corners)
    assert margin >= 0, "Margin should be non-negative"
    
    expanded_corners = expand_obb(corners, margin)
    assert expanded_corners.shape == (4, 2), "Invalid expanded corners shape"
    
    final_margin = calculate_expansion_margin(expanded_corners, gt_corners)
    print("All tests passed")
    
    return {
        'original_corners': corners,
        'gt_corners': gt_corners,
        'expanded_corners': expanded_corners,
        'margin': margin,
        'final_margin': final_margin
    }


if __name__ == '__main__':
    test_geometry_utils()