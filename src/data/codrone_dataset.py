
import os
import torch
import torch.utils.data
import numpy as np
from PIL import Image
import json
from typing import List, Dict, Any
import glob
import cv2
import random

import torchvision

from src.core import register

__all__ = ['CODroneDetection']

@register
class CODroneDetection(torch.utils.data.Dataset):
    """CODrone1024 oriented object detection dataset (DOTA-style annotations)."""
    
    __inject__ = []
    __share__ = []
    
    # 12 classes defined by CODrone
    CLASSES = (
        'car', 'truck', 'traffic-sign', 'people', 'motor', 'bicycle',
        'traffic-light', 'tricycle', 'bridge', 'bus', 'boat', 'ship'
    )
    
    def __init__(self, img_folder, ann_folder, split='train', 
                 patch_size=1024, debug_mode=False, max_samples=None, use_difficult=False):
        """
        Args:
            img_folder: path to images
            ann_folder: path to DOTA txt annotations
            split: 'train' | 'val' | 'test'
            patch_size: expected image size (1024 for CODrone1024)
            debug_mode: enable verbose warnings
            max_samples: optional max number of samples to load
        """
        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.split = split
        self.patch_size = patch_size
        self.debug_mode = debug_mode
        self.max_samples = max_samples
        self.use_difficult = use_difficult
        
        # class name -> index
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.CLASSES)}
        
        # collect samples
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Collect image-annotation pairs."""
        samples = []
        
        # get image files (case-insensitive)
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        img_files = []
        for ext in img_extensions:
            img_files.extend(glob.glob(os.path.join(self.img_folder, f'*{ext}')))
            img_files.extend(glob.glob(os.path.join(self.img_folder, f'*{ext.upper()}')))
        
        # deduplicate and sort for deterministic order
        img_files = sorted(list(set(img_files)))
        
        for img_path in img_files:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            ann_path = os.path.join(self.ann_folder, f'{img_name}.txt')
            
            if os.path.exists(ann_path):
                samples.append({
                    'img_path': img_path,
                    'ann_path': ann_path,
                    'img_name': img_name
                })
                if self.max_samples and len(samples) >= self.max_samples:
                    break
            else:
                if not getattr(self, 'debug_mode', False):
                    print(f"[WARN] Missing annotation: {img_path}")
        
        print(f"Loaded {len(samples)} samples for split={self.split}")
        return samples
        
    def _parse_dota_annotation(self, ann_path):
        """Parse DOTA txt: x1 y1 x2 y2 x3 y3 x4 y4 class difficulty."""
        objects = []
        
        with open(ann_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line or line.startswith('imagesource') or line.startswith('gsd'):
                continue
                
            parts = line.split()
            if len(parts) < 9:  # 8 coords + class (+ optional difficulty)
                continue
                
            try:
                coords = [float(x) for x in parts[:8]]
                class_name = parts[8]
                difficulty = int(parts[9]) if len(parts) > 9 else 0
                
                
                if not self.use_difficult and difficulty != 0:
                    continue
                
                # unknown class
                if class_name not in self.class_to_idx:
                    continue
                
                # ignore special labels
                if class_name in ('ignore', 'ignored'):
                    continue
                    
                polygon = np.array(coords).reshape(4, 2)
                
                objects.append({
                    'polygon': polygon,
                    'class_name': class_name,
                    'label': self.class_to_idx[class_name],
                    'difficulty': difficulty
                })
                
            except (ValueError, IndexError) as e:
                print(f"[WARN] Bad line: {line} ({e})")
                continue
                
            
        return objects
    
    def _polygon_to_obb(self, polygon):
        """Convert 4-point polygon to [cx, cy, w, h, angle] (le90)."""
        try:
            points = polygon.astype(np.float32)
            rect = cv2.minAreaRect(points)  # ((cx, cy), (w, h), angle_deg)
            (cx, cy), (w, h), angle_deg = rect
            
            if w < h:
                w, h = h, w
                angle_deg += 90
            
            angle = np.deg2rad(angle_deg)

            while angle >= np.pi/2:
                angle -= np.pi
            while angle < -np.pi/2:
                angle += np.pi
            
            return np.array([cx, cy, abs(w), abs(h), angle], dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"minAreaRect failed for {polygon}: {e}")
    
    def _polygon_area(self, polygon):
        """Polygon area (shoelace)."""
        if len(polygon) < 3:
            return 0.0
        x = polygon[:, 0]
        y = polygon[:, 1]
        return 0.5 * abs(float(np.sum(x * np.roll(y, 1) - y * np.roll(x, 1))))
    
    def _apply_horizontal_flip(self, image, boxes):
        """Horizontal flip image and OBBs (normalized [0,1])."""
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        if len(boxes) > 0:
            flipped_boxes = boxes.copy()
            flipped_boxes[:, 0] = 1.0 - boxes[:, 0]  # cx
            flipped_boxes[:, 4] = -boxes[:, 4]       # angle
            angles = flipped_boxes[:, 4]
            angles = np.where(angles >= np.pi/2, angles - np.pi, angles)
            angles = np.where(angles < -np.pi/2, angles + np.pi, angles)
            flipped_boxes[:, 4] = angles
            return flipped_image, flipped_boxes
        else:
            return flipped_image, boxes
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = Image.open(sample['img_path']).convert('RGB')
        
        if image.size != (self.patch_size, self.patch_size):
            image = image.resize((self.patch_size, self.patch_size))
        
        objects = self._parse_dota_annotation(sample['ann_path'])
        
        if objects:
            boxes = np.array([self._polygon_to_obb(obj['polygon']) for obj in objects])
            labels = np.array([obj['label'] for obj in objects])
            difficulties = np.array([obj['difficulty'] for obj in objects])
            
            if len(boxes) > 0:
                img_w, img_h = image.size
                boxes[:, 0] = np.clip(boxes[:, 0] / img_w, 0.0, 1.0)
                boxes[:, 1] = np.clip(boxes[:, 1] / img_h, 0.0, 1.0)
                boxes[:, 2] = np.clip(boxes[:, 2] / img_w, 1e-6, 1.0)
                boxes[:, 3] = np.clip(boxes[:, 3] / img_h, 1e-6, 1.0)
        else:
            boxes = np.zeros((0, 5), dtype=np.float32)
            labels = np.array([], dtype=np.int64)
            difficulties = np.array([], dtype=np.int64)
        
        # simple augmentation: 50% horizontal flip on train split
        if self.split == 'train' and random.random() < 0.5:
            image, boxes = self._apply_horizontal_flip(image, boxes)
        
        # to tensor [C,H,W], normalized to [0,1]
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': torch.as_tensor([self._polygon_area(obj['polygon']) for obj in objects] if objects else [], dtype=torch.float32),
            'iscrowd': torch.zeros(len(labels), dtype=torch.int64),
            'difficulties': torch.as_tensor(difficulties, dtype=torch.int64),
            'orig_size': torch.as_tensor([image.size[1], image.size[0]]),
            'size': torch.as_tensor([image.size[1], image.size[0]])
        }
        
        return image_tensor, target
    
    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_folder: {self.ann_folder}\n'
        s += f' split: {self.split}\n image_size: {self.patch_size}x{self.patch_size}\n'
        s += f' use_difficult: {self.use_difficult}\n'
        s += f' hflip_aug: {"train_only_50%" if self.split == "train" else "off"}\n'
        s += f' direct_tensor: True\n minAreaRect: True\n'
        return s