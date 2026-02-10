import os
import sys
import time
import argparse
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


sys.path.insert(0, str(Path(__file__).parent))

from src.core import YAMLConfig
from src.misc import dist

try:
    from fvcore.nn import FlopCountMode, flop_count
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

HAS_PTFLOPS = False

try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


warnings.filterwarnings('ignore')


def create_paper_benchmark_script():

    script_content = '''#!/usr/bin/env python3

import subprocess
import sys

# Test config
config = {
    "config_file": "configs/eavdetr/r50vd_codrone.yml",
    "device": "cuda",
    "trained_weights": None, 
    "num_warmup": 100,
    "num_test": 1000,
    "output": "fair_benchmark_results.json"
}

cmd = [
    sys.executable, "benchmark_model.py",
    "--config", config["config_file"],
    "--device", config["device"],
    "--num-warmup", str(config["num_warmup"]),
    "--num-test", str(config["num_test"]),
    "--output", config["output"]
]

# Add weights if provided
if config["trained_weights"]:
    cmd.extend(["--trained-weights", config["trained_weights"]])

print("Run fair FPS test...")
print("- Warmup: images [0:{0})".format(config['num_warmup']))
print("- Test: images [{0}:{1})".format(config['num_warmup'], config['num_warmup']+config['num_test']))
print("- No overlap between warmup and test")
print("- Same image order across models")
print("Hint: set 'trained_weights' to use trained weights")
print(f"Command: {' '.join(cmd)}")
subprocess.run(cmd)
'''
    
    with open("fair_benchmark.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("Created fair_benchmark.py")


class ModelBenchmark:
    """Model benchmarking utility."""
    
    def __init__(self, config_path: str, device: str = 'cuda', trained_weights: str = None):
        """
        Initialize benchmark.
        Args:
            config_path: config path
            device: 'cuda' or 'cpu'
            trained_weights: optional trained weights path
        """
        self.device = torch.device(device)
        self.config_path = config_path
        self.trained_weights = trained_weights
        
        print("Init benchmark")
        print(f"Config: {config_path}")
        print(f"Device: {device}")
        print(f"Weights: {trained_weights if trained_weights else 'random (none)'}")
        
        # Load config
        self.cfg = YAMLConfig(config_path)
        
        # Set device
        self.cfg.device = self.device
        
        # Clear tuning path
        self.cfg.tuning = ''
        
        # Init model
        self._setup_model()
        
        # Get input size
        self.input_size = self._get_input_size()
        print(f"Input size: {self.input_size}")
        
        # Dataset for real images
        self._setup_dataset()
        
        # Postprocessor for full FPS
        self._setup_postprocessor()
        
    def _setup_model(self):
        """Setup model"""
        try:
            # Build model
            self.model = self.cfg.model.to(self.device)
            
            # Load trained weights if provided
            if self.trained_weights and os.path.exists(self.trained_weights):
                print(f"Load weights: {self.trained_weights}")
                checkpoint = torch.load(self.trained_weights, map_location=self.device)
                
                # Handle common checkpoint formats
                if 'model' in checkpoint:
                    model_state = checkpoint['model']
                elif 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    model_state = checkpoint['state_dict']
                else:
                    model_state = checkpoint
                
                # Strict load
                self.model.load_state_dict(model_state, strict=True)
                print("Weights loaded")
                
                # Optional training info
                if 'epoch' in checkpoint:
                    print(f"Epoch: {checkpoint['epoch']}")
                if 'best_map' in checkpoint:
                    print(f"Best mAP: {checkpoint['best_map']:.4f}")
                    
            elif self.trained_weights:
                print(f"Warning: weights not found: {self.trained_weights}")
                print("Use random weights")
            else:
                print("Use random weights (no weights provided)")
            
            # To device and eval
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Log device
            model_device = next(self.model.parameters()).device
            print(f"Model ready on: {model_device}")
            
        except Exception as e:
            print(f"Model init failed: {e}")
            raise
    
    def _get_input_size(self):
        """Get input size"""
        # From config
        if hasattr(self.cfg.yaml_cfg, 'HybridEncoder') and 'eval_spatial_size' in self.cfg.yaml_cfg['HybridEncoder']:
            size = self.cfg.yaml_cfg['HybridEncoder']['eval_spatial_size']
            return (3, size[0], size[1])
        elif hasattr(self.cfg.yaml_cfg, 'RTDETRTransformer') and 'eval_spatial_size' in self.cfg.yaml_cfg['RTDETRTransformer']:
            size = self.cfg.yaml_cfg['RTDETRTransformer']['eval_spatial_size']
            return (3, size[0], size[1])
        else:
            # Default
            return (3, 1024, 1024)
    
    def _setup_dataset(self):
        """Setup dataset for real images"""
        try:
            print("\nInit test dataset...")
            
            # Use test_dataloader config only
            if 'test_dataloader' in self.cfg.yaml_cfg and 'dataset' in self.cfg.yaml_cfg['test_dataloader']:
                test_dataset_config = self.cfg.yaml_cfg['test_dataloader']['dataset']
                print("Found test_dataloader config")
                
                # Check paths
                img_folder = test_dataset_config['img_folder']
                ann_folder = test_dataset_config['ann_folder']
                
                if not os.path.exists(img_folder):
                    print(f"Image folder not found: {img_folder}")
                    raise FileNotFoundError(f"Image folder not found: {img_folder}")
                    
                if not os.path.exists(ann_folder):
                    print(f"Annotation folder not found: {ann_folder}")
                    raise FileNotFoundError(f"Annotation folder not found: {ann_folder}")
                
                # Create dataset
                from src.data.codrone_dataset import CODroneDetection
                self.test_dataset = CODroneDetection(
                    img_folder=img_folder,
                    ann_folder=ann_folder,
                    split='test',
                    patch_size=test_dataset_config.get('patch_size', 1024),
                    debug_mode=False,
                    max_samples=None,
                    use_difficult=test_dataset_config.get('use_difficult', False)
                )
                
                print(f"Dataset loaded: {len(self.test_dataset)} images")
                
                # Ensure non-empty
                if len(self.test_dataset) == 0:
                    print("Empty dataset")
                    self.test_dataset = None
                    
            else:
                print("No test_dataloader config found")
                print("Add 'test_dataloader' in the config")
                self.test_dataset = None
                
        except Exception as e:
            print(f"Dataset init failed: {e}")
            print("Will use random data")
            self.test_dataset = None
    
    def _setup_postprocessor(self):
        """Setup postprocessor for full FPS"""
        try:
            print("\nInit postprocessor...")
            
            # From config
            if hasattr(self.cfg.yaml_cfg, 'postprocessor') and self.cfg.yaml_cfg['postprocessor']:
                postprocessor_config = self.cfg.yaml_cfg['postprocessor']
                print("Use configured postprocessor")
                
                # Create
                from src.zoo.eavdetr.postprocessor import RTDETRPostProcessor
                self.postprocessor = RTDETRPostProcessor(
                    num_classes=postprocessor_config.get('num_classes', 12),
                    use_focal_loss=postprocessor_config.get('use_focal_loss', True),
                    num_top_queries=postprocessor_config.get('num_top_queries', 300),
                    remap_mscoco_category=postprocessor_config.get('remap_mscoco_category', False)
                ).to(self.device)
                
            else:
                # Default
                print("Use default postprocessor config")
                from src.zoo.eavdetr.postprocessor import RTDETRPostProcessor
                self.postprocessor = RTDETRPostProcessor(
                    num_classes=12,
                    use_focal_loss=True,
                    num_top_queries=300,
                    remap_mscoco_category=False
                ).to(self.device)
            
            self.postprocessor.eval()
            
        except Exception as e:
            print(f"Postprocessor init failed: {e}")
            print("FPS will include model forward only")
            self.postprocessor = None
    
    def count_parameters(self):
        """Count model parameters"""
        print("\n" + "="*50)
        print("Parameter Count")
        print("="*50)
        
        total_params = 0
        trainable_params = 0
        
        # Per-module stats
        module_stats = {}
        
        for name, module in self.model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            module_stats[name] = {
                'total': module_params,
                'trainable': module_trainable
            }
            
            total_params += module_params
            trainable_params += module_trainable
            
            print(f"  {name:15s}: {module_params:>12,} ({module_trainable:>12,} trainable)")
        
        print("-" * 50)
        print(f"  {'Total':15s}: {total_params:>12,} ({trainable_params:>12,} trainable)")
        print(f"  Model size: {total_params * 4 / (1024**2):.2f} MB (FP32)")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024**2),
            'module_stats': module_stats
        }
    
    def measure_flops_fvcore(self, input_tensor):
        """Measure FLOPs with fvcore"""
        if not HAS_FVCORE:
            return None
            
        print("\nFLOPs (fvcore)...")
        try:
            with torch.no_grad():
                flops_dict, _ = flop_count(
                    self.model, 
                    (input_tensor,),
                    supported_ops=None
                )
            
            total_flops = sum(flops_dict.values())
            print(f"  FLOPs: {total_flops:,}")
            print(f"  GFLOPs: {total_flops / 1e9:.2f}")
            
            return {
                'total_flops': total_flops,
                'gflops': total_flops / 1e9,
                'flops_dict': flops_dict
            }
        except Exception as e:
            print(f"  fvcore failed: {e}")
            return None
    

    
    def measure_flops_thop(self, input_tensor):
        """Measure FLOPs with thop"""
        if not HAS_THOP:
            return None
            
        print("\nFLOPs (thop)...")
        try:
            # Ensure same device
            model_device = next(self.model.parameters()).device
            input_tensor_same_device = input_tensor.to(model_device)
            
            # To device
            self.model = self.model.to(model_device)
            
            with torch.no_grad():
                flops, params = profile(
                    self.model, 
                    inputs=(input_tensor_same_device,),
                    verbose=False
                )
            
            flops_str, params_str = clever_format([flops, params], "%.3f")
            
            print(f"  FLOPs: {flops_str}")
            print(f"  Params: {params_str}")
            print(f"  GFLOPs: {flops / 1e9:.2f}")
            
            return {
                'flops': flops,
                'gflops': flops / 1e9,
                'params': params,
                'flops_str': flops_str,
                'params_str': params_str
            }
            
        except Exception as e:
            print(f"  thop failed: {e}")
            print(f"  Note: thop may be incompatible with some models")
            return None
    
    def measure_fps(self, num_warmup=100, num_test=1000, batch_size=1):
        """
        FPS on real images (includes post-processing if available).
        """
        # Force batch_size=1
        if batch_size != 1:
            print(f"Force batch_size=1 (was {batch_size})")
            batch_size = 1
        
        # Limit test images to 1000
        max_test_images = min(num_test, 1000)
        if num_test > 1000:
            print(f"Limit num_test to 1000 (was {num_test})")
            num_test = max_test_images
        
        # Total images needed
        total_images_needed = num_warmup + num_test
        
        print(f"\nFPS test (warmup: {num_warmup}, test: {num_test})")
        print("="*50)
        pipeline_info = "inference+postproc" if self.postprocessor is not None else "inference-only"
        print(f"Measure: {pipeline_info} (no data loading)")
        
        # Prepare data
        all_test_images = []
        if self.test_dataset and len(self.test_dataset) > 0:
            print(f"Dataset: {len(self.test_dataset)} images, need {total_images_needed}")
            
            # Check size
            if len(self.test_dataset) < total_images_needed:
                print(f"Not enough images: need {total_images_needed}, have {len(self.test_dataset)}")
                print(f"Use available {len(self.test_dataset)} images")
                total_images_needed = len(self.test_dataset)
                # Adjust split
                if total_images_needed >= 200:
                    num_warmup = min(num_warmup, total_images_needed // 2)
                    num_test = total_images_needed - num_warmup
                    print(f"Adjust: warmup={num_warmup}, test={num_test}")
                else:
                    print(f"Too few images for reliable test")
                    return None
            
            # Preload tensors (not timed)
            print(f"\nPreload tensors (0-{total_images_needed-1})...")
            
            import time
            preload_start = time.time()
            
            for i in range(total_images_needed):
                try:
                    # __getitem__ includes preprocessing
                    image_tensor, target = self.test_dataset[i]
                    # Add batch dim and move to device
                    image_batch = image_tensor.unsqueeze(0).to(self.device)  # [1, C, H, W]
                    all_test_images.append(image_batch)
                    
                    if (i + 1) % 200 == 0:
                        print(f"    Preloaded: {i+1}/{total_images_needed}")
                        
                except Exception as e:
                    print(f"    Skip image {i}: {e}")
                    continue
            
            preload_time = time.time() - preload_start
            print(f"Preloaded {len(all_test_images)} images in {preload_time:.2f}s")
            
        else:
            print(f"Use random data x {total_images_needed}")
            
            # 创建随机数据
            input_shape = (batch_size, *self.input_size)
            for i in range(total_images_needed):
                dummy_input = torch.randn(input_shape, device=self.device)
                all_test_images.append(dummy_input)
        
        # CUDA cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Warmup (model only)
        if all_test_images and len(all_test_images) >= total_images_needed:
            print(f"\nWarmup ({num_warmup} images)...")
            warmup_images = all_test_images[:num_warmup]  # 取前num_warmup张图片
            
            # Ensure device
            self.model = self.model.to(self.device)
            
            with torch.no_grad():
                for i, warmup_img in enumerate(warmup_images):
                    # Ensure same device
                    warmup_img = warmup_img.to(self.device)
                    
                    # Forward only
                    _ = self.model(warmup_img)
                    
                    if self.device.type == 'cuda' and (i + 1) % 50 == 0:
                        torch.cuda.synchronize()
                        print(f"    Warmup: {i+1}/{num_warmup}")
        
        # Sync
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Timing (includes postproc)
        test_images = all_test_images[num_warmup:num_warmup + num_test]  # 取后num_test张图片
        print(f"\nMeasure FPS... (images {num_warmup}-{num_warmup + len(test_images)-1}, total {len(test_images)})")
        
        # Prepare model_input_sizes (for postprocessor)
        model_input_sizes = torch.tensor([[self.input_size[1], self.input_size[2]]], 
                                       device=self.device, dtype=torch.float32)  # [1, 2] [h, w]
        
        if self.device.type == 'cuda':
            # CUDA events timing
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            times = []
            
            with torch.no_grad():
                for i, image_batch in enumerate(test_images):
                    # image_batch already on device
                    # image_batch = image_batch.to(self.device)  # 已预加载到GPU
                    
                    # Start
                    starter.record()
                    
                    # Forward
                    outputs = self.model(image_batch)  # 推理单张图片 (batch_size=1)
                    
                    # Postproc (if any)
                    if self.postprocessor is not None:
                        _ = self.postprocessor(outputs, model_input_sizes)
                    
                    # End
                    ender.record()
                    torch.cuda.synchronize()
                    
                    # Seconds
                    gpu_time = starter.elapsed_time(ender) / 1000.0
                    times.append(gpu_time)
                    
                    # Progress
                    if (i + 1) % 200 == 0:
                        current_fps = batch_size / np.mean(times[-100:])
                        print(f"    Progress: {i+1}/{len(test_images)}, FPS: {current_fps:.2f}")
            
            timing_method = f"CUDA Events (inference{' + postproc' if self.postprocessor else ''} - real images)"
            
        else:
            # CPU timing
            times = []
            with torch.no_grad():
                for i, image_batch in enumerate(test_images):
                    # image_batch already on device
                    # image_batch = image_batch.to(self.device)  # 已预加载到设备
                    
                    start_time = time.perf_counter()
                    
                    # Forward
                    outputs = self.model(image_batch)
                    
                    # Postproc (if any)
                    if self.postprocessor is not None:
                        _ = self.postprocessor(outputs, model_input_sizes)
                    
                    end_time = time.perf_counter()
                    
                    times.append(end_time - start_time)
                    
                    if (i + 1) % 200 == 0:
                        current_fps = batch_size / np.mean(times[-100:])
                        print(f"    Progress: {i+1}/{len(test_images)}, FPS: {current_fps:.2f}")
            
            timing_method = f"CPU time.perf_counter() (inference{' + postproc' if self.postprocessor else ''} - real images)"
        
        # Stats
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # FPS
        fps_mean = batch_size / mean_time
        fps_min = batch_size / max_time  
        fps_max = batch_size / min_time
        
        # Report
        print("\n" + "="*50)
        print("FPS Results")
        print("="*50)
        
        print(f"\nLatency:")
        print(f"  Mean: {mean_time*1000:.3f} ms")
        print(f"  Std: {std_time*1000:.3f} ms")
        print(f"  Min: {min_time*1000:.3f} ms")
        print(f"  Max: {max_time*1000:.3f} ms")
        
        print(f"\nFPS: {fps_mean:.3f} (mean) | {fps_max:.3f} (max) | {fps_min:.3f} (min)")
        
        measurement_type = "inference+postproc" if self.postprocessor is not None else "inference-only"
        print(f"Measure: {measurement_type} ({len(times)} images)")
        print(f"Timing: {timing_method}")
        
        return {
            'mean_time': mean_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'fps_mean': fps_mean,
            'fps_min': fps_min,
            'fps_max': fps_max,
            'batch_size': batch_size,
            'timing_method': timing_method,
            'total_images_processed': len(test_images),
            'warmup_images_used': num_warmup,
            'real_images_used': bool(self.test_dataset),
            'test_image_indices': f"[{num_warmup}:{num_warmup + len(test_images)-1}]",
            'times': times.tolist()
        }
    
    def run_benchmark(self, num_warmup=100, num_test=1000, batch_size=1):
        """Run full benchmark"""
        print("Start benchmark")
        print("="*70)
        
        results = {}
        
        # 1) Params
        results['params'] = self.count_parameters()
        
        # 2) FLOPs
        print("\n" + "="*50)
        print("FLOPs")
        print("="*50)
        
        # Dummy input
        dummy_input = torch.randn(1, *self.input_size, device=self.device)
        
        # Multiple methods
        results['flops'] = {}
        
        # fvcore
        fvcore_result = self.measure_flops_fvcore(dummy_input)
        if fvcore_result:
            results['flops']['fvcore'] = fvcore_result
        
        # thop
        thop_result = self.measure_flops_thop(dummy_input)
        if thop_result:
            results['flops']['thop'] = thop_result
        
        # 3) FPS
        results['fps'] = self.measure_fps(num_warmup, num_test, batch_size)
        
        # 4) Summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results):
        """Print summary"""
        print("\n" + "="*70)
        print("Benchmark Summary")
        print("="*70)
        
        # Model info
        print(f"Config: {self.config_path}")
        print(f"Input size: {self.input_size}")
        print(f"Device: {self.device}")
        
        # Params
        if 'params' in results:
            params = results['params']
            print(f"\nParams:")
            print(f"  Total: {params['total_params']:,}")
            print(f"  Trainable: {params['trainable_params']:,}")
            print(f"  Size: {params['model_size_mb']:.2f} MB")
        
        # FLOPs
        print(f"\nFLOPs:")
        flops_results = results.get('flops', {})
        
        if 'fvcore' in flops_results:
            gflops = flops_results['fvcore']['gflops']
            print(f"  fvcore: {gflops:.2f} G")
        
        if 'thop' in flops_results:
            gflops = flops_results['thop']['gflops']
            print(f"  thop: {gflops:.2f} G")
            print(f"  FLOPs: {flops_results['thop']['flops_str']}")
            print(f"  Params: {flops_results['thop']['params_str']}")
        
        # FPS
        if 'fps' in results:
            fps = results['fps']
            pipeline_type = "inference+postproc" if self.postprocessor is not None else "inference-only"
            print(f"\nFPS:")
            print(f"  Mean FPS: {fps['fps_mean']:.3f}")
            print(f"  Mean Latency: {fps['mean_time']*1000:.3f} ms ({pipeline_type})")
            print(f"  Images: {fps.get('total_images_processed', 0)} real images")
            print(f"  Timing: {fps['timing_method']}")
        
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='RT-DETR Benchmark - real images')
    parser.add_argument('--config', type=str, 
                       default='configs/eavdetr/r50vd_codrone.yml',
                       help='config file path')
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'], help='device to test on')
    parser.add_argument('--trained-weights', type=str, default=None,
                       help='trained weights path (optional)')
    parser.add_argument('--batch-size', type=int, default=1, 
                       help='batch size (forced to 1)')
    parser.add_argument('--num-warmup', type=int, default=100, 
                       help='warmup images')
    parser.add_argument('--num-test', type=int, default=1000, 
                       help='test images (max 1000)')
    parser.add_argument('--output', type=str, default=None,
                       help='output json path')
    
    args = parser.parse_args()
    
    # Check config
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        return
    
    # Check weights
    if args.trained_weights and not os.path.exists(args.trained_weights):
        print(f"Weights not found: {args.trained_weights}")
        print("Use random weights")
        args.trained_weights = None
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU")
        args.device = 'cpu'
    
    # Force batch size
    if args.batch_size != 1:
        print(f"Force batch_size=1 (was {args.batch_size})")
        args.batch_size = 1
    
    try:
        # Create benchmark
        benchmark = ModelBenchmark(
            config_path=args.config,
            device=args.device,
            trained_weights=args.trained_weights
        )
        
        # Run
        results = benchmark.run_benchmark(
            num_warmup=args.num_warmup,
            num_test=args.num_test,
            batch_size=args.batch_size
        )
        
        # Save
        if args.output:
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved: {args.output}")
        
        print("\nDone")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
