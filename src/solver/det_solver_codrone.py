
import os
import time
from pathlib import Path
from collections import OrderedDict

import torch
import torch.amp
from torch.nn.parallel import DistributedDataParallel as DDP

from .solver import BaseSolver
# Import oriented detection evaluation engine
from .det_engine_codrone import train_one_epoch, evaluate_oriented
from src.misc import dist
from src.misc.dist import reduce_dict


class DetSolverCODrone(BaseSolver):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # CODrone settings
        self.use_oriented_eval = True
        self.patch_level_eval = True
        self.print_freq = getattr(cfg, 'print_freq', 10)
        
        # Best metrics tracking
        self.best_map = 0.0
        self.best_model_path = None
        self.best_epoch = -1
        self._best_info_restored_from_checkpoint = False
        
        # Num classes
        num_classes = getattr(cfg.yaml_cfg, 'num_classes', 12)
        
        print(f"[DetSolverCODrone] {num_classes} classes, single best model file")
    
    def fit(self):

        print("[DetSolverCODrone] Training with oriented IoU eval...")
        
        # Full initialization
        self.train()
        
        print(f"Start from epoch {self.last_epoch + 1} to {self.cfg.epoches}")
        print(f"Current best mAP: {getattr(self, 'best_map', 0):.4f}")
        
        # Try restore best model file info
        self._smart_restore_best_model_info()
        
        # Planned snapshot epochs
        expected_snapshots = []
        for e in range(self.last_epoch + 1, self.cfg.epoches):
            if (e + 1) % self.cfg.snapshot_epoch == 0:
                expected_snapshots.append(e)
        print(f"Snapshots at epochs: {expected_snapshots}")
        
        for epoch in range(self.last_epoch + 1, self.cfg.epoches):
            # Update last_epoch
            self.last_epoch = epoch
            
            epoch_start_time = time.time()
            
            # Epoch header
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.cfg.epoches-1}")
            print(f"Best mAP: {getattr(self, 'best_map', 0):.4f}")
            if hasattr(self, 'best_epoch') and self.best_epoch >= 0:
                print(f"Best epoch: {self.best_epoch}")
            print(f"{'='*60}")
            
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # Train
            train_start_time = time.time()
            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch,
                self.cfg.clip_max_norm, 
                print_freq=self.print_freq,
                ema=self.ema,
                scaler=self.scaler
            )
            train_time = time.time() - train_start_time
            
            # LR step
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # Evaluate (oriented)
            if self.val_dataloader is not None:
                print(f"\n=== Oriented IoU eval @ epoch {epoch} ===")
                eval_start_time = time.time()
                test_stats, oriented_evaluator = self._evaluate_oriented()
                eval_time = time.time() - eval_start_time
                
                # Update best model if improved
                if 'oriented_mAP' in test_stats:
                    current_map = test_stats['oriented_mAP']
                    previous_best = getattr(self, 'best_map', 0)
                    
                    print(f"mAP: {current_map:.4f} (best {previous_best:.4f})")
                    
                    if current_map > previous_best:
                        # Update best
                        self.best_map = current_map
                        old_best_epoch = getattr(self, 'best_epoch', -1)
                        self.best_epoch = epoch
                        
                        # Save new best
                        self._save_best_model_safely(epoch)
                        
                        print(f"[Best] mAP {current_map:.4f} @ epoch {epoch}")
                        if old_best_epoch >= 0:
                            print(f"[Best] Prev best epoch {old_best_epoch}")
                    else:
                        print(f"No improvement (> {previous_best:.4f} needed)")
                
                # Log stats with timing
                epoch_total_time = time.time() - epoch_start_time
                timing_stats = {
                    'train_time': train_time,
                    'eval_time': eval_time,
                    'epoch_total_time': epoch_total_time
                }
                self._log_stats(epoch, train_stats, test_stats, timing_stats)
                
                # Save eval details
                if dist.is_main_process() and oriented_evaluator is not None:
                    self._save_evaluation_results(epoch, oriented_evaluator)
            
            # Always save latest
            self._save_checkpoint(epoch, filename='latest.pth')
            print(f"Saved latest.pth @ epoch {epoch}")
            
            # Snapshot save
            if self.cfg.snapshot_epoch > 0:
                if (epoch + 1) % self.cfg.snapshot_epoch == 0:
                    self._save_checkpoint(epoch, filename=f'checkpoint_epoch_{epoch}.pth')
                    print(f"Saved snapshot checkpoint @ epoch {epoch}")
            
            # Final epoch save
            if epoch == self.cfg.epoches - 1:
                self._save_checkpoint(epoch, filename=f'final_epoch_{epoch}.pth')
                print(f"Saved final checkpoint @ epoch {epoch}")
                
        print(f"[DetSolverCODrone] Done. Best mAP: {getattr(self, 'best_map', 0):.4f}")
        if hasattr(self, 'best_epoch') and self.best_epoch >= 0:
            print(f"[DetSolverCODrone] Best epoch {self.best_epoch}: {self.best_model_path}")
    
    def _smart_restore_best_model_info(self):
        if not dist.is_main_process():
            return
        checkpoint_has_best_info = (
            hasattr(self, 'best_epoch') and self.best_epoch >= 0 and
            hasattr(self, 'best_map') and self.best_map >= 0
        )
        if checkpoint_has_best_info:
            print(f"Restored best model info from checkpoint:")
            print(f"   Best epoch: {self.best_epoch}")
            print(f"   Best mAP: {self.best_map:.4f}")
            expected_best_path = self.output_dir / f'best_oriented_model_epoch_{self.best_epoch}.pth'
            if expected_best_path.exists():
                self.best_model_path = expected_best_path
                print(f"   Found best model file: {expected_best_path}")
            else:
                print(f"   Missing best model file: {expected_best_path}")
                self._fallback_restore_from_filesystem()
        else:
            print("No best info in checkpoint, scanning filesystem...")
            self._fallback_restore_from_filesystem()
        self._cleanup_extra_best_models()
    
    def _fallback_restore_from_filesystem(self):
        pattern = "best_oriented_model_epoch_*.pth"
        existing_best_files = list(self.output_dir.glob(pattern))
        if existing_best_files:
            latest_best_file = max(existing_best_files, key=lambda x: x.stat().st_mtime)
            try:
                filename = latest_best_file.stem
                epoch_str = filename.split('_')[-1]
                file_best_epoch = int(epoch_str)
                if not (hasattr(self, 'best_epoch') and self.best_epoch >= 0):
                    self.best_epoch = file_best_epoch
                    print(f"Restored best epoch from filename: {file_best_epoch}")
                self.best_model_path = latest_best_file
                print(f"Best model file: {latest_best_file}")
            except (ValueError, IndexError) as e:
                print(f"Warn: cannot parse epoch from {latest_best_file}: {e}")
                self.best_epoch = -1
                self.best_model_path = None
        else:
            print("No existing best model files found")
    
    def _cleanup_extra_best_models(self):
        if not self.best_model_path:
            return
        pattern = "best_oriented_model_epoch_*.pth"
        existing_files = list(self.output_dir.glob(pattern))
        for old_file in existing_files:
            if old_file != self.best_model_path:
                print(f"Cleaning extra best model: {old_file}")
                try:
                    old_file.unlink()
                except Exception as e:
                    print(f"Warn: cannot remove {old_file}: {e}")
    
    def _save_best_model_safely(self, epoch):
        if not dist.is_main_process():
            return
        new_best_filename = f'best_oriented_model_epoch_{epoch}.pth'
        new_best_path = self.output_dir / new_best_filename
        try:
            self._save_checkpoint(epoch, is_best=True, filename=new_best_filename)
            if not new_best_path.exists():
                raise RuntimeError(f"Failed to save new best model: {new_best_path}")
            old_best_path = getattr(self, 'best_model_path', None)
            if old_best_path and old_best_path.exists() and old_best_path != new_best_path:
                print(f"Removing old best model: {old_best_path}")
                try:
                    old_best_path.unlink()
                except Exception as e:
                    print(f"Warn: cannot remove old best model file: {e}")
            pattern = "best_oriented_model_epoch_*.pth"
            existing_files = list(self.output_dir.glob(pattern))
            for old_file in existing_files:
                if old_file != new_best_path:
                    print(f"Cleaning old best model: {old_file}")
                    try:
                        old_file.unlink()
                    except Exception as e:
                        print(f"Warn: cannot remove {old_file}: {e}")
            self.best_model_path = new_best_path
            print(f"Best model updated: {new_best_filename}")
        except Exception as e:
            print(f"Error saving best model: {e}")
            raise
    
    def _evaluate_oriented(self):
        """
        Evaluate with oriented IoU.
        """
        # Use EMA model if available, fallback to regular model
        module = self.ema.module if hasattr(self, 'ema') and self.ema else self.model
        
        # Base dataset for eval
        base_ds = self.val_dataloader.dataset
        
        test_stats, oriented_evaluator = evaluate_oriented(
            module, 
            self.criterion, 
            self.postprocessor,
            self.val_dataloader, 
            base_ds, 
            self.device, 
            self.output_dir
        )
        
        return test_stats, oriented_evaluator
    
    def _log_stats(self, epoch, train_stats, test_stats, timing_stats=None):
        """
        Log train/eval stats.
        """
        if not dist.is_main_process():
            return
            
        # Prepare log
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        if timing_stats:
            log_stats.update(timing_stats)
        
        # Print key metrics
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_stats.get('loss', 0):.4f}")
        if 'oriented_mAP' in test_stats:
            print(f"  Oriented mAP: {test_stats['oriented_mAP']:.4f}")
        if hasattr(self, 'best_map'):
            print(f"  Best mAP: {self.best_map:.4f}")
        
        # Timing
        if timing_stats:
            print(f"  Training Time: {timing_stats.get('train_time', 0):.2f}s")
            print(f"  Evaluation Time: {timing_stats.get('eval_time', 0):.2f}s")
            print(f"  Total Epoch Time: {timing_stats.get('epoch_total_time', 0):.2f}s")
        
        # Save to log file
        log_file = self.output_dir / 'log.txt'
        with open(log_file, 'a') as f:
            f.write(f"{log_stats}\n")
    
    def _save_evaluation_results(self, epoch, oriented_evaluator):
        """
        Save detailed eval results.
        """
        if not hasattr(oriented_evaluator, 'eval_results'):
            return
            
        eval_dir = self.output_dir / 'eval'
        eval_dir.mkdir(exist_ok=True)
        
        # Save metrics
        eval_file = eval_dir / f'oriented_eval_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'mean_ap': getattr(oriented_evaluator, 'mean_ap', 0),
            'eval_results': getattr(oriented_evaluator, 'eval_results', []),
            'stats': getattr(oriented_evaluator, 'stats', None)
        }, eval_file)
        
        print(f"Saved oriented eval to {eval_file}")
    
    def _save_checkpoint(self, epoch, is_best=False, filename=None):
        """
        Save model checkpoint (with best tracking)
        """
        if not dist.is_main_process():
            return
            
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pth'
        
        checkpoint_path = self.output_dir / filename
        
        # Build checkpoint
        checkpoint = self.state_dict(epoch)  # 传入当前epoch
        
        # Extra fields
        checkpoint.update({
            'epoch': epoch,
            'last_epoch': epoch,
            'best_map': getattr(self, 'best_map', 0),
            'best_epoch': getattr(self, 'best_epoch', -1),
            'config': self.cfg,
            'is_best': is_best,
            'save_time': time.time(),
            'training_info': {
                'completed_epochs': epoch + 1,
                'remaining_epochs': self.cfg.epoches - epoch - 1
            }
        })
        
        # Log save
        print(f"💾 Saving checkpoint:")
        print(f"  📁 File: {filename}")
        print(f"  📊 Epoch: {epoch}")
        print(f"  🎯 Best mAP: {checkpoint['best_map']:.4f}")
        print(f"  ⭐ Is best: {is_best}")
        
        temp_path = checkpoint_path.with_suffix('.tmp')
        try:
            # Save to temp
            torch.save(checkpoint, temp_path)
    
            # Windows-friendly atomic rename
            if checkpoint_path.exists():
                checkpoint_path.unlink()
    
            temp_path.rename(checkpoint_path)
            print(f"✅ Checkpoint saved: {checkpoint_path}")
    
        except Exception as e:
            # Cleanup temp
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            print(f"❌ Error saving checkpoint: {e}")
            raise
        
        # Verify
        try:
            saved_state = torch.load(checkpoint_path, map_location='cpu')
            saved_epoch = saved_state.get('epoch', -1)
            saved_last_epoch = saved_state.get('last_epoch', -1)
            saved_best_map = saved_state.get('best_map', 0)
            saved_best_epoch = saved_state.get('best_epoch', -1)
            print(f"✅ Checkpoint verification:")
            print(f"  📊 Saved epoch: {saved_epoch}")
            print(f"  📊 Saved last_epoch: {saved_last_epoch}")
            print(f"  🎯 Saved best_map: {saved_best_map:.4f}")
            print(f"  🏆 Saved best_epoch: {saved_best_epoch}")
        except Exception as e:
            print(f"Warn: cannot verify saved checkpoint: {e}")
    
    def val(self):
        """
        Validation with oriented IoU.
        """
        print("Running oriented IoU validation...")
        
        # Init for val-only
        if not hasattr(self, 'model') or self.model is None:
            print("Initializing model for validation...")
            self.setup()
            self.val_dataloader = dist.warp_loader(self.cfg.val_dataloader, 
                                                  shuffle=self.cfg.val_dataloader.shuffle)
            
            # Resume if specified
            if self.cfg.resume:
                print(f'Resume from {self.cfg.resume}')
                self.resume(self.cfg.resume)
                print(f"Resume completed")
        
        # Use EMA model if available, fallback to regular model
        module = self.ema.module if hasattr(self, 'ema') and self.ema else self.model
        base_ds = self.val_dataloader.dataset
        
        test_stats, oriented_evaluator = evaluate_oriented(
            module, 
            self.criterion, 
            self.postprocessor,
            self.val_dataloader, 
            base_ds, 
            self.device, 
            self.output_dir
        )
        
        if dist.is_main_process():
            print("Oriented IoU validation completed")
            
            # Save val results
            if oriented_evaluator is not None:
                eval_file = self.output_dir / "validation_results.pth"
                torch.save({
                    'test_stats': test_stats,
                    'mean_ap': getattr(oriented_evaluator, 'mean_ap', 0),
                    'eval_results': getattr(oriented_evaluator, 'eval_results', [])
                }, eval_file)
                print(f"Saved validation results to {eval_file}")
        
        return test_stats, oriented_evaluator 