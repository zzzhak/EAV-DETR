
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from src.core import register


__all__ = ['AdamW', 'SGD', 'Adam', 'MultiStepLR', 'CosineAnnealingLR', 'OneCycleLR', 'LambdaLR', 'SequentialLR', 'LinearLR', 'WarmupCosineLR']



SGD = register(optim.SGD)
Adam = register(optim.Adam)
AdamW = register(optim.AdamW)


MultiStepLR = register(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register(lr_scheduler.OneCycleLR)
LambdaLR = register(lr_scheduler.LambdaLR)

SequentialLR = register(lr_scheduler.SequentialLR)
LinearLR = register(lr_scheduler.LinearLR)

@register
class WarmupCosineLR:
    
    def __init__(self, optimizer, T_max, warmup_epochs=3, eta_min=1e-7, warmup_start_factor=0.1):

        # Warmup scheduler
        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer, 
            start_factor=warmup_start_factor,
            total_iters=warmup_epochs
        )
        
        # Cosine scheduler
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max - warmup_epochs,
            eta_min=eta_min
        )

        self.scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    
    def step(self, epoch=None):
        return self.scheduler.step()
    
    def get_last_lr(self):
        return self.scheduler.get_last_lr()
    
    def state_dict(self):
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.scheduler.load_state_dict(state_dict)
