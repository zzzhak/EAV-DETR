import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
            
        x = self.backbone(x)
        
        encoder_output = self.encoder(x)
        if isinstance(encoder_output, tuple):
            x, encoder_aux_pred = encoder_output
        else:
            x = encoder_output
            encoder_aux_pred = None
            
        x = self.decoder(x, targets)

        if encoder_aux_pred is not None and self.training:
            if isinstance(x, dict):
                x['encoder_aux_pred'] = encoder_aux_pred
            else:
                x = {'decoder_output': x, 'encoder_aux_pred': encoder_aux_pred}

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 