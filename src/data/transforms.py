import torch 
import torch.nn as nn 

import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from PIL import Image 
from typing import Any, Dict, List, Optional

from src.core import register, GLOBAL_CONFIG


# compat: torchvision.datapoints
try:
    from torchvision import datapoints
except ImportError:
    class _Datapoints:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("torchvision.datapoints is not available in this version.")
        @classmethod
        def __getattr__(cls, name):
            raise NotImplementedError(f"torchvision.datapoints.{name} is not available.")
    datapoints = _Datapoints

__all__ = ['Compose', ]

# ToImage compat
if hasattr(T, 'ToImage'):
    _BaseToImage = T.ToImage
else:
    _BaseToImage = T.ToImageTensor
ToImage = register('ToImage')(_BaseToImage)

# ConvertDtype compat
if hasattr(T, 'ToDtype'):
    ConvertDtype = register('ConvertDtype')(T.ToDtype)
else:
    ConvertDtype = register('ConvertDtype')(T.ConvertDtype)

# SanitizeBoundingBoxes compat
if hasattr(T, 'SanitizeBoundingBoxes'):
    SanitizeBoundingBoxes = register('SanitizeBoundingBoxes')(T.SanitizeBoundingBoxes)
else:
    SanitizeBoundingBoxes = register('SanitizeBoundingBoxes')(T.SanitizeBoundingBox)

RandomPhotometricDistort = register(T.RandomPhotometricDistort)
RandomZoomOut = register(T.RandomZoomOut)
# RandomIoUCrop = register(T.RandomIoUCrop)
RandomHorizontalFlip = register(T.RandomHorizontalFlip)
Resize = register(T.Resize)
RandomCrop = register(T.RandomCrop)
Normalize = register(T.Normalize)


@register
class Compose(T.Compose):
    def __init__(self, ops) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    class_name = GLOBAL_CONFIG[name]['_name']
                    transfom = getattr(GLOBAL_CONFIG[name]['_pymodule'], class_name)(**op)
                    transforms.append(transfom)
                elif isinstance(op, nn.Module):
                    transforms.append(op)
                else:
                    raise ValueError('Invalid transform spec')
        else:
            transforms =[EmptyTransform(), ]
        super().__init__(transforms=transforms)


@register
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register
class PadToSize(T.Pad):
    _transformed_types = (
        Image.Image,
        datapoints.Image,
        datapoints.Video,
        datapoints.Mask,
        datapoints.BoundingBox,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sz = F.get_spatial_size(flat_inputs[0])
        h, w = self.spatial_size[0] - sz[0], self.spatial_size[1] - sz[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, spatial_size, fill=0, padding_mode='constant') -> None:
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        self.spatial_size = spatial_size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:        
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p 

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        return super().forward(*inputs)


@register
class ConvertBox(T.Transform):
    _transformed_types = (
        datapoints.BoundingBox,
    )
    def __init__(self, out_fmt='', normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize
        self.data_fmt = {
            'xyxy': datapoints.BoundingBoxFormat.XYXY,
            'cxcywh': datapoints.BoundingBoxFormat.CXCYWH
        }

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        if self.out_fmt:
            spatial_size = inpt.spatial_size
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.out_fmt)
            inpt = datapoints.BoundingBox(inpt, format=self.data_fmt[self.out_fmt], spatial_size=spatial_size)
        if self.normalize:
            inpt = inpt / torch.tensor(inpt.spatial_size[::-1]).tile(2)[None]
        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)


@register
class ConvertOrientedBox(T.Transform):
    """Oriented bbox converter (cx,cy,w,h,angle), optional normalization."""
    _transformed_types = (
        datapoints.BoundingBox,
    )
    
    def __init__(self, out_fmt='cxcywha', normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        result = inpt if (hasattr(inpt, 'format') and inpt.format == 'cxcywha') else inpt
        if self.normalize and len(result.shape) > 1 and result.shape[-1] >= 4:
            if hasattr(inpt, 'spatial_size'):
                spatial_size = torch.tensor(inpt.spatial_size[::-1])  # [w, h]
            else:
                spatial_size = torch.tensor([1024, 1024])
            normalized = result.clone()
            normalized[..., 0] /= spatial_size[0]
            normalized[..., 1] /= spatial_size[1]
            normalized[..., 2] /= spatial_size[0]
            normalized[..., 3] /= spatial_size[1]
            result = normalized
        return result

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)
    
