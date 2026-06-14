"""
DiffMorphologyDisk  - soft disk SE, learnable radius.
DiffMorphologyLine  - soft line SE, learnable length + angle.
Uses log-sum-exp dilation/erosion so gradients reach all parameters.
"""

import math
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from operations.operation import Operation

_ALPHA = 8.0
_TAU   = 8.0


def _soft_dilate(x, se):
    B, C, H, W = x.shape
    kh, kw  = se.shape
    padded  = F.pad(x, (kw//2, kw//2, kh//2, kh//2), mode='replicate')
    unf     = F.unfold(padded, kernel_size=(kh, kw)).view(B, C, kh*kw, H*W)
    se_flat = se.flatten().clamp(min=1e-10).view(1, 1, -1, 1)
    log_w   = _TAU * unf + torch.log(se_flat)
    log_w   = log_w - log_w.detach().max(dim=2, keepdim=True).values
    w       = torch.exp(log_w)
    w       = w / (w.sum(dim=2, keepdim=True) + 1e-10)
    return (w * unf).sum(dim=2).view(B, C, H, W)


def _soft_erode(x, se):
    B, C, H, W = x.shape
    kh, kw  = se.shape
    padded  = F.pad(x, (kw//2, kw//2, kh//2, kh//2), mode='replicate')
    unf     = F.unfold(padded, kernel_size=(kh, kw)).view(B, C, kh*kw, H*W)
    se_flat = se.flatten().clamp(min=1e-10).view(1, 1, -1, 1)
    log_w   = -_TAU * unf + torch.log(se_flat)
    log_w   = log_w - log_w.detach().max(dim=2, keepdim=True).values
    w       = torch.exp(log_w)
    w       = w / (w.sum(dim=2, keepdim=True) + 1e-10)
    return (w * unf).sum(dim=2).view(B, C, H, W)


def _apply_operation(x, se, operation):
    if operation == 'dilation':     return _soft_dilate(x, se)
    elif operation == 'erosion':    return _soft_erode(x, se)
    elif operation == 'opening':    return _soft_dilate(_soft_erode(x, se), se)
    elif operation == 'closing':    return _soft_erode(_soft_dilate(x, se), se)
    elif operation == 'gradient':   return _soft_dilate(x, se) - _soft_erode(x, se)
    elif operation == 'top_hat':    return x - _soft_dilate(_soft_erode(x, se), se)
    elif operation == 'bottom_hat': return _soft_erode(_soft_dilate(x, se), se) - x
    else: raise ValueError(f'Unknown operation: {operation}')


class DiffMorphologyDisk(Operation):
    SUPPORTED_OPS = ['dilation','erosion','opening','closing','gradient','top_hat','bottom_hat']

    def __init__(self, operation='closing', se_size=7):
        super().__init__('DiffMorphologyDisk')
        assert operation in self.SUPPORTED_OPS
        self.operation = operation
        self.se_size   = se_size
        self.params['radius']       = torch.nn.Parameter(torch.tensor(2.0))
        self.param_ranges['radius'] = [0.1, se_size/2.0-0.5]
        center = se_size // 2
        coords = torch.arange(se_size, dtype=torch.float32)
        gi, gj = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('dist', torch.sqrt((gi-center)**2 + (gj-center)**2))

    def _get_se(self):
        r = torch.clamp(self.params['radius'], *self.param_ranges['radius'])
        return torch.sigmoid(_ALPHA * (r - self.dist))

    def forward(self, x):
        self.clamp_params()
        out = _apply_operation(x, self._get_se(), self.operation)
        return (out - out.min()) / (out.max() - out.min() + 1e-8)


class DiffMorphologyLine(Operation):
    SUPPORTED_OPS = ['dilation','erosion','opening','closing','gradient','top_hat','bottom_hat']
    LINE_WIDTH = 0.8

    def __init__(self, operation='closing', se_size=11):
        super().__init__('DiffMorphologyLine')
        assert operation in self.SUPPORTED_OPS
        self.operation = operation
        self.se_size   = se_size
        self.params['length']       = torch.nn.Parameter(torch.tensor(3.0))
        self.params['angle']        = torch.nn.Parameter(torch.tensor(math.pi/4.0))
        self.param_ranges['length'] = [0.5, float(se_size//2)]
        self.param_ranges['angle']  = [0.0, math.pi]
        center = se_size // 2
        coords = torch.arange(se_size, dtype=torch.float32) - center
        gi, gj = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('gi', gi)
        self.register_buffer('gj', gj)

    def _get_se(self):
        length = torch.clamp(self.params['length'], *self.param_ranges['length'])
        angle  = torch.clamp(self.params['angle'],  *self.param_ranges['angle'])
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        u = self.gj * cos_a + self.gi * sin_a
        v = -self.gj * sin_a + self.gi * cos_a
        return (torch.sigmoid(_ALPHA*(u+length)) * torch.sigmoid(_ALPHA*(length-u)) *
                torch.sigmoid(_ALPHA*(v+self.LINE_WIDTH)) * torch.sigmoid(_ALPHA*(self.LINE_WIDTH-v)))

    def forward(self, x):
        self.clamp_params()
        out = _apply_operation(x, self._get_se(), self.operation)
        return (out - out.min()) / (out.max() - out.min() + 1e-8)
