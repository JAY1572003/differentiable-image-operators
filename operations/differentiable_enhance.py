"""
Differentiable Enhancement Operations
======================================

These are drop-in replacements for Kornia's enhancement operations
that actually support gradient-based parameter optimization.

The original Kornia enhance functions don't produce gradients w.r.t. their
factor parameters, making them unusable for optimization. These versions do!
"""

import torch
from operations.operation import Operation


class DiffBrightness(Operation):
    """
    Differentiable brightness adjustment via multiplication.
    Unlike Kornia's adjust_brightness, this DOES support gradients!
    
    factor > 1.0 = brighter
    factor < 1.0 = darker
    factor = 1.0 = no change
    """
    def __init__(self):
        super().__init__('DiffBrightness')
        self.params['factor'] = torch.nn.Parameter(torch.tensor([1.0]))
        self.param_ranges['factor'] = [0.01, 2.0]
    
    def forward(self, input):
        self.clamp_params()
        # Multiplicative - this IS differentiable!
        return torch.clamp(input * self.params['factor'], 0.0, 1.0)


class DiffContrast(Operation):
    """
    Differentiable contrast adjustment by scaling deviation from mean.
    
    factor > 1.0 = more contrast
    factor < 1.0 = less contrast
    factor = 1.0 = no change
    """
    def __init__(self):
        super().__init__('DiffContrast')
        self.params['factor'] = torch.nn.Parameter(torch.tensor([1.0]))
        self.param_ranges['factor'] = [0.01, 3.0]
    
    def forward(self, input):
        self.clamp_params()
        # Scale around mean - this IS differentiable!
        mean = input.mean(dim=[2, 3], keepdim=True)
        output = mean + (input - mean) * self.params['factor']
        return torch.clamp(output, 0.0, 1.0)


class DiffGamma(Operation):
    """
    Differentiable gamma correction.
    
    gamma > 1.0 = darker
    gamma < 1.0 = brighter
    gamma = 1.0 = no change
    
    NOTE: Uses epsilon to prevent NaN gradients when input has near-zero values.
    The derivative of x^gamma is gamma * x^(gamma-1), which can explode for small x
    when gamma < 1.0.
    """
    def __init__(self):
        super().__init__('DiffGamma')
        self.params['gamma'] = torch.nn.Parameter(torch.tensor([1.0]))
        self.param_ranges['gamma'] = [0.5, 2.5]  # Restrict to safer range
    
    def forward(self, input):
        self.clamp_params()
        # Add small epsilon to prevent gradient explosion for near-zero inputs
        # when gamma < 1.0 (derivative is gamma * x^(gamma-1) which explodes)
        eps = 1e-6
        safe_input = torch.clamp(input, min=eps)
        # Power function - this IS differentiable!
        return torch.clamp(safe_input.pow(self.params['gamma']), 0.0, 1.0)


class DiffSaturation(Operation):
    """
    Differentiable saturation adjustment.
    
    factor > 1.0 = more saturated
    factor < 1.0 = less saturated (more gray)
    factor = 0.0 = completely grayscale
    factor = 1.0 = no change
    """
    def __init__(self):
        super().__init__('DiffSaturation')
        self.params['factor'] = torch.nn.Parameter(torch.tensor([1.0]))
        self.param_ranges['factor'] = [0.0, 2.0]
    
    def forward(self, input):
        self.clamp_params()
        # Convert to grayscale
        # RGB -> Gray: 0.299*R + 0.587*G + 0.114*B
        weights = torch.tensor([0.299, 0.587, 0.114], device=input.device).view(1, 3, 1, 1)
        gray = (input * weights).sum(dim=1, keepdim=True)
        gray = gray.repeat(1, 3, 1, 1)
        
        # Blend between grayscale and original
        # factor=0 -> grayscale, factor=1 -> original
        output = gray + (input - gray) * self.params['factor']
        return torch.clamp(output, 0.0, 1.0)


class DiffBrightnessAdditive(Operation):
    """
    Additive brightness (closer to Kornia's original, but differentiable).
    
    offset > 0 = brighter
    offset < 0 = darker
    offset = 0 = no change
    """
    def __init__(self):
        super().__init__('DiffBrightnessAdditive')
        self.params['offset'] = torch.nn.Parameter(torch.tensor([0.0]))
        self.param_ranges['offset'] = [-0.5, 0.5]
    
    def forward(self, input):
        self.clamp_params()
        # Additive - this IS differentiable!
        return torch.clamp(input + self.params['offset'], 0.0, 1.0)


# For easier migration, create aliases with common names
MultiplicativeBrightness = DiffBrightness
MultiplicativeContrast = DiffContrast
