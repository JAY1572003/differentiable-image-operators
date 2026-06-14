"""
DiffMultiscaleContrast - multi-scale local contrast operator.
Professor-verified. IoU on tile.png = 0.5105.
"""

import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from operations.operation import Operation


class DiffMultiscaleContrast(Operation):
    def __init__(self, sigma_small=1.0, sigma_large=5.0, epsilon=1e-6):
        super().__init__('DiffMultiscaleContrast')
        self.epsilon = epsilon
        self.params['sigma_small'] = torch.nn.Parameter(torch.tensor(float(sigma_small)))
        self.params['sigma_large'] = torch.nn.Parameter(torch.tensor(float(sigma_large)))
        self.param_ranges['sigma_small'] = [0.5,  5.0]
        self.param_ranges['sigma_large'] = [2.0, 15.0]

    def _gaussian_kernel(self, sigma):
        sigma  = torch.clamp(sigma, 0.3, 15.0)
        k      = int(6 * sigma.item() + 1) | 1
        ax     = torch.arange(-k//2+1, k//2+1, dtype=torch.float32, device=sigma.device)
        kernel = torch.exp(-ax**2 / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        return (kernel[:,None] * kernel[None,:]).unsqueeze(0).unsqueeze(0), k

    def _local_stats(self, x, sigma):
        kernel, k = self._gaussian_kernel(sigma)
        mean    = F.conv2d(x,      kernel, padding=k//2)
        mean_sq = F.conv2d(x**2,   kernel, padding=k//2)
        return mean, torch.sqrt(torch.clamp(mean_sq - mean**2, min=self.epsilon))

    def forward(self, x):
        self.clamp_params()
        sig_s = torch.clamp(self.params['sigma_small'], *self.param_ranges['sigma_small'])
        sig_l = torch.clamp(self.params['sigma_large'], *self.param_ranges['sigma_large'])
        mean_s, std_s = self._local_stats(x, sig_s)
        mean_l, std_l = self._local_stats(x, sig_l)
        contrast = torch.abs(mean_l - mean_s) + 0.5 * (std_l - std_s)
        return (contrast - contrast.min()) / (contrast.max() - contrast.min() + self.epsilon)
