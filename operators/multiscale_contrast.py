"""
DiffMultiscaleContrast - multi-scale local contrast operator.
Interface: kit v2 (1-element param tensors, no per-image normalization).
"""

import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from operations.operation import Operation


class DiffMultiscaleContrast(Operation):
    def __init__(self, sigma_small=1.0, sigma_large=5.0, scale=5.0, epsilon=1e-6):
        super().__init__('DiffMultiscaleContrast')
        self.epsilon = epsilon

        self.params['sigma_small'] = torch.nn.Parameter(torch.tensor([float(sigma_small)]))
        self.params['sigma_large'] = torch.nn.Parameter(torch.tensor([float(sigma_large)]))
        self.params['scale']       = torch.nn.Parameter(torch.tensor([float(scale)]))
        self.param_ranges['sigma_small'] = [0.5, 5.0]
        self.param_ranges['sigma_large'] = [2.0, 8.0]
        self.param_ranges['scale']       = [0.5, 20.0]

        # Fixed kernel sizes, chosen once from the MAX of each sigma's range.
        # sigma itself only shapes the kernel VALUES below (still differentiable);
        # it never determines the kernel SIZE at runtime (that would break gradients).
        self.k_small = self._fixed_kernel_size(self.param_ranges['sigma_small'][1])
        self.k_large = self._fixed_kernel_size(self.param_ranges['sigma_large'][1])

    @staticmethod
    def _fixed_kernel_size(max_sigma):
        k = int(6 * max_sigma + 1)
        if k % 2 == 0:
            k += 1
        return k

    def _gaussian_kernel(self, sigma, k):
        ax = torch.arange(-(k // 2), k // 2 + 1, dtype=torch.float32, device=sigma.device)
        kernel = torch.exp(-ax**2 / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        return (kernel[:, None] * kernel[None, :]).unsqueeze(0).unsqueeze(0)

    def _local_stats(self, x, sigma, k):
        kernel = self._gaussian_kernel(sigma, k)
        mean    = F.conv2d(x,    kernel, padding=k // 2)
        mean_sq = F.conv2d(x**2, kernel, padding=k // 2)
        return mean, torch.sqrt(torch.clamp(mean_sq - mean**2, min=self.epsilon))

    def forward(self, x):
        self.clamp_params()
        sig_s = self.params['sigma_small']
        sig_l = self.params['sigma_large']
        scale = self.params['scale']

        mean_s, std_s = self._local_stats(x, sig_s, self.k_small)
        mean_l, std_l = self._local_stats(x, sig_l, self.k_large)

        contrast = torch.abs(mean_l - mean_s) + 0.5 * (std_l - std_s)
        # Replaces per-image min-max normalization: a learnable scale + clamp
        # keeps output in [0, 1] without depending on this image's global min/max.
        return torch.clamp(contrast * scale, 0.0, 1.0)