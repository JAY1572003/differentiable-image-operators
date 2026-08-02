"""
DiffHaralick - fully differentiable Haralick/GLCM texture-statistic operator.

Investigation summary (per spec section 5):
  - Checked Kornia: no GLCM/Haralick implementation exists in the library.
  - Checked scikit-image (graycomatrix/graycoprops) and pyradiomics: both
    standard, widely-used, but built on hard gray-level quantization and
    discrete co-occurrence counting -> not differentiable.
  - No off-the-shelf differentiable version exists; built from scratch below
    using the soft-binning direction the spec describes.

Approach:
  - Soft-bin each pixel's intensity into `n_bins` bin centers via a
    softmax over squared distance to each center (temperature = 'softness').
    This replaces hard quantization with a differentiable soft assignment.
  - The "neighbor" pixel at a continuous (dx, dy) offset is obtained via
    grid_sample (bilinear, sub-pixel), which is differentiable w.r.t. the
    displacement magnitude and angle -- this replaces the discrete pixel
    shift a normal GLCM uses.
  - The co-occurrence matrix at each pixel's local window is the outer
    product of the two soft membership maps, box-filtered over the window.
    This replaces discrete counting with a sum of products of soft
    memberships, exactly as the spec describes.
  - Three standard Haralick statistics (contrast, homogeneity, energy) are
    computed from this per-pixel soft GLCM, producing three output maps.

Learnable params (3, per spec's 2-3 param scope):
  - softness:     controls how "hard" the bin assignment is
  - displacement: distance (in pixels) to the neighbor pixel
  - angle:        direction (radians) to the neighbor pixel

NOTE on spec section 3: this operator's output is 3 channels (one per
texture statistic) regardless of input channel count, since its purpose is
to produce multiple texture-statistic maps (spec section 5 explicitly asks
for "1-3 texture-statistic maps"). This is an intentional, documented
deviation from the "output same shape as input" convention, which was
written with single-channel enhancement/detection operators in mind.
"""

import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from operations.operation import Operation


class DiffHaralick(Operation):
    N_BINS = 6       # fixed, not learnable -- number of soft gray-level bins
    WINDOW = 9        # fixed, not learnable -- local window size for the co-occurrence stats

    def __init__(self, softness=0.15, displacement=1.5, angle=0.3, scale=5.0):
        super().__init__('DiffHaralick')
        self.params['softness']     = torch.nn.Parameter(torch.tensor([float(softness)]))
        self.params['displacement'] = torch.nn.Parameter(torch.tensor([float(displacement)]))
        self.params['angle']        = torch.nn.Parameter(torch.tensor([float(angle)]))
        self.params['scale']        = torch.nn.Parameter(torch.tensor([float(scale)]))
        self.param_ranges['softness']     = [0.03, 0.15]
        self.param_ranges['angle']        = [0.0, 3.14159]
        self.param_ranges['displacement'] = [1.5, 5.0]
        self.param_ranges['scale']        = [1.0, 20.0]
        self.register_buffer('bin_centers', torch.linspace(0.0, 1.0, self.N_BINS))

    def clamp_params(self):
        # Same small-margin override used in diff_morphology.py, for the
        # same reason: exact-boundary clamping gives zero gradient once a
        # param drifts to the wall. Doesn't touch operations/operation.py.
        with torch.no_grad():
            for key, p in self.params.items():
                lo, hi = self.param_ranges[key]
                margin = (hi - lo) * 0.01
                p.data.clamp_(lo + margin, hi - margin)

    def _soft_bin_membership(self, x, softness):
        # x: (B,1,H,W) -> (B, n_bins, H, W), soft membership summing to 1 over bins
        diff = x.unsqueeze(-1) - self.bin_centers.view(1, 1, 1, 1, -1)
        logits = -(diff ** 2) / (2.0 * softness ** 2 + 1e-8)
        mem = torch.softmax(logits, dim=-1)
        return mem.squeeze(1).permute(0, 3, 1, 2)

    @staticmethod
    def _shift_image(x, dx, dy):
        # Differentiable sub-pixel shift via grid_sample.
        B, C, H, W = x.shape
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        dx_norm = 2.0 * dx / max(W - 1, 1)
        dy_norm = 2.0 * dy / max(H - 1, 1)
        grid = torch.stack([xs + dx_norm, ys + dy_norm], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)

    def forward(self, x):
        self.clamp_params()
        softness     = self.params['softness']
        displacement = self.params['displacement']
        angle        = self.params['angle']

        dx = displacement * torch.cos(angle)
        dy = displacement * torch.sin(angle)

        results = []
        for c in range(x.shape[1]):
            xc = x[:, c:c+1, :, :]
            xc_shifted = self._shift_image(xc, dx, dy)

            mem_c = self._soft_bin_membership(xc, softness)          # (B,K,H,W)
            mem_n = self._soft_bin_membership(xc_shifted, softness)  # (B,K,H,W)

            B, K, H, W = mem_c.shape
            prod = (mem_c.unsqueeze(2) * mem_n.unsqueeze(1)).reshape(B, K * K, H, W)

            local = F.avg_pool2d(prod, kernel_size=self.WINDOW, stride=1,
                                  padding=self.WINDOW // 2, count_include_pad=False)
            local = local.reshape(B, K, K, H, W)
            local = local / (local.sum(dim=(1, 2), keepdim=True) + 1e-8)

            i_idx = torch.arange(K, dtype=torch.float32, device=x.device).view(K, 1, 1, 1)
            j_idx = torch.arange(K, dtype=torch.float32, device=x.device).view(1, K, 1, 1)

            contrast_map    = (local * (i_idx - j_idx) ** 2).sum(dim=(1, 2)) / ((K - 1) ** 2)
            homogeneity_map = (local / (1.0 + (i_idx - j_idx) ** 2)).sum(dim=(1, 2))
            energy_map      = (local ** 2).sum(dim=(1, 2))

            out_c = torch.stack([contrast_map, homogeneity_map, energy_map], dim=1)  # (B,3,H,W)
            results.append(out_c)

        # For multi-channel input, average texture maps across channels.
        out = torch.stack(results, dim=0).mean(dim=0)
        # Learnable scale (spec section 3): contrast's natural range is
        # small (observed max ~0.1) since it's normalized by a large
        # theoretical ceiling (n_bins-1)^2 that real textures rarely
        # approach. Scale brings it into a meaningful [0,1] range instead
        # of leaving it too small for the standard 0.5 threshold.
        return torch.clamp(out * self.params['scale'], 0.0, 1.0)