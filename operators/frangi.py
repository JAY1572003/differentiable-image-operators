"""
DiffFrangi - fully differentiable Frangi vesselness filter in PyTorch.
Learnable: sigma (scale), beta (anisotropy), gamma (background).
"""

import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from operations.operation import Operation


class DiffFrangi(Operation):
    def __init__(self, sigma=2.0, beta=0.5, gamma=5.0, black_ridges=True):
        super().__init__('DiffFrangi')
        self.black_ridges = black_ridges
        self.params['sigma'] = torch.nn.Parameter(torch.tensor(float(sigma)))
        self.params['beta']  = torch.nn.Parameter(torch.tensor(float(beta)))
        self.params['gamma'] = torch.nn.Parameter(torch.tensor(float(gamma)))
        self.param_ranges['sigma'] = [0.5, 6.0]
        self.param_ranges['beta']  = [0.1, 2.0]
        self.param_ranges['gamma'] = [1.0, 20.0]

    @staticmethod
    def _gauss1d(sigma, order):
        sig = torch.clamp(sigma, 0.3, 8.0)
        k   = int(6 * sig.item() + 1) | 1
        ax  = torch.arange(-k//2+1, k//2+1, dtype=torch.float32, device=sigma.device)
        g   = torch.exp(-ax**2 / (2.0 * sig**2))
        if order == 0:
            return g / g.sum()
        elif order == 1:
            dg = -(ax / sig**2) * g
            return dg / (dg.abs().sum() / 2.0 + 1e-10)
        else:
            d2g = (ax**2 / sig**4 - 1.0 / sig**2) * g
            return d2g / (d2g.abs().sum() / 2.0 + 1e-10)

    def _hessian(self, x, sigma):
        g0, g1, g2 = self._gauss1d(sigma,0), self._gauss1d(sigma,1), self._gauss1d(sigma,2)
        k   = g0.shape[0]
        pad = k // 2
        s2  = sigma ** 2
        kxx = (g2[:,None]*g0[None,:]).unsqueeze(0).unsqueeze(0)
        kyy = (g0[:,None]*g2[None,:]).unsqueeze(0).unsqueeze(0)
        kxy = (g1[:,None]*g1[None,:]).unsqueeze(0).unsqueeze(0)
        return (s2*F.conv2d(x,kxx,padding=pad),
                s2*F.conv2d(x,kxy,padding=pad),
                s2*F.conv2d(x,kyy,padding=pad))

    @staticmethod
    def _eigenvalues(Hxx, Hxy, Hyy):
        trace = Hxx + Hyy
        disc  = torch.sqrt(((Hxx-Hyy)/2.0)**2 + Hxy**2 + 1e-10)
        return trace/2.0 - disc, trace/2.0 + disc

    def forward(self, x):
        self.clamp_params()
        sigma = torch.clamp(self.params['sigma'], *self.param_ranges['sigma'])
        beta  = torch.clamp(self.params['beta'],  *self.param_ranges['beta'])
        gamma = torch.clamp(self.params['gamma'], *self.param_ranges['gamma'])
    
    # Process each channel independently
        results = []
        for c in range(x.shape[1]):
            xc   = x[:, c:c+1, :, :]
            img  = 1.0 - xc if self.black_ridges else xc
            Hxx, Hxy, Hyy = self._hessian(img, sigma)
            lam1, lam2    = self._eigenvalues(Hxx, Hxy, Hyy)
            ridge_mask = torch.sigmoid(20.0 * lam2)
            Rb2 = (lam1 / (lam2 + 1e-8)) ** 2
            S2  = lam1**2 + lam2**2
            v = torch.exp(-Rb2/(2.0*beta**2)) * (1.0 - torch.exp(-S2/(2.0*gamma**2)))
            v = v * ridge_mask
            v = (v - v.min()) / (v.max() - v.min() + 1e-8)
            results.append(v)
        return torch.cat(results, dim=1)
