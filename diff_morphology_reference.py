"""Professor's reference implementation of DiffMorphologyDisk."""
import torch
import torch.nn.functional as F

ALPHA = 5.0
LARGE = 1e2

def _soft_dilation(input, se):
    B, C, H, W = input.shape
    kh, kw = se.shape
    padded   = F.pad(input, (kw//2,kw//2,kh//2,kh//2), mode='replicate')
    unfolded = F.unfold(padded, kernel_size=(kh,kw)).view(B,C,kh*kw,H*W)
    se_flat  = se.flatten().view(1,1,-1,1)
    return (unfolded*se_flat + (1-se_flat)*(-LARGE)).max(dim=2).values.view(B,C,H,W)

def _soft_erosion(input, se):
    B, C, H, W = input.shape
    kh, kw = se.shape
    padded   = F.pad(input, (kw//2,kw//2,kh//2,kh//2), mode='replicate')
    unfolded = F.unfold(padded, kernel_size=(kh,kw)).view(B,C,kh*kw,H*W)
    se_flat  = se.flatten().view(1,1,-1,1)
    return (unfolded*se_flat + (1-se_flat)*LARGE).min(dim=2).values.view(B,C,H,W)

class DiffMorphologyDisk(torch.nn.Module):
    def __init__(self, operation='closing', se_size=5):
        super().__init__()
        self.operation = operation
        self.radius    = torch.nn.Parameter(torch.tensor([2.0]))
        center = se_size // 2
        coords = torch.arange(se_size, dtype=torch.float32)
        gi, gj = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('dist', torch.sqrt((gi-center)**2 + (gj-center)**2))
    def forward(self, x):
        se = torch.sigmoid(ALPHA * (self.radius - self.dist))
        if self.operation == 'dilation':  return _soft_dilation(x, se)
        elif self.operation == 'erosion': return _soft_erosion(x, se)
        elif self.operation == 'closing': return _soft_erosion(_soft_dilation(x, se), se)
        elif self.operation == 'opening': return _soft_dilation(_soft_erosion(x, se), se)
