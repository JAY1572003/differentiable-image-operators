"""
Functional check (professor's spec, section 6):
Build a synthetic image with a uniform bright background and one thin dark
line. Check the operator's mean response ON the line is >= 5x the mean
response OFF the line, at default parameters.

This tells us if an operator can "see" a crack at all, separate from any
training/optimization problem. Run this FIRST when training seems stuck.
"""

import torch
import sys
sys.path.insert(0, '.')

from operators.frangi import DiffFrangi
from operators.multiscale_contrast import DiffMultiscaleContrast
from operators.diff_morphology import DiffMorphologyDisk, DiffMorphologyLine


def make_synthetic_image(size=400, line_width=2):
    """Uniform bright background (0.8) with one thin dark vertical line (0.1)."""
    img = torch.full((1, 1, size, size), 0.8)
    mid = size // 2
    img[:, :, :, mid:mid + line_width] = 0.1
    line_mask = torch.zeros(1, 1, size, size, dtype=torch.bool)
    line_mask[:, :, :, mid:mid + line_width] = True
    return img, line_mask


def check_operator(name, op, img, line_mask):
    with torch.no_grad():
        out = op(img)
    on_line  = out[line_mask].mean().item()
    off_line = out[~line_mask].mean().item()
    ratio = on_line / (off_line + 1e-8)
    passed = ratio >= 5.0
    print(f"{name:22s} | on-line={on_line:.4f} | off-line={off_line:.4f} | "
          f"ratio={ratio:.2f}x | {'PASS' if passed else 'FAIL'}")
    return passed


img, line_mask = make_synthetic_image()

print("Functional check: response on crack line vs off it (need >= 5x)\n")
check_operator("DiffFrangi",            DiffFrangi(),            img, line_mask)
check_operator("DiffMultiscaleContrast", DiffMultiscaleContrast(), img, line_mask)
check_operator("DiffMorphologyDisk",    DiffMorphologyDisk(),    img, line_mask)
check_operator("DiffMorphologyLine",    DiffMorphologyLine(),    img, line_mask)