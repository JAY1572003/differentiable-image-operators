"""
Tests for differentiable operators — enhance ops + crack-detection ops.

For EVERY op in DIFF_ENHANCE_OPS, we verify:
0. has_params       — at least one nn.Parameter exists
1. gradient_flow    — backward pass produces non-None gradients for all params
2. output_shape     — output tensor has the expected shape
3. no_nan_inf       — no NaN/Inf in output or gradients
4. convergence      — added manually per op in TestConvergence below
"""

import pytest
import torch
import torch.nn as nn

# ----- Enhance ops under test -----
from operations.differentiable_enhance import (
    DiffBrightness, DiffContrast, DiffGamma, DiffSaturation, DiffBrightnessAdditive,
)

# ----- Crack-detection ops under test -----
from operators.frangi import DiffFrangi
from operators.multiscale_contrast import DiffMultiscaleContrast
from operators.diff_morphology import DiffMorphologyDisk, DiffMorphologyLine


# ============================================================
# Helpers
# ============================================================

def _run_forward_backward(op, input_img):
    input_img = input_img.clone().detach().requires_grad_(True)
    output = op(input_img)
    loss = output.sum()
    loss.backward()
    return output, loss


def _run_convergence(op, input_img, target, max_epochs=50, lr=0.05):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(op.parameters(), lr=lr)

    with torch.no_grad():
        initial_output = op(input_img)
        initial_loss = loss_fn(initial_output, target).item()

    for _ in range(max_epochs):
        optimizer.zero_grad()
        output = op(input_img)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_output = op(input_img)
        final_loss = loss_fn(final_output, target).item()

    return initial_loss, final_loss


# ============================================================
# Ops to test
# ============================================================

DIFF_ENHANCE_OPS = [
    ("DiffBrightness",         DiffBrightness,         {}),
    ("DiffContrast",           DiffContrast,           {}),
    ("DiffGamma",              DiffGamma,              {}),
    ("DiffSaturation",         DiffSaturation,         {}),
    ("DiffBrightnessAdditive", DiffBrightnessAdditive, {}),
    ("DiffFrangi",             DiffFrangi,             {}),
    ("DiffMultiscaleContrast", DiffMultiscaleContrast, {}),
    ("DiffMorphologyDisk",     DiffMorphologyDisk,     {}),
    ("DiffMorphologyLine",     DiffMorphologyLine,     {}),
]


# ============================================================
# 0. Learnable parameters exist
# ============================================================

class TestHasLearnableParams:
    @pytest.mark.parametrize("name,cls,kwargs", DIFF_ENHANCE_OPS,
                             ids=[x[0] for x in DIFF_ENHANCE_OPS])
    def test_has_learnable_params(self, name, cls, kwargs):
        op = cls(**kwargs)
        assert len(list(op.parameters())) > 0, (
            f"{name}: list(op.parameters()) is empty — no nn.Parameter anywhere."
        )
        assert hasattr(op, 'params') and len(op.params) > 0, (
            f"{name}: op.params dict is missing or empty."
        )


# ============================================================
# 1. Gradient flow tests
# ============================================================

class TestGradientFlow:
    @pytest.mark.parametrize("name,cls,kwargs", DIFF_ENHANCE_OPS,
                             ids=[x[0] for x in DIFF_ENHANCE_OPS])
    def test_all_params_receive_gradients(self, name, cls, kwargs, gray_image, rgb_image):
        op = cls(**kwargs)
        # Frangi/MultiscaleContrast/Morphology ops are built for single-channel
        # (grayscale) crack images; enhance ops expect RGB.
        img = gray_image if name in (
            "DiffFrangi", "DiffMultiscaleContrast",
            "DiffMorphologyDisk", "DiffMorphologyLine"
        ) else rgb_image

        op.zero_grad()
        _run_forward_backward(op, img)

        for param_name, param in op.params.items():
            assert param.grad is not None, (
                f"{name}: param '{param_name}' has None gradient after backward pass!"
            )
            assert param.grad.abs().sum() > 0, (
                f"{name}: param '{param_name}' has all-zero gradient."
            )


# ============================================================
# 2. Output shape tests
# ============================================================

class TestOutputShape:
    @pytest.mark.parametrize("name,cls,kwargs", DIFF_ENHANCE_OPS,
                             ids=[x[0] for x in DIFF_ENHANCE_OPS])
    def test_shape_preserved(self, name, cls, kwargs, gray_image, rgb_image):
        op = cls(**kwargs)
        img = gray_image if name in (
            "DiffFrangi", "DiffMultiscaleContrast",
            "DiffMorphologyDisk", "DiffMorphologyLine"
        ) else rgb_image
        output = op(img)
        assert output.shape == img.shape, (
            f"{name}: expected output shape {img.shape}, got {output.shape}"
        )


# ============================================================
# 3. No NaN/Inf tests
# ============================================================

class TestNoNanInf:
    @pytest.mark.parametrize("name,cls,kwargs", DIFF_ENHANCE_OPS,
                             ids=[x[0] for x in DIFF_ENHANCE_OPS])
    def test_output_finite(self, name, cls, kwargs, gray_image, rgb_image):
        op = cls(**kwargs)
        img = gray_image if name in (
            "DiffFrangi", "DiffMultiscaleContrast",
            "DiffMorphologyDisk", "DiffMorphologyLine"
        ) else rgb_image
        output = op(img)
        assert torch.isfinite(output).all(), f"{name}: output contains NaN or Inf values"

    @pytest.mark.parametrize("name,cls,kwargs", DIFF_ENHANCE_OPS,
                             ids=[x[0] for x in DIFF_ENHANCE_OPS])
    def test_gradients_finite(self, name, cls, kwargs, gray_image, rgb_image):
        op = cls(**kwargs)
        img = gray_image if name in (
            "DiffFrangi", "DiffMultiscaleContrast",
            "DiffMorphologyDisk", "DiffMorphologyLine"
        ) else rgb_image
        op.zero_grad()
        _run_forward_backward(op, img)
        for param_name, param in op.params.items():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"{name}: param '{param_name}' has NaN/Inf in gradient"
                )


# ============================================================
# 4. Convergence tests
# ============================================================

class TestConvergence:

    def test_brightness_convergence(self, rgb_image):
        op = DiffBrightness()
        with torch.no_grad():
            target_op = DiffBrightness()
            target_op.set_params({'factor': [0.5]})
            target = target_op(rgb_image)
        initial_loss, final_loss = _run_convergence(op, rgb_image, target)
        assert final_loss < initial_loss * 0.1

    def test_contrast_convergence(self, rgb_image):
        op = DiffContrast()
        with torch.no_grad():
            target_op = DiffContrast()
            target_op.set_params({'factor': [1.8]})
            target = target_op(rgb_image)
        initial_loss, final_loss = _run_convergence(op, rgb_image, target)
        assert final_loss < initial_loss * 0.1

    def test_gamma_convergence(self, rgb_image):
        op = DiffGamma()
        with torch.no_grad():
            target_op = DiffGamma()
            target_op.set_params({'gamma': [0.7]})
            target = target_op(rgb_image)
        initial_loss, final_loss = _run_convergence(op, rgb_image, target)
        assert final_loss < initial_loss * 0.1

    def test_saturation_convergence(self, rgb_image):
        op = DiffSaturation()
        with torch.no_grad():
            target_op = DiffSaturation()
            target_op.set_params({'factor': [0.3]})
            target = target_op(rgb_image)
        initial_loss, final_loss = _run_convergence(op, rgb_image, target)
        assert final_loss < initial_loss * 0.1

    def test_brightness_additive_convergence(self, rgb_image):
        op = DiffBrightnessAdditive()
        with torch.no_grad():
            target_op = DiffBrightnessAdditive()
            target_op.set_params({'offset': [0.2]})
            target = target_op(rgb_image)
        initial_loss, final_loss = _run_convergence(op, rgb_image, target)
        assert final_loss < initial_loss * 0.1

    # --- Crack-detection operators ---

    def test_frangi_convergence(self, gray_image):
        op = DiffFrangi()
        with torch.no_grad():
            target_op = DiffFrangi()
            target_op.set_params({'sigma': [3.5]})
            target = target_op(gray_image)
        initial_loss, final_loss = _run_convergence(op, gray_image, target, lr=0.05)
        assert final_loss < initial_loss * 0.5, (
            f"DiffFrangi: loss didn't converge. Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
        )

    def test_multiscale_contrast_convergence(self, gray_image):
        op = DiffMultiscaleContrast()
        with torch.no_grad():
            target_op = DiffMultiscaleContrast()
            target_op.set_params({'sigma_small': [2.0], 'sigma_large': [8.0]})
            target = target_op(gray_image)
        initial_loss, final_loss = _run_convergence(op, gray_image, target, lr=0.05)
        assert final_loss < initial_loss * 0.5, (
            f"DiffMultiscaleContrast: loss didn't converge. Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
        )

    def test_morphology_disk_convergence(self, gray_image):
        op = DiffMorphologyDisk()
        with torch.no_grad():
            target_op = DiffMorphologyDisk()
            target_op.set_params({'radius': [1.0]})
            target = target_op(gray_image)
        initial_loss, final_loss = _run_convergence(op, gray_image, target, lr=0.05)
        assert final_loss < initial_loss * 0.5, (
            f"DiffMorphologyDisk: loss didn't converge. Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
        )

    # Angle is non-convex, so per the spec: test convergence from several
    # (3+) different starting angles, not just one.
    @staticmethod
    def _line_test_image(size=48):
        """
        Synthetic image with a few short line segments at different angles,
        rather than random noise. Pure noise gives the angle parameter a weak,
        ambiguous gradient signal, which is why far-away angle starts were
        getting stuck (a real non-convexity issue, not a code bug). An image
        with actual line structure gives a much clearer direction to follow —
        closer to what a real crack image looks like anyway.
        """
        img = torch.full((1, 1, size, size), 0.8)
        img[:, :, size // 3, :] = 0.1                      # horizontal segment
        for i in range(size):
            j = size // 3 + (i - size // 2) // 2
            if 0 <= j < size:
                img[:, :, j, i] = 0.1                       # diagonal segment
        return img

    # Angle is non-convex, so per the spec: test convergence from several
    # (3+) different starting angles, not just one.
    @pytest.mark.parametrize("start_angle", [0.3, 1.2, 2.5])
    def test_morphology_line_convergence(self, start_angle):
        line_image = self._line_test_image()
        op = DiffMorphologyLine()
        op.set_params({'angle': [start_angle], 'length': [2.0]})
        with torch.no_grad():
            target_op = DiffMorphologyLine()
            target_op.set_params({'angle': [1.571], 'length': [3.5]})
            target = target_op(line_image)
        initial_loss, final_loss = _run_convergence(op, line_image, target,
                                                     max_epochs=300, lr=0.08)
        assert final_loss < initial_loss * 0.5, (
            f"DiffMorphologyLine (start_angle={start_angle}): loss didn't converge. "
            f"Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
        )