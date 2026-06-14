"""
Tests for differentiable enhance operations.

This simplified test suite demonstrates how to test custom differentiable operators.
For EVERY op, we verify:

0. has_params       — at least one nn.Parameter exists (else there is nothing to learn)
1. gradient_flow    — backward pass produces non-None gradients for all learnable params
2. output_shape     — output tensor has the expected shape
3. no_nan_inf       — no NaN/Inf in output or gradients
4. convergence      — mini optimization loop: loss decreases from wrong params toward target
                      (NOT automatic — add one test per op in TestConvergence)

To extend with your own ops:
1. Implement your op in differentiable_enhance.py (inherit from nn.Module)
2. Add it to DIFF_ENHANCE_OPS below
3. Run: pytest test_differentiable_enhance_only.py
"""

import pytest
import torch
import torch.nn as nn

# ----- Ops under test -----
from operations.differentiable_enhance import (
    DiffBrightness, DiffContrast, DiffGamma, DiffSaturation, DiffBrightnessAdditive,
)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from operators.frangi import DiffFrangi
from operators.diff_morphology import DiffMorphologyDisk, DiffMorphologyLine

# ============================================================
# Helpers
# ============================================================

def _run_forward_backward(op, input_img):
    """Run forward + backward, return output and loss."""
    input_img = input_img.clone().detach().requires_grad_(True)
    output = op(input_img)
    loss = output.sum()
    loss.backward()
    return output, loss


def _run_convergence(op, input_img, target, max_epochs=50, lr=0.05):
    """Run a mini optimization loop. Returns (initial_loss, final_loss)."""
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(op.parameters(), lr=lr)

    # Initial loss
    with torch.no_grad():
        initial_output = op(input_img)
        initial_loss = loss_fn(initial_output, target).item()

    # Optimize
    for _ in range(max_epochs):
        optimizer.zero_grad()
        output = op(input_img)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

    # Final loss
    with torch.no_grad():
        final_output = op(input_img)
        final_loss = loss_fn(final_output, target).item()

    return initial_loss, final_loss


# ============================================================
# Ops to test (ADD YOUR NEW OPS HERE)
# ============================================================

DIFF_ENHANCE_OPS = [
    ("DiffBrightness",        DiffBrightness,        {}),
    ("DiffContrast",          DiffContrast,          {}),
    ("DiffGamma",             DiffGamma,             {}),
    ("DiffSaturation",        DiffSaturation,        {}),
    ("DiffBrightnessAdditive",DiffBrightnessAdditive,{}),
    ("DiffFrangi",            DiffFrangi,            {}),
    ("DiffMorphologyDisk",    DiffMorphologyDisk,    {"operation": "closing"}),
    ("DiffMorphologyLine",    DiffMorphologyLine,    {"operation": "closing"}),
]


# ============================================================
# 0. Learnable parameters exist
# ============================================================

class TestHasLearnableParams:
    """Verify the op has at least one learnable parameter.

    Without this check, the gradient-flow test below passes VACUOUSLY for an
    op with an empty params dict (the for-loop body never runs). An operator
    with no nn.Parameters gives gradient descent nothing to optimize, even if
    its forward pass runs in PyTorch and passes gradients through the image.
    """

    @pytest.mark.parametrize("name,cls,kwargs", DIFF_ENHANCE_OPS,
                             ids=[x[0] for x in DIFF_ENHANCE_OPS])
    def test_has_learnable_params(self, name, cls, kwargs):
        op = cls(**kwargs)
        assert len(list(op.parameters())) > 0, (
            f"{name}: list(op.parameters()) is empty — no nn.Parameter anywhere. "
            f"There is nothing for the optimizer to learn. Tunable values must be "
            f"nn.Parameters, not plain ints/floats or fixed banks."
        )
        assert hasattr(op, 'params') and len(op.params) > 0, (
            f"{name}: op.params dict is missing or empty. Expose every learnable "
            f"parameter in self.params so the tests below can check each one."
        )


# ============================================================
# 1. Gradient flow tests
# ============================================================

class TestGradientFlow:
    """Verify that backward pass produces non-None gradients for ALL learnable params."""

    @pytest.mark.parametrize("name,cls,kwargs", DIFF_ENHANCE_OPS,
                             ids=[x[0] for x in DIFF_ENHANCE_OPS])
    def test_all_params_receive_gradients(self, name, cls, kwargs, rgb_image):
        """Every learnable param must receive non-zero gradients after backward()."""
        op = cls(**kwargs)
        op.zero_grad()
        _run_forward_backward(op, rgb_image)

        for param_name, param in op.params.items():
            assert param.grad is not None, (
                f"{name}: param '{param_name}' has None gradient after backward pass! "
                f"This means gradients are NOT flowing through this parameter."
            )
            assert param.grad.abs().sum() > 0, (
                f"{name}: param '{param_name}' has all-zero gradient. "
                f"Gradients reach the param but are zero — check the forward computation."
            )


# ============================================================
# 2. Output shape tests
# ============================================================

class TestOutputShape:
    """Verify output tensors have expected shapes."""

    @pytest.mark.parametrize("name,cls,kwargs", DIFF_ENHANCE_OPS,
                             ids=[x[0] for x in DIFF_ENHANCE_OPS])
    def test_shape_preserved(self, name, cls, kwargs, rgb_image):
        """Output shape must match input shape."""
        op = cls(**kwargs)
        output = op(rgb_image)
        assert output.shape == rgb_image.shape, (
            f"{name}: expected output shape {rgb_image.shape}, got {output.shape}"
        )


# ============================================================
# 3. No NaN/Inf tests
# ============================================================

class TestNoNanInf:
    """Verify no NaN or Inf values in outputs or gradients."""

    @pytest.mark.parametrize("name,cls,kwargs", DIFF_ENHANCE_OPS,
                             ids=[x[0] for x in DIFF_ENHANCE_OPS])
    def test_output_finite(self, name, cls, kwargs, rgb_image):
        """Output must contain only finite values (no NaN, Inf)."""
        op = cls(**kwargs)
        output = op(rgb_image)
        assert torch.isfinite(output).all(), (
            f"{name}: output contains NaN or Inf values"
        )

    @pytest.mark.parametrize("name,cls,kwargs", DIFF_ENHANCE_OPS,
                             ids=[x[0] for x in DIFF_ENHANCE_OPS])
    def test_gradients_finite(self, name, cls, kwargs, rgb_image):
        """Gradients must be finite (no NaN, Inf)."""
        op = cls(**kwargs)
        op.zero_grad()
        _run_forward_backward(op, rgb_image)

        for param_name, param in op.params.items():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"{name}: param '{param_name}' has NaN/Inf in gradient"
                )


# ============================================================
# 4. Convergence tests
# ============================================================

class TestConvergence:
    """
    Verify that gradient descent can actually optimize each op's parameters.

    Strategy: apply op with known target params to get a target image,
    then start from different params and optimize toward the target.
    Loss should decrease significantly.
    """

    def test_brightness_convergence(self, rgb_image):
        op = DiffBrightness()
        with torch.no_grad():
            target_op = DiffBrightness()
            target_op.params['factor'].data.fill_(0.5)
            target = target_op(rgb_image)
        initial_loss, final_loss = _run_convergence(op, rgb_image, target)
        assert final_loss < initial_loss * 0.1, (
            f"DiffBrightness: loss didn't converge. Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
        )

    def test_contrast_convergence(self, rgb_image):
        op = DiffContrast()
        with torch.no_grad():
            target_op = DiffContrast()
            target_op.params['factor'].data.fill_(1.8)
            target = target_op(rgb_image)
        initial_loss, final_loss = _run_convergence(op, rgb_image, target)
        assert final_loss < initial_loss * 0.1, (
            f"DiffContrast: loss didn't converge. Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
        )

    def test_gamma_convergence(self, rgb_image):
        op = DiffGamma()
        with torch.no_grad():
            target_op = DiffGamma()
            target_op.params['gamma'].data.fill_(0.7)
            target = target_op(rgb_image)
        initial_loss, final_loss = _run_convergence(op, rgb_image, target)
        assert final_loss < initial_loss * 0.1, (
            f"DiffGamma: loss didn't converge. Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
        )

    def test_saturation_convergence(self, rgb_image):
        op = DiffSaturation()
        with torch.no_grad():
            target_op = DiffSaturation()
            target_op.params['factor'].data.fill_(0.3)
            target = target_op(rgb_image)
        initial_loss, final_loss = _run_convergence(op, rgb_image, target)
        assert final_loss < initial_loss * 0.1, (
            f"DiffSaturation: loss didn't converge. Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
        )

    def test_brightness_additive_convergence(self, rgb_image):
        op = DiffBrightnessAdditive()
        with torch.no_grad():
            target_op = DiffBrightnessAdditive()
            target_op.params['offset'].data.fill_(0.2)
            target = target_op(rgb_image)
        initial_loss, final_loss = _run_convergence(op, rgb_image, target)
        assert final_loss < initial_loss * 0.1, (
            f"DiffBrightnessAdditive: loss didn't converge. Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
        )

    def test_frangi_convergence(self, gray_image):
        op = DiffFrangi()
        with torch.no_grad():
            target_op = DiffFrangi()
            target_op.params['sigma'].data.fill_(3.5)
            target = target_op(gray_image)
        initial_loss, final_loss = _run_convergence(op, gray_image, target, lr=0.05)
        assert final_loss < initial_loss * 0.5, (
            f"DiffFrangi: loss didn't converge. Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
        )

    def test_morphology_disk_convergence(self, gray_image):
        op = DiffMorphologyDisk(operation="closing")
        with torch.no_grad():
            target_op = DiffMorphologyDisk(operation="closing")
            target_op.params['radius'].data.fill_(1.0)
            target = target_op(gray_image)
        initial_loss, final_loss = _run_convergence(op, gray_image, target, lr=0.05)
        assert final_loss < initial_loss * 0.5, (
            f"DiffMorphologyDisk: loss didn't converge. Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
        )

    def test_morphology_line_convergence(self, gray_image):
        op = DiffMorphologyLine(operation="closing")
        op.params['angle'].data.fill_(1.2)
        op.params['length'].data.fill_(2.0)
        with torch.no_grad():
            target_op = DiffMorphologyLine(operation="closing")
            target_op.params['angle'].data.fill_(1.571)
            target_op.params['length'].data.fill_(3.5)
            target = target_op(gray_image)
        initial_loss, final_loss = _run_convergence(op, gray_image, target,
                                                     max_epochs=200, lr=0.05)
        assert final_loss < initial_loss * 0.5, (
            f"DiffMorphologyLine: loss didn't converge. Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
        )