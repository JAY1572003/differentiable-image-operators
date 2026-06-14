"""
Shared fixtures for student differentiable op tests.

Adds the param_optimizer root to sys.path so that imports like
`from operations.differentiable_enhance import DiffBrightness` work.
"""

import sys
import os
import pytest
import torch

# Add param_optimizer root to path (matches how the main code imports)
PARAM_OPTIMIZER_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PARAM_OPTIMIZER_ROOT)


@pytest.fixture
def rgb_image():
    """Dummy RGB image: (1, 3, 32, 32), values in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(1, 3, 32, 32)


@pytest.fixture
def gray_image():
    """Dummy grayscale image: (1, 1, 32, 32), values in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(1, 1, 32, 32)
