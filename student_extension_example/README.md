# Student Extension Example: Differentiable Operators

This folder contains a **minimal, runnable example** for students to extend the differentiable operator vocabulary.

## What's included

- `conftest.py` — pytest fixtures (RGB/grayscale test images)
- `test_differentiable_enhance_only.py` — stripped test suite (enhance ops only)
- `../operations/` — the `Operation` base class and `differentiable_enhance.py`, the base module you'll extend (when shipped standalone; inside the full project this is the main operations package)

## How to add a new enhancement operator

### 1. Implement your operator in `differentiable_enhance.py`

```python
import torch
import torch.nn as nn

class DiffYourNewOp(nn.Module):
    """Your op description."""
    
    def __init__(self):
        super().__init__()
        # Define learnable parameters
        self.register_parameter('param_name', nn.Parameter(torch.tensor([1.0])))
        # Store as dict so tests can access them
        self.params = {'param_name': self.param_name}
    
    def forward(self, x):
        """Apply the operation to input tensor x (shape: [B, C, H, W])."""
        # Your differentiable computation here
        return result
    
    def set_params(self, param_dict):
        """Set parameter values from a dict."""
        for name, values in param_dict.items():
            if hasattr(self, name):
                with torch.no_grad():
                    getattr(self, name).copy_(torch.tensor(values))
```

### 2. Add your op to the test suite

In `test_differentiable_enhance_only.py`, add one line to `DIFF_ENHANCE_OPS`:

```python
DIFF_ENHANCE_OPS = [
    ("DiffBrightness", DiffBrightness, {}),
    ("DiffContrast", DiffContrast, {}),
    # ... existing ops ...
    ("DiffYourNewOp", DiffYourNewOp, {}),  # ← Add this line
]
```

### 3. Run the tests

```bash
pytest test_differentiable_enhance_only.py -v
```

Your op will automatically be tested for:
- ✅ **Has learnable params** — at least one nn.Parameter exists (grid-searched ints don't count)
- ✅ **Gradient flow** — gradients actually flow to all params
- ✅ **Output shape** — output shape matches input shape
- ✅ **Finite values** — no NaN/Inf in outputs or gradients

⚠️ **Convergence is NOT automatic** — you must add one small convergence test
per op in `TestConvergence` (copy any of the existing five as a template).
An op does not count as differentiable until its convergence test passes:
gradient descent must be able to move the params toward a target.

## What each test verifies

| Test | What it checks | Why it matters |
|---|---|---|
| `test_has_learnable_params` | `list(op.parameters())` is non-empty and `op.params` is populated | An op with no nn.Parameters gives the optimizer nothing to learn — even if the forward pass runs in PyTorch |
| `test_all_params_receive_gradients` | Every learnable param gets non-None, non-zero gradients | Catches detached params (like the old GaussianBlur2D bug). Note: gradients reaching the *image* is not the same as gradients reaching the *params* |
| `test_shape_preserved` | Output shape == input shape | Ops must not change image dimensions |
| `test_output_finite` | No NaN or Inf in output | Numeric stability |
| `test_gradients_finite` | No NaN or Inf in gradients | Optimizer won't diverge |
| `TestConvergence` (add yours manually) | Loss decreases when optimizing from wrong → target params | Proves the op is actually learnable. For non-convex params (angles, scales), test from several initializations |

## Tips for implementing a new op

1. **Use only differentiable PyTorch ops** — avoid `.numpy()`, `.item()`, discrete operations
2. **Clamp or normalize outputs** if needed — keep values in `[0, 1]` to avoid output explosion
3. **Avoid discontinuous operations** — e.g., `torch.round()` breaks gradients. Use soft approximations instead
4. **Test with small inputs first** — `(1, 3, 32, 32)` catches bugs faster than large tensors
5. **Add docstrings** — future students need to understand what your op does

## Example: Adding a simple gamma correction op

```python
class DiffGamma(nn.Module):
    """Gamma correction: x^(1/gamma)."""
    
    def __init__(self):
        super().__init__()
        self.register_parameter('gamma', nn.Parameter(torch.tensor([1.0])))
        self.params = {'gamma': self.gamma}
    
    def forward(self, x):
        # Clamp to avoid log(0)
        x_safe = torch.clamp(x, min=1e-6)
        # Apply gamma: x^(1/gamma)
        gamma_safe = torch.clamp(self.gamma, min=0.1, max=2.0)
        return torch.pow(x_safe, 1.0 / gamma_safe)
    
    def set_params(self, param_dict):
        if 'gamma' in param_dict:
            with torch.no_grad():
                self.gamma.copy_(torch.tensor(param_dict['gamma']))
```

Then add to test:
```python
DIFF_ENHANCE_OPS = [
    # ... others ...
    ("DiffGamma", DiffGamma, {}),
]
```

Run:
```bash
pytest test_differentiable_enhance_only.py::TestConvergence::test_gamma_convergence -v
```

## Common issues

| Issue | Cause | Fix |
|---|---|---|
| `param has None gradient` | Op doesn't use the param in forward() | Check forward() computation uses `self.param` |
| `param has all-zero gradient` | Param is used but doesn't affect loss | Check the computation is differentiable |
| `NaN or Inf in output` | Division by zero, log(negative), etc. | Add clamping/safeguards in forward() |
| `loss doesn't decrease` | Optimizer is too slow or learning rate is wrong | Try higher `lr` in `_run_convergence()` |

## Files to share with others

Send these **3 files** to let others extend the op vocabulary:
1. `conftest.py` — pytest fixtures
2. `test_differentiable_enhance_only.py` — test suite
3. `differentiable_enhance.py` — base ops (copy from parent folder)

```bash
# Example: copy the files
cp ../differentiable_enhance.py .
# Now others can run: pytest test_differentiable_enhance_only.py
```