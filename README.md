# Differentiable Crack-Detection Operators — Integration Deliverable

Delivered per `integration_spec.md`. Covers Priority A (4 operators, kit v2
interface). Priority B (Haralick) in progress — see bottom of this file.

## Operators

### DiffFrangi
Differentiable Frangi vesselness filter. Detects thin ridge-like structures
via Hessian eigenvalue analysis.

| Param | Range | Default |
|---|---|---|
| sigma | [0.5, 6.0] | 2.0 |
| beta | [0.1, 2.0] | 0.5 |
| gamma | [1.0, 20.0] | 5.0 |

Train/test IoU (train 0–3, test 4–5, BCEDice, no post-processing): **0.1226**

Note: an eigenvalue-ordering bug was found and fixed during this integration
pass — the operator was previously ordering eigenvalues by signed value
instead of absolute value, which silently zeroed its response on real
ridges. Confirmed via the functional check below (was 0.00x, now 500x).

### DiffMultiscaleContrast
Compares local mean/std at two Gaussian scales.

| Param | Range | Default |
|---|---|---|
| sigma_small | [0.5, 5.0] | 1.0 |
| sigma_large | [2.0, 8.0] | 5.0 |
| scale | [0.5, 20.0] | 5.0 |

Train/test IoU: **0.0895**

### DiffMorphologyDisk
Soft morphological bottom-hat with a learnable disk structuring element
(log-sum-exp dilation/erosion).

| Param | Range | Default |
|---|---|---|
| radius | [0.1, 7.0] | 2.0 |

Train/test IoU: **0.0353**

Note: se_size was increased from 7 to 15 during this pass. At se_size=7,
radius consistently trained toward its ceiling and stalled there — the
structuring element was too small to represent the crack thickness in
these images (840x840 px). At se_size=15 the parameter converges to a
stable value well inside its range instead of hitting the wall.

### DiffMorphologyLine
Soft morphological bottom-hat with a learnable line structuring element
(length + angle, built analytically).

| Param | Range | Default |
|---|---|---|
| length | [0.5, 5.0] | 3.0 |
| angle | [0.0, pi] | pi/4 |

Train/test IoU: **0.0031**

Note: this operator converges correctly (loss decreases smoothly and
plateaus, confirmed via gradient-level debugging), but IoU is low because
the operator has a single global angle for the whole image. Training tiles
contain cracks at different orientations, so one global angle cannot fit
all of them simultaneously — it converges to a compromise angle that
doesn't match either held-out test crack well. This is a design limitation
of a single-angle operator, not a training bug.

## Functional check (spec section 6)

Synthetic image: uniform bright background, one thin dark line. Checks
mean response ON the line is >= 5x the mean response OFF the line.

```
Functional check: response on crack line vs off it (need >= 5x)
DiffFrangi             | on-line=0.0766 | off-line=0.0002 | ratio=500.45x | PASS
DiffMultiscaleContrast | on-line=0.9955 | off-line=0.1254 | ratio=7.94x | PASS
DiffMorphologyDisk     | on-line=0.6972 | off-line=0.0000 | ratio=69718915.22x | PASS
DiffMorphologyLine     | on-line=0.6983 | off-line=0.0000 | ratio=69825822.11x | PASS
```

All 4 operators pass.

## Test suite (spec section 6)

56/56 tests passing: has-params, gradient-flow, output-shape, no-NaN/Inf,
and convergence (including DiffMorphologyLine angle convergence from 3
different starting angles, per spec).

```
============================= test session starts ==============================
platform darwin -- Python 3.14.5, pytest-9.0.3, pluggy-1.6.0
collected 56 items

test_differentiable_enhance_only.py::TestHasLearnableParams::test_has_learnable_params[DiffBrightness] PASSED [  1%]
test_differentiable_enhance_only.py::TestHasLearnableParams::test_has_learnable_params[DiffContrast] PASSED [  3%]
test_differentiable_enhance_only.py::TestHasLearnableParams::test_has_learnable_params[DiffGamma] PASSED [  5%]
test_differentiable_enhance_only.py::TestHasLearnableParams::test_has_learnable_params[DiffSaturation] PASSED [  7%]
test_differentiable_enhance_only.py::TestHasLearnableParams::test_has_learnable_params[DiffBrightnessAdditive] PASSED [  8%]
test_differentiable_enhance_only.py::TestHasLearnableParams::test_has_learnable_params[DiffFrangi] PASSED [ 10%]
test_differentiable_enhance_only.py::TestHasLearnableParams::test_has_learnable_params[DiffMultiscaleContrast] PASSED [ 12%]
test_differentiable_enhance_only.py::TestHasLearnableParams::test_has_learnable_params[DiffMorphologyDisk] PASSED [ 14%]
test_differentiable_enhance_only.py::TestHasLearnableParams::test_has_learnable_params[DiffMorphologyLine] PASSED [ 16%]
test_differentiable_enhance_only.py::TestGradientFlow::test_all_params_receive_gradients[DiffBrightness] PASSED [ 17%]
test_differentiable_enhance_only.py::TestGradientFlow::test_all_params_receive_gradients[DiffContrast] PASSED [ 19%]
test_differentiable_enhance_only.py::TestGradientFlow::test_all_params_receive_gradients[DiffGamma] PASSED [ 21%]
test_differentiable_enhance_only.py::TestGradientFlow::test_all_params_receive_gradients[DiffSaturation] PASSED [ 23%]
test_differentiable_enhance_only.py::TestGradientFlow::test_all_params_receive_gradients[DiffBrightnessAdditive] PASSED [ 25%]
test_differentiable_enhance_only.py::TestGradientFlow::test_all_params_receive_gradients[DiffFrangi] PASSED [ 26%]
test_differentiable_enhance_only.py::TestGradientFlow::test_all_params_receive_gradients[DiffMultiscaleContrast] PASSED [ 28%]
test_differentiable_enhance_only.py::TestGradientFlow::test_all_params_receive_gradients[DiffMorphologyDisk] PASSED [ 30%]
test_differentiable_enhance_only.py::TestGradientFlow::test_all_params_receive_gradients[DiffMorphologyLine] PASSED [ 32%]
test_differentiable_enhance_only.py::TestOutputShape::test_shape_preserved[DiffBrightness] PASSED [ 33%]
test_differentiable_enhance_only.py::TestOutputShape::test_shape_preserved[DiffContrast] PASSED [ 35%]
test_differentiable_enhance_only.py::TestOutputShape::test_shape_preserved[DiffGamma] PASSED [ 37%]
test_differentiable_enhance_only.py::TestOutputShape::test_shape_preserved[DiffSaturation] PASSED [ 39%]
test_differentiable_enhance_only.py::TestOutputShape::test_shape_preserved[DiffBrightnessAdditive] PASSED [ 41%]
test_differentiable_enhance_only.py::TestOutputShape::test_shape_preserved[DiffFrangi] PASSED [ 42%]
test_differentiable_enhance_only.py::TestOutputShape::test_shape_preserved[DiffMultiscaleContrast] PASSED [ 44%]
test_differentiable_enhance_only.py::TestOutputShape::test_shape_preserved[DiffMorphologyDisk] PASSED [ 46%]
test_differentiable_enhance_only.py::TestOutputShape::test_shape_preserved[DiffMorphologyLine] PASSED [ 48%]
test_differentiable_enhance_only.py::TestNoNanInf::test_output_finite[DiffBrightness] PASSED [ 50%]
test_differentiable_enhance_only.py::TestNoNanInf::test_output_finite[DiffContrast] PASSED [ 51%]
test_differentiable_enhance_only.py::TestNoNanInf::test_output_finite[DiffGamma] PASSED [ 53%]
test_differentiable_enhance_only.py::TestNoNanInf::test_output_finite[DiffSaturation] PASSED [ 55%]
test_differentiable_enhance_only.py::TestNoNanInf::test_output_finite[DiffBrightnessAdditive] PASSED [ 57%]
test_differentiable_enhance_only.py::TestNoNanInf::test_output_finite[DiffFrangi] PASSED [ 58%]
test_differentiable_enhance_only.py::TestNoNanInf::test_output_finite[DiffMultiscaleContrast] PASSED [ 60%]
test_differentiable_enhance_only.py::TestNoNanInf::test_output_finite[DiffMorphologyDisk] PASSED [ 62%]
test_differentiable_enhance_only.py::TestNoNanInf::test_output_finite[DiffMorphologyLine] PASSED [ 64%]
test_differentiable_enhance_only.py::TestNoNanInf::test_gradients_finite[DiffBrightness] PASSED [ 66%]
test_differentiable_enhance_only.py::TestNoNanInf::test_gradients_finite[DiffContrast] PASSED [ 67%]
test_differentiable_enhance_only.py::TestNoNanInf::test_gradients_finite[DiffGamma] PASSED [ 69%]
test_differentiable_enhance_only.py::TestNoNanInf::test_gradients_finite[DiffSaturation] PASSED [ 71%]
test_differentiable_enhance_only.py::TestNoNanInf::test_gradients_finite[DiffBrightnessAdditive] PASSED [ 73%]
test_differentiable_enhance_only.py::TestNoNanInf::test_gradients_finite[DiffFrangi] PASSED [ 75%]
test_differentiable_enhance_only.py::TestNoNanInf::test_gradients_finite[DiffMultiscaleContrast] PASSED [ 76%]
test_differentiable_enhance_only.py::TestNoNanInf::test_gradients_finite[DiffMorphologyDisk] PASSED [ 78%]
test_differentiable_enhance_only.py::TestNoNanInf::test_gradients_finite[DiffMorphologyLine] PASSED [ 80%]
test_differentiable_enhance_only.py::TestConvergence::test_brightness_convergence PASSED [ 82%]
test_differentiable_enhance_only.py::TestConvergence::test_contrast_convergence PASSED [ 83%]
test_differentiable_enhance_only.py::TestConvergence::test_gamma_convergence PASSED [ 85%]
test_differentiable_enhance_only.py::TestConvergence::test_saturation_convergence PASSED [ 87%]
test_differentiable_enhance_only.py::TestConvergence::test_brightness_additive_convergence PASSED [ 89%]
test_differentiable_enhance_only.py::TestConvergence::test_frangi_convergence PASSED [ 91%]
test_differentiable_enhance_only.py::TestConvergence::test_multiscale_contrast_convergence PASSED [ 92%]
test_differentiable_enhance_only.py::TestConvergence::test_morphology_disk_convergence PASSED [ 94%]
test_differentiable_enhance_only.py::TestConvergence::test_morphology_line_convergence[0.3] PASSED [ 96%]
test_differentiable_enhance_only.py::TestConvergence::test_morphology_line_convergence[1.2] PASSED [ 98%]
test_differentiable_enhance_only.py::TestConvergence::test_morphology_line_convergence[2.5] PASSED [100%]

============================== 56 passed in 2.10s ==============================
```

## Priority B — DiffHaralick

In progress. Investigation and results to be added here.
