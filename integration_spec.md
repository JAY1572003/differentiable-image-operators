# Integration spec — final deliverable (deadline: August 15)

The goal of the remaining project time is to make your operators usable inside
our pipeline system. No new operators. The deliverable is a folder that we can
drop into our codebase with minimal changes. Requirements below; each one
exists because our system depends on it.

## 1. Base class

Use `operations/operation.py` **from the kit (v2) unchanged** — not your
rewritten version. Your operators must work with its interface:

- Learnable values live in `self.params` as **1-element tensors**:
  `self.params['radius'] = torch.nn.Parameter(torch.tensor([2.0]))`
  (not 0-d `torch.tensor(2.0)`).
- Every param has an entry in `self.param_ranges`.
- `op.set_params({'radius': [2.0]})` must work for every param — our system
  uses this to inject LLM-proposed initial values.
- `forward()` starts with `self.clamp_params()`.

## 2. Parameter names are a contract

Our LLM prompt schema references params by exact name, and `set_params`
**silently ignores unknown keys** — a name mismatch means the LLM's values are
silently dropped. So: for each operator, document in the final report a table
of: operator name, param names, ranges, defaults. These names are final once
delivered.

## 3. Forward-pass conventions

- Input and output: float tensors, shape (B, C, H, W), values in [0, 1],
  output same shape as input.
- **Remove the per-image min-max normalization** (`(out - min) / (max - min)`)
  from the operator outputs. It makes the output depend on global image
  content, which breaks composition with downstream operators. If an
  operator's raw response can leave [0, 1], control it with a learnable
  `scale` parameter or a fixed constant, and document the expected range.
- Keep 2–3 learnable params per operator.

## 4. Known gradient rules (from our codebase)

- No kernel sizes derived from a learnable param via `int(param.item())`
  where avoidable: fix the kernel size at construction (large enough for the
  max of the param range) and let the learnable value appear only inside the
  analytic kernel formula. (Your Frangi's `int(6 * sig.item())` should become
  a fixed size derived from `param_ranges['sigma'][1]`.)
- Angular parameters: keep the analytic sigmoid construction you have; the
  convergence test must pass from at least 3 different angle initializations.

## 5. Operators to deliver

Priority A — make the existing operators deliverable (do this first):

1. `DiffMultiscaleContrast` (sigma_small, sigma_large)
2. `DiffMorphologyDisk` (radius) — keep your log-sum-exp core if you prefer,
   but document the temperature constant and remove output normalization
3. `DiffMorphologyLine` (length, angle)
4. `DiffFrangi` (sigma, beta, gamma) — see the functional check below; in its
   current state it does not satisfy it

Priority B — new work, once A is in good shape:

5. `DiffHaralick` — differentiable Haralick/GLCM texture features. The two
   gradient blockers in the standard version are the hard quantization into
   gray-level bins and the discrete counting in the co-occurrence matrix; the
   known direction is soft binning (soft assignment of each pixel to bin
   centers, so the co-occurrence matrix becomes a sum of products of soft
   memberships). Scope: one operator, 2-3 learnable params (e.g. bin
   softness, displacement scale), outputting 1-3 texture-statistic maps
   (contrast / homogeneity / energy). Check Kornia and other libraries first.
   Evaluate on the carpet/cut pairs you received (texture-disruption defect —
   this is the category the operator exists for). If soft binning turns out
   not to work, a documented negative result with evidence (what was tried,
   where the gradient dies, measurements) is an acceptable outcome — an
   undocumented "it didn't work" is not.

## 6. Tests (acceptance criteria)

- All four operators added to `DIFF_ENHANCE_OPS` in the kit test suite.
- One convergence test per operator (angle: from 3+ inits).
- **Functional check per operator**: a small script that builds a synthetic
  image (uniform bright background, one thin dark line) and shows the
  operator's mean response ON the line is at least 5x the mean response OFF
  the line, at default parameters. An operator that cannot see a synthetic
  crack cannot learn to see a real one — run this check FIRST when training
  seems stuck; it separates "operator is blind" from "optimization problem".
- Full suite green. Include the pytest output and the functional-check output
  in the final report.

## 7. README in the delivered folder (this is what I need — keep it short)

- Per-operator: param table (names, ranges, defaults) — these names go
  directly into our LLM schema, so they must exactly match the code — plus
  train/test IoU on the tile/crack split (train 0–3, test 4–5, BCEDice, no
  post-processing). The real numbers, whatever they are.
- Haralick: what exists in libraries, your soft-binning design in a few
  sentences, results on carpet/cut under the same protocol — or the
  documented negative result per section 5.
- Pytest output and functional-check output pasted at the bottom.

The formal project report you submit to the university by August 15 is a
separate document and your own responsibility; reuse whatever you like from
this README for it.