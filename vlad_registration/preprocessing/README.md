# Preprocessing

Shared preprocessing steps for the LSFM/PS-OCT comparison matrix, meant to run
*before* an image reaches either registration pipeline
(`vlad_registration/baseline` or `vlad_registration/voxelmorph`). Neither
pipeline is modified by anything here - these are standalone array-in,
array-out functions you call on a loaded image before passing it along.

## Intensity clipping preprocessing (`intensity_clipping_preprocessing.py`)

Percentile-based intensity clipping: clips an image to a percentile range
computed over its positive-valued pixels, then rescales to `[0, 1]`. This is
the same computation `compare_097.normalize()` already does internally
(hardcoded to 1-99), generalized to any range and toggleable.

```python
from intensity_clipping_preprocessing import clip_intensity, PRESETS

out = clip_intensity(raw_img)                                  # default: 1-99
out = clip_intensity(raw_img, percentile_range=(2, 98))         # standard
out = clip_intensity(raw_img, percentile_range=PRESETS["aggressive"])  # 5-95
out = clip_intensity(raw_img, enabled=False)                    # no-op passthrough
```

`enabled=False` returns the image unchanged (cast to float32) so a step can
be switched off from one call site without touching the registration code
downstream of it.

### Using it ahead of the existing pipelines

Both pipelines already call `compare_097.normalize()` (fixed at 1-99)
internally, so the clean way to try a different percentile range is to
pre-clip the array yourself and feed that into the same entry points - no
edits to `compare_097.py`, `voxelmorph_097.py`, etc. needed:

```python
import sys
sys.path.insert(0, "../voxelmorph")
from mrb_dataset import load_subject
from intensity_clipping_preprocessing import clip_intensity

d = load_subject("097")
lsfm = clip_intensity(d["lsfm"], percentile_range=(2, 98))
psoct = clip_intensity(d["psoct"], percentile_range=(2, 98))
# lsfm/psoct now feed into fit_affine/affine_warp/bspline_refine/
# voxelmorph_refine exactly like the raw arrays did before
```

### Adding your own preprocessing step

Follow the same shape so steps compose: a function that takes an image array
as its first argument and returns a processed array of the same shape.
`apply_steps()` chains any mix of these in order:

```python
from intensity_clipping_preprocessing import clip_intensity, apply_steps

def my_denoise(img, strength=1.0):
    ...
    return img

out = apply_steps(raw_img, [
    (clip_intensity, {"percentile_range": (2, 98)}),
    (my_denoise, {"strength": 0.5}),
])
```

## Validation

`test_intensity_clipping.py` runs `clip_intensity` against one real section
(subject 097, both LSFM and PS-OCT) and checks: output shape matches input,
`enabled=False` is a true no-op, every preset range produces finite values in
`[0, 1]`, an invalid range raises `ValueError` instead of misbehaving
silently, and `apply_steps` matches calling the function directly.

```bash
cd vlad_registration/preprocessing
python3 test_intensity_clipping.py
```

This is basic validation on one subject, not a run of the full comparison
matrix.

## VoxelMorph experiment (`voxelmorph_clipping_experiment.py`)

Runs VoxelMorph on sections 077/099/103 under two conditions - the
pipeline's existing default (`compare_097.normalize()`, 1-99) vs this
module's `clip_intensity()` at a different range (2-98) - keeping every
other setting identical, and scores both with Mattes MI, Dice, and
per-landmark TRE. Does not modify `voxelmorph_097.py`: it defines its own
training driver that takes the preprocessing function as a parameter.

```bash
cd vlad_registration/preprocessing
python3 voxelmorph_clipping_experiment.py --smoke   # fast correctness check, section 077 only
python3 voxelmorph_clipping_experiment.py           # full run, sections 077/099/103
```

Results land in `clipping_experiment_results/`: `results.json` (full detail,
including per-point TRE) and `summary.csv` (one row per section x condition -
ready to drop into the team's 4x7 comparison matrix).
