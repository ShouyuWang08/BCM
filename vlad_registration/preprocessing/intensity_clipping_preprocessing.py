# Percentile-based intensity clipping - a reusable preprocessing step for
# the LSFM/PS-OCT comparison matrix, usable ahead of either the baseline
# (vlad_registration/baseline) or VoxelMorph (vlad_registration/voxelmorph)
# registration pipelines. Does NOT modify either pipeline: call it on a
# loaded image array before handing that array to fit_affine/affine_warp/
# bspline_refine/voxelmorph_refine, same way compare_097.normalize() is
# already used internally by those pipelines (1-99 percentile, hardcoded).
# This generalizes that same computation to any percentile range and lets
# it be switched off entirely without touching caller code.
import numpy as np

# named presets for the comparison matrix, so rows can reference a label
# instead of repeating raw percentile tuples
PRESETS = {
    "wide": (1, 99),
    "standard": (2, 98),
    "aggressive": (5, 95),
}


def clip_intensity(img, enabled=True, percentile_range=(1, 99)):
    """Percentile-based intensity clipping.

    Clips `img` to the [lo_pct, hi_pct] percentile range computed over its
    positive-valued pixels (these LSFM/PS-OCT scans use hard 0 as the
    background value, same convention compare_097.normalize relies on),
    then rescales the clipped range to [0, 1]. Image dimensions are never
    changed - this is a pure intensity-value transform on the array you
    pass in, and it never reads or writes any metadata (e.g. an nrrd
    header), so anything a caller tracks separately from the array is
    simply untouched by this call.

    Args:
        img: 2D (or ND) array-like of raw intensities.
        enabled: if False, returns img unchanged (cast to float32) - lets a
            pipeline toggle this step on/off without any other code change.
        percentile_range: (lo_pct, hi_pct), e.g. (1, 99), (2, 98), (5, 95),
            or any custom pair. See PRESETS for named shortcuts.

    Returns:
        float32 array, same shape as img.
    """
    img = np.asarray(img, dtype=np.float32)
    if not enabled:
        return img

    lo_pct, hi_pct = percentile_range
    if not (0 <= lo_pct < hi_pct <= 100):
        raise ValueError(f"percentile_range must satisfy 0 <= lo < hi <= 100, got {percentile_range}")

    positive = img[img > 0]
    if positive.size == 0:
        return np.zeros_like(img)

    lo, hi = np.percentile(positive, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(img)

    return np.clip((img - lo) / (hi - lo), 0, 1)


def apply_steps(img, steps):
    """Apply a sequence of preprocessing steps in order.

    Shared convention for the team's preprocessing comparison matrix: each
    step is either a bare callable `img -> img`, or an `(callable, kwargs)`
    tuple. Every step in this module (and any teammate adds) should follow
    this array-in/array-out shape so steps compose by simple chaining:

        from intensity_clipping import clip_intensity, apply_steps
        out = apply_steps(raw_img, [
            (clip_intensity, {"percentile_range": (2, 98)}),
            teammates_denoise_fn,   # e.g. a bare callable, no extra kwargs
        ])
    """
    out = np.asarray(img, dtype=np.float32)
    for step in steps:
        if isinstance(step, tuple):
            fn, kwargs = step
            out = fn(out, **kwargs)
        else:
            out = step(out)
    return out
