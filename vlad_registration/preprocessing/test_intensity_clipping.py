# Basic validation for clip_intensity on one real LSFM/PS-OCT section
# (subject 097). Not a full run of the comparison matrix - just confirms
# the function behaves correctly before teammates build on it: shape/dtype
# preserved, enabled=False is a true no-op, every requested percentile
# range runs without error and produces values in [0, 1].
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "voxelmorph"))
from mrb_dataset import load_subject  # noqa: E402

from intensity_clipping_preprocessing import PRESETS, clip_intensity, apply_steps  # noqa: E402

SUBJECT = "097"


def check(label, cond):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {label}")
    assert cond, label


def main():
    print(f"loading subject {SUBJECT}...")
    d = load_subject(SUBJECT)
    lsfm, psoct = d["lsfm"], d["psoct"]

    for name, arr in [("lsfm", lsfm), ("psoct", psoct)]:
        print(f"\n--- {name}: shape={arr.shape} dtype={arr.dtype} ---")

        # disabled = true no-op (identical values, just cast to float32)
        passthrough = clip_intensity(arr, enabled=False)
        check("disabled: shape preserved", passthrough.shape == arr.shape)
        check("disabled: values unchanged", np.array_equal(passthrough, arr.astype(np.float32)))

        for label, pct_range in PRESETS.items():
            out = clip_intensity(arr, enabled=True, percentile_range=pct_range)
            check(f"{label} {pct_range}: shape preserved", out.shape == arr.shape)
            check(f"{label} {pct_range}: dtype is float32", out.dtype == np.float32)
            check(f"{label} {pct_range}: no NaN/Inf", np.isfinite(out).all())
            check(f"{label} {pct_range}: values in [0, 1]", out.min() >= 0.0 and out.max() <= 1.0)

        # a custom (non-preset) range should also work
        custom = clip_intensity(arr, percentile_range=(0.5, 99.5))
        check("custom (0.5, 99.5): shape preserved", custom.shape == arr.shape)

        # invalid range should raise, not silently misbehave
        try:
            clip_intensity(arr, percentile_range=(99, 1))
            check("invalid range raises ValueError", False)
        except ValueError:
            check("invalid range raises ValueError", True)

        # chains with a second no-op step via apply_steps, same convention
        # a teammate's own preprocessing function would follow
        chained = apply_steps(arr, [(clip_intensity, {"percentile_range": (2, 98)}), lambda x: x])
        direct = clip_intensity(arr, percentile_range=(2, 98))
        check("apply_steps matches direct call", np.array_equal(chained, direct))

    print("\nall checks passed")


if __name__ == "__main__":
    main()
