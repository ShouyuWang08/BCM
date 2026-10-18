# Generic loader for the Slicer .mrb scene bundles in data/mrb/. Each .mrb is
# just a zip containing one LSFM volume, one PSOCT volume, and matching
# LSFM.mrk.json / PSOCT.mrk.json fiducial lists (20 points each, same Slicer
# template used for V_097_20pts.mrb). The internal folder naming is NOT
# consistent across bundles (e.g. "V_77/Data/...", "100/Data/...",
# "V_98mrb/Data/..."), so files are found by suffix/substring match rather
# than a fixed path. Extracted nrrd files are cached on disk next to data/mrb
# (data/extracted/<id>/...) since re-unzipping ~50-180MB members on every
# load would dominate runtime once this is used in a training loop.
import json
import re
import zipfile
from pathlib import Path

import nrrd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MRB_DIR = REPO_ROOT / "data" / "mrb"
CACHE_DIR = REPO_ROOT / "data" / "extracted"

# subject_id -> mrb filename
SUBJECTS = {
    "097": "V_097_20pts.mrb",
    "077": "V_77.mrb",
    "078": "V_78.mrb",
    "079": "V_79.mrb",
    "096": "V_96.mrb",
    "098": "V_98.mrb",
    "099": "V_99.mrb",
    "100": "V_100.mrb",
    "101": "V_101.mrb",
    "102": "V_102.mrb",
    "103": "V_103.mrb",
}


def _find_member(names, must_contain=None, must_end=None):
    matches = [
        n for n in names
        if (must_contain is None or must_contain in n)
        and (must_end is None or n.endswith(must_end))
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly 1 match (contain={must_contain!r}, end={must_end!r}), got {matches}")
    return matches[0]


def _extract_cached(zf, member, subject_id):
    out_path = CACHE_DIR / subject_id / Path(member).name
    if not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(zf.read(member))
    return out_path


def _load_points(zf, member):
    data = json.loads(zf.read(member))
    control_points = data["markups"][0]["controlPoints"]
    # label order (LSFM-1, LSFM-2, ...) is what pairs an LSFM point with its
    # PSOCT counterpart of the same index - sort explicitly rather than
    # trusting file order, in case Slicer ever reorders them on save
    def label_idx(cp):
        m = re.search(r"(\d+)$", cp["label"])
        return int(m.group(1)) if m else 0
    control_points = sorted(control_points, key=label_idx)
    return np.array([cp["position"][:2] for cp in control_points])


def load_subject(subject_id):
    # returns dict with:
    #   lsfm, psoct      - float32 2D arrays (moving, fixed)
    #   lsfm_px, psoct_px - (20, 2) pixel-space control points, index-matched
    mrb_path = MRB_DIR / SUBJECTS[subject_id]
    with zipfile.ZipFile(mrb_path) as zf:
        names = zf.namelist()
        lsfm_nrrd_member = next(
            (n for n in names if n.endswith(".nrrd") and ("LSFM" in n or "488_small" in n)), None
        )
        psoct_nrrd_member = next((n for n in names if n.endswith("_R.nrrd")), None)
        lsfm_mrk_member = _find_member(names, must_end="LSFM.mrk.json")
        psoct_mrk_member = _find_member(names, must_end="PSOCT.mrk.json")
        if lsfm_nrrd_member is None or psoct_nrrd_member is None:
            raise ValueError(f"{subject_id}: couldn't identify LSFM/PSOCT nrrd members in {names}")

        lsfm_path = _extract_cached(zf, lsfm_nrrd_member, subject_id)
        psoct_path = _extract_cached(zf, psoct_nrrd_member, subject_id)
        lsfm_px = _load_points(zf, lsfm_mrk_member)
        psoct_px = _load_points(zf, psoct_mrk_member)

    lsfm_arr, _ = nrrd.read(str(lsfm_path))
    psoct_arr, _ = nrrd.read(str(psoct_path))
    return {
        "lsfm": np.squeeze(lsfm_arr).astype(np.float32),
        "psoct": np.squeeze(psoct_arr).astype(np.float32),
        "lsfm_px": lsfm_px,
        "psoct_px": psoct_px,
    }


if __name__ == "__main__":
    # quick smoke test: load every subject, print shapes/point counts
    for sid in SUBJECTS:
        d = load_subject(sid)
        print(f"{sid}: lsfm={d['lsfm'].shape} psoct={d['psoct'].shape} "
              f"n_points=({len(d['lsfm_px'])},{len(d['psoct_px'])})")
