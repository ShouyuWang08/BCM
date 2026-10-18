# 102 - runs all 4 methods (No transform / Affine / B-spline / VoxelMorph),
# then exports downsized PNGs + a results.json (per-landmark TRE in
# full-res px + Mattes MI per method) for an interactive opacity-blend
# comparison viewer. Does not modify any existing script - reuses
# mrb_dataset for loading, the generic fit_affine/affine_warp from
# voxelmorph_loocv.py, voxelmorph_097.voxelmorph_refine, and
# compare_097.py's normalize/bspline_refine/mattes_mi/no_transform/
# invert_affine_point/invert_bspline_point as-is.
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "baseline"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # mrb_dataset/voxelmorph_loocv/voxelmorph_097 now live one level up
import compare_097 as c10  # noqa: E402

from mrb_dataset import load_subject  # noqa: E402
from voxelmorph_loocv import fit_affine, affine_warp  # noqa: E402
from voxelmorph_097 import voxelmorph_refine  # noqa: E402

SUBJECT = "102"
DOWNSAMPLE = c10.DOWNSAMPLE
EXPORT_WIDTH = 900  # resize exported PNGs to this width for the web viewer
OUT_DIR = Path(__file__).resolve().parent / "compare_102_assets"


def invert_voxelmorph_point(target_xy, flow_full, n_iter=50, tol=1e-3):
    # same fixed-point iteration as compare_097.invert_bspline_point, just
    # sampling the dense flow field (bilinear) instead of calling a sitk
    # transform - flow_full is (2,H,W) = (dx, dy) in (x,y)-indexed order
    q = np.array(target_xy, dtype=float)
    for _ in range(n_iter):
        dx = map_coordinates(flow_full[0], [[q[1]], [q[0]]], order=1, mode='nearest')[0]
        dy = map_coordinates(flow_full[1], [[q[1]], [q[0]]], order=1, mode='nearest')[0]
        mapped = np.array([q[0] + dx, q[1] + dy])
        residual = np.array(target_xy) - mapped
        if np.linalg.norm(residual) < tol:
            break
        q = q + residual
    return q


def dice_score(fixed_arr, moving_arr, thresh=0.02):
    # coarse tissue-silhouette Dice via intensity threshold - there's no
    # manual segmentation in this dataset, so this approximates anatomical
    # footprint overlap using the hard-zero background both scans already
    # have (same convention normalize() uses to find "real" pixels)
    f = c10.normalize(fixed_arr) > thresh
    m = c10.normalize(moving_arr) > thresh
    inter = np.logical_and(f, m).sum()
    return float(2.0 * inter / (f.sum() + m.sum() + 1e-8))


def export_png(arr, path, scale):
    img = c10.normalize(arr)
    img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(img8, mode="L")
    new_size = (max(1, int(round(im.width * scale))), max(1, int(round(im.height * scale))))
    im = im.resize(new_size, Image.LANCZOS)
    im.save(path)
    return new_size


def main():
    OUT_DIR.mkdir(exist_ok=True)
    d = load_subject(SUBJECT)
    lsfm_full, psoct_full = d["lsfm"], d["psoct"]
    psoct_px, lsfm_px = d["psoct_px"], d["lsfm_px"]

    psoct = psoct_full[::DOWNSAMPLE, ::DOWNSAMPLE]
    shape = psoct.shape  # (i=row count, j=col count)
    scale = EXPORT_WIDTH / shape[1]

    M, offset = fit_affine(psoct_px, lsfm_px)
    affine_result = affine_warp(lsfm_full, M, offset, shape, DOWNSAMPLE)
    no_xform = c10.no_transform(lsfm_full, shape, DOWNSAMPLE)

    print("running bspline...")
    bspline_result, bspline_transform = c10.bspline_refine(affine_result, psoct)

    print("running voxelmorph...")
    vm_result, flow_full = voxelmorph_refine(affine_result, psoct)

    export_size = export_png(psoct, OUT_DIR / "fixed.png", scale)
    export_png(no_xform, OUT_DIR / "no_transform.png", scale)
    export_png(affine_result, OUT_DIR / "affine.png", scale)
    export_png(bspline_result, OUT_DIR / "bspline.png", scale)
    export_png(vm_result, OUT_DIR / "voxelmorph.png", scale)

    def landing_no_transform(p):
        return p / DOWNSAMPLE

    def landing_affine(p):
        return c10.invert_affine_point(p, M, offset) / DOWNSAMPLE

    def landing_bspline(p):
        pre = c10.invert_affine_point(p, M, offset) / DOWNSAMPLE
        i0, j0 = pre
        j_f, i_f = c10.invert_bspline_point((j0, i0), bspline_transform)
        return np.array([i_f, j_f])

    def landing_voxelmorph(p):
        pre = c10.invert_affine_point(p, M, offset) / DOWNSAMPLE
        i0, j0 = pre
        j_f, i_f = invert_voxelmorph_point((j0, i0), flow_full)
        return np.array([i_f, j_f])

    methods = {
        "no_transform": (no_xform, landing_no_transform),
        "affine": (affine_result, landing_affine),
        "bspline": (bspline_result, landing_bspline),
        "voxelmorph": (vm_result, landing_voxelmorph),
    }

    psoct_px_ds = psoct_px / DOWNSAMPLE  # (i, j) in the output/comparison grid
    results = {
        "subject": SUBJECT,
        "image_size": list(export_size),  # (width, height) px, matches the exported PNGs
        "psoct_points": [[j * scale, i * scale] for i, j in psoct_px_ds],  # (x, y) px in exported image
        "methods": {},
    }

    for name, (img, landing_fn) in methods.items():
        score = c10.mattes_mi(psoct, img)
        dice = dice_score(psoct, img)
        dists_full_px, landing_xy = [], []
        for k in range(len(lsfm_px)):
            i_land, j_land = landing_fn(lsfm_px[k])
            i_psoct, j_psoct = psoct_px_ds[k]
            dist_full_px = float(np.linalg.norm([i_land - i_psoct, j_land - j_psoct]) * DOWNSAMPLE)
            dists_full_px.append(dist_full_px)
            landing_xy.append([j_land * scale, i_land * scale])
        dists_full_px = np.array(dists_full_px)
        results["methods"][name] = {
            "mattes_mi": float(score),
            "dice": dice,
            "landing_points": landing_xy,
            "tre_px": dists_full_px.tolist(),
            "tre_mean": float(dists_full_px.mean()),
            "tre_median": float(np.median(dists_full_px)),
            "tre_max": float(dists_full_px.max()),
        }
        print(f"{name}: MI={score:.4f}  Dice={dice:.4f}  TRE mean={dists_full_px.mean():.1f}px "
              f"median={np.median(dists_full_px):.1f}px max={dists_full_px.max():.1f}px")

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved export to: {OUT_DIR}")


if __name__ == "__main__":
    main()
