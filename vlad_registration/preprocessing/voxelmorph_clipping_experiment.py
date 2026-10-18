# Intensity-clipping preprocessing experiment for VoxelMorph, sections
# 077/099/103 (097 is excluded on purpose - its segmentation/validation
# work is assigned separately, and this script does not touch
# compare_097.py or anything 097-specific).
#
# Does not modify the existing voxelmorph_097.py / voxelmorph_loocv.py /
# compare_102_export.py pipeline, which other work already depends on.
# voxelmorph_refine() in voxelmorph_097.py hardcodes compare_097.normalize()
# (1-99 percentile clip) as its preprocessing step, so instead of editing
# that shared function this script defines its own training driver,
# voxelmorph_refine_custom(), which is identical except the preprocessing
# call is a parameter. That lets two conditions run through the exact same
# training code, differing only in which function preprocesses the images:
#   - baseline   : compare_097.normalize()      (the pipeline's existing,
#                  unmodified default - not literally "no preprocessing",
#                  but no *additional* preprocessing beyond what every other
#                  script here already gets for free)
#   - clipping   : intensity_clipping_preprocessing.clip_intensity() at a different
#                  percentile range (CLIP_PERCENTILE below)
#
# Usage:
#   python3 voxelmorph_clipping_experiment.py --smoke            # fast correctness check, section 077 only
#   python3 voxelmorph_clipping_experiment.py                    # full run, sections 077/099/103
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))
import compare_097 as c10  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "voxelmorph"))
from mrb_dataset import load_subject  # noqa: E402
from voxelmorph_loocv import fit_affine, affine_warp  # noqa: E402
from voxelmorph_097 import (  # noqa: E402
    VoxelMorphUNet, warp, ncc_loss, smoothness_loss, to_tensor,
    DOWNSAMPLE, VM_DOWNSAMPLE, LAMBDA_SMOOTH, LR, N_ITERS,
)

from intensity_clipping_preprocessing import clip_intensity  # noqa: E402

torch.manual_seed(7)
np.random.seed(7)

SECTIONS = ["077", "099", "103"]
CLIP_PERCENTILE = (2, 98)  # the one range used for this comparison; others available via clip_intensity directly
OUT_DIR = Path(__file__).resolve().parent / "clipping_experiment_results"


def dice_score(fixed_arr, moving_arr, thresh=0.02):
    # same thresholded tissue-silhouette overlap used in
    # voxelmorph/compare_102_export.py - no manual segmentation exists for
    # this dataset, so this approximates footprint overlap, not true
    # structure-level Dice
    f = c10.normalize(fixed_arr) > thresh
    m = c10.normalize(moving_arr) > thresh
    inter = np.logical_and(f, m).sum()
    return float(2.0 * inter / (f.sum() + m.sum() + 1e-8))


def invert_voxelmorph_point(target_xy, flow_full, n_iter=50, tol=1e-3):
    # same fixed-point iteration as voxelmorph/compare_102_export.py's
    # invert_voxelmorph_point - duplicated rather than imported since that
    # module is section-102-specific and this experiment is deliberately
    # kept independent of it
    from scipy.ndimage import map_coordinates
    q = np.array(target_xy, dtype=float)
    for _ in range(n_iter):
        dx = map_coordinates(flow_full[0], [[q[1]], [q[0]]], order=1, mode="nearest")[0]
        dy = map_coordinates(flow_full[1], [[q[1]], [q[0]]], order=1, mode="nearest")[0]
        mapped = np.array([q[0] + dx, q[1] + dy])
        residual = np.array(target_xy) - mapped
        if np.linalg.norm(residual) < tol:
            break
        q = q + residual
    return q


def voxelmorph_refine_custom(lsfm_affine_warped, psoct, preprocess_fn, n_iters, vm_downsample=VM_DOWNSAMPLE):
    # identical to voxelmorph_097.voxelmorph_refine except the fixed/moving
    # preprocessing call is a parameter instead of a hardcoded c10.normalize()
    fixed_full = preprocess_fn(psoct)
    moving_full = preprocess_fn(lsfm_affine_warped)

    fixed_small = fixed_full[::vm_downsample, ::vm_downsample]
    moving_small = moving_full[::vm_downsample, ::vm_downsample]
    fixed_t = to_tensor(fixed_small)
    moving_t = to_tensor(moving_small)

    net = VoxelMorphUNet()
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    t0 = time.time()
    for it in range(n_iters):
        opt.zero_grad()
        flow = net(torch.cat([moving_t, fixed_t], dim=1))
        warped = warp(moving_t, flow)
        loss = ncc_loss(fixed_t, warped) + LAMBDA_SMOOTH * smoothness_loss(flow)
        loss.backward()
        opt.step()
        if it % max(1, n_iters // 4) == 0 or it == n_iters - 1:
            print(f"      iter {it:4d}/{n_iters}: loss={loss.item():.4f}")
    print(f"      training took {time.time() - t0:.1f}s")

    with torch.no_grad():
        flow = net(torch.cat([moving_t, fixed_t], dim=1))
        flow_full = F.interpolate(flow, size=fixed_full.shape, mode="bilinear", align_corners=True) * vm_downsample
        warped_full = warp(to_tensor(moving_full), flow_full)

    return warped_full.squeeze().numpy(), flow_full.squeeze(0).numpy()


def run_condition(section_id, condition_name, preprocess_fn, affine_result, psoct, shape,
                   M, offset, psoct_px, lsfm_px, n_iters):
    print(f"    condition={condition_name} ...")
    vm_result, flow_full = voxelmorph_refine_custom(affine_result, psoct, preprocess_fn, n_iters)

    mi = c10.mattes_mi(psoct, vm_result)
    dice = dice_score(psoct, vm_result)

    psoct_px_ds = psoct_px / DOWNSAMPLE
    dists = []
    for k in range(len(lsfm_px)):
        pre = c10.invert_affine_point(lsfm_px[k], M, offset) / DOWNSAMPLE
        i0, j0 = pre
        j_f, i_f = invert_voxelmorph_point((j0, i0), flow_full)
        i_psoct, j_psoct = psoct_px_ds[k]
        dists.append(float(np.linalg.norm([i_f - i_psoct, j_f - j_psoct]) * DOWNSAMPLE))
    dists = np.array(dists)

    print(f"      MI={mi:.4f}  Dice={dice:.4f}  TRE mean={dists.mean():.1f}px "
          f"median={np.median(dists):.1f}px max={dists.max():.1f}px")

    return {
        "section": section_id,
        "condition": condition_name,
        "mattes_mi": float(mi),
        "dice": dice,
        "tre_mean_px": float(dists.mean()),
        "tre_median_px": float(np.median(dists)),
        "tre_max_px": float(dists.max()),
        "tre_px_per_point": dists.tolist(),
    }


def run_section(section_id, n_iters):
    print(f"\n=== section {section_id} ===")
    d = load_subject(section_id)
    lsfm_full, psoct_full = d["lsfm"], d["psoct"]
    psoct_px, lsfm_px = d["psoct_px"], d["lsfm_px"]

    psoct = psoct_full[::DOWNSAMPLE, ::DOWNSAMPLE]
    shape = psoct.shape
    M, offset = fit_affine(psoct_px, lsfm_px)
    affine_result = affine_warp(lsfm_full, M, offset, shape, DOWNSAMPLE)

    conditions = [
        ("baseline_normalize_1_99", c10.normalize),
        (f"intensity_clipping_{CLIP_PERCENTILE[0]}_{CLIP_PERCENTILE[1]}",
         lambda img: clip_intensity(img, percentile_range=CLIP_PERCENTILE)),
    ]

    rows = []
    for name, fn in conditions:
        rows.append(run_condition(section_id, name, fn, affine_result, psoct, shape,
                                   M, offset, psoct_px, lsfm_px, n_iters))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="fast correctness check: section 077 only, 20 iters")
    parser.add_argument("--sections", type=str, default=None, help="comma-separated section ids (default: 077,099,103)")
    parser.add_argument("--iters", type=int, default=None, help="override training iterations (default: pipeline's N_ITERS)")
    args = parser.parse_args()

    if args.smoke:
        sections = ["077"]
        n_iters = 20
    else:
        sections = args.sections.split(",") if args.sections else SECTIONS
        n_iters = args.iters if args.iters is not None else N_ITERS

    print(f"sections={sections}  n_iters={n_iters}  clip_percentile={CLIP_PERCENTILE}")

    OUT_DIR.mkdir(exist_ok=True)
    all_rows = []
    per_section = {}
    for sid in sections:
        rows = run_section(sid, n_iters)
        all_rows.extend(rows)
        per_section[sid] = rows

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({"clip_percentile": list(CLIP_PERCENTILE), "n_iters": n_iters, "sections": per_section}, f, indent=2)

    csv_path = OUT_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["section", "condition", "mattes_mi", "dice", "tre_mean_px", "tre_median_px", "tre_max_px"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nsaved: {OUT_DIR / 'results.json'}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
