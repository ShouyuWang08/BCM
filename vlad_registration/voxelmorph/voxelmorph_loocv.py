# VoxelMorph with leave-one-out cross-validation (LOOCV) across the 11
# subjects in data/mrb/ (097 + the 10 new pairs). Each fold trains one
# network from scratch on 10 subjects and evaluates it on the 1 held-out
# subject it never saw during training - this is what makes VoxelMorph's
# reported score a genuine test of generalization, unlike the B-spline /
# affine / TPS baselines, which optimize directly on the pair being scored
# (not a flaw in this comparison, just a different operating mode - both
# numbers are printed side by side so that distinction stays visible).
#
# Hyperparameters (N_ITERS_PER_FOLD, LAMBDA_SMOOTH, LR, VM_DOWNSAMPLE) are
# fixed across every fold, reusing the defaults already validated in
# voxelmorph_097.py - they are NOT tuned per fold, so the held-out score is
# never used to pick the hyperparameters that produced it.
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))
import compare_097 as c10  # reused as-is: normalize(), no_transform(), mattes_mi(), bspline_refine()

from mrb_dataset import SUBJECTS, load_subject
from voxelmorph_097 import (
    VoxelMorphUNet, warp, ncc_loss, smoothness_loss, to_tensor,
    DOWNSAMPLE, VM_DOWNSAMPLE, LAMBDA_SMOOTH, LR,
)

torch.manual_seed(7)
np.random.seed(7)

N_ITERS_PER_FOLD = 2000  # fixed across all folds - see module docstring


def fit_affine(psoct_px, lsfm_px):
    n = psoct_px.shape[0]
    X = np.hstack([psoct_px, np.ones((n, 1))])
    A, *_ = np.linalg.lstsq(X, lsfm_px, rcond=None)
    M = np.array([[A[0, 0], A[1, 0]], [A[0, 1], A[1, 1]]])
    offset = A[2, :]
    return M, offset


def affine_warp(lsfm_full, M, offset, out_shape, downsample):
    ii, jj = np.meshgrid(
        np.arange(out_shape[0]) * downsample,
        np.arange(out_shape[1]) * downsample,
        indexing='ij',
    )
    coords = np.stack([ii.ravel(), jj.ravel()], axis=1)
    mapped = coords @ M.T + offset
    i_lsfm = mapped[:, 0].reshape(out_shape)
    j_lsfm = mapped[:, 1].reshape(out_shape)
    return ndimage.map_coordinates(lsfm_full, [i_lsfm, j_lsfm], order=1, mode='constant', cval=0)


def prepare_pair(subject_id):
    # raw-intensity psoct (fixed) + affine-pre-aligned lsfm (moving), both at
    # the DOWNSAMPLE comparison grid - same convention as baseline/compare_097.py
    d = load_subject(subject_id)
    psoct = d["psoct"][::DOWNSAMPLE, ::DOWNSAMPLE]
    shape = psoct.shape
    M, offset = fit_affine(d["psoct_px"], d["lsfm_px"])
    affine_result = affine_warp(d["lsfm"], M, offset, shape, DOWNSAMPLE)
    return {"lsfm_full": d["lsfm"], "psoct": psoct, "affine_result": affine_result, "shape": shape}


def train_fold(train_pairs, n_iters=N_ITERS_PER_FOLD):
    net = VoxelMorphUNet()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    ids = list(train_pairs.keys())
    t0 = time.time()
    for it in range(n_iters):
        sid = ids[np.random.randint(len(ids))]
        pair = train_pairs[sid]
        fixed_full = c10.normalize(pair["psoct"])
        moving_full = c10.normalize(pair["affine_result"])
        fixed_small = fixed_full[::VM_DOWNSAMPLE, ::VM_DOWNSAMPLE]
        moving_small = moving_full[::VM_DOWNSAMPLE, ::VM_DOWNSAMPLE]
        fixed_t = to_tensor(fixed_small)
        moving_t = to_tensor(moving_small)

        opt.zero_grad()
        flow = net(torch.cat([moving_t, fixed_t], dim=1))
        warped = warp(moving_t, flow)
        loss = ncc_loss(fixed_t, warped) + LAMBDA_SMOOTH * smoothness_loss(flow)
        loss.backward()
        opt.step()
        if it % 200 == 0 or it == n_iters - 1:
            print(f'    iter {it:4d}/{n_iters} (subject {sid}): loss={loss.item():.4f}')
    print(f'    fold training took {time.time() - t0:.1f}s over {n_iters} iters / {len(ids)} subjects')
    return net


def apply_net(net, pair):
    fixed_full = c10.normalize(pair["psoct"])
    moving_full = c10.normalize(pair["affine_result"])
    fixed_small = fixed_full[::VM_DOWNSAMPLE, ::VM_DOWNSAMPLE]
    moving_small = moving_full[::VM_DOWNSAMPLE, ::VM_DOWNSAMPLE]
    with torch.no_grad():
        flow = net(torch.cat([to_tensor(moving_small), to_tensor(fixed_small)], dim=1))
        flow_full = F.interpolate(flow, size=fixed_full.shape, mode='bilinear', align_corners=True) * VM_DOWNSAMPLE
        warped_full = warp(to_tensor(moving_full), flow_full)
    return warped_full.squeeze().numpy()


def run_loocv(subject_ids, n_iters=N_ITERS_PER_FOLD):
    print("preparing all pairs (affine pre-alignment)...")
    pairs = {sid: prepare_pair(sid) for sid in subject_ids}

    rows = []
    for held_out in subject_ids:
        print(f"\n=== fold: held-out subject {held_out} ===")
        train_pairs = {sid: p for sid, p in pairs.items() if sid != held_out}
        net = train_fold(train_pairs, n_iters)

        test_pair = pairs[held_out]
        no_xform = c10.no_transform(test_pair["lsfm_full"], test_pair["shape"], DOWNSAMPLE)
        bspline_result, _ = c10.bspline_refine(test_pair["affine_result"], test_pair["psoct"])
        vm_result = apply_net(net, test_pair)

        scores = {
            "subject": held_out,
            "no_transform": c10.mattes_mi(test_pair["psoct"], no_xform),
            "affine": c10.mattes_mi(test_pair["psoct"], test_pair["affine_result"]),
            "bspline": c10.mattes_mi(test_pair["psoct"], bspline_result),
            "voxelmorph_loocv": c10.mattes_mi(test_pair["psoct"], vm_result),
        }
        rows.append(scores)
        print(f"    {held_out}: no_xform={scores['no_transform']:.4f}  affine={scores['affine']:.4f}  "
              f"bspline={scores['bspline']:.4f}  voxelmorph(held-out)={scores['voxelmorph_loocv']:.4f}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=N_ITERS_PER_FOLD,
                         help="training iterations per fold (fixed across folds, not tuned per fold)")
    parser.add_argument("--subjects", type=str, default=None,
                         help="comma-separated subject ids to run folds over (default: all 11)")
    parser.add_argument("--out", type=str, default="loocv_results.csv")
    args = parser.parse_args()

    subject_ids = args.subjects.split(",") if args.subjects else list(SUBJECTS.keys())
    rows = run_loocv(subject_ids, n_iters=args.iters)

    print("\n=== LOOCV summary (Mattes MI, more negative = better fit) ===")
    print(f"{'subject':>8} {'no_xform':>10} {'affine':>10} {'bspline':>10} {'voxelmorph':>12}")
    for r in rows:
        print(f"{r['subject']:>8} {r['no_transform']:>10.4f} {r['affine']:>10.4f} "
              f"{r['bspline']:>10.4f} {r['voxelmorph_loocv']:>12.4f}")
    means = {k: np.mean([r[k] for r in rows]) for k in ["no_transform", "affine", "bspline", "voxelmorph_loocv"]}
    print(f"{'mean':>8} {means['no_transform']:>10.4f} {means['affine']:>10.4f} "
          f"{means['bspline']:>10.4f} {means['voxelmorph_loocv']:>12.4f}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "no_transform", "affine", "bspline", "voxelmorph_loocv"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
