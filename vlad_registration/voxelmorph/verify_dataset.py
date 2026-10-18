# Sanity-check all 11 subjects in data/mrb/ before building anything more on
# top of them: for each pair, fit the same least-squares affine as
# baseline/control_points.py uses (PSOCT control points -> LSFM control
# points) and report the per-point residual. Big residuals would mean a
# mislabeled point set or a wrong image pairing - catch that now rather than
# after a long VoxelMorph training run.
import numpy as np

from mrb_dataset import SUBJECTS, load_subject


def fit_affine(src_px, dst_px):
    n = src_px.shape[0]
    X = np.hstack([src_px, np.ones((n, 1))])
    A, *_ = np.linalg.lstsq(X, dst_px, rcond=None)
    pred = X @ A
    resid = np.linalg.norm(pred - dst_px, axis=1)
    return A, resid


def main():
    print(f"{'subject':>8} {'lsfm shape':>16} {'psoct shape':>16} {'mean resid (px)':>16} {'max resid (px)':>15}")
    for sid in SUBJECTS:
        d = load_subject(sid)
        _, resid = fit_affine(d["psoct_px"], d["lsfm_px"])
        print(f"{sid:>8} {str(d['lsfm'].shape):>16} {str(d['psoct'].shape):>16} "
              f"{resid.mean():>16.1f} {resid.max():>15.1f}")


if __name__ == "__main__":
    main()
