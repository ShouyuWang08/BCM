# 097 / Vlad - warp LSFM onto PSOCT grid using the affine from Sarah's
# 10 control points, then overlay to check alignment
import nrrd
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

BASE = "/Users/sophiaaa/Desktop/CMC_project/V_097/V_097/Data"

# control points from Slicer, RAS space (r, a)
# Slicer IJK->RAS here is R=-I, A=-J (spacing=1mm placeholder, origin=0)
# so pixel index = -RAS
PSOCT_RAS = np.array([
    [-2713.68, -4261.86], [-4351.38, -4804.27], [-7837.02, -611.282],
    [-5518.94, -5392.31], [-10023,   -483.447], [-12034.3, -5051.42],
    [-350.143, -4889.5],  [-1372.82, -3704.89], [-6298.74, -474.925],
    [-1815.99, -4710.53],
])
LSFM_RAS = np.array([
    [-7177.8,  -2749.65], [-6074.97, -3134.76], [-3808.06, -438.974],
    [-5225.97, -3620.53], [-2075.05, -548.381], [-237.011, -3725.56],
    [-8827.66, -2859.06], [-7882.38, -2088.83], [-4748.96, -224.536],
    [-8004.92, -2885.32],
])

def load(name):
    arr, _ = nrrd.read(f"{BASE}/{name}")
    return np.squeeze(arr).astype(np.float32)

def normalize(img):
    lo, hi = np.percentile(img[img > 0], [1, 99])
    return np.clip((img - lo) / (hi - lo), 0, 1)

def fit_affine(src_px, dst_px):
    n = src_px.shape[0]
    X = np.hstack([src_px, np.ones((n, 1))])
    A, *_ = np.linalg.lstsq(X, dst_px, rcond=None)
    pred = X @ A
    resid = np.linalg.norm(pred - dst_px, axis=1)
    return A, resid

def main():
    lsfm = load("097_LSFM.nrrd")
    psoct = load("097_PSOCT_R.nrrd")  # retardance channel

    psoct_px = -PSOCT_RAS
    lsfm_px = -LSFM_RAS

    A, resid = fit_affine(psoct_px, lsfm_px)
    print("per-point residual (pixels):", np.round(resid, 1))
    print(f"mean residual: {resid.mean():.1f} px, max: {resid.max():.1f} px")

    # scipy wants [i_lsfm, j_lsfm] = M @ [i_psoct, j_psoct] + offset
    M = np.array([[A[0, 0], A[1, 0]], [A[0, 1], A[1, 1]]])
    offset = A[2, :]

    warped_lsfm = ndimage.affine_transform(
        lsfm, matrix=M, offset=offset, output_shape=psoct.shape, order=1
    )

    overlay = np.zeros((*psoct.shape, 3))
    overlay[..., 0] = normalize(psoct)        # red = psoct
    overlay[..., 1] = normalize(warped_lsfm)  # green = warped lsfm

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].imshow(normalize(psoct).T, cmap='gray')
    axes[0].set_title('PSOCT Retardance'); axes[0].axis('off')
    axes[1].imshow(normalize(warped_lsfm).T, cmap='gray')
    axes[1].set_title('LSFM warped into PSOCT grid'); axes[1].axis('off')
    axes[2].imshow(overlay.transpose(1, 0, 2))
    axes[2].set_title(f'Overlay after affine warp (mean resid={resid.mean():.0f}px)')
    axes[2].axis('off')
    plt.tight_layout()
    out_path = "/Users/sophiaaa/Desktop/nibabel_project/warp_097.png"
    plt.savefig(out_path, dpi=120)
    print("saved:", out_path)

if __name__ == "__main__":
    main()
