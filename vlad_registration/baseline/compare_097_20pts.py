# 097 / Vlad - comparing 4 registration approaches for LSFM -> PS-OCT
# scored with Mattes MI (SimpleITK), lower/more negative = better fit
# same as compare_097.py but fit on the 20-point set (10 original + 10 more
# from professor, V_097_20pts.mrb / LSFM.mrk.json + PSOCT.mrk.json)
import nrrd
import numpy as np
from scipy import ndimage
from scipy.interpolate import RBFInterpolator
import SimpleITK as sitk
import matplotlib.pyplot as plt

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)  # required for exact reproducibility with a fixed seed

BASE = "/Users/sophiaaa/Desktop/CMC_project/V_097/V_097/Data"
DOWNSAMPLE = 6

PSOCT_RAS = np.array([
    [-2713.68, -4261.86], [-4351.38, -4804.27], [-7837.02, -611.282],
    [-5518.94, -5392.31], [-10023.0, -483.447], [-12034.3, -5051.42],
    [-350.143, -4889.5],  [-1372.82, -3704.89],  [-6298.74, -474.925],
    [-1815.99, -4710.53], [-10863.0, -3910.67],  [-12620.3, -3550.2],
    [-10164.6, -2671.54], [-10975.6, -812.854],  [-6212.88, -3129.31],
    [-416.869, -3597.13], [-2262.71, -2749.86],  [-7573.28, -4035.9],
    [-4093.41, -5579.13], [-6362.88, -1902.58],
])
LSFM_RAS = np.array([
    [-7177.8,  -2749.65], [-6074.97, -3134.76], [-3808.06, -438.974],
    [-5225.97, -3620.53], [-2075.05, -548.381], [-237.011, -3725.56],
    [-8827.66, -2859.06], [-7882.38, -2088.83], [-4748.96, -224.536],
    [-8004.92, -2885.32], [-2782.79, -2346.11], [-241.327, -2514.14],
    [-2278.7,  -1856.02], [-1165.5,  -784.823], [-4946.19, -1968.04],
    [-8601.43, -1864.51], [-7349.78, -1404.22], [-4014.74, -2663.95],
    [-6437.29, -3713.71], [-4741.5,  -1186.19],
])

def load(name):
    arr, _ = nrrd.read(f"{BASE}/{name}")
    return np.squeeze(arr).astype(np.float32)

def normalize(img):
    positive = img[img > 0]
    if positive.size == 0:
        return np.zeros_like(img)
    lo, hi = np.percentile(positive, [1, 99])
    return np.clip((img - lo) / (hi - lo), 0, 1)

def mattes_mi(fixed_arr, moving_arr):
    fixed = sitk.Cast(sitk.GetImageFromArray(normalize(fixed_arr)), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.GetImageFromArray(normalize(moving_arr)), sitk.sitkFloat32)
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.5, seed=7)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetInitialTransform(sitk.TranslationTransform(2), inPlace=False)
    return method.MetricEvaluate(fixed, moving)

def out_grid(out_shape, downsample):
    # output pixel (i,j) -> full-res PSOCT pixel (i*downsample, j*downsample)
    ii, jj = np.meshgrid(
        np.arange(out_shape[0]) * downsample,
        np.arange(out_shape[1]) * downsample,
        indexing='ij'
    )
    return ii, jj

def no_transform(lsfm_full, out_shape, downsample):
    ii, jj = out_grid(out_shape, downsample)
    out = np.zeros(out_shape, dtype=np.float32)
    valid = (ii < lsfm_full.shape[0]) & (jj < lsfm_full.shape[1])
    out[valid] = lsfm_full[ii[valid], jj[valid]]
    return out

def fit_affine():
    psoct_px, lsfm_px = -PSOCT_RAS, -LSFM_RAS
    n = psoct_px.shape[0]
    X = np.hstack([psoct_px, np.ones((n, 1))])
    A, *_ = np.linalg.lstsq(X, lsfm_px, rcond=None)
    M = np.array([[A[0, 0], A[1, 0]], [A[0, 1], A[1, 1]]])
    offset = A[2, :]
    return M, offset

def affine_warp(lsfm_full, out_shape, downsample):
    M, offset = fit_affine()
    ii, jj = out_grid(out_shape, downsample)
    coords = np.stack([ii.ravel(), jj.ravel()], axis=1)
    mapped = coords @ M.T + offset
    i_lsfm = mapped[:, 0].reshape(out_shape)
    j_lsfm = mapped[:, 1].reshape(out_shape)
    return ndimage.map_coordinates(lsfm_full, [i_lsfm, j_lsfm], order=1, mode='constant', cval=0)

def tps_warp(lsfm_full, out_shape, downsample):
    psoct_px, lsfm_px = -PSOCT_RAS, -LSFM_RAS
    tps = RBFInterpolator(psoct_px, lsfm_px, kernel='thin_plate_spline', degree=1)
    ii, jj = out_grid(out_shape, downsample)
    query = np.stack([ii.ravel(), jj.ravel()], axis=1)
    mapped = tps(query)
    i_lsfm = mapped[:, 0].reshape(out_shape)
    j_lsfm = mapped[:, 1].reshape(out_shape)
    return ndimage.map_coordinates(lsfm_full, [i_lsfm, j_lsfm], order=1, mode='constant', cval=0)

def bspline_refine(lsfm_affine_warped, psoct, mesh_size=8):
    fixed = sitk.Cast(sitk.GetImageFromArray(normalize(psoct)), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.GetImageFromArray(normalize(lsfm_affine_warped)), sitk.sitkFloat32)
    transform_init = sitk.BSplineTransformInitializer(fixed, [mesh_size, mesh_size])
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.3, seed=7)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsLBFGSB(numberOfIterations=100)
    method.SetInitialTransform(transform_init, inPlace=True)
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    final_transform = method.Execute(fixed, moving)
    resampled = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
    return sitk.GetArrayFromImage(resampled)

def main():
    lsfm_full = load("097_LSFM.nrrd")
    psoct_full = load("097_PSOCT_R.nrrd")

    # psoct is the fixed image so just decimate it directly; lsfm has to be
    # sampled from full res or the affine/tps coords won't line up
    psoct = psoct_full[::DOWNSAMPLE, ::DOWNSAMPLE]
    shape = psoct.shape

    affine_result = affine_warp(lsfm_full, shape, DOWNSAMPLE)
    titles = ['No transform', 'Affine (20 pts)', 'TPS (20 pts)', 'B-Spline (on affine)']
    warped = [
        no_transform(lsfm_full, shape, DOWNSAMPLE),
        affine_result,
        tps_warp(lsfm_full, shape, DOWNSAMPLE),
        bspline_refine(affine_result, psoct),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(28, 8))
    for ax, title, img in zip(axes, titles, warped):
        score = mattes_mi(psoct, img)
        overlay = np.zeros((*shape, 3))
        overlay[..., 0] = normalize(psoct)
        overlay[..., 1] = normalize(img) if img.max() > 0 else img
        ax.imshow(overlay.transpose(1, 0, 2))
        ax.set_title(title)
        ax.axis('off')
        print(f'{title}: Mattes MI = {score:.4f}')

    fig.text(0.5, 0.02, "20-point comparison, metric TBD w/ sponsor",
              ha='center', va='bottom', fontsize=11, style='italic')
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    out_path = "compare_097_20pts.png"
    plt.savefig(out_path, dpi=110)
    print("saved:", out_path)

if __name__ == "__main__":
    main()
