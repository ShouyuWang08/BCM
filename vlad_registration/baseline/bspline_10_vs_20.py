# 097 / Vlad - B-Spline only: 10 original control points vs the 20-point
# set professor added (V_097_20pts.mrb). Does NOT modify compare_097.py or
# compare_097_20pts.py - reuses their point sets and affine fit as-is, only
# adds a bspline variant that also returns the fitted transform so
# per-point error can be printed and overlaid on the images.
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib as mpl

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)  # required for exact reproducibility with a fixed seed

import compare_097 as c10
import compare_097_20pts as c20

DOWNSAMPLE = c10.DOWNSAMPLE

def bspline_refine_with_transform(lsfm_affine_warped, psoct, mesh_size=8):
    # same registration as compare_097.bspline_refine, but also hands back
    # final_transform so individual control points can be mapped through it
    fixed = sitk.Cast(sitk.GetImageFromArray(c10.normalize(psoct)), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.GetImageFromArray(c10.normalize(lsfm_affine_warped)), sitk.sitkFloat32)
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
    return sitk.GetArrayFromImage(resampled), final_transform

def bspline_predict_point(point_psoct_full, M, offset, bspline_transform, downsample):
    # psoct full-res point -> downsampled grid -> bspline correction (defined
    # on that grid) -> back to full-res -> same affine used to build the
    # bspline's moving image -> predicted lsfm full-res point
    i_ds, j_ds = point_psoct_full[0] / downsample, point_psoct_full[1] / downsample
    j_ds2, i_ds2 = bspline_transform.TransformPoint((j_ds, i_ds))  # sitk point order is (x, y)
    proxy_full = np.array([i_ds2 * downsample, j_ds2 * downsample])
    return proxy_full @ M.T + offset

def run_bspline(mod, lsfm_full, psoct, shape):
    psoct_px, lsfm_px = -mod.PSOCT_RAS, -mod.LSFM_RAS
    M, offset = mod.fit_affine()
    affine_img = mod.affine_warp(lsfm_full, shape, DOWNSAMPLE)
    bspline_img, bspline_transform = bspline_refine_with_transform(affine_img, psoct)
    return psoct_px, lsfm_px, bspline_img, M, offset, bspline_transform

def lsfm_landing(lsfm_point_full, M, offset, bspline_transform, downsample):
    # inverse of bspline_predict_point: where the originally-marked LSFM point
    # actually ends up in the *displayed* output grid (post-affine,
    # post-bspline), so it lands in the same coordinate frame as the PSOCT
    # target x-marks and can be compared directly on the image
    pre_bspline_ds = c10.invert_affine_point(lsfm_point_full, M, offset) / downsample
    i0, j0 = pre_bspline_ds
    j_final, i_final = c10.invert_bspline_point((j0, i0), bspline_transform)  # sitk order is (x, y)
    return np.array([i_final, j_final])

def scatter_with_error(ax, psoct_px, lsfm_px, M, offset, bspline_transform, downsample, colors):
    errs = []
    for k in range(len(psoct_px)):
        i_ds, j_ds = lsfm_landing(lsfm_px[k], M, offset, bspline_transform, downsample)
        pred = bspline_predict_point(psoct_px[k], M, offset, bspline_transform, downsample)
        err = np.linalg.norm(pred - lsfm_px[k])
        errs.append(err)
        ax.scatter([i_ds], [j_ds], c=[colors[k]], s=60, edgecolors='white', linewidths=1)
        ax.annotate(f'{k}:{err:.0f}px', (i_ds, j_ds), color='white', fontsize=8,
                    fontweight='bold', xytext=(5, 5), textcoords='offset points')
    return np.array(errs)

def mark_all_target_points(ax, psoct_px_20, downsample):
    # ground-truth PSOCT target locations for all 20 points (professor's full
    # set), overlaid on every panel regardless of how many points that panel's
    # affine/bspline was actually fit on - lets you check by eye whether the
    # extra 10 points correspond to sane anatomy / are placed where expected
    for k in range(len(psoct_px_20)):
        i_ds, j_ds = psoct_px_20[k, 0] / downsample, psoct_px_20[k, 1] / downsample
        color = 'red' if k >= 10 else 'yellow'
        ax.scatter([i_ds], [j_ds], marker='x', c=color, s=90, linewidths=2)
        ax.annotate(str(k), (i_ds, j_ds), color=color, fontsize=9, fontweight='bold',
                    xytext=(-10, -10), textcoords='offset points')

def main():
    lsfm_full = c10.load("097_LSFM.nrrd")
    psoct_full = c10.load("097_PSOCT_R.nrrd")
    psoct = psoct_full[::DOWNSAMPLE, ::DOWNSAMPLE]
    shape = psoct.shape

    psoct_px_20 = -c20.PSOCT_RAS

    cases = [('10 pts', c10), ('20 pts', c20)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, (label, mod) in zip(axes, cases):
        psoct_px, lsfm_px, bspline_img, M, offset, bspline_transform = run_bspline(mod, lsfm_full, psoct, shape)
        cmap = mpl.colormaps['tab10'] if len(psoct_px) <= 10 else mpl.colormaps['tab20']
        colors = cmap(np.linspace(0, 1, len(psoct_px)))
        overlay = np.zeros((*shape, 3))
        overlay[..., 0] = c10.normalize(psoct)
        overlay[..., 1] = c10.normalize(bspline_img) if bspline_img.max() > 0 else bspline_img
        ax.imshow(overlay.transpose(1, 0, 2))
        errs = scatter_with_error(ax, psoct_px, lsfm_px, M, offset, bspline_transform, DOWNSAMPLE, colors)
        mark_all_target_points(ax, psoct_px_20[:len(psoct_px)], DOWNSAMPLE)
        ax.set_title(f'B-Spline ({label})\nTRE: mean={errs.mean():.0f}px  median={np.median(errs):.0f}px')
        ax.axis('off')
        print(f'B-Spline ({label}): TRE mean={errs.mean():.1f}px  median={np.median(errs):.1f}px  max={errs.max():.1f}px')
        for k, e in enumerate(errs):
            print(f'  landmark {k}: {e:.0f}px')

    fig.text(0.5, 0.01,
              "dots = predicted LSFM landing after this panel's own warp, labels show error in full-res px  |  "
              "x = true PSOCT target (yellow=original 10, red=professor's new 10)  |  gap between dot and x = misregistration",
              ha='center', va='bottom', fontsize=11, style='italic')
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    out_path = "bspline_10_vs_20.png"
    plt.savefig(out_path, dpi=110)
    print("saved:", out_path)

if __name__ == "__main__":
    main()
