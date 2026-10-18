# 097 / Vlad - 20-point set, clean pipeline visualization, single combined figure:
#   1. PSOCT (fixed), its own 20 points marked x
#   2. LSFM (native, unwarped), its own 20 points marked o
#   3. PSOCT (red) + Affine-warped LSFM (green), PSOCT points (x) and LSFM
#      points after affine (o)
#   4. PSOCT (red) + Affine+B-Spline LSFM (green), PSOCT points (x) and LSFM
#      points after bspline (o)
# Same color = same point index across all 4 panels, so a mismatched pair
# (x and o far apart) is easy to spot by eye.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import compare_097 as c10
import compare_097_20pts as c20
from bspline_10_vs_20 import bspline_refine_with_transform, lsfm_landing

DOWNSAMPLE = c10.DOWNSAMPLE
COLORS = mpl.colormaps['tab20'](np.linspace(0, 1, 20))

def mark_points(ax, points, marker):
    for k, (x, y) in enumerate(points):
        ax.scatter([x], [y], marker=marker, c=[COLORS[k]], s=90,
                   edgecolors='white' if marker == 'o' else None, linewidths=1.5)
        ax.annotate(str(k), (x, y), color=COLORS[k], fontsize=8, fontweight='bold',
                    xytext=(6, 6), textcoords='offset points')

def draw_single(ax, img, points, marker, title):
    ax.imshow(c10.normalize(img).T, cmap='gray')
    mark_points(ax, points, marker)
    ax.set_title(title)
    ax.axis('off')

def draw_overlay(ax, psoct, moving_img, psoct_points, lsfm_points, title):
    overlay = np.zeros((*psoct.shape, 3))
    overlay[..., 0] = c10.normalize(psoct)
    overlay[..., 1] = c10.normalize(moving_img) if moving_img.max() > 0 else moving_img
    ax.imshow(overlay.transpose(1, 0, 2))
    mark_points(ax, psoct_points, 'x')
    mark_points(ax, lsfm_points, 'o')
    ax.set_title(title)
    ax.axis('off')

def main():
    lsfm_full = c10.load("097_LSFM.nrrd")
    psoct_full = c10.load("097_PSOCT_R.nrrd")
    psoct = psoct_full[::DOWNSAMPLE, ::DOWNSAMPLE]
    lsfm_native = lsfm_full[::DOWNSAMPLE, ::DOWNSAMPLE]
    shape = psoct.shape

    psoct_px, lsfm_px = -c20.PSOCT_RAS, -c20.LSFM_RAS
    M, offset = c20.fit_affine()
    affine_img = c20.affine_warp(lsfm_full, shape, DOWNSAMPLE)
    bspline_img, bspline_transform = bspline_refine_with_transform(affine_img, psoct)

    psoct_pts_ds = [tuple(psoct_px[k] / DOWNSAMPLE) for k in range(len(psoct_px))]
    lsfm_pts_native = [tuple(lsfm_px[k] / DOWNSAMPLE) for k in range(len(lsfm_px))]
    lsfm_pts_affine = [tuple(c10.invert_affine_point(lsfm_px[k], M, offset) / DOWNSAMPLE)
                        for k in range(len(lsfm_px))]
    lsfm_pts_bspline = [tuple(lsfm_landing(lsfm_px[k], M, offset, bspline_transform, DOWNSAMPLE))
                         for k in range(len(lsfm_px))]

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    draw_single(axes[0, 0], psoct, psoct_pts_ds, 'x', "PSOCT (fixed, 20 points)")
    draw_single(axes[0, 1], lsfm_native, lsfm_pts_native, 'o', "LSFM (native, unwarped, 20 points)")
    draw_overlay(axes[1, 0], psoct, affine_img, psoct_pts_ds, lsfm_pts_affine,
                 "PSOCT + Affine(LSFM), 20 pts")
    draw_overlay(axes[1, 1], psoct, bspline_img, psoct_pts_ds, lsfm_pts_bspline,
                 "PSOCT + Affine+B-Spline(LSFM), 20 pts")

    fig.text(0.5, 0.005, "x = PSOCT points, o = LSFM points (same color = same index)",
              ha='center', va='bottom', fontsize=11, style='italic')
    plt.tight_layout(rect=(0, 0.01, 1, 1))
    out_path = "registration_stages_20pts.png"
    plt.savefig(out_path, dpi=110)
    print("saved:", out_path)

if __name__ == "__main__":
    main()
