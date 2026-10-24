"""
make_figure1_panels.py -- export Figure 1 panels from a finished CarDpy run.

Reads CarDpy_Output/ (stage folders, 11_Segmentation, 12_Quantitative_Results)
and writes one PNG per panel, all cropped to the same heart-centred box:

  stage_<Name>.png      mean diffusion-weighted image after each stage
  dti_<MAP>_full.png    MD / FA over the whole FOV, no DWI underneath
  dti_<MAP>_rings.png   MD / FA inside the myocardium only, transparent outside
  cdti_<MAP>.png        MD / FA / HA in the myocardium on top of the DWI
  colorbar_<MAP>.png    standalone colorbars for the maps

Edit the settings block, then:  python make_figure1_panels.py
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import nrrd
import numpy as np
from scipy.io import loadmat

# ---------------- settings ----------------
OUTPUT_DIR = '/Users/dahniazarroug/Desktop/Dahnia/V001/CarDpy_Output'
SAVE_DIR   = os.path.join(OUTPUT_DIR, 'Figure1_Panels')
SLICE      = 0       # slice index within the processed data (0 = first processed slice)
MARGIN     = 0.35    # extra space around the LV, as a fraction of the LV box size
DPI        = 300
STAGES     = ['Original', 'Unrung', 'Rejected', 'Registered',
              'Averaged', 'Denoised', 'Interpolated']
MAPS       = {'MD': (0, 3), 'FA': (0, 1), 'HA': (-90, 90)}
# ------------------------------------------

try:
    from cardpy.Colormaps import cDTI_Colormaps_Generator
    CMAPS = cDTI_Colormaps_Generator()
except Exception:
    CMAPS = {'MD': 'viridis', 'FA': 'magma', 'HA': 'twilight'}


def stage_mean_dwi(name):
    """Mean of the diffusion-weighted (b > min) volumes for SLICE, or None if the stage wasn't saved."""
    folders = sorted(glob.glob(os.path.join(OUTPUT_DIR, '*_' + name)))
    if not folders:
        return None
    nii = glob.glob(os.path.join(folders[0], '*.nii*'))
    bvl = glob.glob(os.path.join(folders[0], '*.bvals'))
    if not nii or not bvl:
        return None
    data = np.asarray(nib.load(nii[0]).get_fdata())        # [rows, cols, slices, volumes]
    b = np.loadtxt(bvl[0]).ravel()
    return np.nanmean(np.abs(data[:, :, SLICE, b > b.min()]), axis=2)


def stage_mean_all(name):
    folders = sorted(glob.glob(os.path.join(OUTPUT_DIR, '*_' + name)))
    nii = glob.glob(os.path.join(folders[0], '*.nii*'))
    data = np.asarray(nib.load(nii[0]).get_fdata())
    return np.nanmean(np.abs(data[:, :, SLICE, :]), axis=2)


def crop_box(mask):
    """Heart box as fractions of the image, so it applies to native and interpolated grids alike."""
    rows, cols = np.nonzero(mask)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    size = max(r1 - r0, c1 - c0) * (1 + 2 * MARGIN)
    rc, cc = (r0 + r1) / 2, (c0 + c1) / 2
    h, w = mask.shape
    return ((rc - size / 2) / h, (rc + size / 2) / h, (cc - size / 2) / w, (cc + size / 2) / w)


def crop(img, box):
    h, w = img.shape[:2]
    r0, r1 = int(round(max(box[0], 0) * h)), int(round(min(box[1], 1) * h))
    c0, c1 = int(round(max(box[2], 0) * w)), int(round(min(box[3], 1) * w))
    return img[r0:r1, c0:c1]


def window(img):
    lo, hi = np.nanpercentile(img, [1, 99.5])
    return lo, hi


def save_panel(path, layers, transparent=False):
    """layers: list of (image, kwargs) drawn in order on a borderless square axis."""
    h, w = layers[0][0].shape[:2]
    fig = plt.figure(figsize=(3, 3 * h / w))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    for img, kw in layers:
        ax.imshow(img, interpolation='nearest', **kw)
    fig.savefig(path, dpi=DPI, transparent=transparent, facecolor='none' if transparent else 'black')
    plt.close(fig)


def save_colorbar(path, cmap, vmin, vmax, label):
    fig = plt.figure(figsize=(0.6, 3))
    ax = fig.add_axes([0.1, 0.05, 0.3, 0.9])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, cax=ax).set_label(label)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', transparent=True)
    plt.close(fig)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    mask_file = glob.glob(os.path.join(OUTPUT_DIR, '11_Segmentation', 'LV_Myocardium.nrrd'))[0]
    mask = nrrd.read(mask_file)[0][:, :, SLICE] > 0
    box = crop_box(mask)

    # Stage panels: mean DWI after each stage, each windowed on its own crop.
    for name in STAGES:
        img = stage_mean_dwi(name)
        if img is None:
            print('skipped (not saved):', name)
            continue
        img = crop(img, box)
        lo, hi = window(img)
        save_panel(os.path.join(SAVE_DIR, 'stage_%s.png' % name),
                   [(img, dict(cmap='gray', vmin=lo, vmax=hi))])

    # Maps (on the final, interpolated grid, same as the mask).
    qr = os.path.join(OUTPUT_DIR, '12_Quantitative_Results')
    dti = loadmat(os.path.join(qr, 'Standard_DTI_Metrics.mat'))
    cdti = loadmat(os.path.join(qr, 'cDTI_Metrics.mat'))
    maps = {'MD': dti['MD'][:, :, SLICE], 'FA': dti['FA'][:, :, SLICE], 'HA': cdti['HASF'][:, :, SLICE]}
    background = crop(stage_mean_all('Interpolated'), box)
    bg_lo, bg_hi = window(background)
    m = crop(mask, box)

    for key, (vmin, vmax) in MAPS.items():
        cmap = CMAPS[key]
        full = crop(maps[key], box)
        ring = np.where(m, full, np.nan)
        if key in ('MD', 'FA'):
            save_panel(os.path.join(SAVE_DIR, 'dti_%s_full.png' % key),
                       [(full, dict(cmap=cmap, vmin=vmin, vmax=vmax))])
            save_panel(os.path.join(SAVE_DIR, 'dti_%s_rings.png' % key),
                       [(ring, dict(cmap=cmap, vmin=vmin, vmax=vmax))], transparent=True)
        save_panel(os.path.join(SAVE_DIR, 'cdti_%s.png' % key),
                   [(background, dict(cmap='gray', vmin=bg_lo, vmax=bg_hi)),
                    (ring, dict(cmap=cmap, vmin=vmin, vmax=vmax))])
        units = {'MD': 'MD (µm²/ms)', 'FA': 'FA', 'HA': 'HA (°)'}[key]
        save_colorbar(os.path.join(SAVE_DIR, 'colorbar_%s.png' % key), cmap, vmin, vmax, units)

    print('Panels written to', SAVE_DIR)


if __name__ == '__main__':
    main()
