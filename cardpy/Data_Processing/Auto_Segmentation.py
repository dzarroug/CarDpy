"""
Auto_Segmentation.py -- segmentation mode = 'auto'

Runs Sascha Stocker's nnUNet_cDTI pipeline (YOLO heart crop -> nnUNet LV
5-fold ensemble -> YOLO RV insertion points), bundled in nnunet_cdti/, on
CarDpy's average DWI and MD maps, then converts the result into CarDpy's
contour format:
  endo/epi : 200-point closed curves, x = column, y = row (same as New_GUI)
  RVIPs    : one (x, y) point each, or None if not detected

Weights (~4 GB, gated Hugging Face repo sash99666/cdti-lv-baseline):
request access on the model page, run `hf auth login` once; the first run
downloads them into ~/.cache/cardpy/cdti-lv-baseline (or CARDPY_AUTOSEG_WEIGHTS).
"""
import os
import sys
import shutil
import tempfile
import subprocess

import numpy as np

N_CONTOUR_POINTS = 200
DEFAULT_WEIGHTS = os.path.join(os.path.expanduser('~'), '.cache', 'cardpy', 'cdti-lv-baseline')


def _clean_ring(lv):
    """Largest LV component with exactly one cavity; None if the ring is open."""
    from scipy import ndimage
    labels, n = ndimage.label(lv)
    if n == 0:
        return None
    sizes = ndimage.sum(lv, labels, range(1, n + 1))
    lv = labels == (np.argmax(sizes) + 1)
    filled = ndimage.binary_fill_holes(lv)
    holes, n_holes = ndimage.label(filled & ~lv)
    if n_holes == 0:
        return None
    hole_sizes = ndimage.sum(holes > 0, holes, range(1, n_holes + 1))
    cavity = holes == (np.argmax(hole_sizes) + 1)
    return filled & ~cavity, filled, cavity


def _boundary_curve(region):
    """Closed 200-point periodic spline around a binary region, as (x=col, y=row)."""
    from skimage.measure import find_contours
    from scipy.interpolate import splprep, splev
    padded = np.pad(region.astype(float), 1)
    curves = find_contours(padded, 0.5)
    curve = max(curves, key=len) - 1.0
    xy = curve[:, ::-1]
    if np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    step = max(1, len(xy) // 40)
    xy = xy[::step]
    closed = np.vstack((xy, xy[:1]))
    tck, _ = splprep(closed.T, s=len(xy) * 0.25, per=1)
    out = np.column_stack(splev(np.linspace(0, 1, N_CONTOUR_POINTS), tck))
    out[-1] = out[0]
    return out


def _point(layer):
    from scipy import ndimage
    if not np.any(layer):
        return None
    row, col = ndimage.center_of_mass(layer)
    return [float(col), float(row)]


def auto_segment(avg_dwi, md, weights_dir=None, work_dir=None, case='CarDpy'):
    """
    avg_dwi, md : [rows, cols, slices] on the contouring grid (MD in um^2/ms, not normalized).
    Returns a list (one entry per slice) of dicts with keys
      mask (bool, model LV after ring cleanup), endo, epi (N x 2), antRVIP, infRVIP ([x, y] or None),
    or None for a slice where no usable closed LV ring was found.
    """
    import nibabel as nib
    from cardpy.Data_Processing.nnunet_cdti import deployable as deploy

    weights = weights_dir or os.environ.get('CARDPY_AUTOSEG_WEIGHTS') or DEFAULT_WEIGHTS
    weights = os.path.abspath(os.path.expanduser(os.fspath(weights)))
    if not deploy._weights_present(weights):
        print('Downloading auto segmentation weights (~4 GB) into %s' % weights)
        try:
            deploy.download_weights(weights)
        except Exception as error:
            raise RuntimeError('Could not download weights from huggingface.co/%s. Request access on '
                               'that page, then run `hf auth login`.' % deploy.HF_REPO) from error

    temporary = tempfile.TemporaryDirectory() if work_dir is None else None
    root = temporary.name if temporary is not None else work_dir
    in_dir, out_dir = os.path.join(root, 'input'), os.path.join(root, 'output')
    for folder in (in_dir, out_dir):
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)

    try:
        n_slices = avg_dwi.shape[2]
        for slc in range(n_slices):
            for channel, image in ((0, avg_dwi), (1, md)):
                array = np.nan_to_num(np.real(image[:, :, slc]), nan=0.0, posinf=0.0, neginf=0.0)
                nib.save(nib.Nifti1Image(array.astype(np.float32), np.eye(4)),
                         os.path.join(in_dir, '%s_slice%02d_%04d.nii.gz' % (case, slc + 1, channel)))

        # Run the model in a separate Python process. torch, dipy and scikit-learn each ship
        # their own OpenMP runtime on macOS; loading them in one process can segfault.
        command = [sys.executable, '-m', 'cardpy.Data_Processing.nnunet_cdti.deployable',
                   '--input', in_dir, '--output', out_dir, '--weights', weights]
        if subprocess.run(command).returncode != 0:
            raise RuntimeError('Auto segmentation subprocess failed (see output above).')

        results = []
        for slc in range(n_slices):
            path = os.path.join(out_dir, '%s_slice%02d.nii.gz' % (case, slc + 1))
            if not os.path.isfile(path):
                print('Auto segmentation: no output for slice %d.' % (slc + 1))
                results.append(None)
                continue
            layers = np.asarray(nib.load(path).get_fdata()) > 0
            ring = _clean_ring(layers[..., 0])
            if ring is None:
                print('Auto segmentation: slice %d has no closed LV ring.' % (slc + 1))
                results.append(None)
                continue
            mask, filled, cavity = ring
            results.append({
                'mask':    mask,
                'epi':     _boundary_curve(filled),
                'endo':    _boundary_curve(cavity),
                'antRVIP': _point(layers[..., 1]),
                'infRVIP': _point(layers[..., 2]),
            })
        return results
    finally:
        if temporary is not None:
            temporary.cleanup()