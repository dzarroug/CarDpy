"""Reject implausibly high apparent diffusivity within each encoding."""
# Original implementation by Tyler E. Cork, CMR Group, Stanford University, 2022.
import numpy as np


def ADC_Filter(original_matrix, original_bvals=None, original_bvecs=None,
               operation_type='Magnitude', max_diffusivity=3.0):
    """Filter repetitions using their actual b-value difference from low b.

    ########## Definition Inputs ##############################################
    original_matrix : Sorted 5D data or an encoding dictionary.
    original_bvals  : One measured b-value per encoding (array input).
    original_bvecs  : One three-component b-vector per encoding (array input).
    ########## Definition Outputs #############################################
    Filtered data, b-values, and b-vectors; rejected samples are NaN.

    Low-shell images are pooled into a magnitude reference. The threshold is
    in µm²/ms. If every repetition exceeds it at a voxel, retain the lowest
    ADC repetition, as in the original filter. Encoding dictionaries preserve
    variable repetition counts, including directional low-b acquisitions.
    """
    if operation_type not in ('Magnitude', 'Complex'):
        raise ValueError("operation_type must be 'Magnitude' or 'Complex'.")
    if not np.isfinite(max_diffusivity) or max_diffusivity <= 0:
        raise ValueError('max_diffusivity must be positive and finite.')
    dictionary = isinstance(original_matrix, dict)
    if dictionary:
        if not original_matrix:
            raise ValueError('The encoding dictionary is empty.')
        entries = list(original_matrix.values())
        bvals = np.asarray([e['bval'] for e in entries])
        images = [np.asarray(e['images']) for e in entries]
    else:
        data = np.asarray(original_matrix)
        bvals = np.asarray(original_bvals)
        if data.ndim != 5 or bvals.shape != (data.shape[3],):
            raise ValueError('ADC filtering requires matching 5D data and encodings.')
        images = [data[..., dif, :] for dif in range(data.shape[3])]
    if not np.isfinite(bvals).all() or len(np.unique(bvals)) < 2:
        raise ValueError('ADC filtering requires at least two finite b-value shells.')
    low = float(np.min(bvals))
    reference = np.nanmean(np.concatenate(
        [np.abs(image) for image, bval in zip(images, bvals) if bval == low], axis=3), axis=3)
    filtered = []
    for image, bval in zip(images, bvals):
        out = np.array(np.abs(image) if operation_type == 'Magnitude' else image,
                       dtype=np.complex128 if operation_type == 'Complex' and np.iscomplexobj(image) else float,
                       copy=True)
        if bval > low:
            with np.errstate(divide='ignore', invalid='ignore'):
                adc = -1000.0 * np.log(np.abs(image) / reference[..., None]) / (bval - low)
            rejected = adc >= max_diffusivity
            all_rejected = np.all(rejected, axis=3)
            best = np.argmin(np.where(np.isnan(adc), np.inf, adc), axis=3)
            # Unreject the best shot only where all shots were rejected.
            best_mask = np.arange(image.shape[3]) == best[..., None]
            rejected &= ~(all_rejected[..., None] & best_mask)
            out[rejected] = np.nan
        filtered.append(out)
    if dictionary:
        encodings = {key: {**encoding, 'images': image}
                     for (key, encoding), image in zip(original_matrix.items(), filtered)}
        return [encodings, bvals, np.asarray([e['bvec'] for e in entries])]
    return [np.stack(filtered, axis=3), original_bvals, original_bvecs]
