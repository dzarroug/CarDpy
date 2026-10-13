"""Phase correction and equal-weight averaging of diffusion repetitions."""
import numpy as np
from cardpy.FT_Operators import fft2c, ifft2c


def remove_background_phase(matrix):
    """Correct each 2D repetition before voxel-wise masking or averaging.

    Fully missing shots/slices are allowed and stay missing. Partially masked
    images must be phase-corrected before masking; NaNs cannot enter the FFT.
    """
    data = np.asarray(matrix)
    if data.ndim != 5 or any(size == 0 for size in data.shape):
        raise ValueError('Phase correction requires a nonempty 5D array.')
    if not np.iscomplexobj(data):
        raise ValueError('Complex phase correction requires complex input.')
    rows, columns = data.shape[:2]
    pads = [int(np.rint(np.nextafter(n / 8, n / 8 + 1))) for n in (rows, columns)]
    window = np.sqrt(np.outer(np.hamming(rows - 2*pads[0]),
                              np.hamming(columns - 2*pads[1])))
    window = np.pad(window, [(pads[0], pads[0]), (pads[1], pads[1])])
    corrected = np.full(data.shape, np.nan, dtype=np.complex128)
    for slc, dif, rep in np.ndindex(data.shape[2:]):
        image = data[:, :, slc, dif, rep]
        finite = np.isfinite(image)
        if not finite.any():
            if np.isinf(image).any():
                raise ValueError('Phase correction does not accept infinite signals.')
            continue
        if not finite.all():
            raise ValueError('Apply phase correction before voxel-wise masking or ADC filtering.')
        smooth = ifft2c(fft2c(image) * window)
        corrected[:, :, slc, dif, rep] = image * np.exp(-1j * np.angle(smooth))
    return corrected


def average(original_matrix, original_bvals=(), original_bvecs=(),
            operation_type='Magnitude', *, phase_corrected=False):
    """Average available repetitions once each, returning a singleton average.

    Complex mode returns the signed real component of the corrected mean.
    Set phase_corrected=True only for already corrected complex repetitions;
    this allows voxel-wise ADC masks without repeating FFT phase estimation.
    """
    if operation_type not in ('Magnitude', 'Complex'):
        raise ValueError('operation_type must be Magnitude or Complex.')
    if phase_corrected and operation_type != 'Complex':
        raise ValueError('phase_corrected is only valid for Complex averaging.')
    if isinstance(original_matrix, dict):
        if not original_matrix:
            raise ValueError('The encoding dictionary is empty.')
        encodings = {}
        for key, encoding in original_matrix.items():
            images = np.asarray(encoding['images'])
            if images.ndim != 4 or any(size == 0 for size in images.shape):
                raise ValueError('Dictionary images must be nonempty 4D arrays.')
            result, _, _ = average(images[:, :, :, None, :],
                                   operation_type=operation_type,
                                   phase_corrected=phase_corrected)
            encodings[key] = {**encoding, 'images': result[:, :, :, 0, :]}
        return [encodings, np.asarray([e['bval'] for e in encodings.values()]),
                np.asarray([e['bvec'] for e in encodings.values()])]
    data = np.asarray(original_matrix)
    if data.ndim != 5 or any(size == 0 for size in data.shape):
        raise ValueError('Averaging requires a nonempty 5D array.')
    if np.isinf(data).any():
        raise ValueError('Averaging does not accept infinite signals.')
    if operation_type == 'Complex':
        if not np.iscomplexobj(data):
            raise ValueError('Complex averaging requires complex input; magnitude input has no phase.')
        data = data if phase_corrected else remove_background_phase(data)
    elif np.iscomplexobj(data):
        data = np.abs(data)
    valid = np.isfinite(data)
    counts = valid.sum(axis=4, keepdims=True)
    if np.any(counts == 0):
        raise ValueError('No available repetitions at one or more voxels; cannot average.')
    result = np.sum(np.where(valid, data, 0), axis=4, keepdims=True) / counts
    return [np.real(result), original_bvals, original_bvecs]
