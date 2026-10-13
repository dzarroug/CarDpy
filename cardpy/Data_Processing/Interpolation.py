"""Fourier interpolation on the image's in-plane sampling grid."""
import numpy as np
from scipy.signal import resample


def zero_filled(original_matrix, original_bvals=(), original_bvecs=(),
                operation_type='Magnitude', kaiser_beta=None):
    """Double rows/columns without interpolating wrapped phase angles.

    Complex data uses untapered Fourier zero filling by default. Optional
    Kaiser apodization trades resolution for reduced ringing. Magnitude mode
    retains Kaiser smoothing (beta=24.6) by default. scipy handles odd sizes
    and the even-size Nyquist term; original pixel i maps to output pixel 2*i.
    Interpolation adds sampling points, not acquired spatial resolution.
    """
    data = np.asarray(original_matrix)
    if data.ndim != 5 or min(data.shape[:2]) < 1:
        raise ValueError('Expected [rows, columns, slices, directions, averages].')
    if operation_type not in ('Magnitude', 'Complex'):
        raise ValueError('operation_type must be Magnitude or Complex.')
    beta = (0.0 if operation_type == 'Complex' else 24.6) if kaiser_beta is None else float(kaiser_beta)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError('kaiser_beta must be finite and nonnegative.')
    # Never offset signed real/imaginary values or mutate the acquired input.
    source = data.astype(np.complex128) if operation_type == 'Complex' else np.abs(data)
    window = None if beta == 0 else ('kaiser', beta)
    result = resample(source, 2 * data.shape[0], axis=0, window=window)
    result = resample(result, 2 * data.shape[1], axis=1, window=window)
    if operation_type == 'Magnitude':
        result = np.abs(result)
    return [result, original_bvals, original_bvecs]
