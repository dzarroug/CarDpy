"""Fourier interpolation on the image's in-plane sampling grid."""
import numpy as np
from scipy.signal import resample


def zero_filled(original_matrix, original_bvals=(), original_bvecs=(),
                operation_type='Magnitude', kaiser_beta=None):
    """Double rows/columns without interpolating wrapped phase angles.

    Both complex and magnitude modes default to a square-root separable
    Kaiser window (beta=24.6) on the
    doubled k-space grid, matching the original ZIP filter. Applying a full
    Kaiser window to the source grid instead causes excessive smoothing.
    Set kaiser_beta=0 for untapered Fourier zero filling. scipy handles odd
    sizes and the even-size Nyquist term; pixel i maps to output pixel 2*i.
    Interpolation adds sampling points, not acquired spatial resolution.
    """
    data = np.asarray(original_matrix)
    if data.ndim != 5 or min(data.shape[:2]) < 1:
        raise ValueError('Expected [rows, columns, slices, directions, averages].')
    if operation_type not in ('Magnitude', 'Complex'):
        raise ValueError('operation_type must be Magnitude or Complex.')
    beta = 24.6 if kaiser_beta is None else float(kaiser_beta)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError('kaiser_beta must be finite and nonnegative.')
    # Never offset signed real/imaginary values or mutate the acquired input.
    source = data.astype(np.complex128) if operation_type == 'Complex' else np.abs(data)
    # Resample without a taper first so scipy preserves odd-size alignment and
    # splits the even-size Nyquist bin correctly. Apodize only on the doubled
    # spectrum; the same beta on the source grid is a much stronger filter.
    result = resample(source, 2 * data.shape[0], axis=0)
    result = resample(result, 2 * data.shape[1], axis=1)
    if beta != 0:
        axes = (0, 1)
        window = np.sqrt(np.outer(np.kaiser(result.shape[0], beta),
                                  np.kaiser(result.shape[1], beta)))
        window = window.reshape((*result.shape[:2], 1, 1, 1))
        spectrum = np.fft.fftshift(np.fft.fft2(result, axes=axes), axes=axes)
        result = np.fft.ifft2(np.fft.ifftshift(spectrum * window, axes=axes),
                              axes=axes)
    if operation_type == 'Magnitude':
        result = np.abs(result)
    return [result, original_bvals, original_bvecs]
