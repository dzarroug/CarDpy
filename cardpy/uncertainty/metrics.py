"""Vectorized versions of the Voxel Bank uncertainty definitions.

Bootstrap samples occupy the final axis. Angular inputs have component axis -2.
Order statistics intentionally match the supplied notebooks at N=400.
"""
import numpy as np


def _order_statistic(samples, fraction):
    # Historical round(p*N)-1, bounded for small test/pilot sample sizes.
    count = np.isfinite(samples).sum(axis=-1)
    index = np.clip(np.rint(fraction * count).astype(int) - 1, 0,
                    samples.shape[-1] - 1)
    ordered = np.sort(np.where(np.isfinite(samples), samples, np.inf), axis=-1)
    result = np.take_along_axis(ordered, index[..., None], axis=-1)[..., 0]
    return np.where(count > 0, result, np.nan)


def invariant_uncertainty(samples):
    """Median, lower/upper 95% bounds, width, and finite sample count.

The upper index follows Invariant_Uncertainty.py: N-round(.025*N)-1.
At 400 samples these are zero-based indices 9 and 389, respectively.
"""
    values = np.asarray(samples, dtype=float)
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError('At least two bootstrap samples are required.')
    values = np.where(np.isfinite(values), values, np.nan)
    count = np.isfinite(values).sum(axis=-1)
    lower = _order_statistic(values, .025)
    ordered = np.sort(np.where(np.isfinite(values), values, np.inf), axis=-1)
    index = np.clip(count - np.rint(.025 * count).astype(int) - 1,
                    0, values.shape[-1] - 1)
    upper = np.take_along_axis(ordered, index[..., None], axis=-1)[..., 0]
    upper = np.where(count > 0, upper, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(values, axis=-1)
    usable = count >= 2
    return dict(median=np.where(usable, median, np.nan),
                lower=np.where(usable, lower, np.nan),
                upper=np.where(usable, upper, np.nan),
                width=np.where(usable, upper - lower, np.nan), count=count)


def angular_uncertainty(samples):
    """Dyadic mean, coherence, and one-sided 95% cone half-angle in degrees.

Eigenvectors are axes: v and -v are identical. Missing or zero vectors are
excluded; fewer than two vectors or a degenerate mean gives an undefined cone.
"""
    vectors = np.asarray(samples, dtype=float)
    if vectors.ndim < 2 or vectors.shape[-2] != 3 or vectors.shape[-1] < 2:
        raise ValueError('Expected [..., 3, bootstrap_samples], with >=2 samples.')
    norm = np.linalg.norm(vectors, axis=-2)
    valid = np.isfinite(vectors).all(axis=-2) & (norm > 0)
    unit = np.divide(vectors, norm[..., None, :], out=np.zeros_like(vectors),
                     where=valid[..., None, :])
    count = valid.sum(axis=-1)
    dyad = np.einsum('...ib,...jb->...ij', unit, unit) / np.maximum(count, 1)[..., None, None]
    values, axes = np.linalg.eigh(dyad)
    mean = axes[..., :, -1]
    dot = np.einsum('...i,...ib->...b', mean, unit)
    angles = np.where(valid, np.degrees(np.arccos(np.clip(np.abs(dot), 0, 1))), np.nan)
    usable = (count >= 2) & ((values[..., -1] - values[..., -2]) > 1e-10)
    coherence = 1 - np.sqrt(np.maximum(values[..., 0] + values[..., 1], 0)
                            / np.maximum(2 * values[..., 2], 1e-30))
    return dict(mean=np.where(usable[..., None], mean, np.nan),
                coherence=np.where(count >= 2, coherence, np.nan),
                cone=np.where(usable, _order_statistic(angles, .95), np.nan),
                count=count)
