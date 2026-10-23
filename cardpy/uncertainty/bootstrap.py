"""Notebook-style whole-image resampling, with explicit post-rejection support."""
import math
import numpy as np


def encoding_groups(bvals, bvecs):
    groups = {}
    for index, (bval, bvec) in enumerate(zip(bvals, bvecs)):
        key = (round(float(bval), 5), *np.round(bvec, 5))
        groups.setdefault(key, []).append(index)
    return list(groups.values())


def make_plan(data, bvals, bvecs, *, iterations=400, desired_averages=None, seed=0):
    """Draw unique composite rows, then group rows into bootstrap datasets.

    As in diffusion_bootstrap in the MEP notebooks, each composite row selects
    one repetition independently per encoding; a dataset contains
    desired_averages such rows. Composite rows cannot repeat across datasets.
    Images are sampled as wholes, never pixel by pixel. Different slices can
    have different retained pools after rejection, so sampling is per slice.
    """
    data = np.asarray(data)
    bvals, bvecs = np.asarray(bvals), np.asarray(bvecs)
    if (data.ndim != 4 or bvals.shape != (data.shape[-1],)
            or bvecs.shape != (data.shape[-1], 3)):
        raise ValueError('Expected stacked [rows, columns, slices, volumes] and matching gradients.')
    if not np.isfinite(bvals).all() or not np.isfinite(bvecs).all() or np.any(bvals < 0):
        raise ValueError('Invalid diffusion encodings.')
    finite = np.isfinite(data)
    available = finite.all(axis=(0, 1))
    if np.isinf(data).any() or np.any(finite.any(axis=(0, 1)) != available):
        raise ValueError('Input must contain finite whole images or wholly NaN rejected images.')
    if not isinstance(iterations, int) or iterations < 2:
        raise ValueError('iterations must be an integer >=2.')
    groups = encoding_groups(bvals, bvecs)
    pools = [[np.array([i for i in group if available[slc, i]], dtype=int)
              for group in groups] for slc in range(data.shape[2])]
    counts = np.array([[len(pool) for pool in row] for row in pools])
    if np.any(counts == 0):
        raise ValueError('At least one slice/encoding has no retained images.')
    # Notebook default: desired_averages is the original repetition-axis size.
    averages = max(map(len, groups)) if desired_averages is None else desired_averages
    if not isinstance(averages, int) or averages < 1:
        raise ValueError('desired_averages must be a positive integer.')
    rng = np.random.default_rng(seed)
    plan = np.empty((iterations, data.shape[2], len(groups), averages), dtype=np.int32)
    required = iterations * averages
    for slc, pool_row in enumerate(pools):
        capacity = math.prod(map(len, pool_row))
        if required > capacity:
            raise ValueError(f'Slice {slc}: need {required} unique composite rows, '
                             f'but retained repetitions allow {capacity}. Reduce iterations '
                             'or desired_averages; uniqueness cannot be satisfied.')
        # Sparse partial Fisher-Yates avoids the notebook rejection loop hanging
        # near exhaustion, and also supports products above int64.
        swaps = {}
        def randbelow(limit):
            if limit <= np.iinfo(np.int64).max:
                return int(rng.integers(limit))
            bits = limit.bit_length()
            while True:
                value = int.from_bytes(rng.bytes((bits + 7) // 8), 'little') & ((1 << bits) - 1)
                if value < limit:
                    return value
        for row_index in range(required):
            remaining = capacity - row_index
            selected = randbelow(remaining)
            code = swaps.get(selected, selected)
            swaps[selected] = swaps.get(remaining - 1, remaining - 1)
            iteration, draw = divmod(row_index, averages)
            for encoding, pool in enumerate(pool_row):
                code, digit = divmod(code, len(pool))
                plan[iteration, slc, encoding, draw] = pool[digit]
    return plan, counts


def materialize(data, bvals, bvecs, indices):
    """Materialize one [slice, encoding, draw] plan without averaging it."""
    groups = encoding_groups(bvals, bvecs)
    shape = (*data.shape[:3], len(groups), indices.shape[-1])
    sampled = np.empty(shape, dtype=data.dtype)
    for slc, encoding in np.ndindex(indices.shape[:2]):
        sampled[:, :, slc, encoding, :] = np.take(data[:, :, slc, :], indices[slc, encoding], axis=-1)
    first = np.array([group[0] for group in groups])
    from cardpy.Data_Sorting import sorted2stacked
    return sorted2stacked(sampled, bvals[first], bvecs[first])
