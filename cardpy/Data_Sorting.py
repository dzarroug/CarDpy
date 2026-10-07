import numpy as np


def sorted2stacked(sorted_matrix, sorted_bvals, sorted_bvecs):
    """Stack in repetition-major order, preserving dtype and encoding metadata."""
    matrix = np.asarray(sorted_matrix)
    bvals, bvecs = np.asarray(sorted_bvals), np.asarray(sorted_bvecs)
    if matrix.ndim != 5 or any(size == 0 for size in matrix.shape):
        raise ValueError('sorted_matrix must be a nonempty 5D array.')
    if bvals.shape != (matrix.shape[3],) or bvecs.shape != (matrix.shape[3], 3):
        raise ValueError('Sorted encodings must match b-values and b-vectors.')
    stacked = matrix.swapaxes(3, 4).reshape(*matrix.shape[:3], -1)
    return [stacked, np.tile(bvals, matrix.shape[4]), np.tile(bvecs, (matrix.shape[4], 1))]


def stacked2layout(stacked_matrix, sorted_shape):
    """Undo stacking after a volume-wise operation; never regroup encodings."""
    matrix = np.asarray(stacked_matrix)
    if matrix.shape != tuple(sorted_shape[:3]) + (sorted_shape[3] * sorted_shape[4],):
        raise ValueError('Processed stacked data do not match the original layout.')
    return matrix.reshape(*sorted_shape[:3], sorted_shape[4], sorted_shape[3]).swapaxes(3, 4)


def sorted2dictionary(matrix, bvals, bvecs):
    """Regroup a sorted array by encoding, including singleton repetitions."""
    return stacked2dictionary(*sorted2stacked(matrix, bvals, bvecs))


def dictionary2stacked(encodings):
    """Flatten all repetitions with matching per-image encoding metadata."""
    if not encodings:
        raise ValueError('The encoding dictionary is empty.')
    images, bvals, bvecs = [], [], []
    for encoding in encodings.values():
        data = np.asarray(encoding['images'])
        if data.ndim != 4 or data.shape[3] == 0:
            raise ValueError('Dictionary images must be nonempty 4D arrays.')
        images.append(data)
        bvals.extend([encoding['bval']] * data.shape[3])
        bvecs.extend([encoding['bvec']] * data.shape[3])
    return [np.concatenate(images, axis=3), np.asarray(bvals), np.asarray(bvecs)]


def stacked2dictionary(stacked_matrix, stacked_bvals, stacked_bvecs, decimals=5):
    """Group stacked volumes by both b-value and b-vector without losing repetitions."""
    stacked_bvals = np.asarray(stacked_bvals, dtype=float)
    stacked_bvecs = np.asarray(stacked_bvecs, dtype=float)
    if stacked_matrix.ndim != 4 or stacked_bvals.ndim != 1:
        raise ValueError('stacked_matrix must have shape [rows, columns, slices, volumes].')
    if stacked_matrix.shape[3] != len(stacked_bvals) or stacked_bvecs.shape != (len(stacked_bvals), 3):
        raise ValueError('Image volumes, b-values, and b-vectors must have matching lengths.')

    if stacked_matrix.shape[3] == 0 or not np.isfinite(stacked_bvals).all() or not np.isfinite(stacked_bvecs).all() or np.any(stacked_bvals < 0):
        raise ValueError('Diffusion encodings must be nonempty, finite, and have nonnegative b-values.')

    encodings = {}
    for idx, (bval, bvec) in enumerate(zip(stacked_bvals, stacked_bvecs)):
        key = (round(float(bval), decimals), *np.round(bvec, decimals).tolist())
        if key not in encodings:
            encodings[key] = {'bval': float(bval), 'bvec': bvec.copy(), 'images': []}
        encodings[key]['images'].append(stacked_matrix[:, :, :, idx])

    for encoding in encodings.values():
        encoding['images'] = np.stack(encoding['images'], axis=3)
    return encodings


def dictionary2sorted(encodings, mode='strict', operation_type='Magnitude'):
    """Convert an encoding dictionary to CarDpy's 5D sorted representation.

    ``strict`` preserves repetitions and requires equal counts. ``average`` uses
    every repetition through the array averaging implementation and returns one
    average per encoding; set operation_type="Complex" for phase correction. ``individual`` keeps
    every image but represents each as an encoding with a singleton average;
    this is useful for registration before regrouping and averaging.
    """
    if not encodings:
        raise ValueError('The encoding dictionary is empty.')
    if mode not in ('strict', 'average', 'individual'):
        raise ValueError("mode must be 'strict', 'average', or 'individual'.")

    if mode == 'average':
        from cardpy.Data_Processing.Averaging import average
        encodings, _, _ = average(encodings, operation_type=operation_type)

    if mode == 'individual':
        matrix, bvals, bvecs = dictionary2stacked(encodings)
        return [matrix[..., np.newaxis], bvals, bvecs]

    counts = [encoding['images'].shape[3] for encoding in encodings.values()]
    if mode == 'strict' and len(set(counts)) != 1:
        raise ValueError(
            'Unequal repetitions across (b-value, b-vector) encodings: %s. '
            "Use stacked2dictionary() and dictionary2sorted(..., mode='average') "
            'to retain all repetitions.' % counts)

    matrices, bvals, bvecs = [], [], []
    for encoding in encodings.values():
        images = encoding['images']
        matrices.append(images)
        bvals.append(encoding['bval'])
        bvecs.append(encoding['bvec'])
    matrix = np.stack(matrices, axis=3)
    return [matrix, np.asarray(bvals), np.asarray(bvecs)]


def stacked2sorted(stacked_matrix, stacked_bvals, stacked_bvecs):
    """
    ########## Definition Inputs ##################################################################################################################
    # stacked_matrix        : Stacked diffusion data (4D - [rows, columns, slices, directions]).
    # stacked_bvals         : Stacked b-values.
    # stacked_bvecs         : Stacked b-vectors.
    ########## Definition Outputs #################################################################################################################
    # sorted_matrix         : Sorted diffusion data (5D - [rows, columns, slices, directions, averages]).
    # sorted_bvals          : Sorted b-values.
    # sorted_bvecs          : Sorted b-vectors.
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    encodings = stacked2dictionary(stacked_matrix, stacked_bvals, stacked_bvecs)
    return dictionary2sorted(encodings, mode='strict')
