"""Magnitude-driven slice or volume registration, applied to original signals."""
# Original implementation by Tyler E. Cork, CMR Group, Stanford University, 2022.
import numpy as np


def _estimate_transform(static, moving, algorithm, grid, level_iters):
    from dipy.align.imaffine import (
        AffineRegistration, MutualInformationMetric, transform_centers_of_mass,
    )
    from dipy.align.transforms import (
        TranslationTransform2D, RigidTransform2D, AffineTransform2D,
        TranslationTransform3D, RigidTransform3D,
    )
    from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from dipy.align.metrics import EMMetric

    dimension = static.ndim
    initial = transform_centers_of_mass(static, grid, moving, grid)
    registration = AffineRegistration(
        metric=MutualInformationMetric(nbins=32), level_iters=list(level_iters),
        sigmas=[3.0, 2.0, 0.0], factors=[2, 1, 1], verbosity=0,
    )
    transforms = ((TranslationTransform2D, RigidTransform2D, AffineTransform2D)
                  if dimension == 2 else
                  (TranslationTransform3D, RigidTransform3D))
    count = {'Translation': 1, 'Rigid': 2, 'Affine': 3, 'Elastic': 3}[algorithm]
    mapping = initial
    for transform in transforms[:count]:
        mapping = registration.optimize(
            static, moving, transform(), None,
            static_grid2world=grid, moving_grid2world=grid,
            starting_affine=mapping.affine,
        )
    if algorithm == 'Elastic':
        # SDR's prealign composes the affine with the deformation. Estimating
        # on an affine-warped image and applying only the deformation loses it.
        registration = SymmetricDiffeomorphicRegistration(
            EMMetric(dimension), level_iters=[100, 50, 25],
        )
        registration.verbosity = 0
        mapping = registration.optimize(
            static, moving, static_grid2world=grid, moving_grid2world=grid,
            prealign=mapping.affine,
        )
    return mapping


def _apply_transform(mapping, image):
    if np.iscomplexobj(image):
        return mapping.transform(image.real) + 1j * mapping.transform(image.imag)
    return mapping.transform(image)


def register(original_matrix, original_bvals, original_bvecs,
             registration_algorithm='Rigid', temporary_denoising='OFF',
             operation_type='Magnitude', mode='2D', voxel_size=None,
             level_iters=(10000, 1000, 100), b0_threshold=0):
    """Register to a lowest-b encoding without changing acquisition order.

    ########## Definition Inputs ##############################################
    original_matrix : Sorted data [rows, columns, slices, encodings, repetitions].
    original_bvals  : One measured b-value per encoding.
    original_bvecs  : One three-component b-vector per encoding.
    ########## Definition Outputs #############################################
    Registered data with the original shape, b-values, and b-vectors.

    Each repetition is first aligned to its lowest-b reference, then to the
    first repetition's reference. A singleton repetition needs only one pass.
    3D supports only Rigid and requires multiple contiguous slices; ``voxel_size`` supplies physical
    axis spacing in mm (unit spacing when omitted). Complex signals share one
    magnitude-estimated transform for their real and imaginary components.

    B-values and b-vectors retain the existing CarDpy convention: image
    registration does not reorient gradients. This is not a replacement for
    diffusion motion correction with per-volume gradient reorientation.
    """
    from dipy.align.imaffine import AffineMap

    if mode not in ('2D', '3D'):
        raise ValueError("mode must be '2D' or '3D'.")
    if registration_algorithm not in ('Translation', 'Rigid', 'Affine', 'Elastic'):
        raise ValueError('Unknown registration algorithm: %s' % registration_algorithm)
    if mode == '3D' and registration_algorithm != 'Rigid':
        raise ValueError('3D registration supports only Rigid; set registration_algorithm="Rigid".')
    if operation_type not in ('Magnitude', 'Complex'):
        raise ValueError("operation_type must be 'Magnitude' or 'Complex'.")
    if temporary_denoising not in ('ON', 'OFF'):
        raise ValueError("temporary_denoising must be 'ON' or 'OFF'.")
    data = np.asarray(original_matrix)
    bvals, bvecs = np.asarray(original_bvals), np.asarray(original_bvecs)
    if data.ndim != 5 or any(size == 0 for size in data.shape):
        raise ValueError('Registration requires a nonempty 5D sorted matrix.')
    if bvals.shape != (data.shape[3],) or bvecs.shape != (data.shape[3], 3):
        raise ValueError('Registration encodings must match b-values and b-vectors.')
    if not np.isfinite(bvals).all() or not np.isfinite(bvecs).all():
        raise ValueError('Registration encodings must be finite.')
    if mode == '3D' and data.shape[2] < 2:
        raise ValueError('3D registration requires multiple contiguous slices; use mode="2D" for a single slice.')
    spacing = np.ones(3) if voxel_size is None else np.asarray(voxel_size, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('voxel_size must contain three positive finite spacings.')
    if len(level_iters) != 3 or any(int(value) != value or value <= 0 for value in level_iters):
        raise ValueError('level_iters must contain three positive integer iteration counts.')
    data = np.abs(data) if operation_type == 'Magnitude' else data
    if not np.isfinite(data).all():
        raise ValueError('Registration signals must be finite.')
    guide = np.abs(data)
    if temporary_denoising == 'ON':
        from cardpy.Data_Processing.Denoising import denoise
        denoised, _, _ = denoise(
            guide, bvals, bvecs, numCoils=20, denoising_algorithm='Patch2Self',
            operation_type='Magnitude', b0_threshold=b0_threshold,
        )
        guide = np.abs(denoised)
    reference = int(np.argmin(bvals))
    dtype = np.complex128 if np.iscomplexobj(data) else np.float64
    registered = np.empty(data.shape, dtype=dtype)
    dimension = 2 if mode == '2D' else 3
    grid = np.diag(np.r_[spacing[:dimension], 1.0])
    spatial_indices = range(data.shape[2]) if mode == '2D' else [None]
    print('Registering %s data to lowest b-value (%g).' % (mode, bvals[reference]))
    for slc in spatial_indices:
        spatial = (slice(None), slice(None), slc) if mode == '2D' else (slice(None),) * 3
        first = guide[spatial + (reference, 0)]
        if not np.any(first > 0):
            raise ValueError('The lowest-b registration reference is empty.')
        for avg in range(data.shape[4]):
            static = guide[spatial + (reference, avg)]
            if not np.any(static > 0):
                raise ValueError('A lowest-b registration reference is empty.')
            average_map = None if avg == 0 or np.array_equal(first, static) else _estimate_transform(
                first, static, registration_algorithm, grid, level_iters)
            for dif in range(data.shape[3]):
                index = spatial + (dif, avg)
                moving, signal = guide[index], data[index]
                if not np.any(moving > 0):
                    registered[index] = signal
                    continue
                direction_map = None if dif == reference or np.array_equal(static, moving) else _estimate_transform(
                    static, moving, registration_algorithm, grid, level_iters)
                if direction_map is not None and average_map is not None and registration_algorithm != 'Elastic':
                    # Pullback: first reference -> repetition reference -> image.
                    combined = AffineMap(
                        direction_map.affine @ average_map.affine,
                        domain_grid_shape=first.shape, domain_grid2world=grid,
                        codomain_grid_shape=moving.shape, codomain_grid2world=grid,
                    )
                    signal = _apply_transform(combined, signal)
                else:
                    if direction_map is not None:
                        signal = _apply_transform(direction_map, signal)
                    if average_map is not None:
                        signal = _apply_transform(average_map, signal)
                registered[index] = signal
    return [registered, original_bvals, original_bvecs]
