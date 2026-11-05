"""Whole-shot availability and replacement, separate from voxel-wise ADC masks."""
import numpy as np


METHODS = ('none', 'random_valid', 'first_valid', 'mean_valid')
POLICIES = ('error', 'reject', 'replace')


def validate_options(missing=None, replacement=None):
    missing = dict(missing or {})
    replacement = dict(replacement or {})
    policy = missing.get('policy', 'error')
    method = replacement.get('method', 'none')
    if policy not in POLICIES:
        raise ValueError('missing_shots.policy must be error, reject, or replace.')
    if method not in METHODS:
        raise ValueError('shot_replacement.method must be none, random_valid, first_valid, or mean_valid.')
    if policy == 'replace' and method == 'none':
        raise ValueError('missing_shots.policy=replace requires a replacement method.')
    if replacement.get('all_rejected', 'first_valid') not in ('error', 'first_valid'):
        raise ValueError('shot_replacement.all_rejected must be error or first_valid.')
    seed = replacement.get('seed')
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError('shot_replacement.seed must be a nonnegative integer or None.')
    return policy, method


def availability(matrix):
    """A shot is available only if all its pixels are finite."""
    data = np.asarray(matrix)
    if data.ndim != 5 or any(n == 0 for n in data.shape):
        raise ValueError('Shot handling requires nonempty 5D sorted data.')
    return np.isfinite(data).all(axis=(0, 1))


def replace_shots(matrix, bvals, bvecs, keep, *, method='none',
                  all_rejected='first_valid', seed=None, rng=None, log=None,
                  stage='rejection'):
    """Apply decisions within each slice/encoding, recording donor identities.

    Missing positions can never be donors. No cross-encoding or cross-slice
    replacement is allowed. A fallback remains rejected in the decision mask.
    """
    validate_options(replacement=dict(method=method, all_rejected=all_rejected, seed=seed))
    rng = rng if rng is not None else (np.random.default_rng(seed) if seed is not None else np.random)
    if isinstance(matrix, dict):
        output = {}
        for dif, (key, encoding) in enumerate(matrix.items()):
            local_log = []
            images = np.asarray(encoding['images'])[:, :, :, None, :]
            filled = replace_shots(images, [encoding['bval']], [encoding['bvec']],
                np.asarray(keep[key])[:, None, :], method=method,
                all_rejected=all_rejected, seed=seed, rng=rng,
                log=local_log, stage=stage)
            output[key] = {**encoding, 'images': filled[:, :, :, 0, :]}
            for item in local_log:
                item['encoding_index'] = dif
            if log is not None:
                log.extend(local_log)
        return output
    valid = availability(matrix)
    keep = np.asarray(keep)
    if keep.shape != valid.shape or not np.isin(keep, [0, 1]).all():
        raise ValueError('Rejection mask must match slices, encodings and repetitions.')
    accepted = valid & keep.astype(bool)
    output = np.array(matrix, dtype=np.result_type(matrix, float), copy=True)
    output[:, :, ~accepted] = np.nan
    rng = rng if rng is not None else (np.random.default_rng(seed) if seed is not None else np.random)
    for slc, dif in np.ndindex(valid.shape[:2]):
        donors = np.flatnonzero(accepted[slc, dif])
        bad = np.flatnonzero(~accepted[slc, dif])
        if not len(bad):
            continue
        fallback = False
        if not len(donors):
            measured = np.flatnonzero(valid[slc, dif])
            if all_rejected == 'error' or not len(measured):
                raise ValueError(f'No accepted measured donor for slice {slc+1}, encoding {dif+1}.')
            donors = measured[:1]
            fallback = True
        if method == 'none':
            targets = donors if fallback else []
        else:
            targets = bad
        for rep in targets:
            sources = (donors if method == 'mean_valid' else
                       np.asarray([rng.choice(donors) if method == 'random_valid' else donors[0]]))
            output[:, :, slc, dif, rep] = np.mean(matrix[:, :, slc, dif, sources], axis=-1)
            if log is not None:
                log.append(dict(stage=stage, slice_index=slc, encoding_index=dif,
                    repetition_index=int(rep), donor_repetition_indices=sources.tolist(),
                    method=method, retained_as_fallback=fallback,
                    reason='missing_input' if not valid[slc, dif, rep] else 'rejected_measured_shot',
                    bval=float(bvals[dif]), bvec=np.asarray(bvecs[dif]).tolist()))
    return output


def prepare_missing_shots(matrix, bvals, bvecs, missing=None, replacement=None, *, log=None, rng=None):
    policy, method = validate_options(missing, replacement)
    seed = (replacement or {}).get('seed')
    rng = rng if rng is not None else (np.random.default_rng(seed) if seed is not None else np.random)
    if isinstance(matrix, dict):
        output = {}
        # Prepare encodings separately because padding is not an acquisition.
        for dif, (key, encoding) in enumerate(matrix.items()):
            local_log = []
            images = prepare_missing_shots(np.asarray(encoding['images'])[:, :, :, None, :],
                [encoding['bval']], [encoding['bvec']], missing, replacement, log=local_log, rng=rng)
            output[key] = {**encoding, 'images': images[:, :, :, 0, :]}
            for item in local_log:
                item['encoding_index'] = dif
            if log is not None:
                log.extend(local_log)
        return output
    valid = availability(matrix)
    if valid.all():
        return matrix
    if policy == 'error':
        raise ValueError('Input contains missing/nonfinite shots; choose missing_shots.policy=reject or replace.')
    if not valid.any(axis=-1).all():
        raise ValueError('An encoding has no measured repetition in a slice; missing shots cannot be supplied.')
    if policy == 'reject':
        output = np.array(matrix, dtype=np.result_type(matrix, float), copy=True)
        output[:, :, ~valid] = np.nan
        return output
    options = dict(replacement or {})
    return replace_shots(matrix, bvals, bvecs, valid, method=method,
        all_rejected='error', seed=options.get('seed'), rng=rng, log=log, stage='input')


def reject_missing_shots(core, matrix, bvals, bvecs, options):
    """Score only measured shots, then restore keep masks to their original slots."""
    dictionary = isinstance(matrix, dict)
    if dictionary:
        entries = matrix
    else:
        entries = {i: dict(images=matrix[:, :, :, i, :], bval=bvals[i], bvec=bvecs[i])
                   for i in range(matrix.shape[3])}
    slices = next(iter(entries.values()))['images'].shape[2]
    keep = {key: np.zeros(e['images'].shape[2:], dtype=np.uint8) for key, e in entries.items()}
    coordinates = [[], [], [], []]
    for slc in range(slices):
        compact, positions = {}, {}
        for key, encoding in entries.items():
            images = np.asarray(encoding['images'])
            present = np.isfinite(images[:, :, slc]).all(axis=(0, 1))
            positions[key] = np.flatnonzero(present)
            if not present.any():
                raise ValueError(f'No measured shots for slice {slc+1}, encoding {key}.')
            compact[key] = {**encoding, 'images': images[:, :, slc:slc+1, positions[key]]}
        kwargs = dict(options)
        if kwargs.get('diagnostics_path'):
            import os
            kwargs['diagnostics_path'] = os.path.join(kwargs['diagnostics_path'], 'Missing_Shot_Rejection', f'Slice_{slc+1:03d}')
        ml_source = kwargs.pop('ml_input', None)
        if kwargs.get('rejection_mode', 'algorithm') == 'algorithm':
            result = core(compact, bvals, bvecs, **kwargs)
        else:
            # Manual/ML core accepts sorted data. Each measured volume occupies
            # one column; restore its decision into the original encoding below.
            images = np.concatenate([e['images'] for e in compact.values()], axis=3)
            bb = np.concatenate([np.repeat(e['bval'], e['images'].shape[3]) for e in compact.values()])
            vv = np.concatenate([np.repeat(np.asarray(e['bvec'])[None], e['images'].shape[3], axis=0) for e in compact.values()])
            if ml_source is not None:
                if dictionary:
                    from cardpy.Data_Sorting import sorted2dictionary
                    source_entries = ml_source if isinstance(ml_source, dict) else sorted2dictionary(ml_source, bvals, bvecs)
                else:
                    source_entries = {i: dict(images=ml_source[:, :, :, i, :]) for i in entries}
                ml_images = np.concatenate([source_entries[key]['images'][:, :, slc:slc+1, positions[key]] for key in compact], axis=3)
                kwargs['ml_input'] = ml_images[:, :, :, :, None]
            result = core(images[:, :, :, :, None], bb, vv, **kwargs)
            decisions = result[5][0, :, 0]
            offset, mapped = 0, {}
            for key, indices in positions.items():
                mapped[key] = decisions[None, offset:offset+len(indices)]
                offset += len(indices)
            result[5] = mapped
        for key in entries:
            keep[key][slc, positions[key]] = result[5][key][0]
        for axis in range(4):
            coordinates[axis].append(int(np.asarray(result[3][axis]).ravel()[0]))
    if dictionary:
        stats = {key: np.mean(k, axis=-1)*100 for key, k in keep.items()}
    else:
        keep = np.stack(list(keep.values()), axis=1)
        stats = np.mean(keep, axis=-1)*100
    return coordinates, stats, keep


def unrung_present(core, matrix, bvals, bvecs, **kwargs):
    """Gibbs removal is independent per image: process measured images only."""
    valid = availability(matrix)
    if valid.all():
        return core(matrix, bvals, bvecs, **kwargs)
    rows, cols, slices, directions, repetitions = matrix.shape
    output = np.full(matrix.shape, np.nan, dtype=np.result_type(matrix, float))
    for slc in range(slices):
        present = np.flatnonzero(valid[slc].ravel())
        if not len(present):
            continue
        plane = matrix[:, :, slc].reshape(rows, cols, -1)
        compact = plane[:, :, present][:, :, None, :, None]
        corrected = core(compact, np.repeat(bvals, repetitions)[present],
            np.repeat(bvecs, repetitions, axis=0)[present], **kwargs)[0]
        restored = np.full(plane.shape, np.nan, dtype=output.dtype)
        restored[:, :, present] = corrected[:, :, 0, :, 0]
        output[:, :, slc] = restored.reshape(rows, cols, directions, repetitions)
    return [output, bvals, bvecs]
