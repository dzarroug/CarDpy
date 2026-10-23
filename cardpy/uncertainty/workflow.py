"""Run each resampled dataset through the current CarDpy numerical pipeline."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .bootstrap import make_plan, materialize
from .metrics import invariant_uncertainty, angular_uncertainty

CHANNELS = ['MD', 'FA', 'E1x', 'E1y', 'E1z', 'E2x', 'E2y', 'E2z', 'E3x', 'E3y', 'E3z']


def processing_code_sha256():
    """Fingerprint the processing code for manifests and notebook output paths."""
    root = Path(__file__).resolve().parents[1]
    code_paths = sorted(root.rglob('*.py'))
    digest = hashlib.sha256()
    for path in code_paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def discover_datasets(root):
    """Find saved CarDpy runs with settings and pre-averaging images."""
    root = Path(root).expanduser().resolve()
    candidates = [root] if (root / 'processing_settings.json').is_file() else sorted(
        p.parent for p in root.rglob('processing_settings.json') if not p.name.startswith('._'))
    return [p for p in candidates if any(p.glob('*_Original/Original.nii*'))
            or any(p.glob('*_Rejected/Rejected.nii*'))]


def load_dataset(directory, config=None):
    from cardpy.Data_Import import NifTi_Reader
    directory = Path(directory).expanduser().resolve()
    settings_path = directory / 'processing_settings.json'
    if config is None:
        if not settings_path.exists():
            raise ValueError('No processing_settings.json; pass the original processing config explicitly.')
        settings = json.loads(settings_path.read_text())
        config = settings.get('cardpy', settings)
    config = deepcopy(config)
    if not all(key in config for key in ('operation_type', 'registration', 'averaging', 'dti')):
        raise ValueError(f'{settings_path} must contain the complete CarDpy configuration.')
    stage = ('Rejected' if config['rejection']['enabled'] else
             'Unrung' if config['gibbs']['enabled'] else 'Original')
    paths = sorted(p for p in directory.glob(f'*_{stage}/{stage}.nii*') if not p.name.startswith('._'))
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one saved {stage} NIfTI in {directory}.')
    image = paths[0]
    stem = image.parent / stage
    data, bvals, bvecs, header, _, _ = NifTi_Reader(str(image), str(stem)+'.bvals',
                                           str(stem)+'.bvecs', str(stem)+'.header', info='OFF')
    if config['operation_type'] == 'Complex' and not np.iscomplexobj(data):
        raise ValueError('Complex processing requires retained complex images; magnitude images cannot recover phase.')
    if not config['averaging']['enabled']:
        raise ValueError('Notebook bootstrap workflow requires averaging.enabled=True.')
    # Resume after the selected pre-averaging stage; never repeat rejection.
    config['gibbs']['enabled'] = False
    config['rejection']['enabled'] = False
    config.setdefault('gui', {}).update(saved_configuration_mode='reuse', segmentation_review=False)
    config.setdefault('uncertainty', {})['enabled'] = False
    config['output']['directory'] = str(directory)  # Read saved respiratory ROI; save=False throughout.
    header.pop('_Shot Metadata', None)  # New bootstrap indices are stored separately.
    sources = [image, Path(str(stem)+'.bvals'), Path(str(stem)+'.bvecs'),
               Path(str(stem)+'.header')]
    if settings_path.exists():
        sources.append(settings_path)
    metadata_path = Path(str(stem)+'.shots.json')
    warnings = []
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        sources.append(metadata_path)
        for row in metadata.get('volumes', []):
            if not row['available']:
                data[:, :, row['slice_index'], row['volume_index']] = np.nan
        fallbacks = [row for row in metadata.get('rejection_decisions', []) if row.get('retained_as_fallback')]
        if fallbacks:
            warnings.append(f'{len(fallbacks)} all-rejected fallback shots retained, matching the original pipeline.')
    respiratory = config['respiratory']
    if respiratory['enabled']:
        crop = directory / '13_Diagnostics' / 'Respiratory_Crop.json'
        coordinates = respiratory.get('crop_coordinates')
        if coordinates is None and not crop.exists():
            raise ValueError(f'Missing saved respiratory ROI: {crop}. Save the ROI in normal processing first.')
        if coordinates is None:
            sources.append(crop)
            coordinates = json.loads(crop.read_text())
        respiratory['crop_coordinates'] = coordinates
    return data, np.asarray(bvals), np.asarray(bvecs), header, config, sources, warnings


def process_realization(data, bvals, bvecs, header, config, indices):
    """Same processing as one dataset; deliberately stops before contours/HA."""
    from cardpy.pipeline import process
    sampled, sampled_b, sampled_v = materialize(data, bvals, bvecs, indices)
    return process('.', deepcopy(config), save=False,
                   input_data=(sampled, sampled_b, sampled_v, dict(header)), stop_after_dti=True)


def _worker(index, data, bvals, bvecs, header, config, indices, output, keep_images):
    output = Path(output)
    target = output / 'samples' / f'{index:04d}.npy'
    if target.exists():
        return index
    # Each worker owns its log and atomic output. Workers never write to source runs.
    with (output / 'logs' / f'{index:04d}.txt').open('w') as log, redirect_stdout(log), redirect_stderr(log):
        result = process_realization(data, bvals, bvecs, header, config, indices)
        metrics, eigenvectors = result['dti_metrics'], result['eigenvectors']
        sample = np.concatenate([metrics['MD'][..., None], metrics['FA'][..., None]] +
                                [eigenvectors[key] for key in ('E1', 'E2', 'E3')], axis=-1)
        temporary = target.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            np.save(handle, sample.astype(np.float32))
        if keep_images:
            image_target = output / 'images' / f'{index:04d}.npz'
            with image_target.with_suffix('.tmp').open('wb') as handle:
                np.savez_compressed(handle, matrix=result['matrix'], bvals=result['bvals'], bvecs=result['bvecs'])
            os.replace(image_target.with_suffix('.tmp'), image_target)
        os.replace(temporary, target)
    return index


def summarize(output, iterations, chunk_voxels=2048):
    """Reduce saved samples in bounded memory; preserve full spatial grid."""
    from scipy.io import savemat
    output = Path(output)
    samples = [np.load(output/'samples'/f'{i:04d}.npy', mmap_mode='r') for i in range(iterations)]
    shape = samples[0].shape[:-1]
    if any(sample.shape != shape + (len(CHANNELS),) for sample in samples):
        raise ValueError('Bootstrap sample shapes are inconsistent.')
    flat = [sample.reshape(-1, len(CHANNELS)) for sample in samples]
    nvox = flat[0].shape[0]
    maps = {}
    for start in range(0, nvox, chunk_voxels):
        end = min(start + chunk_voxels, nvox)
        block = np.stack([sample[start:end] for sample in flat], axis=-1)
        reduced = {}
        for channel, name in enumerate(('MD', 'FA')):
            stats = invariant_uncertainty(block[:, channel, :])
            reduced.update({f'm{name}': stats['median'], f'{name}_lower': stats['lower'],
                            f'{name}_upper': stats['upper'], f'd{name}': stats['width'],
                            f'{name}_n': stats['count']})
        for j in range(3):
            stats = angular_uncertainty(block[:, 2+3*j:5+3*j, :])
            reduced.update({f'dE{j+1}': stats['cone'], f'Psi{j+1}': stats['mean'],
                            f'K{j+1}': stats['coherence'], f'E{j+1}_n': stats['count']})
        for name, values in reduced.items():
            if name not in maps:
                maps[name] = np.empty((nvox,) + values.shape[1:], dtype=values.dtype)
            maps[name][start:end] = values
    maps = {key: values.reshape(shape + values.shape[1:]) for key, values in maps.items()}
    with (output/'uncertainty.tmp').open('wb') as handle:
        np.savez_compressed(handle, **maps)
    os.replace(output/'uncertainty.tmp', output/'uncertainty.npz')
    savemat(output/'Invariant_Uncertainties.mat', {key: maps[key] for key in ('dMD', 'dFA', 'MD_lower', 'MD_upper', 'FA_lower', 'FA_upper')})
    savemat(output/'Invariant_Medians.mat', {key: maps[key] for key in ('mMD', 'mFA')})
    savemat(output/'Angular_Uncertainties.mat', {key: value for key, value in maps.items() if key.startswith(('dE', 'Psi', 'K', 'E'))})
    return output/'uncertainty.npz'


def run_dataset(directory, output=None, *, config=None, iterations=None, n_jobs=None,
                seed=None, desired_averages=None, keep_images=None):
    """Run separately or with config['uncertainty']; save to 14_Uncertainty.

    Explicit keyword options override config. Old runs without saved settings
    require their original full config. Completed samples resume only when the
    inputs, settings, sampling plan, and processing code match.
    """
    from joblib import Parallel, delayed, parallel_config
    directory = Path(directory).expanduser().resolve()
    if config is None:
        settings_path = directory / 'processing_settings.json'
        if settings_path.exists():
            settings = json.loads(settings_path.read_text())
            config = settings.get('cardpy', settings)
    options = (config or {}).get('uncertainty', {})
    iterations = options.get('iterations', 400) if iterations is None else iterations
    n_jobs = options.get('n_jobs', 4) if n_jobs is None else n_jobs
    seed = options.get('seed', 0) if seed is None else seed
    if desired_averages is None:
        desired_averages = options.get('desired_averages')
    keep_images = options.get('keep_images', False) if keep_images is None else keep_images
    if not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError('n_jobs must be a nonzero integer.')
    output = (directory / '14_Uncertainty' if output is None
              else Path(output).expanduser().resolve())
    if output == directory or output in directory.parents:
        raise ValueError('Uncertainty output must not be the source run or its parent.')
    if directory in output.parents and output != directory / '14_Uncertainty':
        raise ValueError('Within a source run, use its dedicated 14_Uncertainty folder.')
    data, bvals, bvecs, header, config, sources, warnings = load_dataset(directory, config=config)
    plan, counts = make_plan(data, bvals, bvecs, iterations=iterations,
                            desired_averages=desired_averages, seed=seed)
    warnings += [f'Slice {s}, encoding {e}: only one retained repetition; no empirical variability for this encoding.'
                 for s, e in np.argwhere(counts == 1)]
    manifest = dict(schema_version=1, source=str(directory), iterations=iterations, seed=seed,
                    desired_averages=int(plan.shape[-1]), channels=CHANNELS,
                    retained_counts=counts.tolist(), warnings=warnings,
                    plan_sha256=hashlib.sha256(plan.tobytes()).hexdigest(),
                    code_sha256=processing_code_sha256(), processing_config=json.loads(
                        json.dumps(config, default=lambda value: value.tolist()
                                   if isinstance(value, np.ndarray) else os.fspath(value))),
                    sources=[dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in sources],
                    md_units='um^2/ms (CarDpy scaling)', fa_units='dimensionless', angle_units='degrees',
                    keep_images=keep_images,
                    resampling='unique whole-image composite rows, grouped into desired_averages draws; per slice',
                    spatial_frame='CarDpy image grid; registration preserves acquired b-vectors',
                    input_note='Saved pre-averaging images; each realization repeats enabled registration and downstream processing',
                    interval_method='Voxel Bank order statistics; scalar indices round(.025*N)-1 and N-round(.025*N)-1; angular round(.95*N)-1')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        changed = sorted(key for key in previous.keys() | manifest.keys()
                         if previous.get(key) != manifest.get(key))
        if changed:
            raise ValueError(f'Output manifest differs from this run ({", ".join(changed)}): '
                             f'{manifest_path}. Choose a new output folder '
                             'and rerun. Existing samples are preserved.')
    elif any(output.iterdir()):
        raise ValueError('Output folder is nonempty without a manifest. Choose a new output folder.')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
        np.savez_compressed(output/'resampling_plan.npz', volume_indices=plan, bvals=bvals, bvecs=bvecs)
    for name in ('samples', 'logs') + (('images',) if keep_images else ()):
        (output/name).mkdir(exist_ok=True)
    status_path = output/'status.json'
    status_path.write_text(json.dumps(dict(status='running', iterations=iterations,
                                           code_sha256=manifest['code_sha256'])))
    print(f'{directory.name}: {iterations} realizations, {plan.shape[-1]} draws/encoding, n_jobs={n_jobs}', flush=True)
    for warning in warnings:
        print('Bootstrap note:', warning, flush=True)
    try:
        with parallel_config(backend='loky', inner_max_num_threads=1):
            results = Parallel(n_jobs=n_jobs, return_as='generator_unordered', batch_size=1)(
                delayed(_worker)(i, data, bvals, bvecs, header, config, plan[i], str(output), keep_images)
                for i in range(iterations))
            for completed, _ in enumerate(results, 1):
                print(f'Completed {completed}/{iterations}', flush=True)
        result = summarize(output, iterations)
    except Exception as exc:
        status_path.write_text(json.dumps(dict(status='failed', error=str(exc),
                                               code_sha256=manifest['code_sha256'])))
        raise
    status_path.write_text(json.dumps(dict(status='complete', iterations=iterations,
                                           code_sha256=manifest['code_sha256'])))
    return result
