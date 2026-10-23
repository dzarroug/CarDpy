"""Resume manual review of processed volunteers: python -m cardpy.manual_rejection.

Only registered, Gibbs-corrected interior slices are reviewed. Source data and
existing rejection outputs are never modified. Decisions index the exported
Review_Input NIfTI directly; skipped edge slices have no decision.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

DEFAULT_ROOT = Path('/Volumes/T7/Smart_Health_Reorganized/01_Processed_Data')
DEFAULT_OUTPUT = DEFAULT_ROOT.parent / '02_Manual_Shot_Rejection'


def discover(root, series=('Blip_Up', 'Blip_Down'), volunteers=None):
    selected = set(volunteers) if volunteers else None
    for volunteer in sorted(Path(root).glob('Healthy_Volunteer_*')):
        if not volunteer.is_dir() or (selected and volunteer.name not in selected):
            continue
        for name in series:
            base = volunteer / name / '01_Unrung'
            paths = [base / 'Unrung.nii', base / 'Unrung.nii.gz']
            source = next((p for p in paths if p.is_file()), None)
            if source is not None:
                yield source
            else:
                print(f'SKIP {volunteer.name}/{name}: no Unrung NIfTI')


def _write_json(path, payload):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def _metadata(source):
    stem = source.parent / 'Unrung'
    bvals = np.atleast_1d(np.loadtxt(str(stem) + '.bvals'))
    bvecs = np.loadtxt(str(stem) + '.bvecs', ndmin=2).T
    image = nib.load(source)
    if len(image.shape) != 4 or image.shape[2] < 3:
        raise ValueError(f'{source}: expected 4D data with at least three slices.')
    if bvals.shape != (image.shape[3],) or bvecs.shape != (image.shape[3], 3):
        raise ValueError(f'{source}: b-values/b-vectors do not match the NIfTI volumes.')
    if not np.isfinite(bvals).all() or not np.isfinite(bvecs).all():
        raise ValueError(f'{source}: nonfinite diffusion metadata.')
    return image, bvals, bvecs


def _register(data, bvals, bvecs, spacing, algorithm):
    from cardpy.Data_Sorting import stacked2sorted, sorted2stacked
    from cardpy.Data_Processing.Registration import register

    matrix, b, v = stacked2sorted(data, bvals, bvecs)
    # Track regrouping explicitly, then return to original NIfTI volume order.
    indices, ib, iv = stacked2sorted(np.arange(len(bvals)).reshape(1, 1, 1, -1), bvals, bvecs)
    permutation = sorted2stacked(indices, ib, iv)[0].ravel()
    aligned, b, v = register(matrix, b, v, registration_algorithm=algorithm,
                             temporary_denoising='OFF', operation_type='Magnitude',
                             mode='2D', voxel_size=spacing)
    stacked = sorted2stacked(aligned, b, v)[0]
    result = np.empty_like(stacked)
    result[..., permutation] = stacked
    return result


def _save_decisions(directory, state, bvals, bvecs):
    """Checkpoint is authoritative; CSV is a readable export of completed slices."""
    _write_json(directory / 'Review_State.json', state)
    path = directory / 'Shot_Decisions.csv'
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['review_slice_index_0based', 'source_slice_index_0based',
                         'source_slice_number_1based', 'volume_index_0based',
                         'tile_number_1based', 'b_value', 'b_x', 'b_y', 'b_z', 'decision'])
        for slc, complete in enumerate(state['completed_slices']):
            if complete:
                original = state['source_slice_indices'][slc]
                for volume, keep in enumerate(state['keep'][slc]):
                    writer.writerow([slc, original, original + 1, volume, volume + 1,
                                     bvals[volume], *bvecs[volume], 'keep' if keep else 'reject'])
    temporary.replace(path)


def review_dataset(source, output_root=DEFAULT_OUTPUT, algorithm='Affine'):
    """Review one series, resuming only slices explicitly applied by the user."""
    from cardpy.Data_Processing.Rejection import _manual_rejection_review
    from cardpy.GUI_Tools._runtime import set_scan_name

    source = Path(source).resolve()
    image, bvals, bvecs = _metadata(source)
    volunteer, series = source.parents[2].name, source.parents[1].name
    directory = Path(output_root).resolve() / volunteer / series
    directory.mkdir(parents=True, exist_ok=True)
    source_slices = list(range(1, image.shape[2] - 1))
    signature = dict(source=str(source), source_size=source.stat().st_size,
                     source_mtime_ns=source.stat().st_mtime_ns,
                     metadata_sha256=hashlib.sha256(bvals.tobytes() + bvecs.tobytes()).hexdigest(),
                     source_shape=list(image.shape), registration_algorithm=algorithm,
                     registration_mode='2D', temporary_denoising='OFF', version=1)
    state_path = directory / 'Review_State.json'
    review_path = directory / 'Review_Input.nii.gz'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['input'] != signature:
            raise ValueError(f'{directory}: input/settings changed; choose a new output folder.')
        review_image = nib.load(review_path)
        expected = (image.shape[0], image.shape[1], len(source_slices), image.shape[3])
        if review_image.shape != expected:
            raise ValueError(f'{review_path}: cached review geometry changed.')
        data = review_image.get_fdata(dtype=np.float32)
    else:
        print(f'{volunteer}/{series}: registering Gibbs-corrected interior slices...', flush=True)
        data = np.abs(np.asanyarray(image.dataobj)[:, :, 1:-1, :])
        if not np.isfinite(data).all():
            raise ValueError(f'{source}: nonfinite input signals.')
        data = np.asarray(_register(data, bvals, bvecs, image.header.get_zooms()[:3], algorithm),
                          dtype=np.float32)
        # Account for removal of source slice zero in the exported world coordinates.
        affine = image.affine.copy()
        affine[:3, 3] = nib.affines.apply_affine(image.affine, [0, 0, 1])
        header = image.header.copy()
        header.set_data_dtype(np.float32)
        header.set_slope_inter(1, 0)
        exported = nib.Nifti1Image(data, affine, header)
        temporary = directory / 'Review_Input.tmp.nii.gz'
        nib.save(exported, temporary)
        temporary.replace(review_path)
        np.savetxt(directory / 'Review_Input.bvals', bvals[None, :], fmt='%.10g')
        np.savetxt(directory / 'Review_Input.bvecs', bvecs.T, fmt='%.10g')
        state = dict(input=signature, source_slice_indices=source_slices,
                     completed_slices=[False] * len(source_slices),
                     keep=np.ones(data.shape[2:], dtype=np.uint8).tolist(),
                     review_nifti=str(review_path),
                     note='keep entries are valid only for completed_slices; edge slices were not reviewed.')
        _save_decisions(directory, state, bvals, bvecs)
        # The viewer always reads the exported file, including on the first run.
        data = nib.load(review_path).get_fdata(dtype=np.float32)
    _save_decisions(directory, state, bvals, bvecs)
    try:
        for slc, original in enumerate(source_slices):
            if state['completed_slices'][slc]:
                continue
            set_scan_name(f'{volunteer} / {series}')
            print(f'{volunteer}/{series}: source slice {original + 1}/{image.shape[2]}', flush=True)
            keep = _manual_rejection_review(
                data[:, :, slc:slc + 1, :], np.ones((1, data.shape[3]), dtype=np.uint8),
                initial_keep=np.asarray(state['keep'][slc:slc + 1]),
                bvals=bvals, bvecs=bvecs, save_decisions=False, show_automatic=False,
                slice_labels=[f'{original + 1}/{image.shape[2]} (original)'], allow_cancel=True)
            state['keep'][slc] = keep[0].tolist()
            state['completed_slices'][slc] = True
            _save_decisions(directory, state, bvals, bvecs)
    finally:
        set_scan_name(None)
    print(f'{volunteer}/{series}: complete. Decisions: {directory / "Shot_Decisions.csv"}', flush=True)
    return state


def run(root=DEFAULT_ROOT, output_root=DEFAULT_OUTPUT, series=('Blip_Up', 'Blip_Down'),
        volunteers=None, algorithm='Affine', dry_run=False):
    sources = list(discover(root, series, volunteers))
    if not sources:
        raise FileNotFoundError(f'No matching Gibbs-corrected datasets found under {root}')
    print(f'{len(sources)} series; outputs: {Path(output_root).resolve()}', flush=True)
    for source in sources:
        if dry_run:
            image, _, _ = _metadata(source)
            print(f'{source.parents[2].name}/{source.parents[1].name}: '
                  f'slices 2–{image.shape[2] - 1}, {image.shape[3]} shots per slice')
        else:
            review_dataset(source, output_root, algorithm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--series', nargs='+', choices=['Blip_Up', 'Blip_Down'],
                        default=['Blip_Up', 'Blip_Down'])
    parser.add_argument('--volunteers', nargs='+', help='e.g. Healthy_Volunteer_006')
    parser.add_argument('--algorithm', choices=['Rigid', 'Translation', 'Affine', 'Elastic'], default='Affine')
    parser.add_argument('--dry-run', action='store_true', help='Validate inventory without writing or opening a GUI.')
    args = parser.parse_args()
    try:
        run(args.root, args.output, args.series, args.volunteers, args.algorithm, args.dry_run)
    except (InterruptedError, KeyboardInterrupt):
        print('Stopped. Applied slices are saved; rerun the same command to resume.')


if __name__ == '__main__':
    main()
