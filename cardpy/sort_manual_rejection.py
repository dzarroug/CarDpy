"""Regroup completed shot reviews by encoding without repeating registration.

python -m cardpy.sort_manual_rejection --root /path/to/data --apply
The original CarDpy_Manual_Rejection directories remain the source of truth;
verified copies are written under CarDpy_Manual_Rejection_Sorted.
"""
import argparse
import copy
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from cardpy.Data_Sorting import stacked2sorted, sorted2stacked, stacked2layout
from cardpy.Data_Processing.Rejection import _load_rejection_configuration


def regroup(configuration):
    """Return the new configuration, sorted layout, and new-to-old shot indices."""
    b = np.asarray(configuration['bvals'])
    v = np.asarray(configuration['bvecs'])
    indices = np.arange(len(b)).reshape(1, 1, 1, -1)
    layout, sb, sv = stacked2sorted(indices, b, v)
    permutation = sorted2stacked(layout, sb, sv)[0].ravel()
    result = copy.deepcopy(configuration)
    result['bvals'] = b[permutation].tolist()
    result['bvecs'] = v[permutation].tolist()
    for key in ('automatic_keep', 'final_keep'):
        before = np.asarray(configuration[key])
        after = before[:, permutation]
        np.testing.assert_array_equal(after[:, np.argsort(permutation)], before)
        result[key] = after.tolist()
    shape = (*configuration['image_shape'][:3], *layout.shape[3:])
    return result, shape, permutation, sb, sv


def remap_decisions(decisions, permutation):
    """Reindex the readable decision records as well as their underlying masks."""
    result = copy.deepcopy(decisions)
    inverse = np.argsort(permutation)
    for key in ('automatic_keep', 'final_keep'):
        result[key] = np.asarray(decisions[key])[:, permutation].tolist()
    for record in result['decisions']:
        old = record['stacked_image_index']
        new = int(inverse[old])
        record['source_stacked_image_index'] = record.get('source_stacked_image_index', old)
        record['stacked_image_index'] = new
    result['decisions'].sort(key=lambda row: (row['slice_index'], row['stacked_image_index']))
    return result


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def convert(source, apply=False):
    source = Path(source)
    destination = source.parent.parent / 'CarDpy_Manual_Rejection_Sorted' / source.name
    config_path = source / '13_Diagnostics' / 'Rejection_Configuration.json'
    original = json.loads(config_path.read_text())
    configuration, shape, permutation, sb, sv = regroup(original)
    original_b = np.asarray(original['bvals'])
    original_v = np.asarray(original['bvecs'])
    stages = []
    for stage, name in (('00_Original', 'Original'), ('01_Unrung', 'Unrung'),
                        ('02_Rejected', 'Rejected')):
        stem = source / stage / name
        image = nib.load(str(stem) + '.nii')
        if image.shape != tuple(original['image_shape']):
            raise ValueError(f'{stem}: image shape differs from saved decisions.')
        np.testing.assert_allclose(np.loadtxt(str(stem) + '.bvals'), original_b, atol=0.051, rtol=0)
        np.testing.assert_allclose(np.loadtxt(str(stem) + '.bvecs').T, original_v, atol=0.0000051, rtol=0)
        stages.append((stage, name, stem, image))
    decision_path = source / '13_Diagnostics' / 'Rejection_Decisions.json'
    decisions = json.loads(decision_path.read_text())
    for key in ('automatic_keep', 'final_keep'):
        np.testing.assert_array_equal(decisions[key], original[key])
    remapped = remap_decisions(decisions, permutation)
    for record in remapped['decisions']:
        slc, idx = record['slice_index'], record['stacked_image_index']
        assert record['manual_decision'] == ('keep' if configuration['final_keep'][slc][idx] else 'reject')
        np.testing.assert_allclose(record['b_value'], configuration['bvals'][idx], atol=1e-12, rtol=0)
        np.testing.assert_allclose(record['b_vector'], configuration['bvecs'][idx], atol=1e-12, rtol=0)
    rejected_count = int(np.count_nonzero(np.asarray(configuration['final_keep']) == 0))
    print(f'{source.parent.parent.name}/{source.name}: {shape[3]} encodings x '
          f'{shape[4]} repetitions; {rejected_count} rejected choices preserved.', flush=True)
    if not apply:
        return destination
    if destination.exists():
        raise FileExistsError(f'Output already exists: {destination}')
    pending = destination.with_name(destination.name + '.pending')
    pending.mkdir(parents=True, exist_ok=False)
    for stage, name, stem, image in stages:
        directory = pending / stage
        directory.mkdir()
        values = np.asanyarray(image.dataobj)[..., permutation]
        exported = nib.Nifti1Image(values, image.affine, image.header.copy())
        target = directory / (name + '.nii')
        nib.save(exported, target)
        # Verify stored voxel values and exact inverse ordering, including NaNs.
        restored = np.asanyarray(nib.load(target).dataobj)
        np.testing.assert_array_equal(restored[..., np.argsort(permutation)],
                                      np.asanyarray(image.dataobj))
        np.testing.assert_array_equal(nib.load(target).affine, image.affine)
        np.savetxt(directory / (name + '.bvals'), np.asarray(configuration['bvals'])[None], fmt='%.17g')
        np.savetxt(directory / (name + '.bvecs'), np.asarray(configuration['bvecs']).T, fmt='%.17g')
        header_path = Path(str(stem) + '.header')
        (directory / (name + '.header')).write_bytes(header_path.read_bytes())
        if name == 'Rejected':
            sorted_values = stacked2layout(values, shape)
            sorted_path = directory / 'Rejected_Sorted_5D.nii.gz'
            nib.save(nib.Nifti1Image(sorted_values, image.affine, image.header.copy()), sorted_path)
            np.testing.assert_array_equal(np.asanyarray(nib.load(sorted_path).dataobj), sorted_values)
    diag = pending / '13_Diagnostics'
    diag.mkdir()
    _write_json(diag / 'Rejection_Configuration.json', configuration)
    _write_json(diag / 'Rejection_Decisions.json', remapped)
    keep = np.asarray(configuration['final_keep'], dtype=np.uint8)
    sorted_keep = stacked2layout(keep[None, None], (1, 1, *shape[2:]))[0, 0]
    np.save(diag / 'Rejection_Keep_Sorted.npy', sorted_keep)
    _write_json(pending / 'Encoding_Layout.json', dict(
        source=str(source.resolve()), sorted_shape=list(shape),
        axes=['rows', 'columns', 'slices', 'encodings', 'repetitions'],
        encoding_bvals=sb.tolist(), encoding_bvecs=sv.tolist(),
        new_to_source_volume_index=permutation.tolist(),
        preserved_rejected_choices=rejected_count,
        note='Voxel values and review choices preserved; registration was not repeated.'))
    # Check that the regular rejection API can reuse the remapped decisions.
    example = np.broadcast_to(np.array(0.), configuration['image_shape'])
    _, _, loaded_keep = _load_rejection_configuration(
        diag, example, np.asarray(configuration['bvals']),
        np.asarray(configuration['bvecs']), configuration['operation_type'])
    np.testing.assert_array_equal(loaded_keep, keep)
    pending.rename(destination)
    print(f'  Verified output: {destination}', flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--apply', action='store_true', help='Write verified sorted copies; otherwise validate only.')
    args = parser.parse_args()
    sources = sorted(args.root.glob('V*/CarDpy_Manual_Rejection/*/13_Diagnostics/Rejection_Configuration.json'))
    if not sources:
        raise FileNotFoundError(f'No completed rejection configurations under {args.root}')
    # Validate the whole batch before writing any output.
    for config in sources:
        convert(config.parents[1])
    if args.apply:
        for config in sources:
            convert(config.parents[1], apply=True)


if __name__ == '__main__':
    main()
