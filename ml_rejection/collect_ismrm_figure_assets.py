#!/usr/bin/env python3
"""Collect only the inputs needed to redesign ISMRM rejection figures.
Run from the repository root OR ml_rejection folder:
    python collect_ismrm_figure_assets.py
Optional explicit result locations:
    python collect_ismrm_figure_assets.py --purl /path/rejection_comparison_purl --cima /path/rejection_comparison_cima
No input files are modified. Output: ismrm_figure_package.zip and folder.
"""
import argparse
import csv
import shutil
import zipfile
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--purl', type=Path, help='PURL rejection_comparison results directory')
ap.add_argument('--cima', type=Path, help='CIMA rejection_comparison results directory')
ap.add_argument('--include-cache-images', action='store_true', help='Include potentially huge cache/X.npy; not needed if supplied images suffice')
ap.add_argument('--output', type=Path, default=Path('ismrm_figure_package'))
args = ap.parse_args()

cwd = Path.cwd().resolve()
ml = cwd if (cwd / 'make_fig4.py').is_file() else cwd / 'ml_rejection'
if not (ml / 'make_fig4.py').is_file():
    ap.error('Run from repo root (containing ml_rejection/) or from ml_rejection/')
repo = ml.parent
out = args.output.resolve()
if out == ml or out == repo:
    ap.error('Output must be a separate directory')
out.mkdir(parents=True, exist_ok=True)
manifest = []
missing = []

def copy(source, destination):
    destination = out / destination
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest.append((str(destination.relative_to(out)), source.stat().st_size, str(source)))
        return True
    missing.append(str(source))
    return False

def locate(explicit, name):
    if explicit:
        d = explicit.expanduser().resolve()
        if not d.is_dir():
            print(f'WARNING: {name} not found at {d}')
            return None
        return d
    candidates = [repo / name, ml / name, repo.parent / name, cwd / name]
    for d in candidates:
        if d.is_dir():
            return d.resolve()
    print(f'WARNING: Could not locate {name}. Use --{ "purl" if "purl" in name else "cima" } /absolute/path')
    return None

figures = ['fig2a.png','fig2b.png','fig2b.gif','fig2b_candidates.png',
           'fig4a_purl.png','fig4a_cima.png','fig4b_cima.png',
           'fig4_values_purl.csv','fig4_values_cima.csv','fig4_values.csv']
for name in figures:
    copy(ml / 'figures' / name, Path('existing_figures') / name)
for png in sorted((ml / 'figures' / 'fig2b_frames').glob('slice_*.png')):
    copy(png, Path('existing_figures/fig2b_frames') / png.name)
for name in ['make_fig2.py', 'make_fig2b_gif.py', 'make_fig4.py']:
    copy(ml / name, Path('scripts') / name)
for path in ['models/crop80/test_predictions.csv', 'cache/meta.csv']:
    copy(ml / path, Path('classifier') / path)
# Keep the checkpoint if present (small relative to cache), useful to validate crop/threshold.
copy(ml / 'models/crop80/best.pt', Path('classifier/models/crop80/best.pt'))
if args.include_cache_images:
    copy(ml / 'cache/X.npy', Path('classifier/cache/X.npy'))

config_map = {'purl': ['none','manual','algorithm','ml'],
              'cima': ['none','algorithm','ml']}
# Explicit participant allowlists prevent picking up unrelated or copied subdirectories.
participants = {
    'cima': [f'V{i:03d}' for i in range(6)],
    'purl': [f'Healthy_Volunteer_{i:03d}' for i in (19, 34, 35, 36, 39)],
}
for ds, foldername, explicit in [('purl','rejection_comparison_purl',args.purl),
                                 ('cima','rejection_comparison_cima',args.cima)]:
    root = locate(explicit, foldername)
    if root is None:
        continue
    expected_names = participants[ds]
    found_names = [n for n in expected_names if (root / n).is_dir()]
    other_names = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in expected_names)
    print(f'{ds.upper()}: using {len(found_names)}/{len(expected_names)} expected participants from {root}')
    print(f'  expected: {", ".join(expected_names)}')
    if other_names:
        print(f'  ignored unexpected folders: {", ".join(other_names)}')
    for name in expected_names:
        study = root / name
        if not study.is_dir():
            missing.append(f'{ds.upper()} PARTICIPANT DIRECTORY MISSING: {study}')
            continue
        for cfg in config_map[ds]:
            relbase = Path('results') / ds / study.name
            copy(study / f'{cfg}.npz', relbase / f'{cfg}.npz')
            copy(study / cfg / 'CarDpy_Output/12_Quantitative_Results/HAP.mat',
                 relbase / cfg / 'CarDpy_Output/12_Quantitative_Results/HAP.mat')
            if ds == 'cima':
                p1 = study / cfg / 'CarDpy_Output/14_Uncertainty/uncertainty.npz'
                p2 = study / f'{cfg}_uncertainty.npz'
                if p1.is_file():
                    copy(p1, relbase / cfg / 'CarDpy_Output/14_Uncertainty/uncertainty.npz')
                elif p2.is_file():
                    copy(p2, relbase / f'{cfg}_uncertainty.npz')
                else:
                    missing.append(f'{study.name}/{cfg}: CIMA uncertainty (either accepted location)')

# Clearly identified placeholders only: no empty .npz/.mat masquerading as data.
placeholders = out / 'placeholders'
placeholders.mkdir(exist_ok=True)
(placeholders / 'CIMA_manual_MISSING.txt').write_text(
    'CIMA manual rejection has not been run. Preserve an empty Manual slot if desired. '
    'Never plot this as measured data.\n')
(placeholders / 'PURL_uncertainty_PENDING.txt').write_text(
    'PURL uncertainties are not yet available. Reserve layout space but do not plot values.\n')
(out / 'README.txt').write_text(
    'ISMRM figure redesign package\n\n'
    'existing_figures/: verified original figure renders; fig4a_purl, fig4a_cima and fig4b_cima are authoritative.\n'
    'results/: selected raw npz, HAP.mat and CIMA uncertainty only.\n'
    'classifier/: test prediction metadata and optional checkpoint; cache/X.npy excluded by default due to size.\n'
    'placeholders/: missing CIMA manual rejection and PURL uncertainty (not synthetic data).\n'
    'Participants: CIMA V000-V005; PURL Healthy_Volunteer_019, _034, _035, _036, _039.\n'
    'Notes: CIMA has no manual rejection results. PURL uncertainty should remain a placeholder.\n'
    'HAP titles should use HAP, not Helix angle pitch.\n'
    'The existing figures can be used directly if some raw data is too big or missing.\n')
with (out / 'manifest.csv').open('w', newline='') as f:
    w = csv.writer(f); w.writerow(['archive_path','bytes','source_path']); w.writerows(manifest)
(out / 'missing_files.txt').write_text('\n'.join(missing) + ('\n' if missing else ''))
zip_path = out.with_suffix('.zip')
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5, allowZip64=True) as z:
    for p in sorted(out.rglob('*')):
        if p.is_file():
            z.write(p, Path(out.name) / p.relative_to(out))
print(f'\nCreated: {zip_path}')
print(f'Files included: {len(manifest)} | Unavailable paths noted: {len(missing)}')
print(f'ZIP size: {zip_path.stat().st_size / 1024**2:.1f} MB')
print('If large, upload just existing_figures/ plus fig4_values CSVs first.')
