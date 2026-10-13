"""Saved run status used by batch notebooks to skip finished outputs."""
import json
import os
from pathlib import Path

STATUS_FILE = 'Processing_Status.json'
QUANTITATIVE_FILES = ('Standard_DTI_Metrics.mat', 'cDTI_Metrics.mat',
                      'DTI_Eigenvectors.mat', 'HAP.mat')


def processing_complete(output_directory):
    """Require final outputs; also recognize complete runs predating status files."""
    root = Path(output_directory)
    status_path = root / STATUS_FILE
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text())
            if status.get('status') != 'complete':
                return False
        except (ValueError, OSError):
            return False
    segmentation = root / '11_Segmentation' / 'Segmentation.json'
    try:
        slices = len(json.loads(segmentation.read_text())['contours']['endo_x'])
    except (ValueError, OSError, KeyError, TypeError):
        return False
    quantitative = root / '12_Quantitative_Results'
    required = [quantitative / name for name in QUANTITATIVE_FILES]
    required.extend(quantitative / f'Quantitative_cDTI_Maps_{slc + 1:02d} Slice.png'
                    for slc in range(slices))
    return slices > 0 and all(path.is_file() and path.stat().st_size > 0 for path in required)


def write_processing_status(output_directory, status):
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    target = root / STATUS_FILE
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({'status': status}, indent=2) + '\n')
    os.replace(temporary, target)
