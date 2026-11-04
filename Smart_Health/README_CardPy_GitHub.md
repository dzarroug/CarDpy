# Smart Health config/process notebooks

- `01_Blip_Up_CardPy_GitHub.ipynb`
- `02_Blip_Down_CardPy_GitHub.ipynb`
- `03_TOPUP_CardPy_GitHub.ipynb`

All three use this interface:

```python
from smart_health_config import get_config
from smart_health_processing import process

config = get_config("blip")  # use "topup" for 03_TOPUP
# Set subject-specific input/output paths in the notebook.
results = process(study_root, config)
```

Edit `smart_health_config.py` in this folder to change shared processing settings.
Its defaults match the original notebooks in
`/Volumes/T7_old/Smart_Health/Scripts`, with the v2 GUI retained from these GitHub
notebooks. Each call returns an independent deep copy. The config is separate
from `cardpy.config.DEFAULT_CONFIG`, so global default changes do not alter these
workflows. Blip uses DICOM input; TOPUP uses corrected NifTis.

The notebooks load processing functions from `~/Documents/GitHub/CarDpy`.
The local `smart_health_processing.process` adapter maps the config to the
original Smart Health workflow. The adapter retains the original registration
passes and TOPUP numbering.

Run from the `Smart_Health` directory using a fresh kernel with the checkout's
Python 3.12+ dependencies. Change CARDPY_REPO in the import cell if it moves.
All original notebooks remain untouched.

The config explicitly preserves Magnitude operation, ON processing flags,
0.75 rejection thresholds, Heart rejection and Stomach respiratory crops,
Affine registration with temporary denoising OFF, 20 coils, and LocalPCA.
All slices are processed (slice_index=None), using v2 interactive tools.
The adapter preserves the original derived Rigid registration before rejection,
Affine registration before respiratory sorting, standalone registration, and
registration after averaging. Outputs retain the original file names, stage
folder names and numbering, and go to the same output locations as before.

Subject selections and patient/Even/Odd branching are retained. In particular,
Blip Up's non-patient input remains under `01_Original_cDTI_Dicom`, while Blip
Down's remains under `02_Anonymized_Data`, matching the original notebooks.

TOPUP reads the existing `*_TOPUP_Final_Data/TOPUP_Corrected` files and begins
saving at `13_Registered`. It does not repeat Gibbs removal, rejection, or
respiratory sorting, even though those unused original settings are retained.
It does not perform FSL TOPUP estimation. The adapter adds no contouring,
quantitative analysis, or standalone final DTI reconstruction.

Results contain the final matrix, bvals, bvecs, and header. New CardPy numerical
implementations can differ from legacy CardPy. The new DICOM reader's defaults
(ascending slice order and modal repetition counts) remain in effect.

Validation: all config values were compared with the originals, input/output
paths checked in both patient modes, processing-call traces compared with the
original sequences (including parameters and save numbering), and imports and
notebook syntax checked. Full imaging processing and interactive GUIs were not
run.

The notebooks use absolute data paths under `/Volumes/T7_old/Smart_Health` so moving this folder does not change input/output locations. Their first cell runs `pip install -e .` in the GitHub checkout using the active kernel Python.

Rejection compatibility: the v2 UI and current automatic rejection decisions are
used, but each rejected repetition is replaced by a random accepted repetition
from the same slice and encoding, matching the legacy replacement loop. Rejected
positions are filled, the number of årepetitions is preserved, and averaging sees
the duplicated accepted shots. Randomness uses NumPy's global RNG as before.
If every repetition for an encoding is rejected, replacement stops with a clear
error because no accepted donor exists. The original singleton branch is also
preserved. Other numerical algorithms still come from the current CardPy checkout;
new automatic rejection decisions need not match the legacy version exactly.
