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


Smart Health compatibility defaults:
- `adc_filter.reference='paired'` restores the original ADC calculation: each
  repetition is compared with the same repetition of the first lowest-b
  encoding, using the full b-value span and threshold 3.0 µm²/ms. When all
  repetitions fail, every tied minimum is retained. Use `'pooled'` explicitly
  to select the newer CardPy pooled-baseline filter.

Shared configurable shot handling now uses the same CardPy implementation as
`cardpy.pipeline.process`. See the repository README for all options.

```python
config['missing_shots'] = {'policy': 'reject'}
config['shot_replacement'] = {
    'method': 'random_valid', 'seed': 0, 'all_rejected': 'error',
}
```

With these settings, missing shots remain NaN during Gibbs removal and
registration, are excluded from rejection scoring, and are filled only during
rejection using accepted shots from the same slice and encoding. No accepted
donor produces a clear error. Audits are in `Shot_Replacements.json`.

Smart Health Blip defaults defer missing shots until rejection (`policy='reject'`)
and use random accepted-shot replacement (`method='random_valid', all_rejected='error'`). The
old policy spelling `random_valid` remains an alias for input replacement.
Deferred `policy='reject'` requires the Blip workflow with rejection enabled;
TOPUP has no rejection stage and defaults to `policy='error'`; select `replace` explicitly if input filling is desired. Set `method` to
`none`, `first_valid`, or `mean_valid` to change how rejected shots are filled.
This adapter ends after primary eigenvector output; it does not run contouring.

Rejection decisions are also configurable: `rejection.mode` accepts `algorithm`,
`manual`, or `ml`. `manual_review` reviews algorithm/ML decisions; manual mode
always opens review. `ml_model_path` and `ml_threshold` configure ML. The shared
GUI saved-configuration mode applies unless overridden in the rejection section.
The adapter forwards these settings to CardPy's rejection function.
Patient notebooks load these choices with `get_config("blip")` and do not
hard-code shot-handling overrides. Restart the kernel after editing the shared
config. Missing slots remain excluded regardless of rejection decision mode.
