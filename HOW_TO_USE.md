# CarDpy — How to Use

The full guide to CarDpy 2.0. Includes installation, pointing at your data,
configuration options, navigating the GUIs, outputs, and troubleshooting. For a short
overview, see the **[README](https://github.com/dzarroug/CarDpy/blob/dev/README.md)**.

---

## 1. Install

```bash
conda create -n cardpy python=3.14 -y
conda activate cardpy
pip install "cardpy-cmr[macos]"
```

Drop `[macos]` on Linux or Windows; it installs `tkmacosx`, which only Macs need.
Check that it worked:

```bash
python -c "import cardpy; from cardpy.pipeline import process; print('CarDpy installed')"
```

Notes:
- **Install size.** `torch`, `nnunetv2`, `ultralytics` and `huggingface_hub`
  are required dependencies, so the install is roughly 1–2 GB.
- **nnUNet-Segment weights** (~4 GB) are only needed for automated
  segmentation. They come from the gated Hugging Face repo
  [`sash99666/cdti-lv-baseline`](https://huggingface.co/sash99666/cdti-lv-baseline).
  Request access on that page, run `hf auth login` once, and CarDpy downloads
  them to `~/.cache/cardpy/cdti-lv-baseline` on first use.
- **CNN-Reject** ships inside the package

---

## 2. Pointing at your data

CarDpy reads a **study folder** containing a DICOM series (or NIfTI files).

**Direct path (recommended).** Pass the folder to `process()`:

```python
results = process("/full/path/to/your/study_folder", config)
```

**Relative path.** If the study folder is in the directory you run from:

```python
results = process("Healthy_Volunteer_007", config)
```

**Environment variable.** The notebooks read `CARDPY_DATA`:

```bash
export CARDPY_DATA="/full/path/to/your/study_folder"
```

An `export` only applies to that terminal session. For VS Code to see it,
launch VS Code from the same terminal (`code .`) or add the line to `~/.zshrc`.

### Expected folder layout

```
your_study_folder/
└── 02_cDTI/
    └── SAX/
        └── your_series_folder/     <- the DICOM files
```

Set the sub-path with `config["dicom_subpath"]`. For NIfTI input set
`config["data_type"] = "NifTis"` (case-sensitive).

### Sample data

`Healthy_Volunteer_007.zip` in the repository is a complete example study. The
notebooks unzip it automatically if `CARDPY_DATA` is not set.

---

## 3. The configuration

`DEFAULT_CONFIG` (in `cardpy/config.py`) holds every setting. Always edit a
**deep copy**: a shallow copy shares the nested dictionaries, so editing it
would change the defaults too.

```python
import copy
from cardpy.config import DEFAULT_CONFIG

config = copy.deepcopy(DEFAULT_CONFIG)
```

Defaults are shown below.

### Input / data
```python
config["data_type"]         = "DICOM"      # "DICOM" or "NifTis"
config["dicom_reader_info"] = "ON"
config["dicom_subpath"]     = "02_cDTI/SAX/your_series_folder_name"
config["operation_type"]    = "Magnitude"  # "Magnitude" or "Complex"
config["slice_index"]       = [3]          # list of slice indices, or None for all
```

### Pipeline stages
Each stage has an `"enabled"` flag:
```python
config["gibbs"]["enabled"]           = True
config["rejection"]["enabled"]       = True
config["respiratory"]["enabled"]     = True
config["registration"]["enabled"]    = True
config["adc_filter"]["enabled"]      = True
config["averaging"]["enabled"]       = True
config["denoising"]["enabled"]       = True
config["interpolation"]["enabled"]   = True
config["extended_matrix"]["enabled"] = True
config["e1"]["enabled"]              = True
```

### Shot rejection
```python
config["rejection"]["mode"]            = "algorithm"  # "algorithm" / "manual" / "ml" (CNN-Reject)
config["rejection"]["manual_review"]   = False        # review automatic decisions before they are applied
config["rejection"]["nrmse_threshold"] = 0.75         # algorithm mode, 0.0–1.0
config["rejection"]["nssim_threshold"] = 0.75         # algorithm mode, 0.0–1.0
config["rejection"]["ml_model_path"]   = None         # ml mode; None = bundled CNN-Reject model
config["rejection"]["ml_threshold"]    = None         # ml mode; None = threshold saved with the model
config["rejection"]["zoom"]            = "ON"
config["rejection"]["interact_zoom"]   = "ON"
config["rejection"]["organ"]           = "Heart"      # label on the crop window only
```
See [section 4](#4-shot-rejection-modes) for how the three modes differ.

### Respiratory sorting
```python
config["respiratory"]["zoom"]          = "ON"
config["respiratory"]["interact_zoom"] = "ON"
config["respiratory"]["organ"]         = "Stomach"   # label on the crop window only
```

### Registration
```python
config["registration"]["algorithm"]           = "Affine"  # "Affine" / "Rigid" / "Translation"
config["registration"]["mode"]                = "2D"      # "2D" / "3D" (3D requires Rigid)
config["registration"]["temporary_denoising"] = "OFF"
```

### Denoising
```python
config["denoising"]["algorithm"]       = "LocalPCA"  # "LocalPCA" / "NLMeans" / "Patch2Self"
config["denoising"]["number_of_coils"] = 20
```

### DTI reconstruction
```python
config["dti"]["tensor_fit"]   = "NLLS"  # "OLS" / "WLS" / "NLLS"
config["dti"]["b0_threshold"] = 0       # only true b=0 is baseline; b=50 keeps its weighting
```

### Segmentation
```python
config["segmentation"]["mode"]           = "manual"  # "manual" / "auto" (nnUNet-Segment)
config["segmentation"]["review"]         = False     # auto mode: open the contour GUI to check/edit
config["segmentation"]["weights_dir"]    = None      # None = ~/.cache/cardpy/cdti-lv-baseline
config["segmentation"]["use_model_mask"] = True      # unreviewed auto slices: model mask (True) or refill from contours
```
See [section 5](#5-segmentation).

### cDTI analysis
```python
config["cdti"]["num_interp_points"] = 200
config["cdti"]["smoothness_level"]  = "Low"    # "Low" / "Medium" / "High" / "Extreme"
config["cdti"]["hap_coordinate"]    = "cubic"  # "cubic" / "centroid_radial" (HAP_mm is always radial)
config["cdti"]["helix_angle_filter"]["linear_outlier_stdev"]      = 1
config["cdti"]["helix_angle_filter"]["spatial_wall_depth_factor"] = 0.25
config["cdti"]["helix_angle_filter"]["spatial_kernel_size"]       = 5   # positive odd integer
```

### GUIs
```python
config["gui"]["version"]                  = "v2"   # "v2" or "legacy"
config["gui"]["saved_configuration_mode"] = "off"  # "off" / "reuse" / "review" (see below)
config["gui"]["point_size"]               = 8.0    # contour marker size, 3–20
config["gui"]["segmentation_zoom"]        = True   # start contouring zoomed to the crop
config["gui"]["md_max"]                   = 3.0    # MD display maximum (µm²/ms)
```

`saved_configuration_mode` lets you rerun a study without redoing the GUIs:
- `"off"`: open every GUI fresh.
- `"reuse"`: reuse the crops, rejection decisions and contours saved in the
  output folder, without opening the GUIs.
- `"review"`: open the GUIs pre-filled with the saved choices so you can adjust them.

`segmentation_saved_configuration_mode` (`None` by default) can override this
for segmentation only.

### Uncertainty
```python
config["uncertainty"]["enabled"]          = False
config["uncertainty"]["iterations"]       = 400    # use 400 for analysis; 2 only to check it runs
config["uncertainty"]["n_jobs"]           = 4
config["uncertainty"]["seed"]             = 0
config["uncertainty"]["desired_averages"] = None   # None keeps the original repetition count
config["uncertainty"]["keep_images"]      = False
```
See [section 8](#8-bootstrapped-uncertainty).

### Output
```python
config["output"]["directory"]        = None   # None = <study_folder>/CarDpy_Output
config["output"]["save_diagnostics"] = True
```

### Missing shots and replacement
See [section 10](#10-missing-shots-and-shot-replacement).

### Running several configs in one script

Deep-copied configs are independent, so you can compare settings on the same data:

```python
config_a = copy.deepcopy(DEFAULT_CONFIG)
config_a["rejection"]["mode"] = "algorithm"
config_a["output"]["directory"] = "/full/path/to/out_algorithm"

config_b = copy.deepcopy(DEFAULT_CONFIG)
config_b["rejection"]["mode"] = "ml"
config_b["output"]["directory"] = "/full/path/to/out_cnn"

results_a = process(study_root, config_a)
results_b = process(study_root, config_b)
```

Give each run its own output directory so results don't overwrite each other.
Manual steps (cropping, contouring) still open for each run unless you use
`saved_configuration_mode="reuse"` or automated segmentation.

---

## 4. Shot rejection modes

Shot rejection identifies corrupted repetitions (averages) and replaces each
rejected one with a retained repetition of the same slice and diffusion
encoding.

| Mode | `config["rejection"]["mode"]` | How decisions are made |
|---|---|---|
| Algorithmic (default) | `"algorithm"` | SSIM/NRMSE between repetitions plus k-means clustering. You crop the heart first. |
| Manual | `"manual"` | A review window opens with every image and you click the ones to reject. |
| CNN-Reject | `"ml"` | A convolutional network classifies each image as keep or reject. Fully automatic. |

**Reviewing automatic decisions.** Set `config["rejection"]["manual_review"] = True`
with `"algorithm"` or `"ml"` to open the review window pre-filled with the
automatic decisions.

**The review window** shows one slice at a time:
- Click a thumbnail to toggle it (green = keep, red = reject).
- **Choose zoom region…** zooms every thumbnail to the same area; **Reset zoom** undoes it.
- The intensity slider sets the display maximum (display only; data are unchanged).
- **Apply slice and continue** (or Enter) saves the slice and moves on.
- **Stop (discard this slice)** ends the review without applying that slice.

If every repetition of an encoding is rejected, the first repetition is kept as
a fallback so processing can continue; the statistics still report 0% acceptance
for that encoding.

Decisions, statistics and diagnostic plots are saved to `13_Diagnostics/`.

---

## 5. Segmentation

**Manual (`"manual"`, default).** The spline contouring window opens once per slice:
- **Draw zoom box**, **Full image** and **Reset to crop** control the view.
- Draw the **epicardium** and **endocardium** by clicking at least 4
  spread-out points, then press the matching **Confirm** button.
- **Undo edit (d)** removes the last point.
- Drag the spline to adjust the contour.
- Click the **anterior** and **inferior** RV insertion points and confirm each.
- **Save contours & Finish** becomes active once all four are confirmed.

If the terminal shows *"Could not fit spline. Click at least 4 spread-out,
non-overlapping points"*, add more spread-out points and confirm again.

**Automated (`"auto"`, nnUNet-Segment).** A YOLO crop, a 5-fold nnU-Net and a
YOLO RV-insertion-point detector run in a separate process and produce the
contours with no clicking. Slices the model cannot segment fall back to the
manual window automatically. Set `config["segmentation"]["review"] = True` to
check and edit the model's contours before analysis.

`results["segmentation_sources"]` records, per slice, whether contours came
from the model or were drawn manually.

---

## 6. Running `process()`

```python
import copy
from cardpy.pipeline import process
from cardpy.config import DEFAULT_CONFIG

config = copy.deepcopy(DEFAULT_CONFIG)
config["dicom_subpath"] = "02_cDTI/SAX/your_series_folder_name"
config["slice_index"]   = None

results = process("/full/path/to/your/study_folder", config)
```

Options:
- `save=False` runs everything without writing files (uncertainty requires `save=True`).
- `stop_after_dti=True` stops after DTI reconstruction, before contouring and
  cDTI analysis.

Pipeline order: registration → Gibbs ringing removal → shot rejection →
respiratory sorting → registration → ADC filtering → averaging → denoising →
interpolation → DTI reconstruction → segmentation → cDTI analysis →
(optional) uncertainty.

---

## 7. Results and outputs

### The returned dictionary

```python
results["dti_metrics"]      # each [rows, cols, slices]: 'MD', 'TR', 'FA', 'MO', 'AD', 'RD'
results["eigenvectors"]     # each [rows, cols, slices, 3]: 'E1', 'E2', 'E3'
results["cardiac_metrics"]  # each [rows, cols, slices]: 'HA', 'HALF' (linear-filtered),
                            #   'HASF' (spatial-filtered), 'E2A', 'TA'
results["HAP"], results["HAP_mm"]   # helix angle pitch (full details in results["hap"])
results["mask"]             # [rows, cols, slices] myocardium mask
results["contours"]         # per slice: 'endo_x', 'endo_y', 'epi_x', 'epi_y', 'antRVIP', 'infRVIP'
results["segmentation_sources"]  # per slice: model or manual
results["header"]           # image header from the reader
results["uncertainty"]      # only when uncertainty is enabled
```

Example:

```python
import matplotlib.pyplot as plt
plt.imshow(results["cardiac_metrics"]["HASF"][:, :, 0], cmap="twilight")
plt.colorbar(); plt.show()
```

### The output folder

`CarDpy_Output/` (or `config["output"]["directory"]`) contains:

- `00_Original` … `10_Primary_Eigenvector`: each pipeline stage as NIfTI
- `11_Segmentation`: myocardium mask (NRRD) and saved contours (`Segmentation.json`)
- `12_Quantitative_Results`: metric maps (`.mat`) and figures (`.png`)
- `13_Diagnostics`: rejection plots and decisions, `Crop_Coordinates.txt`,
  `Rejection_Statistics.txt`, `Rejection_Configuration.json`
- `14_Uncertainty`: only when uncertainty is run
- `processing_settings.json`: the exact config used for the run

---

## 8. Bootstrapped uncertainty

Uncertainty resamples the retained repetitions and reruns registration,
filtering, averaging, denoising, interpolation and DTI fitting for each
realization (no contouring). Run it either way:

```python
config["uncertainty"]["enabled"] = True        # as part of process()
```
```python
from cardpy.uncertainty import run_dataset     # after processing
result = run_dataset("/full/path/to/study/CarDpy_Output", iterations=400, n_jobs=4)
```

Results go to `14_Uncertainty/`: median MD/FA with 95% bounds and widths
(`dMD`, `dFA`), and 95% cone half-angles for E1/E2/E3 (`dE1`, `dE2`, `dE3`,
degrees). Helix angle uncertainty is not calculated. Averaging must be enabled.

- Runs resume where they stopped if the inputs, config and code are unchanged.
  If you change settings, use a new `output=` folder. A partially written
  `14_Uncertainty/` from a different setup blocks reruns ("Output manifest
  differs"); delete it and start again.
- Older runs without a `processing_settings.json` need their original config passed as
  `config=...`. If respiratory sorting was enabled, also provide the crop as
  `config["respiratory"]["crop_coordinates"]` (four lists,
  `[x_start, x_end, y_start, y_end]`, one value per slice), or save it in
  `13_Diagnostics/Respiratory_Crop.json` under the key `"crop_coordinates"`.

See also `cardpy/uncertainty/README.md` and `03_Uncertainty.ipynb`.

---

## 9. The GUIs

With `gui.version = "v2"` (default), all windows size themselves to your screen.

- **Box crop** (shot rejection and respiratory sorting): drag a box around the
  region on each slice. **Back** / **Save & Next** move between slices,
  **Draw new box** starts over, **Full image** resets the crop, and the last
  slice's button reads **Save & Finish**.
- **Shot rejection review**: see [section 4](#4-shot-rejection-modes).
- **Spline contouring**: see [section 5](#5-segmentation).

The pipeline pauses while a window is open and resumes when it closes.
`gui.version = "legacy"` restores the original slider-based tools.

---

## 10. Missing shots and shot replacement

Two independent settings control how missing input images and rejected shots
are handled:

```python
config["missing_shots"]    = {"policy": "error"}   # "error" / "reject" / "replace"
config["shot_replacement"] = {
    "method": "none",               # "none" / "random_valid" / "first_valid" / "mean_valid"
    "seed": None,                   # integer for reproducible random donors
    "all_rejected": "first_valid",  # "error" / "first_valid"
}
```

**Missing-shot policies**
- `"error"` (default): stop if any input shot is missing or non-finite.
- `"reject"`: carry missing shots as NaN through Gibbs removal and registration,
  exclude them from rejection scoring, and mark them rejected. Requires shot
  rejection to be enabled.
- `"replace"`: fill missing inputs before processing using
  `shot_replacement.method` (must not be `"none"`). Filled inputs are scored like
  measured ones.

**Replacement methods** (applied after shot rejection, and at input for policy `"replace"`)
- `"none"`: leave the slot empty (NaN); averaging uses the remaining repetitions.
- `"random_valid"`: copy a random accepted repetition.
- `"first_valid"`: copy the first accepted repetition.
- `"mean_valid"`: use the mean of the accepted repetitions.

Donors always come from the same slice and diffusion encoding (b-value and
b-vector), and missing shots are never used as donors. A partly non-finite
image counts as a missing whole shot. `seed` only affects donor selection, not
clustering or other steps.

If every repetition of an encoding is rejected, `all_rejected="error"` stops
processing and `"first_valid"` keeps the first measured shot as a fallback.

Limitations: missing-shot registration supports 2D with temporary denoising
off; use input replacement for 3D registration or temporary denoising. An
encoding with no measured repetitions at all cannot be recovered.

When saving, `Shot_Replacements.json` records every replacement (zero-based
target and donor indices, method, stage, encoding, fallback status). Copies and
mean fills are not independent measurements and can change weighting during
averaging. With missing inputs, rejection runs per slice on compact
measured-only arrays; those diagnostics are under
`Missing_Shot_Rejection/Slice_*/13_Diagnostics`, and their indices refer to the
compact arrays.

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: torch` (or any package you installed) | You may be in the wrong Python. Run `which python` and activate the `cardpy` environment. |
| `ModuleNotFoundError: ultralytics` during automated segmentation | Install it in the environment actually running CarDpy: `pip install ultralytics`. |
| Weights download fails | Request access at the Hugging Face repo, then run `hf auth login`. |
| Segmentation fault on macOS | In scripts that open no windows, add `import matplotlib; matplotlib.use("Agg")` at the top. If you call torch yourself alongside CarDpy, add `torch.set_num_threads(1)`. |
| `TimeoutError: [Errno 60]` reading data | The files are in iCloud and offloaded. Mark the folder "Keep Downloaded" or keep data outside iCloud. |
| Shape-consistency error at load | CarDpy expects the same number of averages for every direction. Check that the series is complete. |
| "Output manifest differs" (uncertainty) | Delete the partial `14_Uncertainty/` folder or choose a new `output=` folder. |
