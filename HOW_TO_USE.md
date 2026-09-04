# CarDpy — How to Use

A full guide to running CarDpy: pointing it at your data, editing the
configuration, running `process()`, and using the results. For a quick start,
see the **[README.md](https://github.com/dzarroug/CarDpy/blob/dev/README.md)**.

---

## 1. Install

```bash
conda create -n cardpy python=3.14 -y
conda activate cardpy
pip install "cardpy-cmr[macos]"
```

Drop `[macos]` on Linux or Windows (it installs `tkmacosx`, needed only on Mac).
Confirm it worked:

```bash
python -c "import cardpy; print('CarDpy installed')"
```

---

## 2. Pointing at your data

CarDpy reads a **study folder** containing a DICOM series. There are three ways
to give it the path.

**Direct path (recommended).** Pass the folder straight to `process()`:

```python
results = process("/full/path/to/your/study_folder", config)
```

This always works, no matter how you launched Python or VS Code.

**Relative path.** If the study folder sits in the same directory you run the
script or notebook from, just use its name:

```python
results = process("Healthy_Volunteer_007", config)
```

This resolves relative to your current working directory, so it only works when
you run from that location.

**Environment variable (optional).** Set `CARDPY_DATA` and read it in your
script:

```bash
export CARDPY_DATA="/full/path/to/your/study_folder"
```
```python
import os
results = process(os.environ["CARDPY_DATA"], config)
```

Note: an `export` only applies to the terminal session it was run in. For it to
reach VS Code or other code editor, launch it from that same terminal (for VS code with `code .`), or set the
variable permanently by adding the `export` line to `~/.zshrc` and restarting.
Because of this, the direct path above is usually simpler.

### Expected folder layout

The DICOM series lives at a sub-path inside your study folder. By default CarDpy
expects:

```
your_study_folder/
└── 02_cDTI/
    └── SAX/
        └── cDTI_SF_b350_RL_71/     <- the DICOM files
```

Your scan's folder names will differ. Tell CarDpy the correct sub-path with the
`dicom_subpath` setting (section 3). NIfTI input is also supported by setting
`data_type` to `"NifTis"`.

CarDpy assumes each diffusion direction has the same number of averages. Data
where the number of averages varies across directions will stop at an internal
shape-consistency check rather than run.

### Where results are written

CarDpy creates `CarDpy_Output/` inside your study folder:

- `00_Original` … `08_Interpolated` — each pipeline stage as NIfTI
- `09_Extended_Matrix`, `10_Primary_Eigenvector` — additional stage outputs
- `11_Segmentation` — the myocardium mask (NRRD)
- `12_Quantitative_Results` — metric maps (`.mat`) and 9-panel figures (`.png`)
- `13_Diagnostics` — rejection plots, `Crop_Coordinates.txt`, `Rejection_Statistics.txt`

---

## 3. Editing the configuration

`DEFAULT_CONFIG` holds every setting the pipeline uses. For a single run you can use DEFAULT_CONFIG directly. With multiple configs in the same session edit a **deep
copy** so that editing one does not change the others. A shallow copy still shares the nested dictionaries, so only `copy.deepcopy` fully protects the defaults.

```python
from cardpy.config import DEFAULT_CONFIG
import copy

config = copy.deepcopy(DEFAULT_CONFIG)
```

### Full list of options (written with default input)

**Input / data**
```python
config["data_type"]         = "DICOM"     # "DICOM" or "NifTis"
config["dicom_reader_info"] = "ON"
config["dicom_subpath"]     = "02_cDTI/SAX/your_series_folder_name"
config["operation_type"]    = "Magnitude" # "Magnitude" or "Complex"
config["slice_index"]       = [3]         # list of slice indices, or None for all
```

**Pipeline stages** — each has an `"enabled"` flag to turn it on or off:
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

**Rejection**
```python
config["rejection"]["nrmse_threshold"] = 0.75    # 0.0 - 1.0
config["rejection"]["nssim_threshold"] = 0.75    # 0.0 - 1.0
config["rejection"]["zoom"]            = "ON"
config["rejection"]["interact_zoom"]   = "ON"
config["rejection"]["organ"]           = "Heart"   # display label on the crop window only; does not change cropping
```

**Respiratory**
```python
config["respiratory"]["zoom"]          = "ON"
config["respiratory"]["interact_zoom"] = "ON"
config["respiratory"]["organ"]         = "Stomach"   # display label on the crop window only (function default is "Liver")
```

**Registration**
```python
config["registration"]["algorithm"]           = "Affine"  # Affine / Rigid / Translation
config["registration"]["temporary_denoising"]  = "OFF"
```

**Denoising**
```python
config["denoising"]["algorithm"]       = "LocalPCA"  # "Patch2Self", "NLMeans" - Non-Local Means, or "LocalPCA"
config["denoising"]["number_of_coils"] = 20   # defaults to 20
```

**DTI reconstruction**
```python
config["dti"]["tensor_fit"] = "NLLS"  # "OLS" - Ordinary least-squares, "WLS" - Weighted least squeares, "NLLS"- Non-linear least squares  
```

**cDTI analysis**
```python
config["cdti"]["num_interp_points"] = 200
config["cdti"]["smoothness_level"]  = "Low"   # Low, Medium, High, and Extreme
config["cdti"]["helix_angle_filter"]["linear_outlier_stdev"]       = 1     
config["cdti"]["helix_angle_filter"]["spatial_wall_depth_factor"]  = 0.25   
config["cdti"]["helix_angle_filter"]["spatial_kernel_size"]        = 5      # positive odd integer
```

### Running several configs in one script

Each deep-copied config is independent, so you can define and run several without
them interfering. This is useful for comparing settings on the same dataset:

```python
config_a = copy.deepcopy(DEFAULT_CONFIG); config_a["registration"]["algorithm"] = "Affine"
config_b = copy.deepcopy(DEFAULT_CONFIG); config_b["registration"]["algorithm"] = "Rigid"

results_a = process(study_root, config_a)
results_b = process(study_root, config_b)
```

Note: the contouring GUI still runs per slice for each config, so this batches
the *settings*, not the manual contouring.

---

## 4. Running `process()`

```python
from cardpy.pipeline import process
from cardpy.config import DEFAULT_CONFIG
import copy

config = copy.deepcopy(DEFAULT_CONFIG)
config["dicom_subpath"] = "02_cDTI/SAX/your_series_folder_name"
config["slice_index"]   = [3, 4] # for all slices: None

results = process("/full/path/to/your/study_folder", config)
```

To run the full pipeline **without writing any files** (a quick test):

```python
results = process("/full/path/to/your/study_folder", config, save=False)
```

---

## 5. Using the results

`process()` returns a dictionary:

```python
results["dti_metrics"]      # dict of arrays, each [rows, cols, slices]:
                            #   'MD' mean diffusivity, 'TR' trace, 'FA' fractional anisotropy,
                            #   'MO' mode, 'AD' axial diffusivity, 'RD' radial diffusivity
results["eigenvectors"]     # dict of arrays, each [rows, cols, slices, 3]:
                            #   'E1' primary, 'E2' secondary, 'E3' tertiary eigenvector
results["cardiac_metrics"]  # dict of arrays, each [rows, cols, slices]:
                            #   'HA' helix angle, 'HALF' helix angle (linear-filtered),
                            #   'HASF' helix angle (spatial-filtered), 'E2A' E2/sheet angle,
                            #   'TA' transverse angle
results["mask"]             # array [rows, cols, slices] — myocardium mask
results["contours"]         # dict of per-slice point lists:
                            #   'endo_x', 'endo_y', 'epi_x', 'epi_y',
                            #   'antRVIP', 'infRVIP'
results["header"]           # the image header from the reader
```

Reach a specific map by chaining keys, e.g. the spatial-filtered helix angle:

```python
ha = results["cardiac_metrics"]["HASF"]   # [rows, cols, slices]
```

Display the mask for the first slice:

```python
import matplotlib.pyplot as plt
plt.imshow(results["mask"][:, :, 0], cmap="gray")
plt.show()
```

The saved files in `CarDpy_Output/12_Quantitative_Results` hold the same
information for use outside Python.

---

## 6. The GUIs

CarDpy opens two interactive tools during a run. They behave slightly
differently across slices:

- **Crop tool** — one window stays open; crop a slice, then click **Next** to
  move to the next slice within the same window.
- **Contouring tool** — a separate window opens per slice; close each one and
  the next slice's window opens.

In both cases the pipeline resumes only after the last slice is done.

### Contouring tips

- Click **at least 4 points**, spread out around the heart wall (not bunched
  together or in a line), then press the matching **Confirm** button.
- After you press **Confirm**, check the **terminal**. If you see *"Could not fit
  spline. Click at least 4 spread-out, non-overlapping points,"* your points were
  too few or too close. Edit and add more spread-out points and press Confirm again.
- Use the **Edit** buttons to revise a contour you already confirmed, and press
  **d** to delete the most recent point.