# CarDpy

**Cardiac Diffusion in Python**: a Python toolbox for processing cardiac
diffusion tensor imaging (cDTI) data. It runs the full pipeline, from DICOM or
NIfTI input to diffusion tensor and cardiac metrics (MD, FA, helix angle, E2A).

## Pipeline overview

https://github.com/user-attachments/assets/94ed65fc-ec80-4635-a24a-68b779b8f039

## Command-line demo

https://github.com/user-attachments/assets/d49ed1cd-8b80-441a-ae34-f27c8d6ec2c4

## Updates in CarDpy 2.0

- **Three shot rejection modes:** Algorithmic (SSIM/NRMSE clustering, default),
  Manual (click-to-reject review window), and **CNN-Reject** (a bundled deep
  learning classifier).
- **Automated segmentation with nnUNet-Segment:** LV contours and RV insertion
  points without manual drawing, with an optional review step.
- **Bootstrapped uncertainty:** 95% intervals for MD and FA, and cone
  half-angles for E1/E2/E3.
- **Redesigned GUIs:** box cropping, spline contouring with zoom, and a manual
  shot rejection review window.

## Install

```bash
conda create -n cardpy python=3.14 -y
conda activate cardpy
pip install "cardpy-cmr[macos]"    # drop [macos] on Linux/Windows
```

The PyPI name is `cardpy-cmr`; the import name is `cardpy`.

- **Install size:** CarDpy 2.0 depends on `torch`, `nnunetv2`, `ultralytics`
  and `huggingface_hub`, so you can expect a 1–2 GB install.
- **nnUNet-Segment weights** (~4 GB) download on first use from the gated
  Hugging Face repo [`sash99666/cdti-lv-baseline`](https://huggingface.co/sash99666/cdti-lv-baseline).
  Request access on that page, then run `hf auth login` once. Manual
  segmentation needs neither.

## Quick start

```python
import copy
from cardpy.pipeline import process
from cardpy.config import DEFAULT_CONFIG

config = copy.deepcopy(DEFAULT_CONFIG)
config["dicom_subpath"] = "02_cDTI/SAX/your_series_folder_name"
config["slice_index"]   = None              # all slices, or e.g. [3, 4]

config["rejection"]["mode"]    = "algorithm"  # "algorithm" / "manual" / "ml" (CNN-Reject)
config["segmentation"]["mode"] = "manual"     # "manual" / "auto" (nnUNet-Segment)
config["uncertainty"]["enabled"] = False      # True adds bootstrapped uncertainty

results = process("/full/path/to/your/study_folder", config)
```

The GUIs open during the run when needed. When they're done, `results` holds
the DTI metrics, eigenvectors, cardiac metrics and mask, and every stage is
written to `CarDpy_Output/` inside your study folder.

Step-by-step notebooks are in the repository: `01_Data_Processing`,
`02_Post-Processing`, `03_Uncertainty` and `04_Manual_Shot_Rejection`.
A sample dataset, `Healthy_Volunteer_007.zip`, is included.

**Full guide** (every config option, outputs, GUIs, troubleshooting):
**[HOW_TO_USE.md](https://github.com/dzarroug/CarDpy/blob/dev/HOW_TO_USE.md)**

## Associated publication

> Cork, T.E., Hannum, A.J., Loecher, M., Perotti, L.E., Ennis, D.B. (2025).
> *Evaluating the Effect of Post-processing Steps When Analyzing Cardiac
> Diffusion Tensor Data.* In: Functional Imaging and Modeling of the Heart
> (FIMH 2025). Lecture Notes in Computer Science, vol. 15673. Springer, Cham.
> https://doi.org/10.1007/978-3-031-94562-5_13

## Credits and license

CarDpy was originally written by Tyler Cork (CMR Group, Stanford Radiological Sciences Laboratory).
nnUNet-Segment was developed by Sascha Stocker; its code is included in
`cardpy/Data_Processing/nnunet_cdti/` under the Apache 2.0 license. CarDpy is
released under the GNU GPL v3.
