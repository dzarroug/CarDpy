# CarDpy

A Python toolbox for processing cardiac diffusion tensor imaging (cDTI) data.

For a complete guide, see **[HOW_TO_USE.md](https://github.com/dzarroug/CarDpy/blob/dev/HOW_TO_USE.md)**.

# Getting Started

### Environment Setup

Create the Conda environment:

```bash
conda create -n cardpy python=3.14 -y
conda activate cardpy
```

### Installation

Install **CarDpy** with:

```bash
pip install "cardpy-cmr[macos]"
```
Delete `[macos]` if not using Mac.

### Run the pipeline

There are two ways to run CarDpy:

**1. One call with `process()`.** Run the full pipeline from a script:

```python
from cardpy.pipeline import process
from cardpy.config import DEFAULT_CONFIG
import copy

config = copy.deepcopy(DEFAULT_CONFIG)
config["dicom_subpath"] = "02_cDTI/SAX/your_series_folder_name"

results = process("/full/path/to/your/study_folder", config)
```

The contouring GUI opens during the run. Draw the contours, and the pipeline
finishes on its own. `results` is a dictionary of the DTI metrics, eigenvectors,
cardiac metrics, and mask. Outputs are also written to a `CarDpy_Output/` folder
inside your study folder.

Because each config is an independent dictionary, you can define several and run
them in one script (e.g. one per protocol or per patient). This is useful for comparing processing settings on the same data. 
(Contouring still runs per slice for each.)

For the full list of configuration options, the return-value format, and data
setup details, see **[HOW_TO_USE.md](HOW_TO_USE.md)**.

**2. Notebooks or Script (step-by-step).** Write a new script/notebook or use sample notebook. With Sample notebooks start with `01_Data_Processing`, then run
`02_Post-Processing`, using the your data or the sample `Healthy_Volunteer_007` dataset.

## Associated Publication

**Evaluating the Effect of Post-processing Steps When Analyzing Cardiac Diffusion Tensor Data**

> Cork, T.E., Hannum, A.J., Loecher, M., Perotti, L.E., Ennis, D.B. (2025). *Evaluating the Effect of Post-processing Steps When Analyzing Cardiac Diffusion Tensor Data*. In: Chabiniok, R., Zou, Q., Hussain, T., Nguyen, H.H., Zaha, V.G., Gusseva, M. (eds) **Functional Imaging and Modeling of the Heart (FIMH 2025)**. Lecture Notes in Computer Science, vol. 15673. Springer, Cham.

DOI: https://doi.org/10.1007/978-3-031-94562-5_13

---

## Completed Tasks

- preliminary pip install
- GUI compatibility on Python 3.12 + 3.14
- config-driven `process()` pipeline

---

## Ongoing Tasks
When possible:
- Add automated segmentation GUI with Sascha's model
- windows testing

Action Tasks:
4 - call the GUI an additional time to do a "whole-heart crop" which maybe useful for segmentation
    -  Zooming, drag to resize
5 - GUI updates for usability
3 - manual rejection tool to see all the images and click on one to reject
2 - refine pip install set-up (pypi)

## Proposed Merges with Tyler
- New registration WIP
- New Gibb's Ringing Correction 
- New data loading algorithm