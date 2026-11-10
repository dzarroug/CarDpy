# CarDpy

A Python toolbox for processing cardiac diffusion tensor imaging (cDTI) data.

## Pipeline overview

https://github.com/user-attachments/assets/94ed65fc-ec80-4635-a24a-68b779b8f039

Link to figure: [open Figure 1 (pipeline overview)](https://github.com/user-attachments/assets/94ed65fc-ec80-4635-a24a-68b779b8f039).

## Command-line demo

https://github.com/user-attachments/assets/d49ed1cd-8b80-441a-ae34-f27c8d6ec2c4

Link to figure: [open Figure 2a (command-line demo)](https://github.com/user-attachments/assets/d49ed1cd-8b80-441a-ae34-f27c8d6ec2c4).

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

**Optional uncertainty.** Use [03_Uncertainty.ipynb](03_Uncertainty.ipynb) after processing,
or set `config["uncertainty"]["enabled"] = True` before `process()`.
Results are saved separately in `CarDpy_Output/14_Uncertainty/`.

```python
from cardpy.uncertainty import run_dataset
result = run_dataset("/full/path/to/study/CarDpy_Output", iterations=400, n_jobs=4)
```

New runs save their config automatically. For older runs, pass their original
full `config=config`; see the notebook for respiratory crop requirements.

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


## Missing shots and replacement

`cardpy.pipeline.process` accepts two independent configuration sections:

```python
config['missing_shots'] = {'policy': 'reject'}
config['shot_replacement'] = {
    'method': 'random_valid',
    'seed': 0,
    'all_rejected': 'error',
}
```

Missing-shot policies:
- `error`: stop on missing/nonfinite input shots (general pipeline default).
- `reject`: retain missing slots as NaN through Gibbs removal and registration,
  exclude them from rejection scoring, and force their keep decisions to zero.
  Requires rejection enabled.
- `replace`: fill missing inputs before processing, using `shot_replacement.method`.
  Requires a method other than `none`. Filled inputs participate in later scoring.

Replacement methods apply after rejection, and at input for policy `replace`:
- `none`: leave rejected/missing slots as NaN; subsequent averaging uses the available repetitions.
- `random_valid`: copy a random accepted repetition.
- `first_valid`: copy the first accepted repetition.
- `mean_valid`: fill with the mean of accepted repetitions.

Donors always share the target slice and diffusion encoding (b-value plus
b-vector). Missing shots are never scored or used as donors. A partially
nonfinite input image is treated as an unavailable whole shot. This does not
apply to voxel-wise masks introduced later by ADC filtering.

`seed` makes random donor selection reproducible; `None` uses NumPy's global
RNG. It does not seed rejection clustering or other algorithms. If every
measured repetition is rejected, `all_rejected='error'` stops processing;
`'first_valid'` explicitly retains the first measured shot as a rejected
fallback. The general pipeline defaults to `method='none'` and
`all_rejected='first_valid'`, preserving its prior rejection behavior.

Gibbs removal processes present images only. Missing-shot registration supports
2D with temporary denoising OFF; it skips absent moving images and uses an
available measured lowest-b reference without filling the missing slot. A
slice without any lowest-b reference fails clearly. Use input replacement for
3D registration or temporary denoising. Missing encodings without any measured
repetition cannot be recovered and fail clearly.

When saving, `Shot_Replacements.json` records zero-based target and donor
indices, method, stage, encoding, and fallback status. Shot provenance follows
the donors. Copies and mean fills do not add independent measurements and can
change weighting during subsequent averaging. For missing inputs, rejection
runs per slice with compact measured-only arrays; diagnostics are under
`Missing_Shot_Rejection/Slice_*/13_Diagnostics` and decision indices there refer
to the compact arrays. The replacement audit uses the original layout.

Direct `shot_rejection` calls also accept `replacement_method`,
`replacement_seed`, `all_rejected`, and a mutable `replacement_log` list.
