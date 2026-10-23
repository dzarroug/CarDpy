# Optional uncertainty

Use `03_Uncertainty.ipynb`, or enable `config["uncertainty"]["enabled"]` before
`process()`. The normal pipeline leaves uncertainty off.

```python
from cardpy.uncertainty import run_dataset
result = run_dataset("/path/to/CarDpy_Output", iterations=400, n_jobs=4)
# Older runs: run_dataset("/path/to/CarDpy_Output", config=original_config)
```

Results go to `14_Uncertainty`, after quantitative results and diagnostics.
`uncertainty.npz` and MATLAB files contain median MD/FA, their lower/upper
95% bounds and widths (`dMD`, `dFA`), and E1/E2/E3 95% cone half-angles
(`dE1`, `dE2`, `dE3`, degrees), mean axes, coherence, and valid sample counts.
MD uses CarDpy's µm²/ms scaling. HA uncertainty is not calculated.

The supplied code's whole-image resampling and order statistics are retained:
unique composite rows choose a repetition independently per slice/encoding;
`desired_averages` rows form each realization. Missing rejected shots are excluded.
Each realization repeats enabled registration, phase correction, ADC filtering,
averaging, denoising, interpolation, and DTI fitting without contouring.
Averaging must be enabled. Too few unique combinations raises an error;
one retained repetition cannot estimate variability for its encoding.
Use 400 realizations for analysis; 2 is an execution check only.

New processing runs save `processing_settings.json` and the respiratory crop.
Older runs need their original full config. If respiratory sorting was enabled,
provide `config["respiratory"]["crop_coordinates"]` as four lists
`[x_start, x_end, y_start, y_end]`, one value per slice, or save the same lists in
`13_Diagnostics/Respiratory_Crop.json`. Complex processing requires complex images.

Completed samples resume when inputs, config, plan, and code match. Changed
settings or code require a new `output=` folder. The source images and quantitative
results are preserved; logs, samples, plan, and status are saved under uncertainty.
