"""
compare_rejection_configs.py
Process each study under several shot-rejection configurations with identical
contours and crops, bootstrap uncertainty for each, and compare.

  python ml_rejection/compare_rejection_configs.py            # run all configs on all studies, then compare
  python ml_rejection/compare_rejection_configs.py --compare  # re-make tables/figures from saved results

Per study, the FIRST config ('algorithm') opens the GUIs: heart crop, stomach crop,
contours. Later configs reuse the same contours and stomach crop, so the only
difference between configs is the rejection step.

Bootstrapping: off by default (BOOTSTRAP = False). Each config's complete run is kept in
rejection_comparison/<study>/<config>/CarDpy_Output, so it can be bootstrapped later with
cardpy.uncertainty.run_dataset(<that folder>, iterations=400) and picked up by --compare.

Outputs in rejection_comparison/<study>/:
  <config>.npz                metrics + mask + background
  <config>_uncertainty.npz    dMD / dFA / dE1 maps (copied from 14_Uncertainty)
  <config>_CarDpy_maps/       CarDpy's own map figures
  <config>_Rejection_Statistics.txt
  maps_slice<N>.png           MD / FA / HA, one column per config
  uncertainty_slice<N>.png    dMD / dFA / dE1, one column per config
and rejection_comparison/summary_per_study.csv, summary.csv
"""
import matplotlib
matplotlib.use("Agg")   # must come first: the default macOS backend fights Tk and segfaults the GUIs
import argparse
import copy
import json
import os
import re
import shutil

import numpy as np
import pandas as pd

# ---------------- EDIT THESE ----------------
# name: (study folder, DICOM series path inside it -- or None to auto-detect the one folder in 02_cDTI/SAX)
STUDIES = {
    "Healthy_Volunteer_019": ("/Users/dahniazarroug/Documents/GitHub/CarDpy/PURL_test_raw/Healthy_Volunteer_019", None),
    "Healthy_Volunteer_034": ("/Users/dahniazarroug/Documents/GitHub/CarDpy/PURL_test_raw/Healthy_Volunteer_034", None),
    "Healthy_Volunteer_035": ("/Users/dahniazarroug/Documents/GitHub/CarDpy/PURL_test_raw/Healthy_Volunteer_035", None),
    "Healthy_Volunteer_036": ("/Users/dahniazarroug/Documents/GitHub/CarDpy/PURL_test_raw/Healthy_Volunteer_036", None),
    "Healthy_Volunteer_039": ("/Users/dahniazarroug/Documents/GitHub/CarDpy/PURL_test_raw/Healthy_Volunteer_039", None),
}
DATA_TYPE = "NifTis"       # "DICOM" (needs 02_cDTI/SAX) or "NifTis" (needs <study>/NifTis/data.*)
MODEL = "ml_rejection/models/crop80/best.pt"
SLICES = [3, 5, 7]         # None = all slices; [3, 5, 7] = base / mid / apex for 11-slice PURL data
ITERATIONS = 2           # bootstrap iterations (2 to test)
N_JOBS = 4
OUT = "rejection_comparison_purl"
BOOTSTRAP = False  

# Order matters: the first config draws contours/crops; the rest reuse them.
CONFIGS = [
    "algorithm",
    "none",
    "ml",
    # "manual",   # Rejection_Configuration.json at
    #             # rejection_comparison/<study>/manual_Rejection_Configuration.json
]
# --------------------------------------------

MAP_KEYS = [("MD", "dti_metrics__MD", (0, 3)), ("FA", "dti_metrics__FA", (0, 1)),
            ("HA", "cardiac_metrics__HASF", (-90, 90))]
UNC_KEYS = [("dMD", "dMD"), ("dFA", "dFA"), ("dE1", "dE1")]
LABELS = {"none": "No rejection", "algorithm": "Algorithm", "ml": "ML", "manual": "Manual"}
DISPLAY = ["none", "algorithm", "ml", "manual"]   # column / table order (run order is CONFIGS)


def find_subpath(study_path):
    sax = os.path.join(study_path, "02_cDTI", "SAX")
    series = [d for d in sorted(os.listdir(sax)) if os.path.isdir(os.path.join(sax, d))]
    if len(series) != 1:
        raise SystemExit(f"{study_path}: found {len(series)} series in 02_cDTI/SAX -- set it in STUDIES")
    return os.path.join("02_cDTI", "SAX", series[0])


def run_one(name, study_path, subpath, cfg_name, first):
    from cardpy.config import DEFAULT_CONFIG
    from cardpy.pipeline import process

    study_out = os.path.join(OUT, name)
    # each config gets its own complete output folder (needed to bootstrap it later)
    out_dir = os.path.abspath(os.path.join(study_out, cfg_name, "CarDpy_Output"))
    diag = os.path.join(out_dir, "13_Diagnostics")
    first_out = os.path.abspath(os.path.join(study_out, CONFIGS[0], "CarDpy_Output"))
    if not first:   # copy contours + stomach crop from the first config's run
        for rel in ("11_Segmentation/Segmentation.json", "13_Diagnostics/Respiratory_Crop.json"):
            src = os.path.join(first_out, rel)
            if os.path.exists(src):
                os.makedirs(os.path.dirname(os.path.join(out_dir, rel)), exist_ok=True)
                shutil.copy(src, os.path.join(out_dir, rel))

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["data_type"] = DATA_TYPE
    if subpath:
        cfg["dicom_subpath"] = subpath
    cfg["slice_index"] = SLICES
    cfg["output"]["directory"] = out_dir
    cfg["gui"]["saved_configuration_mode"] = "off"
    # Reuse contours whenever they exist (e.g. from an earlier, interrupted attempt);
    # the first config still opens them for review so you can confirm or edit.
    have_contours = os.path.exists(os.path.join(out_dir, "11_Segmentation", "Segmentation.json"))
    cfg["gui"]["reuse_existing_segmentation"] = have_contours
    cfg["gui"]["segmentation_review"] = first
    cfg["uncertainty"].update(enabled=BOOTSTRAP, iterations=ITERATIONS, n_jobs=N_JOBS, seed=0)
    cfg["rejection"]["manual_review"] = False

    crop_file = os.path.join(diag, "Respiratory_Crop.json")   # reuse the stomach crop once drawn
    if os.path.exists(crop_file):
        saved = json.load(open(crop_file))   # newer CarDpy saves {"image_shape", "organ", "crop_coordinates"}
        cfg["respiratory"]["crop_coordinates"] = saved["crop_coordinates"] if isinstance(saved, dict) else saved

    shutil.rmtree(os.path.join(out_dir, "14_Uncertainty"), ignore_errors=True)   # stale/partial runs

    if cfg_name == "none":
        cfg["rejection"]["enabled"] = False
    elif cfg_name in ("algorithm", "ml"):
        cfg["rejection"]["enabled"] = True
        cfg["rejection"]["mode"] = cfg_name
        cfg["rejection"]["ml_model_path"] = os.path.abspath(MODEL)
    elif cfg_name == "manual":
        src = os.path.join(study_out, "manual_Rejection_Configuration.json")
        if not os.path.exists(src):
            print(f"  {name}: no manual decisions yet ({src}) -- skipping manual")
            return
        os.makedirs(diag, exist_ok=True)
        shutil.copy(src, os.path.join(diag, "Rejection_Configuration.json"))
        cfg["rejection"]["enabled"] = True
        cfg["gui"]["saved_configuration_mode"] = "reuse"   # load her saved decisions

    print(f"\n===== {name}: {cfg_name} =====")
    res = process(study_path, cfg)

    flat = {}
    for group in ("dti_metrics", "cardiac_metrics"):
        for key, val in (res.get(group) or {}).items():
            flat[f"{group}__{key}"] = np.asarray(val, dtype=float)
    flat["mask"] = np.asarray(res["mask"], dtype=float)
    flat["background"] = np.nanmean(np.real(np.asarray(res["matrix"])), axis=(3, 4))
    np.savez_compressed(os.path.join(study_out, f"{cfg_name}.npz"), **flat)

    unc = os.path.join(out_dir, "14_Uncertainty", "uncertainty.npz")
    if os.path.exists(unc):
        shutil.copy(unc, os.path.join(study_out, f"{cfg_name}_uncertainty.npz"))
    elif BOOTSTRAP:
        print(f"  warning: no 14_Uncertainty/uncertainty.npz for {cfg_name}")

    qr = os.path.join(out_dir, "12_Quantitative_Results")
    dest = os.path.join(study_out, f"{cfg_name}_CarDpy_maps")
    os.makedirs(dest, exist_ok=True)
    for f in (os.listdir(qr) if os.path.isdir(qr) else []):
        if f.endswith(".png"):
            shutil.copy(os.path.join(qr, f), os.path.join(dest, f))
    stats = os.path.join(diag, "Rejection_Statistics.txt")
    if os.path.exists(stats) and cfg_name != "none":
        shutil.copy(stats, os.path.join(study_out, f"{cfg_name}_Rejection_Statistics.txt"))


def n_rejected(path):
    if not os.path.exists(path):
        return np.nan
    m = re.search(r"Total averages rejected:\s*(\d+)", open(path).read())
    return int(m.group(1)) if m else np.nan


def _slice(a, s):
    a = np.asarray(a)
    return a[:, :, s] if a.ndim >= 3 else a


def _cmaps():
    try:
        from cardpy.Colormaps import cDTI_Colormaps_Generator
        return cDTI_Colormaps_Generator()
    except Exception:
        return {}


def plot_grid(name, data, mask, bg, rows, fname, cmaps, fixed=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfgs = [c for c in DISPLAY if c in data]
    for s in range(mask.shape[2]):
        m = mask[:, :, s]
        if not m.any():
            continue
        ys, xs = np.where(m)
        pad = 6
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, m.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, m.shape[1])
        fig, axes = plt.subplots(len(rows), len(cfgs), figsize=(3.2 * len(cfgs), 3.0 * len(rows)), squeeze=False)
        for r, row in enumerate(rows):
            label, key = row[0], row[1]
            vals = []
            for c in cfgs:
                if key in data[c] and np.asarray(data[c][key]).shape[:2] == m.shape:
                    vals.append(np.where(m, _slice(data[c][key], s), np.nan)[y0:y1, x0:x1])
                else:
                    vals.append(None)
            if fixed:
                lo, hi = row[2]
            else:
                pool = np.concatenate([v[np.isfinite(v)] for v in vals if v is not None] or [np.array([0, 1])])
                lo, hi = 0, max(float(np.percentile(pool, 99)), 1e-6)
            cmap = (cmaps.get(label, "viridis") if isinstance(cmaps, dict) else "viridis") if fixed else "magma"
            for c_i, (c, v) in enumerate(zip(cfgs, vals)):
                ax = axes[r, c_i]; ax.set_xticks([]); ax.set_yticks([])
                if bg is not None:
                    ax.imshow(_slice(bg, s)[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=100)
                if v is None:
                    ax.set_title(f"{LABELS[c]}: {label} n/a", fontsize=9); continue
                im = ax.imshow(v, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
                ax.set_title(f"{LABELS[c]}: {label}", fontsize=10)
                if c_i == len(cfgs) - 1:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"{name}, slice {s + 1}")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, name, f"{fname}_slice{s + 1}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


def compare():
    cmaps = _cmaps()
    rows = []
    for name in sorted(d for d in os.listdir(OUT) if os.path.isdir(os.path.join(OUT, d))):
        study = os.path.join(OUT, name)
        met = {c: np.load(os.path.join(study, f"{c}.npz")) for c in CONFIGS
               if os.path.exists(os.path.join(study, f"{c}.npz"))}
        if not met:
            continue
        unc = {}
        for c in CONFIGS:
            for f in (os.path.join(study, f"{c}_uncertainty.npz"),
                      os.path.join(study, c, "CarDpy_Output", "14_Uncertainty", "uncertainty.npz")):
                if os.path.exists(f):
                    unc[c] = np.load(f); break
        ref = met.get("algorithm") or next(iter(met.values()))
        mask = ref["mask"] > 0
        if mask.ndim == 2:
            mask = mask[:, :, None]
        bg = ref["background"] if "background" in ref.files else None

        plot_grid(name, met, mask, bg, MAP_KEYS, "maps", cmaps, fixed=True)
        if unc:
            plot_grid(name, unc, mask, bg, UNC_KEYS, "uncertainty", cmaps, fixed=False)

        for c in met:
            row = dict(study=name, config=c,
                       rejected=n_rejected(os.path.join(study, f"{c}_Rejection_Statistics.txt")) if c != "none" else 0)
            for label, key, _ in MAP_KEYS:
                a = np.asarray(met[c][key]) if key in met[c].files else None
                if a is not None and a.shape[:mask.ndim] == mask.shape:
                    v = a[mask & np.isfinite(a)]
                    row[f"{label}_median"] = float(np.median(v)) if v.size else np.nan
            if c in unc:
                for label, key in UNC_KEYS:
                    a = np.asarray(unc[c][key]) if key in unc[c].files else None
                    if a is not None and a.shape[:mask.ndim] == mask.shape:
                        v = a[mask & np.isfinite(a)]
                        row[f"{label}_median"] = float(np.median(v)) if v.size else np.nan
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No results found in", OUT)
        return
    df.to_csv(os.path.join(OUT, "summary_per_study.csv"), index=False)
    summary = df.drop(columns="study").groupby("config").median(numeric_only=True)
    summary = summary.reindex([c for c in DISPLAY if c in summary.index]).round(4)
    summary.to_csv(os.path.join(OUT, "summary.csv"))
    print("\nMedians across studies (myocardium):")
    print(summary.to_string())
    print(f"\nwrote {OUT}/summary_per_study.csv, summary.csv, and <study>/maps_* / uncertainty_* figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare()
    else:
        if not STUDIES:
            raise SystemExit("Fill in STUDIES at the top of the script first.")
        resolved = {}
        for name, (path, sub) in STUDIES.items():
            if DATA_TYPE == "NifTis":
                if not os.path.exists(os.path.join(path, "NifTis", "data.nii")):
                    raise SystemExit(f"{name}: missing {os.path.join(path, 'NifTis', 'data.nii')}")
                resolved[name] = (path, None)
                continue
            sub = sub or find_subpath(path)
            if not os.path.isdir(os.path.join(path, sub)):
                raise SystemExit(f"{name}: DICOM folder not found: {os.path.join(path, sub)}")
            resolved[name] = (path, sub)
        for name, (path, sub) in resolved.items():
            os.makedirs(os.path.join(OUT, name), exist_ok=True)
            for i, c in enumerate(CONFIGS):
                run_one(name, path, sub, c, first=(i == 0))
        compare()
