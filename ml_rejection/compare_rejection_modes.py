"""
compare_rejection_modes.py
Does ML rejection change the final cDTI maps compared with the algorithm?

For each study: run rejection_mode='algorithm' (draw contours), then 'ml' in the
SAME output folder, reusing those contours, so the only difference between the
two runs is the rejection step. Results are saved to ml_comparison/<study>/.

  python ml_rejection/compare_rejection_modes.py            # run both modes on every study
  python ml_rejection/compare_rejection_modes.py --compare  # summarize saved results + map figures
Figures: ml_comparison/<study>/maps_slice<N>.png  (rows MD/FA/HA; cols algorithm | ML | difference),
         drawn like CarDpy's maps (same metrics, ranges, colormaps, background), plus
         ml_comparison/<study>/<mode>_CarDpy_maps/ = CarDpy's own 9-panel figures for each run.

Use the 5 ML TEST participants only -- the model never saw them in training.
"""
import argparse
import copy
import os
import re
import shutil

import numpy as np
import pandas as pd

# ---------------- EDIT THESE ----------------
# name: (full path to the raw study folder, DICOM series path inside it)
# Use the same phase-encode direction (e.g. RL) for every study.
STUDIES = {
    # "Healthy_Volunteer_0XX": ("/full/path/to/Healthy_Volunteer_0XX", "02_cDTI/SAX/cDTI_SF_b350_RL_71"),
}
MODEL = "ml_rejection/models/crop80/best.pt"
SLICES = None                 # None = all slices; e.g. [3] for a quick test
OUT = "ml_comparison"
# --------------------------------------------

ANGLE_PREFIXES = ("HA", "E2A", "TA")      # angles in degrees: compare with wrap-around


def run_one(name, study_path, dicom_subpath, mode, reuse_contours):
    from cardpy.config import DEFAULT_CONFIG
    from cardpy.pipeline import process

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["dicom_subpath"] = dicom_subpath
    cfg["slice_index"] = SLICES
    cfg["rejection"]["enabled"] = True
    cfg["rejection"]["mode"] = mode
    cfg["rejection"]["manual_review"] = False
    cfg["rejection"]["ml_model_path"] = os.path.abspath(MODEL)
    cfg["gui"]["saved_configuration_mode"] = "off"       # fresh rejection every run
    cfg["gui"]["reuse_existing_segmentation"] = reuse_contours
    cfg["gui"]["segmentation_review"] = not reuse_contours
    out_dir = os.path.abspath(os.path.join(OUT, name, "CarDpy_Output"))
    cfg["output"]["directory"] = out_dir

    print(f"\n===== {name}: {mode} =====")
    res = process(study_path, cfg)

    flat = {}
    for group in ("dti_metrics", "cardiac_metrics"):
        for key, val in (res.get(group) or {}).items():
            flat[f"{group}__{key}"] = np.asarray(val, dtype=float)
    flat["mask"] = np.asarray(res["mask"], dtype=float)
    # same background CarDpy uses under its maps: mean of the averaged images
    flat["background"] = np.nanmean(np.real(np.asarray(res["matrix"])), axis=(3, 4))
    np.savez_compressed(os.path.join(OUT, name, f"{mode}.npz"), **flat)

    # keep CarDpy's own 9-panel map figures for this mode (the ml run overwrites the folder)
    qr = os.path.join(out_dir, "12_Quantitative_Results")
    dest = os.path.join(OUT, name, f"{mode}_CarDpy_maps")
    os.makedirs(dest, exist_ok=True)
    for f in os.listdir(qr) if os.path.isdir(qr) else []:
        if f.endswith(".png"):
            shutil.copy(os.path.join(qr, f), os.path.join(dest, f))

    stats = os.path.join(out_dir, "13_Diagnostics", "Rejection_Statistics.txt")
    if os.path.exists(stats):
        shutil.copy(stats, os.path.join(OUT, name, f"{mode}_Rejection_Statistics.txt"))


def n_rejected(path):
    if not os.path.exists(path):
        return np.nan
    m = re.search(r"Total averages rejected:\s*(\d+)", open(path).read())
    return int(m.group(1)) if m else np.nan


# Same metrics, ranges and colormaps as CarDpy's _save_quantitative_maps
# (HA shown is the spatially filtered HASF, as in CarDpy's figure).
MAP_KEYS = [("MD", "dti_metrics__MD", (0, 3)), ("FA", "dti_metrics__FA", (0, 1)),
            ("HA", "cardiac_metrics__HASF", (-90, 90))]


def _cmaps():
    try:
        from cardpy.Colormaps import cDTI_Colormaps_Generator
        return cDTI_Colormaps_Generator()
    except Exception:
        return {}


def plot_maps(name, A, B, cmaps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = A["mask"] > 0
    if mask.ndim == 2:
        mask = mask[:, :, None]
    keys = [(label, k, rng) for label, k, rng in MAP_KEYS if k in A.files and k in B.files]
    bg_all = A["background"] if "background" in A.files else None
    if not keys:
        return
    for s in range(mask.shape[2]):
        m = mask[:, :, s]
        if not m.any():
            continue
        ys, xs = np.where(m)
        pad = 6
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, m.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, m.shape[1])
        fig, axes = plt.subplots(len(keys), 3, figsize=(10, 3.2 * len(keys)), squeeze=False)
        bg = None
        if bg_all is not None:
            bg = (bg_all[:, :, s] if bg_all.ndim == 3 else bg_all)[y0:y1, x0:x1]
        for r, (label, k, (lo, hi)) in enumerate(keys):
            a = np.asarray(A[k]); b = np.asarray(B[k])
            a = a[:, :, s] if a.ndim == 3 else a
            b = b[:, :, s] if b.ndim == 3 else b
            a = np.where(m, a, np.nan)[y0:y1, x0:x1]
            b = np.where(m, b, np.nan)[y0:y1, x0:x1]
            d = np.abs(a - b)
            if label == "HA":
                d = np.minimum(d, 180 - d)
            cmap = cmaps.get(label, "viridis") if isinstance(cmaps, dict) else "viridis"
            for c, (img, title, kw) in enumerate([
                    (a, f"{label}: algorithm", dict(cmap=cmap, vmin=lo, vmax=hi)),
                    (b, f"{label}: ML", dict(cmap=cmap, vmin=lo, vmax=hi)),
                    (d, f"{label}: |difference|", dict(cmap="magma", vmin=0,
                                                      vmax=max(float(np.nanpercentile(d, 99)), 1e-6) if np.isfinite(d).any() else 1))]):
                ax = axes[r, c]
                if bg is not None:
                    ax.imshow(bg, vmin=0, vmax=100, cmap="gray")
                im = ax.imshow(img, interpolation="nearest", **kw)
                ax.set_title(title, fontsize=10); ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"{name}, slice {s + 1}: algorithm vs ML rejection")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, name, f"maps_slice{s + 1}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def compare():
    cmaps = _cmaps()
    rows = []
    for name in sorted(os.listdir(OUT)):
        a_path, b_path = (os.path.join(OUT, name, f"{m}.npz") for m in ("algorithm", "ml"))
        if not (os.path.exists(a_path) and os.path.exists(b_path)):
            continue
        A, B = np.load(a_path), np.load(b_path)
        plot_maps(name, A, B, cmaps)
        mask = A["mask"] > 0
        rej_a = n_rejected(os.path.join(OUT, name, "algorithm_Rejection_Statistics.txt"))
        rej_b = n_rejected(os.path.join(OUT, name, "ml_Rejection_Statistics.txt"))
        for key in A.files:
            if key in ("mask", "background") or key not in B.files or A[key].shape != B[key].shape:
                continue
            a, b = A[key], B[key]
            m = mask if mask.shape == a.shape[:mask.ndim] else np.ones(a.shape, bool)
            if a.ndim > m.ndim:            # e.g. vector-valued maps: skip
                continue
            sel = m & np.isfinite(a) & np.isfinite(b)
            if not sel.any():
                continue
            metric = key.split("__")[1]
            d = np.abs(a[sel] - b[sel])
            if metric.startswith(ANGLE_PREFIXES):
                d = np.minimum(d, 180 - d)
            rows.append(dict(study=name, metric=metric,
                             algorithm_mean=a[sel].mean(), ml_mean=b[sel].mean(),
                             mean_abs_diff=d.mean(), voxels=int(sel.sum()),
                             rejected_algorithm=rej_a, rejected_ml=rej_b))
    df = pd.DataFrame(rows)
    if df.empty:
        print("No completed study pairs found in", OUT)
        return
    df.to_csv(os.path.join(OUT, "comparison_per_study.csv"), index=False)
    summary = (df.groupby("metric")[["algorithm_mean", "ml_mean", "mean_abs_diff"]]
                 .mean().round(4))
    summary.to_csv(os.path.join(OUT, "comparison_summary.csv"))
    print(summary.to_string())
    print("\nimages rejected per study (algorithm vs ml):")
    print(df.drop_duplicates("study")[["study", "rejected_algorithm", "rejected_ml"]].to_string(index=False))
    print(f"\nwrote {OUT}/comparison_per_study.csv, comparison_summary.csv, and <study>/maps_slice*.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare()
    else:
        if not STUDIES:
            raise SystemExit("Fill in STUDIES at the top of the script first.")
        for name, (path, subpath) in STUDIES.items():
            if not os.path.isdir(os.path.join(path, subpath)):
                raise SystemExit(f"{name}: DICOM folder not found: {os.path.join(path, subpath)}")
        for name, (path, subpath) in STUDIES.items():
            os.makedirs(os.path.join(OUT, name), exist_ok=True)
            run_one(name, path, subpath, "algorithm", reuse_contours=False)
            run_one(name, path, subpath, "ml", reuse_contours=True)
        compare()