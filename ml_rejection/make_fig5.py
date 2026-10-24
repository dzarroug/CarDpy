"""
make_fig5.py -- Figure 5: automated vs manual LV segmentation.

Uses the comparison runs: the manual contours are the ones drawn for the
'algorithm' configuration. Step 1 re-runs that exact configuration with
segmentation.mode = "auto" (same crops, same rejection decisions -> no GUIs),
so the ONLY difference is the segmentation. Step 2 compares.

  python ml_rejection/make_fig5.py --run        # step 1 (after the comparison runs finish)
  python ml_rejection/make_fig5.py              # step 2: figure + table

Options: --root rejection_comparison_purl  --raw PURL_test_raw   (Cima.X: --root rejection_comparison_cima --raw /path/to/Dahnia)

Writes  <root>/<study>/auto_seg.npz (+ auto_seg/CarDpy_Output/)
        ml_rejection/figures/fig5.png, fig5_per_slice.csv
"""
import matplotlib
matplotlib.use("Agg")   # must come before cardpy (macOS backend + Tk GUIs segfault)
import argparse
import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
REF = "algorithm"   # the configuration whose manual contours are the reference


def studies(root):
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d, REF, "CarDpy_Output")))


def run_auto(root, raw):
    from cardpy.pipeline import process

    for name in studies(root):
        src = os.path.join(root, name, REF, "CarDpy_Output")
        settings = json.load(open(os.path.join(src, "processing_settings.json")))
        cfg = settings.get("cardpy", settings)
        out = os.path.abspath(os.path.join(root, name, "auto_seg", "CarDpy_Output"))
        if os.path.exists(os.path.join(root, name, "auto_seg.npz")):
            print(f"{name}: already done"); continue
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(os.path.join(out, "13_Diagnostics"), exist_ok=True)
        for f in ("Rejection_Configuration.json", "Respiratory_Crop.json"):   # identical crops + rejection
            p = os.path.join(src, "13_Diagnostics", f)
            if os.path.exists(p):
                shutil.copy(p, os.path.join(out, "13_Diagnostics", f))

        cfg["output"]["directory"] = out
        cfg["gui"]["saved_configuration_mode"] = "reuse"
        cfg["gui"]["reuse_existing_segmentation"] = False
        cfg["gui"]["segmentation_review"] = False
        cfg.setdefault("uncertainty", {})["enabled"] = False
        cfg.setdefault("segmentation", {}).update(mode="auto", review=False, use_model_mask=True)

        print(f"\n===== {name}: auto segmentation =====")
        res = process(os.path.join(raw, name), cfg)
        flat = {f"{g}__{k}": np.asarray(v, dtype=float)
                for g in ("dti_metrics", "cardiac_metrics") for k, v in (res.get(g) or {}).items()}
        flat["mask"] = np.asarray(res["mask"], dtype=float)
        flat["background"] = np.nanmean(np.real(np.asarray(res["matrix"])), axis=(3, 4))
        np.savez_compressed(os.path.join(root, name, "auto_seg.npz"), **flat)
        print(f"{name}: sources = {res.get('segmentation_sources')}")


def hap(path):
    if not os.path.exists(path):
        return None
    from scipy.io import loadmat
    return np.asarray(loadmat(path).get("HAP", np.nan), dtype=float).ravel()


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan


def outline(ax, m, color, label):
    if m.any():
        ax.contour(m.astype(float), levels=[0.5], colors=color, linewidths=1.6)
    ax.plot([], [], color=color, lw=2, label=label)


def compare(root):
    rows, examples = [], []
    for name in studies(root):
        fa, fb = os.path.join(root, name, f"{REF}.npz"), os.path.join(root, name, "auto_seg.npz")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        A, B = np.load(fa), np.load(fb)
        ma, mb = A["mask"] > 0, B["mask"] > 0
        if ma.ndim == 2:
            ma, mb = ma[..., None], mb[..., None]
        ha = hap(os.path.join(root, name, REF, "CarDpy_Output", "12_Quantitative_Results", "HAP.mat"))
        hb = hap(os.path.join(root, name, "auto_seg", "CarDpy_Output", "12_Quantitative_Results", "HAP.mat"))
        for s in range(ma.shape[2]):
            a, b = ma[..., s], mb[..., s]
            if not a.any():
                continue
            row = dict(study=name, slice=s + 1, dice=dice(a, b))
            for k, key in (("MD", "dti_metrics__MD"), ("FA", "dti_metrics__FA")):
                va, vb = A[key][..., s], B[key][..., s]
                row[f"{k}_manual"] = float(np.nanmedian(va[a])) if a.any() else np.nan
                row[f"{k}_auto"] = float(np.nanmedian(vb[b])) if b.any() else np.nan
            if ha is not None and hb is not None and s < len(ha) and s < len(hb):
                row["HAP_manual"], row["HAP_auto"] = float(ha[s]), float(hb[s])
            rows.append(row)
            examples.append((row["dice"], name, s, A["background"], a, b))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No studies with both manual and auto results -- run with --run first.")
    os.makedirs(FIGS, exist_ok=True)
    df.to_csv(os.path.join(FIGS, "fig5_per_slice.csv"), index=False)

    metrics = [m for m in ("MD", "FA", "HAP") if f"{m}_manual" in df]
    units = {"MD": "µm²/ms", "FA": "", "HAP": "°/%"}
    fig = plt.figure(figsize=(4 + 3.3 * (len(metrics) + 1), 4))
    gs = fig.add_gridspec(1, len(metrics) + 2, width_ratios=[1.3, 0.8] + [1] * len(metrics), wspace=0.4)

    # (a) example: the slice closest to the median Dice
    med = df.dice.median()
    _, name, s, bg, a, b = min(examples, key=lambda e: abs(e[0] - med))
    ys, xs = np.where(a | b); pad = 8
    y0, y1 = max(ys.min() - pad, 0), ys.max() + pad + 1
    x0, x1 = max(xs.min() - pad, 0), xs.max() + pad + 1
    ax = fig.add_subplot(gs[0])
    img = bg[..., s] if bg.ndim == 3 else bg
    ax.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=np.nanpercentile(img, 99.5))
    outline(ax, a[y0:y1, x0:x1], "#2ca02c", "Manual")
    outline(ax, b[y0:y1, x0:x1], "#ff7f0e", "Automated")
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8, frameon=False)
    ax.set_title(f"(a) Example (Dice {dice(a, b):.2f})", fontsize=10)

    # (b) Dice per slice
    ax = fig.add_subplot(gs[1])
    rng = np.random.default_rng(0)
    ax.boxplot(df.dice, widths=0.5, showfliers=False, medianprops=dict(color="black", lw=2))
    ax.scatter(1 + rng.uniform(-0.12, 0.12, len(df)), df.dice, s=18, color="#ff7f0e", zorder=3)
    ax.set_ylim(min(0.6, df.dice.min() - 0.05), 1); ax.set_xticks([])
    ax.set_title(f"(b) Dice\nmedian {med:.2f}", fontsize=10); ax.grid(axis="y", alpha=0.3)

    # (c) manual vs automated metrics, one point per slice
    for i, m in enumerate(metrics):
        ax = fig.add_subplot(gs[2 + i])
        x, y = df[f"{m}_manual"], df[f"{m}_auto"]
        ax.scatter(x, y, s=22, color="#1f77b4", edgecolor="black", lw=0.4)
        lo, hi = np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])
        pad = 0.05 * (hi - lo or 1)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray", lw=1)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"Manual {units[m]}".strip()); ax.set_ylabel(f"Automated {units[m]}".strip())
        diff = (y - x)
        ax.set_title(f"{'(c) ' if i == 0 else ''}{m}\nbias {np.nanmean(diff):+.3f}", fontsize=10)
        ax.set_aspect("equal", adjustable="box")
    out = os.path.join(FIGS, "fig5.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)

    print(f"{len(df)} slices from {df.study.nunique()} participants | Dice median {med:.3f} "
          f"(IQR {df.dice.quantile(.25):.3f}-{df.dice.quantile(.75):.3f})")
    for m in metrics:
        d = df[f"{m}_auto"] - df[f"{m}_manual"]
        print(f"{m}: bias {d.mean():+.4f}, 95% limits of agreement {d.mean() - 1.96 * d.std():+.4f} to "
              f"{d.mean() + 1.96 * d.std():+.4f}")
    print("wrote", out, "and fig5_per_slice.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="run auto segmentation first")
    ap.add_argument("--root", default="rejection_comparison_purl")
    ap.add_argument("--raw", default="PURL_test_raw", help="folder containing the raw <study> folders")
    args = ap.parse_args()
    if args.run:
        run_auto(args.root, args.raw)
    compare(args.root)