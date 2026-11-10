"""
make_fig4.py -- Figure 3b: rejection configurations, both datasets in ONE row of panels.

  python ml_rejection/make_fig4.py
  python ml_rejection/make_fig4.py --dataset1 rejection_comparison_purl --dataset2 rejection_comparison_cima

Panels: MD | FA | HAP | dMD | dFA | dE1. One dot per participant-slice (Dataset-1 blue,
Dataset-2 orange); thin lines connect the same slice across configurations; boxes pool all
slices of both datasets. MD/FA/dMD/dFA/dE1 = median over the myocardium of that slice;
HAP = that slice's value from HAP.mat. Configurations: No rejection, Manual, Algorithm,
CNN-Reject (any missing configuration is simply left out of the line).

Writes ml_rejection/figures/fig3b.png + .svg (one row), fig3b_metrics.png/.svg and
fig3b_uncertainty.png/.svg (each three panels), fig3b_slice_values.csv, fig3b_summary.csv.
Prints a summary table per dataset and any missing runs.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
ORDER = ["none", "manual", "algorithm", "ml"]
LABELS = {"none": "No\nrejection", "manual": "Manual", "algorithm": "Algorithm", "ml": "CNN-\nReject"}
DATASET_COLORS = {"Dataset-1": "#1f77b4", "Dataset-2": "#ff7f0e"}
METRICS = [("MD", "Mean diffusivity", "µm²/ms", [1.0, 1.5, 2.0]),
           ("FA", "Fractional anisotropy", "unitless", [0.2, 0.3, 0.4]),
           ("HAP", "Helix angle pitch", "°/% wall depth", [-1.0, -0.5, 0.0])]
UNCS = [("dMD", "dMD", "µm²/ms", None), ("dFA", "dFA", "unitless", None), ("dE1", "dE1", "degrees", None)]

plt.rcParams.update({"svg.fonttype": "none", "font.size": 14, "axes.titlesize": 19,
                     "axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 14})


def as3d(a):
    a = np.asarray(a)
    return a[..., None] if a.ndim == 2 else a


def collect(root, dataset):
    rows, problems = [], []
    for study in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        sdir = os.path.join(root, study)
        for cfg in ORDER:
            f = os.path.join(sdir, f"{cfg}.npz")
            if not os.path.exists(f):
                continue
            with np.load(f) as z:
                mask = as3d(z["mask"]) > 0
                for m, key in (("MD", "dti_metrics__MD"), ("FA", "dti_metrics__FA")):
                    if key not in z:
                        problems.append(f"{dataset} {study} {cfg}: missing {key}"); continue
                    arr = as3d(z[key])
                    if arr.shape != mask.shape:
                        problems.append(f"{dataset} {study} {cfg}: {key} shape {arr.shape} != mask {mask.shape}")
                        continue
                    for s in range(mask.shape[2]):
                        v = arr[..., s][mask[..., s]]; v = v[np.isfinite(v)]
                        if v.size:
                            rows.append((dataset, study, cfg, m, s, float(np.median(v))))
            hp = os.path.join(sdir, cfg, "CarDpy_Output", "12_Quantitative_Results", "HAP.mat")
            if os.path.exists(hp):
                h = np.asarray(loadmat(hp).get("HAP", []), dtype=float).ravel()
                rows += [(dataset, study, cfg, "HAP", s, float(x)) for s, x in enumerate(h) if np.isfinite(x)]
            up = os.path.join(sdir, cfg, "CarDpy_Output", "14_Uncertainty", "uncertainty.npz")
            if not os.path.exists(up):
                up = os.path.join(sdir, f"{cfg}_uncertainty.npz")
            if os.path.exists(up):
                with np.load(up) as u:
                    for m, *_ in UNCS:
                        if m not in u:
                            problems.append(f"{dataset} {study} {cfg}: uncertainty has no {m}"); continue
                        a = as3d(u[m])
                        if a.shape != mask.shape:
                            problems.append(f"{dataset} {study} {cfg}: {m} shape {a.shape} != mask {mask.shape}")
                            continue
                        for s in range(mask.shape[2]):
                            v = a[..., s][mask[..., s]]; v = v[np.isfinite(v)]
                            if v.size:
                                rows.append((dataset, study, cfg, m, s, float(np.median(v))))
    return rows, problems


def panel(ax, d, cfgs, title, unit, ticks):
    for (ds, study, s), g in d.groupby(["dataset", "study", "slice"]):
        y = g.set_index("config").reindex(cfgs)["value"].to_numpy(float)
        ok = np.isfinite(y)
        if ok.any():
            ax.plot(np.arange(len(cfgs))[ok], y[ok], color=DATASET_COLORS[ds], lw=1, alpha=0.3, zorder=1)
    for i, c in enumerate(cfgs):
        vals = d.loc[d.config.eq(c), "value"].to_numpy()
        if not len(vals):
            continue
        bp = ax.boxplot([vals], positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                        manage_ticks=False, zorder=4,
                        medianprops=dict(color="black", lw=2.5),
                        whiskerprops=dict(color="black", lw=1.8), capprops=dict(color="black", lw=1.8))
        bp["boxes"][0].set(facecolor=(0.85, 0.85, 0.85, 0.35), edgecolor="black", linewidth=2)
        for ds, col in DATASET_COLORS.items():
            sub = d[d.config.eq(c) & d.dataset.eq(ds)]
            ax.scatter(np.full(len(sub), i), sub["value"], s=70, color=col, alpha=0.85, lw=0, zorder=3)
    ax.set_xticks(range(len(cfgs)), [LABELS[c] for c in cfgs])
    ax.set_xlim(-0.5, len(cfgs) - 0.5)
    ax.set_title(title); ax.set_ylabel(unit)
    if ticks:
        ax.set_yticks(ticks); ax.set_ylim(ticks[0], ticks[-1])
    else:
        ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)


def figure(df, cfgs, panels, stem, heading):
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 5.2))
    axes = np.atleast_1d(axes)
    for ax, (m, title, unit, ticks) in zip(axes, panels):
        panel(ax, df[df.metric.eq(m)], cfgs, title, unit, ticks)
    if heading:
        fig.suptitle(heading, x=0.01, ha="left", fontsize=22)
    fig.tight_layout()
    fig.legend(handles=[Line2D([], [], marker="o", ms=9, ls="", color=c, label=n)
                        for n, c in DATASET_COLORS.items()],
               loc="lower center", ncol=2, fontsize=15, frameon=False, bbox_to_anchor=(0.5, -0.08))
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGS, f"{stem}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{stem}.png and .svg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset1", default="rejection_comparison_purl")
    ap.add_argument("--dataset2", default="rejection_comparison_cima")
    args = ap.parse_args()
    if args.dataset1 or args.dataset2:
        if not (args.dataset1 and args.dataset2):
            ap.error("Provide both --dataset1 and --dataset2.")
        print("wrote", plot_comparison(args.dataset1, args.dataset2, args.output_dir))
        raise SystemExit(0)
    os.makedirs(FIGS, exist_ok=True)

    rows, problems = [], []
    for name, root in (("Dataset-1", args.dataset1), ("Dataset-2", args.dataset2)):
        if not os.path.isdir(root):
            raise SystemExit(f"{name} folder not found: {root}")
        r, p = collect(root, name); rows += r; problems += p
    df = pd.DataFrame(rows, columns=["dataset", "study", "config", "metric", "slice", "value"])
    if df.empty:
        raise SystemExit("No results found")
    cfgs = [c for c in ORDER if c in set(df.config)]

    # ---- printout ----
    for ds, g in df.groupby("dataset"):
        names = sorted(g.study.unique())
        n_slices = g[g.metric.eq("MD")].groupby("study").slice.nunique().sum()
        print(f"\n{ds}: {len(names)} participants, {n_slices} slices | "
              f"configurations {[c for c in cfgs if c in set(g.config)]}")
        print("  participant numbers:", {n: f"P{i + 1}" for i, n in enumerate(names)})
        table = g.pivot_table(index="config", columns="metric", values="value", aggfunc="median").reindex(cfgs)
        print("  median of slice values:")
        print(table.round(4).to_string())
        have = g[g.metric.isin(["dMD", "dFA", "dE1"])].groupby("study").config.apply(set).to_dict()
        present = [c for c in cfgs if c in set(g.config)]
        missing = [(n, [c for c in present if c not in have.get(n, set())]) for n in names]
        missing = [(n, m) for n, m in missing if m]
        total = len(names) * len(present)
        print(f"  uncertainty runs found: {total - sum(len(m) for _, m in missing)} of {total}")
        for n, m in missing:
            print(f"    {n}: no uncertainty for {m}")
    for p in problems:
        print("  WARNING:", p)

    df.to_csv(os.path.join(FIGS, "fig3b_slice_values.csv"), index=False)
    (df.pivot_table(index=["dataset", "config"], columns="metric", values="value", aggfunc="median")
       .round(4).to_csv(os.path.join(FIGS, "fig3b_summary.csv")))

    figure(df, cfgs, METRICS + UNCS, "fig3b", "")
    figure(df, cfgs, METRICS, "fig3b_metrics", "")
    figure(df, cfgs, UNCS, "fig3b_uncertainty", "")


if __name__ == "__main__":
    main()