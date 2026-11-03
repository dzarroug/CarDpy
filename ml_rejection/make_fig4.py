"""
make_fig4.py -- Figure 4 from the comparison runs.

(a) Box plots per rejection configuration: median myocardial MD
    and FA per slice, and helix angle pitch (one point per slice).
(b) Uncertainty: one line per participant connecting its median dMD, dFA, dE1 across
    configurations (needs bootstrapped runs; panel is skipped until they exist).

  python ml_rejection/make_fig4.py
  python ml_rejection/make_fig4.py --root rejection_comparison_purl
  python ml_rejection/make_fig4.py --dataset1 /path/to/dataset1 --dataset2 /path/to/dataset2

From Python: plot_comparison(dataset1_path, dataset2_path, output_dir=None)

Reads rejection_comparison_purl/<study>/<config>.npz
      .../<study>/<config>/CarDpy_Output/12_Quantitative_Results/HAP.mat
      .../<study>/<config>/CarDpy_Output/14_Uncertainty/uncertainty.npz   (or <config>_uncertainty.npz)
Writes ml_rejection/figures/fig4a.png, fig4b.png, fig4_values.csv
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
ORDER = ["none", "manual", "algorithm", "ml"]
LABELS = {"none": "No\nrejection", "manual": "Manual", "algorithm": "Algorithm", "ml": "ML"}
COLORS = {"none": "#7f7f7f", "manual": "#9467bd", "algorithm": "#1f77b4", "ml": "#ff7f0e"}


def load_hap(study, c):
    f = os.path.join(study, c, "CarDpy_Output", "12_Quantitative_Results", "HAP.mat")
    if not os.path.exists(f):
        return None
    from scipy.io import loadmat
    return np.asarray(loadmat(f).get("HAP", np.nan), dtype=float).ravel()   # one value per slice


def load_unc(study, c):
    for f in (os.path.join(study, c, "CarDpy_Output", "14_Uncertainty", "uncertainty.npz"),
              os.path.join(study, f"{c}_uncertainty.npz")):
        if os.path.exists(f):
            return np.load(f)
    return None


def collect(root, slice_medians=False):
    vox, hap, unc = [], [], []
    for name in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        study = os.path.join(root, name)
        for c in ORDER:
            f = os.path.join(study, f"{c}.npz")
            if not os.path.exists(f):
                continue
            d = np.load(f)
            mask = d["mask"] > 0
            for label, key in (("MD", "dti_metrics__MD"), ("FA", "dti_metrics__FA")):
                if key in d.files and d[key].shape[:mask.ndim] == mask.shape:
                    if slice_medians:
                        for slc in range(mask.shape[2]):
                            v = d[key][:, :, slc][mask[:, :, slc]]
                            v = v[np.isfinite(v)]
                            if v.size:
                                vox.append(pd.DataFrame({"study": [name], "config": [c],
                                    "metric": [label], "slice": [slc], "value": [np.median(v)]}))
                        continue
                    v = d[key][mask]
                    v = v[np.isfinite(v)]
                    vox.append(pd.DataFrame({"study": name, "config": c, "metric": label, "value": v}))
            h = load_hap(study, c)
            if h is not None:
                hdf = pd.DataFrame({"study": name, "config": c, "slice": np.arange(len(h)), "value": h})
                hap.append(hdf[np.isfinite(hdf.value)])
            u = load_unc(study, c)
            if u is not None:
                row = {"study": name, "config": c}
                for k in ("dMD", "dFA", "dE1"):
                    if k in u.files and u[k].shape[:mask.ndim] == mask.shape:
                        x = u[k][mask]
                        row[k] = float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan
                unc.append(row)
    cat = lambda L: pd.concat(L, ignore_index=True) if L else pd.DataFrame()
    return cat(vox), cat(hap), pd.DataFrame(unc)


def collect_datasets(roots, slice_medians=False):
    """Combine named datasets while keeping participant/slice pairing unique."""
    tables = [[], [], []]
    for dataset, root in roots.items():
        for parts, frame in zip(tables, collect(root, slice_medians=slice_medians)):
            if frame.empty:
                continue
            frame = frame.copy()
            frame['dataset'] = dataset
            frame['study'] = dataset + '/' + frame['study']
            parts.append(frame)
    return tuple(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
                 for parts in tables)


def study_colors(names):
    cmap = plt.get_cmap("tab10")
    return {n: cmap(i % 10) for i, n in enumerate(sorted(names))}


def boxpanel(ax, df, cfgs, title, ylabel, rng, colors, max_pts=3000):
    studies = sorted(colors)
    if "slice" in df.columns:
        for (study, slc), g in df.groupby(["study", "slice"]):
            values = g.set_index("config").reindex(cfgs)["value"]
            ax.plot(range(len(cfgs)), values, color=colors[study], lw=1,
                    alpha=0.25, zorder=1)
    for i, c in enumerate(cfgs):
        values = df.loc[df.config == c, "value"].to_numpy()
        if not len(values):
            continue
        bp = ax.boxplot([values], positions=[i], widths=0.5, showfliers=False,
                        patch_artist=True, medianprops=dict(color="black", lw=2.5),
                        whiskerprops=dict(color="black", lw=1.8),
                        capprops=dict(color="black", lw=1.8),
                        manage_ticks=False, zorder=4)
        bp["boxes"][0].set(facecolor=(0.85, 0.85, 0.85, 0.35), edgecolor="black", linewidth=2)
        for study in studies:
            sub = df[(df.config == c) & (df.study == study)]
            if sub.empty:
                continue
            if len(sub) > max_pts:
                sub = sub.iloc[rng.choice(len(sub), max_pts, replace=False)]
            dense = len(sub) > 100
            ax.scatter(np.full(len(sub), i),
                       sub["value"], s=45 if dense else 85,
                       alpha=0.35 if dense else 0.9, color=colors[study],
                       zorder=0 if dense else 3, lw=0)
    ax.set_xticks(range(len(cfgs)), [LABELS[c] for c in cfgs], fontsize=9)
    ax.set_title(title, fontsize=11); ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def paired_panel(ax, df, cfgs, title, ylabel, colors):
    """One line per participant/slice connecting its value across configurations."""
    for (study, slc), g in df.groupby(["study", "slice"]):
        g = g.set_index("config").reindex(cfgs)
        ax.plot(range(len(cfgs)), g["value"], color=colors[study], lw=1,
                alpha=0.25, zorder=1)
        ax.scatter(range(len(cfgs)), g["value"], color=colors[study], s=64,
                   alpha=0.85, edgecolors="black", linewidths=0.4, zorder=3)
    ax.set_xticks(range(len(cfgs)), [LABELS[c] for c in cfgs], fontsize=9)
    ax.set_xlim(-0.4, len(cfgs) - 0.6)
    ax.set_title(title, fontsize=11); ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def participant_legend(fig, colors):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ms=8, ls="", color=col, label=f"Participant {i + 1}")
               for i, (n, col) in enumerate(sorted(colors.items()))]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=14, frameon=False,
               bbox_to_anchor=(0.5, -0.06))


def fig_a(vox, hap, cfgs, output_dir=None):
    rng = np.random.default_rng(0)
    colors = {name: "#1f77b4" for name in vox.study.unique()}
    dataset_colors = {}
    if "dataset" in vox.columns:
        dataset_colors = dict(zip(vox.dataset.unique(), plt.get_cmap("tab10").colors))
        colors = {row.study: dataset_colors[row.dataset]
                  for row in vox[["study", "dataset"]].drop_duplicates().itertuples()}
    panels = [("MD", "Mean diffusivity", "µm²/ms"),
              ("FA", "Fractional anisotropy", "unitless")]
    n = len(panels) + (0 if hap.empty else 1)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    for ax, (m, title, unit) in zip(axes, panels):
        boxpanel(ax, vox[vox.metric == m], cfgs, title, unit, rng, colors)
    if not hap.empty:
        boxpanel(axes[-1], hap, cfgs, "Helix angle pitch", "°/% wall depth", rng, colors)
    for ax, ticks in zip(axes, ([1, 1.5, 2], [0.2, 0.3, 0.4], [-1, -0.5, 0])):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_yticks(ticks)
        ax.tick_params(axis="both", labelsize=15)
        ax.xaxis.label.set_size(17)
        ax.yaxis.label.set_size(17)
        ax.title.set_size(20)
    fig.suptitle("(a) cDTI metrics by shot-rejection configuration", x=0.01, ha="left", fontsize=22)
    fig.tight_layout()
    if len(dataset_colors) > 1:
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([], [], marker="o", ms=8, ls="", color=color,
                                   label=f"Dataset {i + 1}")
                            for i, (name, color) in enumerate(dataset_colors.items())],
                   loc="lower center", ncol=len(dataset_colors), fontsize=14,
                   frameon=False, bbox_to_anchor=(0.5, -0.06))
    out_dir = os.fspath(output_dir) if output_dir is not None else FIGS
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "fig4a.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    return out


def plot_comparison(dataset1_path, dataset2_path, output_dir=None):
    """Save Figure 4a for two dataset folders and return its PNG path.

    Dataset 1 is blue; Dataset 2 is orange. Inputs contain one participant
    directory per study with none.npz, algorithm.npz, and ml.npz results.
    MD/FA dots are slice medians; HAP is read from each run's HAP.mat.
    """
    roots = {"Dataset 1": os.path.abspath(os.path.expanduser(os.fspath(dataset1_path))),
             "Dataset 2": os.path.abspath(os.path.expanduser(os.fspath(dataset2_path)))}
    for name, root in roots.items():
        if not os.path.isdir(root):
            raise FileNotFoundError(f"{name} folder not found: {root}")
    vox, hap, _ = collect_datasets(roots, slice_medians=True)
    present = set(vox.dataset) if not vox.empty else set()
    for name, root in roots.items():
        if name not in present:
            raise ValueError(f"No usable MD/FA results for {name}: {root}")
    cfgs = [c for c in ORDER if c in set(vox.config)]
    out_dir = os.path.expanduser(os.fspath(output_dir)) if output_dir is not None else FIGS
    path = fig_a(vox, hap, cfgs, output_dir=out_dir)
    vox.to_csv(os.path.join(out_dir, "fig4a_slice_medians.csv"), index=False)
    hap.to_csv(os.path.join(out_dir, "fig4a_hap.csv"), index=False)
    return path


def fig_b(unc, cfgs):
    colors = study_colors(unc.study.unique())
    panels = [("dMD", "dMD", "µm²/ms"), ("dFA", "dFA", "unitless"), ("dE1", "dE1", "degrees")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (k, title, unit) in zip(axes, panels):
        if k not in unc:
            ax.set_visible(False); continue
        for name, g in unc.groupby("study"):
            g = g.set_index("config").reindex(cfgs)
            ax.plot(range(len(cfgs)), g[k], "-o", color=colors[name], lw=1.5, ms=6,
                    mec="black", mew=0.5)
        ax.set_xticks(range(len(cfgs)), [LABELS[c] for c in cfgs], fontsize=9)
        ax.set_title(f"{title} [{unit}]", fontsize=11)
        ax.set_ylim(bottom=0); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Median myocardial uncertainty")
    fig.suptitle("(b) Bootstrapped uncertainty (one line per participant)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    participant_legend(fig, colors)
    out = os.path.join(FIGS, "fig4b.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="rejection_comparison_purl")
    ap.add_argument("--dataset1", help="Dataset 1 results folder (blue)")
    ap.add_argument("--dataset2", help="Dataset 2 results folder (orange)")
    ap.add_argument("--output-dir", help="Output folder for the two-dataset figure")
    args = ap.parse_args()
    if args.dataset1 or args.dataset2:
        if not (args.dataset1 and args.dataset2):
            ap.error("Provide both --dataset1 and --dataset2.")
        print("wrote", plot_comparison(args.dataset1, args.dataset2, args.output_dir))
        raise SystemExit(0)
    os.makedirs(FIGS, exist_ok=True)
    vox, hap, unc = collect(args.root, slice_medians=True)
    if vox.empty:
        raise SystemExit(f"No <config>.npz results found under {args.root}")
    cfgs = [c for c in ORDER if c in set(vox.config)]
    print("configs:", cfgs, "| participants:", vox.study.nunique())

    rows = [vox.groupby(["config", "metric"])["value"].median().unstack().add_suffix("_median")]
    if not hap.empty:
        rows.append(hap.groupby("config")["value"].median().rename("HAP_median"))
    if not unc.empty:
        rows.append(unc.groupby("config")[[k for k in ("dMD", "dFA", "dE1") if k in unc]].median()
                    .add_suffix("_median"))
    table = pd.concat(rows, axis=1).reindex(cfgs).round(4)
    table.to_csv(os.path.join(FIGS, "fig4_values.csv"))
    print(table.to_string())

    print("wrote", fig_a(vox, hap, cfgs))
    if unc.empty:
        print("(b) skipped: no bootstrapped uncertainty found yet")
    else:
        print("wrote", fig_b(unc, cfgs))
