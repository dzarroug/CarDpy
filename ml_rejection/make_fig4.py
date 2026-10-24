"""
make_fig4.py -- Figure 4 from the comparison runs.

(a) Box plots per rejection configuration: MD and FA (myocardial voxels pooled across
    participants, with jittered points) and helix angle pitch (one point per slice).
(b) Uncertainty: one line per participant connecting its median dMD, dFA, dE1 across
    configurations (needs bootstrapped runs; panel is skipped until they exist).

  python ml_rejection/make_fig4.py
  python ml_rejection/make_fig4.py --root rejection_comparison_purl

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


def collect(root):
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


def study_colors(names):
    cmap = plt.get_cmap("tab10")
    return {n: cmap(i % 10) for i, n in enumerate(sorted(names))}


def boxpanel(ax, df, cfgs, title, ylabel, rng, colors, max_pts=3000):
    data = [df.loc[df.config == c, "value"].to_numpy() for c in cfgs]
    bp = ax.boxplot(data, positions=range(len(cfgs)), widths=0.5, showfliers=False,
                    patch_artist=True, medianprops=dict(color="black", lw=2))
    for patch in bp["boxes"]:
        patch.set_facecolor("none"); patch.set_edgecolor("black"); patch.set_linewidth(1.5)
    for i, c in enumerate(cfgs):
        sub = df[df.config == c]
        if len(sub) > max_pts:
            sub = sub.iloc[rng.choice(len(sub), max_pts, replace=False)]
        dense = len(sub) > 100
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(sub)), sub["value"], s=4 if dense else 22,
                   alpha=0.35 if dense else 0.9, color=[colors[n] for n in sub["study"]],
                   zorder=0 if dense else 3, lw=0)
    ax.set_xticks(range(len(cfgs)), [LABELS[c] for c in cfgs], fontsize=9)
    ax.set_title(title, fontsize=11); ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def paired_panel(ax, df, cfgs, title, ylabel, colors):
    """One line per participant/slice connecting its value across configurations."""
    for (study, slc), g in df.groupby(["study", "slice"]):
        g = g.set_index("config").reindex(cfgs)
        ax.plot(range(len(cfgs)), g["value"], "-o", color=colors[study], lw=1.2, ms=5,
                alpha=0.85, mec="black", mew=0.4)
    ax.set_xticks(range(len(cfgs)), [LABELS[c] for c in cfgs], fontsize=9)
    ax.set_xlim(-0.4, len(cfgs) - 0.6)
    ax.set_title(title, fontsize=11); ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def participant_legend(fig, colors):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=col, label=f"Participant {i + 1}")
               for i, (n, col) in enumerate(sorted(colors.items()))]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.06))


def fig_a(vox, hap, cfgs):
    rng = np.random.default_rng(0)
    colors = study_colors(vox.study.unique())
    panels = [("MD", "Mean diffusivity", "µm²/ms"), ("FA", "Fractional anisotropy", "unitless")]
    n = len(panels) + (0 if hap.empty else 1)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    for ax, (m, title, unit) in zip(axes, panels):
        boxpanel(ax, vox[vox.metric == m], cfgs, title, unit, rng, colors)
    if not hap.empty:
        paired_panel(axes[-1], hap, cfgs, "Helix angle pitch\n(one line per slice)", "°/% wall depth", colors)
    fig.suptitle("(a) cDTI metrics by shot-rejection configuration", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    participant_legend(fig, colors)
    out = os.path.join(FIGS, "fig4a.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    return out


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
    args = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    vox, hap, unc = collect(args.root)
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