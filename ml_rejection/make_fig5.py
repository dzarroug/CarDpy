"""
make_fig5.py -- Figure 5: automated vs manual LV segmentation.

Uses the comparison runs: the manual contours are the ones you drew for the
'algorithm' configuration. Step 1 re-runs that exact configuration with
segmentation.mode = "auto" (same crops, same rejection decisions -> no GUIs),
so the ONLY difference is the segmentation. Step 2 compares.

  python ml_rejection/make_fig5.py --run        # step 1 (after the comparison runs finish)
  python ml_rejection/make_fig5.py              # step 2: figure + table

Second reader: re-run with saved crops/rejection and draw your own contours (opens the
contour GUI only), saved as <name>.npz next to the others:
  python ml_rejection/make_fig5.py --manual-run ariel_seg --root rejection_comparison_cima --raw /path/to/cima

Final figure, one row per pair in this order -- (A) Reader 1 vs nnUNet-Segment,
(B) Reader 2 vs nnUNet-Segment, (C) Reader 1 vs Reader 2:
  python ml_rejection/make_fig5.py --root rejection_comparison_cima --three-readers
  (Reader 1 = ariel_seg, Reader 2 = algorithm, auto = auto_seg; change with --reader1/--reader2/--auto)

Any single pair can be compared (default: --ref algorithm, --test auto_seg):
  python ml_rejection/make_fig5.py --root rejection_comparison_cima --ref ariel_seg --test auto_seg \
      --ref-label "Reader 2" --test-label "Automated"

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
REF = "algorithm"   # the configuration whose saved crops/rejection/contours are reused


def studies(root):
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d, REF, "CarDpy_Output")))


def run_seg(root, raw, out_name="auto_seg", auto=True):
    """Re-run the REF configuration with identical crops and rejection decisions.
    auto=True: automated segmentation (no GUIs). auto=False: opens the contour GUI for a new reader."""
    from cardpy.pipeline import process

    for name in studies(root):
        src = os.path.join(root, name, REF, "CarDpy_Output")
        settings = json.load(open(os.path.join(src, "processing_settings.json")))
        cfg = settings.get("cardpy", settings)
        out = os.path.abspath(os.path.join(root, name, out_name, "CarDpy_Output"))
        if os.path.exists(os.path.join(root, name, f"{out_name}.npz")):
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
        if auto:
            cfg.setdefault("segmentation", {}).update(mode="auto", review=False, use_model_mask=True)
        else:
            cfg.setdefault("segmentation", {}).update(mode="manual")

        print(f"\n===== {name}: {'auto segmentation' if auto else 'manual segmentation (' + out_name + ')'} =====")
        res = process(os.path.join(raw, name), cfg)
        flat = {f"{g}__{k}": np.asarray(v, dtype=float)
                for g in ("dti_metrics", "cardiac_metrics") for k, v in (res.get(g) or {}).items()}
        flat["mask"] = np.asarray(res["mask"], dtype=float)
        flat["background"] = np.nanmean(np.real(np.asarray(res["matrix"])), axis=(3, 4))
        np.savez_compressed(os.path.join(root, name, f"{out_name}.npz"), **flat)
        print(f"{name}: sources = {res.get('segmentation_sources')}")


def hap(path):
    if not os.path.exists(path):
        return None
    from scipy.io import loadmat
    return np.asarray(loadmat(path).get("HAP", np.nan), dtype=float).ravel()


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan


REF_C, TEST_C, BOTH_C = np.array([0.89, 0.10, 0.11]), np.array([0.12, 0.30, 0.95]), np.array([0.50, 0.12, 0.60])
UNITS = {"MD": "µm²/ms", "FA": "", "HAP": "°/%"}
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11})
plt.rcParams["svg.fonttype"] = "none"  # Keep SVG text editable in Illustrator


def pair_table(root, ref, test):
    """Per-slice Dice + metric medians for one pair; also the example candidates."""
    rows, examples = [], []
    for name in studies(root):
        fa, fb = os.path.join(root, name, f"{ref}.npz"), os.path.join(root, name, f"{test}.npz")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        A, B = np.load(fa), np.load(fb)
        ma, mb = A["mask"] > 0, B["mask"] > 0
        if ma.ndim == 2:
            ma, mb = ma[..., None], mb[..., None]
        ha = hap(os.path.join(root, name, ref, "CarDpy_Output", "12_Quantitative_Results", "HAP.mat"))
        hb = hap(os.path.join(root, name, test, "CarDpy_Output", "12_Quantitative_Results", "HAP.mat"))
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
            bg = A["background"]
            examples.append((row["dice"], name, s, bg[..., s] if bg.ndim == 3 else bg, a, b))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No studies with both {ref}.npz and {test}.npz under {root}.")
    return df, examples


def overlay(ax, img, a, b, ref_label, test_label):
    """Masks: reference only = red, test only = blue, overlap = purple; bold contours on top."""
    ys, xs = np.where(a | b); pad = 8
    y0, y1 = max(ys.min() - pad, 0), ys.max() + pad + 1
    x0, x1 = max(xs.min() - pad, 0), xs.max() + pad + 1
    img, a, b = img[y0:y1, x0:x1], a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    ax.imshow(img, cmap="gray", vmin=0, vmax=np.nanpercentile(img, 99.5))
    rgba = np.zeros(img.shape + (4,))
    rgba[a & ~b, :3], rgba[b & ~a, :3], rgba[a & b, :3] = REF_C, TEST_C, BOTH_C
    rgba[a | b, 3] = 0.65
    ax.imshow(rgba)
    for m, c in ((a, REF_C), (b, TEST_C)):
        if m.any():
            ax.contour(m.astype(float), levels=[0.5], colors=[c], linewidths=2.2)
    ax.set_xticks([]); ax.set_yticks([])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=REF_C, label=ref_label), Patch(color=TEST_C, label=test_label),
                       Patch(color=BOTH_C, label="Overlap")],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9,
              frameon=False, handlelength=1.2, columnspacing=0.8)


def draw_row(axes, df, examples, ref_label, test_label, letter):
    metrics = [m for m in ("MD", "FA", "HAP") if f"{m}_manual" in df]
    med = df.dice.median()
    d, name, s, img, a, b = min(examples, key=lambda e: abs(e[0] - med))   # typical slice
    overlay(axes[0], img, a, b, ref_label, test_label)
    axes[0].set_title(f"({letter}) {ref_label} vs. {test_label}\nexample, Dice {d:.2f}", fontsize=13)

    ax = axes[1]                                                      # Dice per slice
    rng = np.random.default_rng(0)
    ax.boxplot(df.dice, widths=0.5, showfliers=False, medianprops=dict(color="black", lw=2))
    ax.scatter(1 + rng.uniform(-0.12, 0.12, len(df)), df.dice, s=30, color=BOTH_C, zorder=3)
    ax.set_ylim(min(0.6, df.dice.min() - 0.05), 1); ax.set_xticks([])
    ax.set_title(f"Dice\nmedian {med:.2f}"); ax.grid(axis="y", alpha=0.4)

    for i, m in enumerate(metrics):                                   # agreement, one point per slice
        ax = axes[2 + i]
        x, y = df[f"{m}_manual"], df[f"{m}_auto"]
        lo, hi = np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])
        pad = 0.08 * (hi - lo or abs(hi) or 1)
        lo, hi = lo - pad, hi + pad
        from matplotlib.ticker import MaxNLocator
        ticks = [t for t in MaxNLocator(nbins=4).tick_values(lo, hi) if lo <= t <= hi]
        ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1.2, zorder=1)
        ax.scatter(x, y, s=40, color="#1f77b4", edgecolor="black", lw=0.5, zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xticks(ticks); ax.set_yticks(ticks)                    # identical ticks on both axes
        ax.grid(True, alpha=0.4); ax.set_aspect("equal", adjustable="box")
        unit = f" ({UNITS[m]})" if UNITS[m] else ""
        ax.set_xlabel(f"{ref_label}{unit}"); ax.set_ylabel(f"{test_label}{unit}")
        pct = 100 * (y - x) / x.abs()
        ax.set_title(f"{m}\nmedian difference {pct.median():+.1f}%")


def report(df, ref_label, test_label, tag):
    med = df.dice.median()
    print(f"\n{ref_label} vs {test_label}: {len(df)} slices from {df.study.nunique()} participants | "
          f"Dice median {med:.3f} (IQR {df.dice.quantile(.25):.3f}-{df.dice.quantile(.75):.3f})")
    rows = []
    for m in [m for m in ("MD", "FA", "HAP") if f"{m}_manual" in df]:
        a, b = df[f"{m}_manual"], df[f"{m}_auto"]
        pct = 100 * (b - a) / a.abs()
        rows.append({"metric": m,
                     f"{ref_label} median (IQR)": f"{a.median():.3f} ({a.quantile(.25):.3f} to {a.quantile(.75):.3f})",
                     f"{test_label} median (IQR)": f"{b.median():.3f} ({b.quantile(.25):.3f} to {b.quantile(.75):.3f})",
                     "median % difference (IQR)": f"{pct.median():+.1f}% ({pct.quantile(.25):+.1f} to {pct.quantile(.75):+.1f})",
                     "slices within 5%": f"{int((pct.abs() <= 5).sum())}/{len(pct)}"})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(FIGS, f"fig5_{tag}_summary.csv"), index=False)
    df.to_csv(os.path.join(FIGS, f"fig5_{tag}.csv"), index=False)
    print(f"% difference = ({test_label} - {ref_label}) / |{ref_label}|")
    print(summary.to_string(index=False))


def make_figure(root, pairs, stem):
    """pairs: list of (ref, test, ref_label, test_label); one row per pair, in order."""
    os.makedirs(FIGS, exist_ok=True)
    tables = []
    for ref, test, rl, tl in pairs:
        df, ex = pair_table(root, ref, test)
        report(df, rl, tl, f"{os.path.basename(os.path.normpath(root))}_{ref}_vs_{test}")
        tables.append((df, ex, rl, tl))
    ncols = 2 + max(len([m for m in ("MD", "FA", "HAP") if f"{m}_manual" in t[0]]) for t in tables)
    fig = plt.figure(figsize=(4.2 * ncols, 4.6 * len(tables)))
    gs = fig.add_gridspec(len(tables), ncols, width_ratios=[1.25, 0.6] + [1] * (ncols - 2),
                          wspace=0.45, hspace=0.75)
    for r, (df, ex, rl, tl) in enumerate(tables):
        axes = [fig.add_subplot(gs[r, c]) for c in range(ncols)]
        draw_row(axes, df, ex, rl, tl, "ABCDEFGH"[r])
        for ax in axes[2 + len([m for m in ("MD", "FA", "HAP") if f"{m}_manual" in df]):]:
            ax.set_visible(False)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGS, f"{stem}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\nwrote", os.path.join(FIGS, f"{stem}.png"), "and .svg")


def compare(root, ref="algorithm", test="auto_seg", ref_label="Manual", test_label="Automated"):
    tag = f"{os.path.basename(os.path.normpath(root))}_{ref}_vs_{test}"
    make_figure(root, [(ref, test, ref_label, test_label)], f"fig5_{tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="run auto segmentation first")
    ap.add_argument("--root", default="rejection_comparison_purl")
    ap.add_argument("--raw", default="PURL_test_raw", help="folder containing the raw <study> folders")
    ap.add_argument("--manual-run", metavar="NAME", help="draw a second reader's contours, saved as NAME")
    ap.add_argument("--ref", default="algorithm"); ap.add_argument("--test", default="auto_seg")
    ap.add_argument("--ref-label", default="Manual"); ap.add_argument("--test-label", default="Automated")
    ap.add_argument("--three-readers", action="store_true",
                    help="one figure: (A) Reader 1 vs nnUNet-Segment, (B) Reader 2 vs nnUNet-Segment, (C) Reader 1 vs Reader 2")
    ap.add_argument("--reader1", default="ariel_seg"); ap.add_argument("--reader2", default="algorithm")
    ap.add_argument("--auto", default="auto_seg"); ap.add_argument("--auto-label", default="nnUNet-Segment")
    args = ap.parse_args()
    if args.three_readers:
        make_figure(args.root, [(args.reader1, args.auto, "Reader 1", args.auto_label),
                                (args.reader2, args.auto, "Reader 2", args.auto_label),
                                (args.reader1, args.reader2, "Reader 1", "Reader 2")],
                    f"fig5_{os.path.basename(os.path.normpath(args.root))}_three_readers")
        raise SystemExit
    if args.manual_run:
        run_seg(args.root, args.raw, out_name=args.manual_run, auto=False)
        raise SystemExit(f"Saved {args.manual_run}.npz for each study. Compare with --ref/--test.")
    if args.run:
        run_seg(args.root, args.raw, out_name="auto_seg", auto=True)
    compare(args.root, args.ref, args.test, args.ref_label, args.test_label)