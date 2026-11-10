"""Figure 2b: animated shot-rejection decisions across slices of a TEST dataset.

From the CarDpy repository root:
    python ml_rejection/make_fig2b_gif.py
    python ml_rejection/make_fig2b_gif.py --list
    python ml_rejection/make_fig2b_gif.py --dataset "DATASET_NAME"
    python ml_rejection/make_fig2b_gif.py --direction 7

Produces fig2b.gif and per-slice PNG / Illustrator-editable SVG files.
Incomplete slices are skipped; no decisions or images are fabricated.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache_algorithm")
FIGS = os.path.join(HERE, "figures")
GREEN, RED, MARK = "#2ca02c", "#d62728", "#ffd400"

plt.rcParams.update({
    "svg.fonttype": "none",  # preserve text for Illustrator
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.linewidth": 1.8,
    "lines.linewidth": 3.0,
    "savefig.facecolor": "white",
})


def load(run):
    meta = pd.read_csv(os.path.join(CACHE, "meta.csv"))
    preds = pd.read_csv(os.path.join(HERE, "models", run, "test_predictions.csv"))
    test_idx = np.flatnonzero(meta["split"].to_numpy() == "test")
    if len(test_idx) != len(preds):
        raise SystemExit(f"Test rows mismatch: {len(test_idx)} in {CACHE}, {len(preds)} predictions")
    cols = ["participant", "slice", "volume"]
    if not meta.iloc[test_idx][cols].reset_index(drop=True).astype(str).equals(
        preds[cols].reset_index(drop=True).astype(str)
    ):
        raise SystemExit("Test row identities differ; verify the cache and prediction run.")
    preds["cache_idx"] = test_idx
    X = np.load(os.path.join(CACHE, "X.npy"), mmap_mode="r")
    crop = int(run[4:]) if run.startswith("crop") and run[4:].isdigit() else None
    return preds, X, crop


def summary(preds):
    return (preds.groupby("dataset")
            .agg(algorithm_rejected=("y", "sum"), ml_rejected=("pred", "sum"))
            .assign(disagree=preds.assign(disagree=preds["y"].ne(preds["pred"]))
                    .groupby("dataset")["disagree"].sum())
            .sort_values(["algorithm_rejected", "ml_rejected"], ascending=False))


def frame(ds_rows, X, crop, s, encodings, n_avg, enc_labels, title, dpi, svg_path):
    tile = 1.65
    fig = plt.figure(figsize=(n_avg * tile + 2.5, len(encodings) * tile + 1.45), dpi=dpi,
                     facecolor="white")
    # One GridSpec for ALL tiles. Compact lower margin positions legend near images.
    gs = fig.add_gridspec(
        len(encodings), n_avg,
        left=0.16,
        right=0.92,
        top=0.76,
        bottom=0.21,
        wspace=0.06,
        hspace=0.08
    )
    rows = ds_rows.loc[ds_rows["slice"].eq(s)].set_index(["direction", "average"])

    for e_i, d in enumerate(encodings):
        for a in range(n_avg):
            ax = fig.add_subplot(gs[e_i, a])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if e_i == 0:
                ax.set_title(f"Avg {a + 1}", fontsize=17, fontweight="bold", pad=7)
            if a == 0:
                ax.set_ylabel(enc_labels[d], fontsize=14, fontweight="semibold",
                              rotation=0, ha="right", va="center", labelpad=12)

            r = rows.loc[(d, a)]
            img = np.asarray(X[int(r.cache_idx), 0], dtype=np.float32)
            if crop:
                h0, w0 = img.shape
                top, left = (h0 - crop) // 2, (w0 - crop) // 2
                img = img[top:top + crop, left:left + crop]
            ax.imshow(img.T, cmap="gray", origin="lower", vmin=0, vmax=1.0,
                      interpolation="nearest")
            rejected = int(r.pred) == 1
            h, w = img.shape
            ax.add_patch(Rectangle((-0.5, -0.5), w, h, fill=False,
                                   lw=5 if rejected else 4,
                                   ls=(0, (3, 2)) if rejected else "-",
                                   ec=RED if rejected else GREEN, clip_on=False))
            if int(r.pred) != int(r.y):
                inset = 0.07 * min(h, w)
                ax.add_patch(Rectangle((-0.5 + inset, -0.5 + inset),
                                       w - 2 * inset, h - 2 * inset,
                                       fill=False, lw=4, ec=MARK, clip_on=False))

    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.985, linespacing=1.25)
    handles = [
        Line2D([], [], color=GREEN, lw=4, label="CNN accepted"),
        Line2D([], [], color=RED, lw=4, ls=(0, (3, 2)), label="CNN rejected"),
        Line2D([], [], color=MARK, lw=4, label="Algorithm disagrees"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035),
               ncol=3, prop={"size": 13, "weight": "semibold"},
               frameon=False, columnspacing=1.5, handlelength=2.3)

    fig.savefig(svg_path, format="svg", facecolor="white")
    fig.canvas.draw()
    image = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                             bytes(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="crop80")
    ap.add_argument("--dataset", help="Test dataset name (see --list)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--direction", type=int, help="Diffusion direction displayed beside b = 0")
    ap.add_argument("--seconds", type=float, default=2.0, help="Seconds per included slice")
    ap.add_argument("--dpi", type=int, default=180, help="Resolution of PNG/GIF frames")
    args = ap.parse_args()
    if args.seconds <= 0 or args.dpi <= 0:
        ap.error("--seconds and --dpi must be positive")

    preds, X, crop = load(args.run)
    table = summary(preds)
    if args.list:
        print(table.to_string())
        return
    ds = args.dataset or table.index[0]
    if ds not in table.index:
        raise SystemExit(f"{ds!r} is not a test dataset; run with --list")
    rows = preds.loc[preds["dataset"].eq(ds)]
    n_avg = int(rows["average"].max()) + 1
    slices = sorted(rows["slice"].unique())
    bvals = rows.groupby("direction")["bval"].first()
    b0_dirs = [d for d, b in bvals.items() if b < 1]
    dw = rows.loc[~rows["direction"].isin(b0_dirs)]
    if dw.empty:
        raise SystemExit("No diffusion-weighted directions found")
    if args.direction is None:
        scored = dw.assign(score=((dw["pred"].astype(bool) | dw["y"].astype(bool)).astype(int) * 100
                                 + dw["pred"].ne(dw["y"]).astype(int)))
        args.direction = int(scored.groupby("direction")["score"].sum().idxmax())
    if args.direction not in bvals.index or args.direction in b0_dirs:
        raise SystemExit("--direction must identify a diffusion-weighted direction in this dataset")
    encodings = [b0_dirs[0] if b0_dirs else 0, args.direction]
    enc_labels = {
        d: ("b = 0" if bvals.get(d, 1) < 1 else f"Dir {d} (b = {bvals[d]:g})")
        for d in encodings
    }

    out_dir = os.path.join(FIGS, "fig2b_frames")
    os.makedirs(out_dir, exist_ok=True)
    frames, skipped = [], []
    required = {(d, a) for d in encodings for a in range(n_avg)}
    for s in slices:
        sr = rows.loc[rows["slice"].eq(s) & rows["direction"].isin(encodings)]
        available = set(zip(sr["direction"], sr["average"]))
        if not required.issubset(available) or len(sr) != len(required):
            skipped.append(int(s) + 1)
            continue
        title = (f"Slice {int(s) + 1} of {len(slices)}\n"
                 f"CNN rejected: {int(sr.pred.sum())}   |   "
                 f"Algorithm rejected: {int(sr.y.sum())}   |   "
                 f"Disagreements: {int(sr.pred.ne(sr.y).sum())}")
        path = os.path.join(out_dir, f"slice_{int(s) + 1:02d}")
        im = frame(rows, X, crop, s, encodings, n_avg, enc_labels, title,
                   dpi=args.dpi, svg_path=path + ".svg")
        im.save(path + ".png")
        frames.append(im)

    if not frames:
        raise SystemExit("No complete slices found for this direction; try --direction N")
    gif = os.path.join(FIGS, "fig2b.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=round(args.seconds * 1000), loop=0)
    print(f"Dataset: {ds} | encodings: b=0 and direction {args.direction}")
    print(table.loc[ds].to_string())
    print(f"Skipped incomplete slices: {skipped}")
    print(f"Wrote {gif} ({len(frames)} frames, {os.path.getsize(gif) / 1e6:.1f} MB)")
    print(f"PNG and Illustrator-editable SVG frames: {out_dir}")


if __name__ == "__main__":
    main()
