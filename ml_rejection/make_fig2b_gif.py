"""
make_fig2b_gif.py -- Figure 2b: animated GIF stepping through slices of one TEST dataset.

Each frame = one slice. Grid: rows = b = 0 and ONE diffusion direction, columns = averages.
Each tile = the image the model saw (01_Unrung, 80x80 around the heart).
  solid green border  = ML accepted
  dashed red border   = ML rejected
  inner yellow border = the algorithm made the opposite decision
  gray tile           = excluded from evaluation (ambiguous label)

  python ml_rejection/make_fig2b_gif.py                 # auto-picks the test dataset with the most rejections
  python ml_rejection/make_fig2b_gif.py --list          # list test datasets with their counts
  python ml_rejection/make_fig2b_gif.py --dataset "<name from --list>"
  python ml_rejection/make_fig2b_gif.py --direction 7   # choose the direction (default: most rejections)

Writes ml_rejection/figures/fig2b.gif (+ fig2b_frames/slice_XX.png)
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
CACHE = os.path.join(HERE, "cache")
FIGS = os.path.join(HERE, "figures")
GREEN, RED, MARK = "#2ca02c", "#d62728", "#ffd400"


def load(run):
    meta = pd.read_csv(os.path.join(CACHE, "meta.csv"))
    preds = pd.read_csv(os.path.join(HERE, "models", run, "test_predictions.csv"))
    test_idx = np.flatnonzero(meta["split"].to_numpy() == "test")
    if len(test_idx) != len(preds):
        raise SystemExit("test_predictions.csv does not match the cache -- re-run train.py --test")
    preds["cache_idx"] = test_idx
    X = np.load(os.path.join(CACHE, "X.npy"), mmap_mode="r")
    crop = int(run[4:]) if run.startswith("crop") and run[4:].isdigit() else None
    return preds, X, crop


def summary(preds):
    g = preds.groupby("dataset")
    return pd.DataFrame({"algorithm_rejected": g["y"].sum(), "ml_rejected": g["pred"].sum(),
                         "disagree": g.apply(lambda d: int((d["y"] != d["pred"]).sum()))}
                        ).sort_values(["algorithm_rejected", "ml_rejected"], ascending=False)


def frame(ds_rows, X, crop, s, encodings, n_avg, enc_labels, title, dpi=120):
    tile = 1.35
    fig = plt.figure(figsize=(n_avg * tile + 1.2, len(encodings) * tile + 1.3), dpi=dpi)
    gs = fig.add_gridspec(len(encodings), n_avg, left=0.13, right=0.99, top=0.84, bottom=0.14,
                          wspace=0.06, hspace=0.06)
    rows = ds_rows[ds_rows.slice == s].set_index(["direction", "average"])
    for e_i, d in enumerate(encodings):
        for a in range(n_avg):
            ax = fig.add_subplot(gs[e_i, a]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if e_i == 0:
                ax.set_title(f"Avg {a + 1}", fontsize=8, pad=3)
            if a == 0:
                ax.set_ylabel(enc_labels[d], fontsize=8)
            if (d, a) not in rows.index:
                ax.imshow(np.full((2, 2), 0.5), cmap="gray", vmin=0, vmax=1)
                continue
            r = rows.loc[(d, a)]
            img = np.asarray(X[int(r.cache_idx), 0], dtype=np.float32)
            if crop:
                c0 = (img.shape[0] - crop) // 2
                img = img[c0:c0 + crop, c0:c0 + crop]
            ax.imshow(img.T, cmap="gray", origin="lower", vmin=0, vmax=1.0)
            rej = int(r.pred) == 1
            h, w = img.shape
            ax.add_patch(Rectangle((-0.5, -0.5), w, h, fill=False, lw=3 if rej else 2.2,
                                   ls=(0, (3, 2)) if rej else "-", ec=RED if rej else GREEN, clip_on=False))
            if int(r.pred) != int(r.y):   # inner yellow halo: algorithm disagrees
                inset = 0.07 * min(h, w)
                ax.add_patch(Rectangle((-0.5 + inset, -0.5 + inset), w - 2 * inset, h - 2 * inset,
                                       fill=False, lw=2.4, ls="-", ec=MARK, clip_on=False))
    fig.suptitle(title, fontsize=9, y=0.97)
    handles = [Line2D([], [], color=GREEN, lw=2, label="ML accepted"),
               Line2D([], [], color=RED, lw=2, ls=(0, (3, 2)), label="ML rejected"),
               Line2D([], [], color=MARK, lw=2, label="Algorithm disagrees (inner border)"),
               Line2D([], [], marker="s", ls="", mfc="0.5", mec="0.5", label="Excluded (ambiguous label)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.canvas.draw()
    im = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return im


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="crop80")
    ap.add_argument("--dataset", help="test dataset name (see --list)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--direction", type=int, help="diffusion direction to show next to b = 0 (default: most rejections)")
    ap.add_argument("--seconds", type=float, default=2.0, help="seconds per slice")
    ap.add_argument("--dpi", type=int, default=120, help="lower = smaller GIF file")
    args = ap.parse_args()

    preds, X, crop = load(args.run)
    table = summary(preds)
    if args.list:
        print(table.to_string()); raise SystemExit
    ds = args.dataset or table.index[0]
    if ds not in table.index:
        raise SystemExit(f"{ds!r} is not a test dataset; run with --list")
    rows = preds[preds.dataset == ds]
    n_avg = int(rows.average.max()) + 1
    slices = sorted(rows.slice.unique())
    bvals = rows.groupby("direction")["bval"].first()
    b0_dirs = [d for d, b in bvals.items() if b < 1]
    dw = rows[~rows.direction.isin(b0_dirs)]
    if args.direction is None:   # direction with the most rejections (ML or algorithm), then disagreements
        score = dw.groupby("direction").apply(lambda g: (g.pred | g.y).sum() * 100 + (g.pred != g.y).sum())
        args.direction = int(score.idxmax())
    encodings = [b0_dirs[0] if b0_dirs else 0, args.direction]
    enc_labels = {d: ("b = 0" if bvals.get(d, 1) < 1 else f"Dir {d}\n(b = {bvals[d]:g})") for d in encodings}

    out_dir = os.path.join(FIGS, "fig2b_frames")
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for s in slices:
        sr = rows[(rows.slice == s) & rows.direction.isin(encodings)]
        title = (f"Slice {s + 1} of {len(slices)}\nML rejected {int(sr.pred.sum())}  |  "
                 f"algorithm rejected {int(sr.y.sum())}  |  disagreements {int((sr.pred != sr.y).sum())}")
        im = frame(rows, X, crop, s, encodings, n_avg, enc_labels, title, dpi=args.dpi)
        im.save(os.path.join(out_dir, f"slice_{s + 1:02d}.png"))
        frames.append(im)
    gif = os.path.join(FIGS, "fig2b.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=int(args.seconds * 1000), loop=0)
    print(f"dataset: {ds} | encodings shown: b = 0 and direction {args.direction}")
    print(table.loc[ds].to_string())
    print(f"wrote {gif} ({len(frames)} frames, {os.path.getsize(gif) / 1e6:.1f} MB) and {out_dir}/")