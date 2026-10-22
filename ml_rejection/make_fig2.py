"""
make_fig2.py -- Figure 2 (a) and (b) for the abstract, from files you already have.

  python ml_rejection/make_fig2.py                        # candidates sheet + figures with auto-picked examples
  python ml_rejection/make_fig2.py --pick 123 456 789 101 # use your chosen examples (TP FP FN TN, by #id)

(a) confusion matrix + precision-recall curve on the TEST set
(b) one example per category: both reject / ML only / algorithm only / both keep,
    each shown as the image the model saw next to its reference (median of the same slice & direction)

Reads  ml_rejection/models/crop80/test_predictions.csv  (written by train.py --test)
       ml_rejection/models/crop80/best.pt               (threshold, crop)
       ml_rejection/cache/X.npy + meta.csv
Writes ml_rejection/figures/fig2a.png, fig2b.png, fig2b_candidates.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
FIGS = os.path.join(HERE, "figures")

CATS = [("TP", "Both reject"), ("FP", "ML only rejects"),
        ("FN", "Algorithm only rejects"), ("TN", "Both keep")]


def load(run):
    meta = pd.read_csv(os.path.join(CACHE, "meta.csv"))
    preds = pd.read_csv(os.path.join(HERE, "models", run, "test_predictions.csv"))
    test_idx = np.flatnonzero(meta["split"].to_numpy() == "test")
    if len(test_idx) != len(preds):
        raise SystemExit("test_predictions.csv does not match the cache -- re-run train.py --test")
    check = ["participant", "slice", "volume"]
    if not (meta.iloc[test_idx][check].reset_index(drop=True) == preds[check]).all().all():
        raise SystemExit("row order mismatch between cache and test_predictions.csv")
    preds["cache_idx"] = test_idx
    crop = None
    try:
        import torch
        ck = torch.load(os.path.join(HERE, "models", run, "best.pt"), map_location="cpu", weights_only=False)
        crop, thr = ck["args"].get("crop"), float(ck["threshold"])
    except Exception:
        thr = float(preds.loc[preds.pred == 1, "prob"].min())   # fallback: smallest flagged prob
        crop = int(run[4:]) if run.startswith("crop") and run[4:].isdigit() else None
    y, p, pred = preds["y"].to_numpy(), preds["prob"].to_numpy(), preds["pred"].to_numpy()
    preds["cat"] = np.select([(y == 1) & (pred == 1), (y == 0) & (pred == 1),
                              (y == 1) & (pred == 0)], ["TP", "FP", "FN"], "TN")
    X = np.load(os.path.join(CACHE, "X.npy"), mmap_mode="r")
    return preds, X, thr, crop


def fig_a(preds, thr):
    y, p = preds["y"].to_numpy(), preds["prob"].to_numpy()
    counts = preds["cat"].value_counts()
    tp, fp, fn, tn = (int(counts.get(k, 0)) for k in ("TP", "FP", "FN", "TN"))
    ap = average_precision_score(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    op_p, op_r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    cm = np.array([[tp, fn], [fp, tn]])
    ax1.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        frac = v / cm[i].sum()
        ax1.text(j, i, f"{v}\n({100 * frac:.0f}%)", ha="center", va="center", fontsize=12,
                 color="white" if frac > 0.6 else "black")
    ax1.set_xticks([0, 1], ["ML reject", "ML keep"]); ax1.set_yticks([0, 1], ["Algorithm\nreject", "Algorithm\nkeep"])
    ax1.set_title("(a) Test set agreement", loc="left", fontsize=11)

    ax2.plot(rec, prec, lw=2, label=f"CNN (AP = {ap:.2f})")
    ax2.axhline(y.mean(), ls="--", c="gray", label=f"Chance (AP = {y.mean():.2f})")
    ax2.plot(op_r, op_p, "o", ms=8, c="crimson", label=f"Operating point (threshold {thr:.2f})")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
    ax2.legend(loc="best", fontsize=9, frameon=False)
    ax2.set_title("Precision-recall", loc="left", fontsize=11)
    n_part = preds["participant"].nunique()
    fig.suptitle(f"{n_part} held-out participants, {len(y):,} images, {int(y.sum())} rejected by the algorithm",
                 fontsize=10, y=0.02, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = os.path.join(FIGS, "fig2a.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    return out, dict(tp=tp, fp=fp, fn=fn, tn=tn, ap=ap)


def crop_img(img, crop):
    if not crop:
        return img
    c0 = (img.shape[0] - crop) // 2
    return img[c0:c0 + crop, c0:c0 + crop]


def show_pair(ax_img, ax_ref, X, row, crop):
    x = np.asarray(X[int(row.cache_idx)], dtype=np.float32)
    img, ref = crop_img(x[0], crop), crop_img(x[1], crop)
    vmax = max(float(np.percentile(ref, 99.5)), 1e-6)
    for ax, a in ((ax_img, img), (ax_ref, ref)):
        ax.imshow(a.T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])


def candidates(preds, X, crop, n=6, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(4, n, figsize=(2 * n, 8.6))
    for r, (cat, name) in enumerate(CATS):
        pool = preds[preds.cat == cat]
        rows = pool.iloc[rng.choice(len(pool), min(n, len(pool)), replace=False)] if len(pool) else pool
        for c in range(n):
            ax = axes[r, c]; ax.set_xticks([]); ax.set_yticks([])
            if c >= len(rows):
                ax.axis("off"); continue
            row = rows.iloc[c]
            x = np.asarray(X[int(row.cache_idx)], dtype=np.float32)
            img = crop_img(x[0], crop)
            ax.imshow(img.T, cmap="gray", origin="lower", vmin=0,
                      vmax=max(float(np.percentile(crop_img(x[1], crop), 99.5)), 1e-6))
            ax.set_title(f"#{int(row.cache_idx)}  p={row.prob:.2f}", fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.suptitle("Candidates for Figure 2b (image the model saw; pick one #id per row)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig2b_candidates.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


def auto_pick(preds):
    """Clear examples: confident agreements, confident disagreements."""
    pick = {}
    for cat in ("TP", "FP", "FN", "TN"):
        pool = preds[preds.cat == cat]
        if pool.empty:
            continue
        pick[cat] = int(pool.sort_values("prob", ascending=cat in ("FN", "TN")).iloc[0].cache_idx)
    return pick


def fig_b(preds, X, crop, pick):
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.4))
    by_idx = preds.set_index("cache_idx")
    for c, (cat, name) in enumerate(CATS):
        if cat not in pick:
            axes[0, c].axis("off"); axes[1, c].axis("off"); continue
        row = by_idx.loc[pick[cat]]
        row = row.copy(); row["cache_idx"] = pick[cat]
        show_pair(axes[0, c], axes[1, c], X, row, crop)
        b = "b=0" if row.bval < 1 else f"b={row.bval:g}"
        axes[0, c].set_title(f"{name}\nML p = {row.prob:.2f}  ({b})", fontsize=10)
    axes[0, 0].set_ylabel("Image", fontsize=10)
    axes[1, 0].set_ylabel("Reference\n(median of averages)", fontsize=10)
    fig.suptitle("(b) Example decisions", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig2b.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="crop80", help="model folder under models/")
    ap.add_argument("--pick", type=int, nargs=4, metavar=("TP", "FP", "FN", "TN"),
                    help="cache #ids for: both reject, ML only, algorithm only, both keep")
    ap.add_argument("--seed", type=int, default=0, help="reshuffle the candidates sheet")
    args = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)

    preds, X, thr, crop = load(args.run)
    a_path, s = fig_a(preds, thr)
    print(f"(a) {a_path} | TP {s['tp']} FP {s['fp']} FN {s['fn']} TN {s['tn']} | AP {s['ap']:.3f}")
    print(f"candidates: {candidates(preds, X, crop, seed=args.seed)}")
    pick = dict(zip(("TP", "FP", "FN", "TN"), args.pick)) if args.pick else auto_pick(preds)
    bad = [k for k, v in pick.items() if v not in set(preds.cache_idx) or preds.set_index("cache_idx").loc[v, "cat"] != k]
    if bad:
        raise SystemExit(f"--pick ids don't match their categories: {bad} (order is TP FP FN TN)")
    print(f"(b) {fig_b(preds, X, crop, pick)} | examples {pick}")