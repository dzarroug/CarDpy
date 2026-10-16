"""
build_cache.py
Turn rejection_labels.csv + PURL_data into one training-ready array on disk.

Each sample is 2 channels x 128 x 128 (float16):
  ch0 = the 01_Unrung image itself
  ch1 = median of all averages for the same slice/direction (the "consensus")
Both channels are scaled by the 99th percentile of ch1, so relative intensity
(e.g. signal dropout vs. consensus) is preserved.

Ambiguous rows are dropped here. Split is by PARTICIPANT (both blips of a
participant always land in the same split) with a fixed seed.

Usage:
    python ml_rejection/build_cache.py PURL_data rejection_labels.csv
Outputs (in ml_rejection/cache/, gitignored):
    X.npy     float16 [N, 2, 128, 128]  (memory-mapped)
    meta.csv  one row per sample, aligned with X: participant, split, y, ...
"""
import argparse
import glob
import os
import re

import nibabel as nib
import numpy as np
import pandas as pd

SIZE = 128


def fit(img, size=SIZE):
    """Center-crop or zero-pad a 2D image to size x size."""
    out = np.zeros((size, size), dtype=np.float32)
    h, w = img.shape
    ys, xs = max(0, (h - size) // 2), max(0, (w - size) // 2)
    crop = img[ys:ys + size, xs:xs + size]
    oy, ox = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    out[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return out


def participant_of(dataset):
    m = re.search(r"Healthy_Volunteer_\d+", dataset)
    return m.group(0) if m else dataset.split(os.sep)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("labels")
    ap.add_argument("--out", default=os.path.join("ml_rejection", "cache"))
    ap.add_argument("--val", type=int, default=5, help="participants in validation")
    ap.add_argument("--test", type=int, default=5, help="participants in test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    full_by_ds = {k: v for k, v in df.groupby("dataset")}   # all volumes, incl. ambiguous
    n_all = len(df)
    df = df[~df["ambiguous"].astype(bool)].reset_index(drop=True)
    df["y"] = (df["label"] == "rejected").astype(np.int8)
    df["participant"] = df["dataset"].map(participant_of)

    # participant-level split
    people = sorted(df["participant"].unique())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(people)
    test = set(people[:args.test])
    val = set(people[args.test:args.test + args.val])
    df["split"] = ["test" if p in test else "val" if p in val else "train"
                   for p in df["participant"]]

    os.makedirs(args.out, exist_ok=True)
    X = np.lib.format.open_memmap(os.path.join(args.out, "X.npy"), mode="w+",
                                  dtype=np.float16, shape=(len(df), 2, SIZE, SIZE))
    shapes = set()
    for d_i, (ds, g) in enumerate(df.groupby("dataset", sort=False)):
        nii = glob.glob(os.path.join(args.root, ds, "01_Unrung", "*.nii*"))[0]
        U = np.asarray(nib.load(nii).dataobj, dtype=np.float32)
        shapes.add(U.shape[:2])
        full = full_by_ds[ds]
        for (s, d), gg in g.groupby(["slice", "direction"]):
            vols = full[(full.slice == s) & (full.direction == d)]["volume"].unique()
            med = np.median(U[:, :, s, vols], axis=2)
            scale = max(float(np.percentile(med, 99)), 1e-6)
            med_f = fit(med / scale)
            for idx, v in zip(gg.index, gg["volume"]):
                X[idx, 0] = fit(U[:, :, s, v] / scale)
                X[idx, 1] = med_f
        print(f"[{d_i + 1}] {ds}: {len(g)} samples")
    X.flush()

    df.drop(columns=["label"]).to_csv(os.path.join(args.out, "meta.csv"), index=False)

    print("\n=== CACHE ===")
    print(f"samples: {len(df)} (dropped {n_all - len(df)} ambiguous)")
    print(f"in-plane shapes seen: {sorted(shapes)}  (all fit to {SIZE}x{SIZE})")
    for sp in ["train", "val", "test"]:
        part = df[df.split == sp]
        print(f"{sp:5s}: {part.participant.nunique():2d} participants | {len(part):6d} imgs | "
              f"{int(part.y.sum()):5d} rejected ({100 * part.y.mean():.2f}%)")
    tr = df[df.split == "train"]
    print(f"pos_weight (train neg/pos): {(len(tr) - tr.y.sum()) / tr.y.sum():.1f}")
    print(f"wrote {args.out}/X.npy and meta.csv")


if __name__ == "__main__":
    main()