"""
extract_rejection_labels.py
Recover shot-rejection ground truth from CarDpy outputs (PURL yh756nh8583).

shot_rejection replaces a rejected image with a COPY of a randomly chosen good
average from the same slice/direction, so rejected slots show up as exact
duplicates within 02_Rejected. See label_dataset for how the source vs the
rejected copy is decided. Groups with a small correlation margin between the
top two members are flagged 'ambiguous' for manual review.

Comparison is done in stacked space (4D [rows, cols, slices, vols]), so no
dependency on stacked2sorted. direction/average are derived from bvals/bvecs
(order of first appearance) -- VERIFY this ordering matches
cardpy.Data_Sorting.stacked2sorted before using indices for integration.

Usage:
    python extract_rejection_labels.py /path/to/PURL_root -o labels.csv
"""
import argparse
import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd


def _one(folder, pattern):
    hits = sorted(glob.glob(os.path.join(folder, pattern)))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected 1 match for {pattern} in {folder}, got {len(hits)}")
    return hits[0]


def load_folder(folder):
    img = np.asarray(nib.load(_one(folder, "*.nii*")).dataobj, dtype=np.float64)
    bvals = np.atleast_1d(np.loadtxt(_one(folder, "*.bval*")))
    bvecs = np.atleast_2d(np.loadtxt(_one(folder, "*.bvec*")))
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:   # FSL style 3 x N
        bvecs = bvecs.T
    if img.ndim != 4:
        raise ValueError(f"expected 4D stacked image, got {img.shape} in {folder}")
    if not (img.shape[3] == len(bvals) == len(bvecs)):
        raise ValueError(f"volume/bval/bvec count mismatch in {folder}: "
                         f"{img.shape[3]}, {len(bvals)}, {len(bvecs)}")
    return img, bvals, bvecs


def direction_average_index(bvals, bvecs, decimals=4):
    """Map each stacked volume to (direction, average) by unique (bval, bvec)."""
    keys, direction, average, counts = {}, [], [], {}
    for b, g in zip(np.round(bvals, 1), np.round(bvecs, decimals)):
        k = (b, *g)
        if k not in keys:
            keys[k] = len(keys)
        d = keys[k]
        direction.append(d)
        average.append(counts.get(d, 0))
        counts[d] = counts.get(d, 0) + 1
    return np.array(direction), np.array(average)


def _corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def label_dataset(unrung_dir, rejected_dir, ambiguous_margin=0.02):
    """
    Exact-duplicate method. shot_rejection fills a rejected slot with a copy of
    a good average, so in 02_Rejected the copy and its source are identical.
    Within each duplicate group, the source is the member whose 02_Rejected
    image best matches its OWN 01_Unrung image; all other members are rejected.
    (01 -> 02 values are not identical even for kept images -- likely
    registration -- so the duplicate test is done within 02_Rejected.)
    """
    U, bvals, bvecs = load_folder(unrung_dir)
    R, bvals_r, _ = load_folder(rejected_dir)
    if U.shape != R.shape:
        raise ValueError(f"shape mismatch: Unrung {U.shape} vs Rejected {R.shape}")
    if not np.allclose(bvals, bvals_r):
        raise ValueError("bvals differ between 01_Unrung and 02_Rejected")

    direction, average = direction_average_index(bvals, bvecs)
    n_slices, n_vols = U.shape[2], U.shape[3]
    rows = []
    for s in range(n_slices):
        groups = {}
        for v in range(n_vols):
            groups.setdefault(R[:, :, s, v].tobytes(), []).append(v)
        info = {v: dict(label="kept", source_volume=v, margin=np.nan, group_size=1)
                for v in range(n_vols)}
        for members in groups.values():
            if len(members) < 2:
                continue
            scores = {v: _corr(R[:, :, s, v], U[:, :, s, v]) for v in members}
            ranked = sorted(members, key=scores.get, reverse=True)
            src = ranked[0]
            margin = scores[ranked[0]] - scores[ranked[1]]
            for v in members:
                info[v] = dict(label="kept" if v == src else "rejected",
                               source_volume=src, margin=margin,
                               group_size=len(members))
        for v in range(n_vols):
            d = info[v]
            src = d["source_volume"]
            rows.append(dict(
                slice=s, volume=v, direction=int(direction[v]), average=int(average[v]),
                bval=float(bvals[v]), label=d["label"], source_volume=src,
                source_average=int(average[src]),
                same_direction=bool(direction[src] == direction[v]),
                group_size=d["group_size"], margin=d["margin"],
                ambiguous=bool(d["group_size"] > 1 and d["margin"] < ambiguous_margin),
            ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder containing the downloaded datasets")
    ap.add_argument("-o", "--out", default="rejection_labels.csv")
    ap.add_argument("--ambiguous-margin", type=float, default=0.02)
    args = ap.parse_args()

    unrung_dirs = sorted(glob.glob(os.path.join(args.root, "**", "01_Unrung"), recursive=True))
    print(f"Found {len(unrung_dirs)} datasets with 01_Unrung")

    frames, failures = [], []
    for ud in unrung_dirs:
        parent = os.path.dirname(ud)
        name = os.path.relpath(parent, args.root)
        rd = os.path.join(parent, "02_Rejected")
        try:
            df = label_dataset(ud, rd, args.ambiguous_margin)
            df.insert(0, "dataset", name)
            frames.append(df)
            c = df["label"].value_counts()
            print(f"  {name}: {len(df)} imgs | rejected {c.get('rejected', 0)} "
                  f"| ambiguous {int(df['ambiguous'].sum())}")
        except Exception as e:
            failures.append((name, str(e)))
            print(f"  {name}: FAILED -- {e}")

    if not frames:
        print("No datasets labelled.")
        return
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(args.out, index=False)

    n = len(all_df)
    c = all_df["label"].value_counts()
    rej = all_df[all_df["label"] == "rejected"]
    print("\n=== SUMMARY ===")
    print(f"datasets labelled : {len(frames)}  (failed: {len(failures)})")
    print(f"total images      : {n}")
    print(f"rejected          : {c.get('rejected', 0)}  ({100 * c.get('rejected', 0) / n:.2f}%)")
    print(f"ambiguous groups  : {int(all_df['ambiguous'].sum())} rows flagged (check by hand)")
    print(f"cross-direction duplicates (should be 0): {int((~all_df['same_direction']).sum())}")
    print(f"b=0 rejected      : {(rej['bval'] < 1).sum()}")
    print(f"replaced from same direction: {rej['same_direction'].sum()} / {len(rej)}")
    per_ds = all_df.groupby("dataset")["label"].apply(lambda x: (x == "rejected").sum())
    print(f"datasets with >=1 rejection: {(per_ds > 0).sum()} / {len(per_ds)}")
    if c.get("rejected", 0):
        print(f"suggested pos_weight (neg/pos): {(n - c['rejected']) / c['rejected']:.1f}")
    for name, err in failures:
        print(f"FAILED {name}: {err}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()