"""
make_review_root.py
Build a PURL-like folder whose 01_Unrung is Ariel's registered Review_Input.nii.gz,
plus a matching labels CSV, so build_cache.py runs unchanged.

Usage:
    python ml_rejection/make_review_root.py <02_Manual_Shot_Rejection> REVIEW_data rejection_labels_review.csv
"""
import glob
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd

N_ENC = 16
SKIP = {"Healthy_Volunteer_031", "Healthy_Volunteer_014"}  # excluded from the Unrung run too; keep runs comparable

man_root, out_root, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for f in sorted(glob.glob(os.path.join(man_root, "Healthy_Volunteer_*", "Blip_Up", "Shot_Decisions.csv"))):
    p = f.split(os.sep)[-3]
    if p in SKIP:
        continue
    src = os.path.join(os.path.dirname(f), "Review_Input.nii.gz")
    if not os.path.exists(src):
        print(f"skip {p}: no Review_Input"); continue
    shape = nib.load(src).shape
    d = pd.read_csv(f)

    n_rev = d.review_slice_index_0based.nunique()
    if shape[2] == n_rev:
        slc = d.review_slice_index_0based.values
    elif shape[2] > d.source_slice_index_0based.max():
        slc = d.source_slice_index_0based.values
    else:
        sys.exit(f"{p}: Review_Input has {shape[2]} slices, can't match labels")
    if shape[3] != 80:
        sys.exit(f"{p}: expected 80 volumes, got {shape[3]}")

    ds = f"{p}_Processing/{p}/Blip_Up"
    dst = os.path.join(out_root, ds, "01_Unrung")
    os.makedirs(dst, exist_ok=True)
    link = os.path.join(dst, "Unrung.nii.gz")
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(src), link)

    v = d.volume_index_0based.values
    rows.append(pd.DataFrame({
        "dataset": ds, "slice": slc, "volume": v,
        "direction": v % N_ENC, "average": v // N_ENC,
        "bval": d.b_value.values, "ambiguous": False,
        "label": np.where(d.decision.str.strip().str.lower() == "reject", "rejected", "kept"),
    }))
    print(f"{p}: shape {shape}, slice index = {'review' if shape[2] == n_rev else 'source'}")

df = pd.concat(rows, ignore_index=True)
df.to_csv(out_csv, index=False)
print(f"\nwrote {out_csv}: {len(df)} images, {(df.label == 'rejected').sum()} rejected, {df.dataset.nunique()} datasets")