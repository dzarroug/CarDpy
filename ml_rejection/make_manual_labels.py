"""
make_manual_labels.py
Convert Ariel's manual shot decisions (02_Manual_Shot_Rejection/*/Blip_Up/Shot_Decisions.csv)
into the rejection_labels.csv format build_cache.py expects.

Usage:
    python ml_rejection/make_manual_labels.py <02_Manual_Shot_Rejection> PURL_data rejection_labels_manual.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

N_ENC = 16  # b=0 + 15 directions; volume = average * 16 + direction

man_root, purl_root, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows, skipped = [], []
for f in sorted(glob.glob(os.path.join(man_root, "Healthy_Volunteer_*", "Blip_Up", "Shot_Decisions.csv"))):
    p = f.split(os.sep)[-3]
    ds = f"{p}_Processing/{p}/Blip_Up"
    unrung = os.path.join(purl_root, ds, "01_Unrung")
    if not glob.glob(os.path.join(unrung, "*.nii*")):
        skipped.append(p); continue
    d = pd.read_csv(f)

    # sanity check: her b-values must match the Unrung bvals volume-for-volume
    bv = np.loadtxt(glob.glob(os.path.join(unrung, "*.bval*"))[0]).ravel()
    if not np.allclose(bv[d.volume_index_0based.values], d.b_value.values, atol=1):
        sys.exit(f"b-value mismatch in {p} -- volume indexing differs, stop")

    v = d.volume_index_0based.values
    rows.append(pd.DataFrame({
        "dataset": ds,
        "slice": d.source_slice_index_0based.values,
        "volume": v,
        "direction": v % N_ENC,
        "average": v // N_ENC,
        "bval": d.b_value.values,
        "ambiguous": False,
        "label": np.where(d.decision.str.strip().str.lower() == "reject", "rejected", "kept"),
    }))

df = pd.concat(rows, ignore_index=True)
df.to_csv(out, index=False)
n = df.dataset.nunique()
print(f"wrote {out}: {len(df)} images, {(df.label == 'rejected').sum()} rejected "
      f"({100 * (df.label == 'rejected').mean():.1f}%), {n} datasets")
if skipped:
    print(f"SKIPPED (no 01_Unrung on disk): {skipped}")