"""
make_manual_configs.py -- turn Ariel's manual decisions (Shot_Decisions.csv) into the
Rejection_Configuration.json CarDpy reuses, for the 'manual' comparison configuration.

For each study in the comparison folder it starts from YOUR algorithm run's
13_Diagnostics/Rejection_Configuration.json (same image shape, b-values/vectors, crop),
replaces the keep/reject matrix with Ariel's decisions for the analysed slices, and
writes <root>/<study>/manual_Rejection_Configuration.json.

  python ml_rejection/make_manual_configs.py
"""
import json
import os

import numpy as np
import pandas as pd

MANUAL_ROOT = "/Users/dahniazarroug/Desktop/Ariel's Rejections For Training/02_Manual_Shot_Rejection"
ROOT = "rejection_comparison_purl"
SLICES = [0]          # must match SLICES used for the comparison runs (0-based source slices)
BLIP = "Blip_Up"

for study in sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))):
    alg = os.path.join(ROOT, study, "algorithm", "CarDpy_Output", "13_Diagnostics", "Rejection_Configuration.json")
    csv = os.path.join(MANUAL_ROOT, study, BLIP, "Shot_Decisions.csv")
    if not os.path.exists(alg):
        print(f"{study}: no algorithm Rejection_Configuration.json -- skipped"); continue
    if not os.path.exists(csv):
        print(f"{study}: no Shot_Decisions.csv in {MANUAL_ROOT} -- skipped"); continue

    payload = json.load(open(alg))
    rows, cols, n_slc, n_vol = payload["image_shape"]
    if n_slc != len(SLICES):
        raise SystemExit(f"{study}: run has {n_slc} slices but SLICES has {len(SLICES)}")
    bvals = np.asarray(payload["bvals"], dtype=float)

    d = pd.read_csv(csv)
    keep = np.ones((n_slc, n_vol), dtype=int)
    covered = np.zeros(n_slc, dtype=bool)
    for pos, s in enumerate(SLICES):
        sub = d[d["source_slice_index_0based"] == s]
        covered[pos] = len(sub) > 0
        for _, r in sub.iterrows():
            v = int(r["volume_index_0based"])
            if not np.isclose(float(r["b_value"]), bvals[v], atol=1):   # indexing sanity check
                raise SystemExit(f"{study}: b-value mismatch at slice {s}, volume {v} "
                                 f"({r['b_value']} vs {bvals[v]})")
            if str(r["decision"]).strip().lower() == "reject":
                keep[pos, v] = 0
    if not covered.all():
        missing = [SLICES[i] for i in np.flatnonzero(~covered)]
        raise SystemExit(f"{study}: Ariel did not review slice(s) {missing}")

    payload["automatic_keep"] = keep.tolist()
    payload["final_keep"] = keep.tolist()
    out = os.path.join(ROOT, study, "manual_Rejection_Configuration.json")
    json.dump(payload, open(out, "w"), indent=2)
    per_slice = (keep == 0).sum(axis=1).tolist()
    print(f"{study}: manual rejected {int((keep == 0).sum())} of {keep.size} (per slice {per_slice}) -> {out}")
