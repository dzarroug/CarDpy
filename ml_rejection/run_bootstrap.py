import argparse
import glob
import os
import time

from cardpy.uncertainty import run_dataset

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--iterations", type=int, default=400)
ap.add_argument("--n_jobs", type=int, default=8)
args = ap.parse_args()

runs = sorted(glob.glob(os.path.join(args.root, "*", "*", "CarDpy_Output")))
runs = [r for r in runs if "/auto_seg/" not in r]
print(f"{len(runs)} run folders found")
for i, run in enumerate(runs, 1):
    done = os.path.join(run, "14_Uncertainty", "uncertainty.npz")
    if os.path.exists(done):
        print(f"[{i}/{len(runs)}] {run}: already done"); continue
    t0 = time.time()
    print(f"[{i}/{len(runs)}] {run}")
    try:
        run_dataset(run, iterations=args.iterations, n_jobs=args.n_jobs, seed=0)
        print(f"    finished in {(time.time() - t0) / 60:.1f} min")
    except Exception as e:
        print(f"    FAILED: {e}")
