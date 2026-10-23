"""
prep_test_niftis.py -- get the raw (00_Original) NIfTIs for the 5 ML TEST participants
and arrange them the way process() expects with data_type="NifTis":

    PURL_test_raw/<participant>/NifTis/data.nii / data.bvals / data.bvecs / data.header

Downloads one zip at a time (~3 GB each), keeps only <BLIP>/00_Original, deletes the zip.
  python ml_rejection/prep_test_niftis.py
"""
import glob
import os
import shutil
import urllib.parse
import urllib.request
import zipfile

import pandas as pd

DRUID = "yh756nh8583"
BLIP = "Blip_Up"
OUT = "PURL_test_raw"
META = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "meta.csv")

meta = pd.read_csv(META)
people = sorted(meta.loc[meta.split == "test", "participant"].unique())
print("test participants:", people)
os.makedirs(OUT, exist_ok=True)

for p in people:
    dest = os.path.join(OUT, p, "NifTis")
    if os.path.exists(os.path.join(dest, "data.nii")):
        print(f"{p}: already done"); continue
    zname = f"{p}_Processing.zip"
    zpath = os.path.join(OUT, zname)
    if not (os.path.exists(zpath) and zipfile.is_zipfile(zpath)):
        print(f"{p}: downloading {zname}")
        url = f"https://stacks.stanford.edu/file/{DRUID}/{urllib.parse.quote(zname)}"
        with urllib.request.urlopen(url) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f, length=16 * 1024 * 1024)
    tmp = os.path.join(OUT, p, "_tmp")
    with zipfile.ZipFile(zpath) as z:
        members = [m for m in z.namelist() if f"/{BLIP}/00_Original/" in "/" + m and not m.endswith("/")]
        if not members:
            raise SystemExit(f"{p}: no {BLIP}/00_Original in the zip")
        z.extractall(tmp, members=members)
    os.makedirs(dest, exist_ok=True)
    for f in glob.glob(os.path.join(tmp, "**", "00_Original", "*"), recursive=True):
        stem, ext = os.path.splitext(os.path.basename(f))           # Original.nii -> data.nii
        shutil.move(f, os.path.join(dest, "data" + ext))
    shutil.rmtree(tmp)
    os.remove(zpath)
    print(f"{p}: ready -> {sorted(os.listdir(dest))}")

print("\nAdd to STUDIES in compare_rejection_configs.py:")
for p in people:
    print(f'    "{p}": ("{os.path.abspath(os.path.join(OUT, p))}", None),')