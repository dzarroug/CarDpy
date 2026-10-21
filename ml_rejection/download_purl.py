"""
Download the _Processing zips from PURL yh756nh8583 one at a time, keep only
01_Unrung and 02_Rejected (Blip_Up/Blip_Down), then delete the zip.
  python ml_rejection/download_purl.py            # dry run
  python ml_rejection/download_purl.py --download # download + extract
Failed participants are listed at the end and in PURL_data/failed.txt.
"""
import argparse, os, shutil, urllib.parse, urllib.request, zipfile
import xml.etree.ElementTree as ET

DRUID = "yh756nh8583"
OUT = "PURL_data"
KEEP = ("/01_Unrung/", "/02_Rejected/")
RETRIES = 3

ap = argparse.ArgumentParser()
ap.add_argument("--download", action="store_true")
ap.add_argument("--match", default="_Processing")
args = ap.parse_args()

xml = urllib.request.urlopen(f"https://purl.stanford.edu/{DRUID}.xml").read()
files = [(el.get("id"), int(el.get("size", 0)))
         for el in ET.fromstring(xml).iter() if el.tag.endswith("file") and el.get("id")]
wanted = [(f, s) for f, s in files if args.match in f]
print(f"{len(wanted)} zips, {sum(s for _, s in wanted) / 1e9:.1f} GB to stream "
      f"(only 01_Unrung + 02_Rejected kept)")

if not args.download:
    raise SystemExit


def fetch(f, s, zpath):
    url = f"https://stacks.stanford.edu/file/{DRUID}/{urllib.parse.quote(f)}"
    with urllib.request.urlopen(url) as r, open(zpath, "wb") as out:
        shutil.copyfileobj(r, out, length=16 * 1024 * 1024)
    got = os.path.getsize(zpath)
    if got != s:
        raise IOError(f"size mismatch: got {got / 1e9:.2f} GB, expected {s / 1e9:.2f} GB")
    if not zipfile.is_zipfile(zpath):
        raise IOError("downloaded file is not a valid zip")


os.makedirs(OUT, exist_ok=True)
failed = []
for i, (f, s) in enumerate(wanted, 1):
    name = f.removesuffix(".zip")
    done_flag = os.path.join(OUT, name, ".done")
    if os.path.exists(done_flag):
        print(f"[{i}/{len(wanted)}] {name}: already done")
        continue

    zpath = os.path.join(OUT, f)
    ok = os.path.exists(zpath) and os.path.getsize(zpath) == s and zipfile.is_zipfile(zpath)
    for attempt in range(1, RETRIES + 1):
        if ok:
            break
        print(f"[{i}/{len(wanted)}] downloading {f} ({s / 1e9:.1f} GB), attempt {attempt}")
        try:
            fetch(f, s, zpath)
            ok = True
        except Exception as e:
            print(f"    failed: {e}")
    if not ok:
        failed.append((name, "download failed after retries"))
        if os.path.exists(zpath):
            os.remove(zpath)
        continue

    try:
        with zipfile.ZipFile(zpath) as z:
            members = [m for m in z.namelist()
                       if any(k in "/" + m for k in KEEP) and "TopUp" not in m]
            z.extractall(os.path.join(OUT, name), members=members)
        print(f"    extracted {len(members)} files")
        if len(members) != 16:
            failed.append((name, f"extracted {len(members)} files, expected 16"))
        os.remove(zpath)
        os.makedirs(os.path.join(OUT, name), exist_ok=True)
        open(done_flag, "w").close()
    except Exception as e:
        failed.append((name, f"extract failed: {e}"))

if failed:
    with open(os.path.join(OUT, "failed.txt"), "w") as fh:
        for name, why in failed:
            fh.write(f"{name}: {why}\n")
    print("\nFAILED:")
    for name, why in failed:
        print(f"  {name}: {why}")
else:
    print("\nAll participants done.")