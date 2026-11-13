"""Resolve Smart Health series using Up=FH/LR/AP and Down=HF/RL/PA."""
from pathlib import Path
import re

BLIP_DIRECTIONS = {"blipup": ("FH", "LR", "AP"),
                   "blipdown": ("HF", "RL", "PA")}


def _direction(*labels):
    """Recognize direction tokens, including F-H, L-R, and A-P spellings."""
    directions = set()
    for label in labels:
        for match in re.finditer(r"(?<![A-Z0-9])(F[-_ ]?H|H[-_ ]?F|L[-_ ]?R|R[-_ ]?L|A[-_ ]?P|P[-_ ]?A)(?![A-Z0-9])",
                                 str(label).upper()):
            directions.add(re.sub(r"[-_ ]", "", match.group(1)))
    if len(directions) > 1:
        raise ValueError(f"Conflicting phase-encoding direction labels: {labels}")
    return next(iter(directions), None)


def resolve_blip_directory(study_root, distortion, *, even_odd_mode=False, sl_id="Even"):
    """Find one series and check one DICOM SeriesDescription/ProtocolName.

    Supports reorganized 02_cDTI/SAX series and legacy BlipUp/BlipDown
    directories. Multiple matching series require an explicit choice rather
    than silently combining acquisitions. Even/Odd input stays in its named
    acquisition directory. Up=FH/LR/AP and Down=HF/RL/PA are workflow naming
    conventions, not a general physical polarity rule. No pixel data are read.
    """
    import pydicom

    key = distortion.replace("_", "").lower()
    if key not in ("blipup", "blipdown"):
        raise ValueError("distortion must be BlipUp or BlipDown")
    wanted = BLIP_DIRECTIONS[key]
    root = Path(study_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Study directory not found: {root}")
    acquisition = "BlipUp" if key == "blipup" else "BlipDown"
    if even_odd_mode:
        if sl_id not in ("Even", "Odd"):
            raise ValueError("sl_id must be Even or Odd")
        acquisition += sl_id
    legacy = [root / "02_cDTI" / acquisition, root / acquisition]
    # Include underscore spellings used by some older exports.
    legacy += [p.with_name(p.name.replace("Blip", "Blip_")) for p in legacy]
    bases = [p for p in legacy if p.is_dir()]
    if not even_odd_mode:
        cdti = root / "02_cDTI"
        sax = cdti / "SAX"
        bases += [sax] if sax.is_dir() else ([cdti] if cdti.is_dir() else [])
        if (root / "SAX").is_dir():
            bases.append(root / "SAX")

    folders = set()
    for base in bases:
        folders.add(base)
        folders.update(p for p in base.rglob("*") if p.is_dir() and not p.name.startswith("."))
    matches = []
    for folder in sorted(folders):
        # Match the extensions accepted by CardPy's DICOM_Reader.
        files = sorted(p for p in folder.iterdir()
                       if p.is_file() and not p.name.startswith(".")
                       and p.suffix in (".dcm", ".IMA"))
        if not files:
            continue
        # Do not mix a separately acquired Even/Odd series into all-slice input.
        if not even_odd_mode and any(
                re.search(r"(?:even|odd)", part, re.IGNORECASE)
                for part in folder.relative_to(root).parts):
            continue
        named = _direction(folder.name)
        header = pydicom.dcmread(files[0], stop_before_pixels=True,
                                specific_tags=["SeriesDescription", "ProtocolName"])
        described = _direction(header.get("SeriesDescription", ""),
                               header.get("ProtocolName", ""))
        if named and described and named != described:
            raise ValueError(f"Folder and DICOM direction disagree: {folder}")
        direction = described or named
        if direction in wanted or (direction is None and any(
                folder == p or p in folder.parents for p in legacy)):
            matches.append(folder)
    if len(matches) > 1:
        raise ValueError(f"Multiple {wanted} series found; set config['dicom_subpath'] "
                         "explicitly to one series: " + ", ".join(map(str, matches)))
    if not matches:
        raise FileNotFoundError(f"No {acquisition} ({wanted}) DICOM series found under {root}")
    return matches[0]
