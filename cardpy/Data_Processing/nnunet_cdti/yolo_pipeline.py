"""
yolo_pipeline.py
================

Standalone, notebook-facing YOLO side-pipeline for the cDTI LV / RVIP project.

Why this module exists
----------------------
A reviewer asked us to replace the *Stage-1 nnUNet whole-heart crop* (and,
ideally, the RVIP detection) with a cheaper detector. YOLO is a natural fit:
the heart-crop is a single bounding box, and the two RV insertion points (RVIPs)
are well modelled as two *tiny* boxes whose centers we read off.

Design principle
----------------
YOLO here is a drop-in *emitter of the same artifacts the current nnUNet stages
already produce*, so the existing evaluation code (``eval_util.py``,
``calculate_scale_factor``, ``upscale_segmentation_to_original``) keeps working
**unchanged**:

* ``detect_crop_box``  -> a *square* (x_min, x_max, y_min, y_max) box, squared
  with the *exact same* logic as ``make_square_crop`` in ``inference.ipynb``.
* ``crop_box_to_mask`` -> a binary mask byte-for-byte compatible with the
  "cleaned" Stage-1 crop mask that the rest of the pipeline consumes.
* ``detect_rvips``     -> two points (or ``None`` on low confidence -> a *clean*
  failure instead of a wild 1250 mm outlier).
* ``rvips_to_mask``    -> disks written as labels **1** (anterior / IP1) and **2**
  (inferior / IP2), matching Dataset105_HannumSmartHealthDataIPs (IPs-only, no LV).

RVIP detection uses the **tiny-bbox + center** formulation (chosen over a single
keypoint): a box carries spatial extent and a confidence score, so a weak
detection lands *near* the right region and can be *rejected* rather than thrown
to a random corner of the image.

Coordinate conventions  (read this once, it removes all the confusion)
----------------------------------------------------------------------
The repo's ``make_square_crop`` does ``coords = np.argwhere(mask > 0)`` then
``x_min, y_min = coords.min(axis=0)``. So throughout *this* repo:

    "x"  ==  array axis 0  ==  row    (vertical)
    "y"  ==  array axis 1  ==  col    (horizontal)

YOLO, by contrast, speaks in image pixels where:

    YOLO "x" (cx, x1, x2)  ==  column  (horizontal)
    YOLO "y" (cy, y1, y2)  ==  row     (vertical)

Every conversion below is explicit about this swap. Boxes and points returned by
this module are always in the **repo convention** (x = row/axis-0,
y = col/axis-1) so they slot straight into the existing code.

Dependency
----------
``ultralytics`` (YOLOv8) is required for training/inference but is imported
*lazily* inside the functions that need it, so you can import this module (and
use the pure-numpy adapters / label exporters) without it installed.
Install with: ``pip install ultralytics`` (pulls torch, already present for
nnUNet). The default ``yolov8n.pt`` weights download on first use.
"""

from __future__ import annotations

import os
import glob
from typing import Iterable, Optional, Sequence

import numpy as np
import nibabel as nib


# ---------------------------------------------------------------------------
# Class-id conventions
# ---------------------------------------------------------------------------
# Two *separate* YOLO models are assumed (one for the crop, one for the RVIPs),
# matching the plan. If you ever unify them into a single model, shift the RVIP
# class ids to 1 and 2 and reserve 0 for the heart.
CROP_CLASS_ID = 0          # single class for the whole-heart crop model
ANTERIOR_CLASS_ID = 0      # 2-class RVIP model, class 0 -> anterior insertion point
INFERIOR_CLASS_ID = 1      # 2-class RVIP model, class 1 -> inferior insertion point
IP_CLASS_ID = 0            # single-class RVIP model: one "insertion_point" class

# Anterior vs inferior are distinguished by *position*, not local appearance, so a
# tiny-box classifier can't separate them (and fliplr augmentation scrambles the
# cue). The single-class model detects both points as IP_CLASS_ID and we assign
# anterior/inferior afterwards with assign_anterior_inferior() -- see that function.

# RVIP GT label ids. These match Dataset105_HannumSmartHealthDataIPs (IPs-only,
# no LV): background 0, IP1 = 1 (anterior insertion point), IP2 = 2 (inferior).
# NOTE: an *older* combined-mask convention used LV=1, anterior=2, inferior=3 (and
# eval_util.py still hardcodes 2/3). Dataset105 does NOT follow that; it is 1/2.
# Both GT reading and predicted-mask writing below use these ids, so YOLO output is
# directly comparable to the Dataset105 GT.
ANTERIOR_LABEL = 1         # IP1 in dataset.json
INFERIOR_LABEL = 2         # IP2 in dataset.json


# ---------------------------------------------------------------------------
# Multi-contrast input convention
# ---------------------------------------------------------------------------
# nnUNet stores each contrast of a case as a separate file with a 4-digit
# channel suffix: ``CASE_0000.nii.gz`` (contrast 1), ``CASE_0001.nii.gz`` ...
# YOLOv8 expects a 3-channel (RGB-like) image, so we map up to three nnUNet
# contrasts onto R/G/B. ``CHANNELS`` selects *which* contrast index feeds each
# of the three YOLO channels:
#
#   (0, 0, 0)  -> "version 3": replicate the first contrast into all 3 channels
#                 (grayscale-as-RGB; identical information to the old behaviour).
#   (0, 1, 2)  -> "version 1": three *distinct* contrasts as R, G, B, so the
#                 detector actually sees multi-contrast information.
#
# Any 3 indices are allowed (e.g. (0, 2, 3) to pick a subset of four contrasts).
DEFAULT_CHANNELS = (0, 0, 0)


# ===========================================================================
# Geometry helpers (pure numpy, no YOLO needed)
# ===========================================================================
def make_square_crop(mask: np.ndarray, target_size: int = 256):
    """Square bounding box around the non-zero pixels of ``mask``.

    This is an exact re-implementation of ``make_square_crop`` from
    ``inference.ipynb`` (cell 2) so YOLO crops are squared *identically* to the
    nnUNet pipeline. Returns ``(x_min, x_max, y_min, y_max, scale_factor)`` in
    repo convention (x = axis 0 / row, y = axis 1 / col).
    """
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("make_square_crop: mask is empty (no non-zero pixels).")
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0) + 1  # exclusive

    side_length = int(max(x_max - x_min, y_max - y_min))

    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    x_min = int(max(0, center_x - side_length // 2))
    x_max = int(x_min + side_length)
    y_min = int(max(0, center_y - side_length // 2))
    y_max = int(y_min + side_length)

    scale_factor = side_length / target_size
    return x_min, x_max, y_min, y_max, scale_factor


def _box_to_binary_mask(x_min, x_max, y_min, y_max, shape) -> np.ndarray:
    """Render a (repo-convention) box as a uint8 binary mask of ``shape``."""
    mask = np.zeros(shape, dtype=np.uint8)
    x_min = max(0, int(x_min)); y_min = max(0, int(y_min))
    x_max = min(shape[0], int(x_max)); y_max = min(shape[1], int(y_max))
    mask[x_min:x_max, y_min:y_max] = 1
    return mask


def _bbox_repo_to_yolo(x_min, x_max, y_min, y_max, img_h, img_w):
    """Repo box (x=row, y=col, exclusive max) -> normalized YOLO (cx, cy, w, h).

    Remember the swap: YOLO cx is horizontal (col / y-axis here), cy is vertical
    (row / x-axis here).
    """
    row_center = (x_min + x_max) / 2.0
    col_center = (y_min + y_max) / 2.0
    cx = col_center / img_w          # horizontal -> col
    cy = row_center / img_h          # vertical   -> row
    w = (y_max - y_min) / img_w      # width  spans cols
    h = (x_max - x_min) / img_h      # height spans rows
    return cx, cy, w, h


def _bbox_yolo_to_repo(cx, cy, w, h, img_h, img_w):
    """Normalized YOLO (cx, cy, w, h) -> repo box (x_min, x_max, y_min, y_max).

    Inverse of ``_bbox_repo_to_yolo``. Returns integer, exclusive-max bounds.
    """
    col_center = cx * img_w
    row_center = cy * img_h
    box_w = w * img_w
    box_h = h * img_h
    x_min = int(round(row_center - box_h / 2.0))
    x_max = int(round(row_center + box_h / 2.0))
    y_min = int(round(col_center - box_w / 2.0))
    y_max = int(round(col_center + box_w / 2.0))
    return x_min, x_max, y_min, y_max


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 255] uint8 (matches normalize_image in repo)."""
    image = np.asarray(image, dtype=np.float64)
    lo, hi = image.min(), image.max()
    if hi > lo:
        image = (image - lo) / (hi - lo)
    else:
        image = np.zeros_like(image)
    return (image * 255.0).round().astype(np.uint8)


def load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI volume as a numpy array (squeezed to 2D when possible)."""
    data = nib.load(path).get_fdata()
    return np.squeeze(data)


def _channel_path(image_path: str, channel: int) -> str:
    """Sibling path for a given nnUNet channel index.

    Given the channel-0 file (``.../CASE_0000.nii.gz``) return the path for
    ``channel`` (``.../CASE_0001.nii.gz`` etc.). If the filename has no 4-digit
    channel token the original path is returned unchanged.
    """
    dirname, fname = os.path.split(image_path)
    for ext in (".nii.gz", ".nii"):
        if fname.endswith(ext):
            stem = fname[: -len(ext)]
            break
    else:
        stem, ext = os.path.splitext(fname)
    if len(stem) >= 5 and stem[-5] == "_" and stem[-4:].isdigit():
        stem = stem[:-4] + f"{channel:04d}"
    return os.path.join(dirname, stem + ext)


def load_channels(image_path: str,
                  channels: Sequence[int] = DEFAULT_CHANNELS) -> list:
    """Load the requested nnUNet contrast(s) for a case as a list of 2D arrays.

    ``image_path`` is the channel-0 file; each index in ``channels`` is resolved
    to its sibling file via :func:`_channel_path` and loaded. With ``(0, 0, 0)``
    the same file is loaded three times (cheap; version 3); with ``(0, 1, 2)``
    three distinct contrasts are loaded (version 1).
    """
    arrays = []
    for c in channels:
        p = _channel_path(image_path, c)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"load_channels: contrast {c} not found at {p} "
                f"(derived from {image_path}). Check your CHANNELS vs the "
                f"number of contrasts in the dataset."
            )
        arrays.append(load_nifti(p))
    return arrays


def _stack_rgb(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Per-channel min-max normalize 2D arrays and stack into (H, W, 3) uint8.

    Accepts 1..N arrays: fewer than 3 are padded by repeating the last channel
    (so a single array -> classic grayscale-as-RGB); more than 3 use the first
    three. Each channel is normalized *independently* so a weak contrast still
    uses the full 8-bit range.
    """
    arrays = list(arrays)
    if not arrays:
        raise ValueError("_stack_rgb: need at least one array.")
    chans = [_normalize_to_uint8(a) for a in arrays[:3]]
    while len(chans) < 3:
        chans.append(chans[-1])
    return np.stack(chans, axis=-1)


def _prepare_rgb(image) -> np.ndarray:
    """Coerce assorted inputs into an (H, W, 3) uint8 RGB image for YOLO.

    * a single 2D array          -> replicated to 3 channels (version 3);
    * a list/tuple of 2D arrays  -> stacked as R/G/B (version 1);
    * an (H, W, C) array         -> its channels stacked (C may be 1, 3, ...).
    """
    if isinstance(image, (list, tuple)):
        return _stack_rgb(image)
    image = np.asarray(image)
    if image.ndim == 2:
        return _stack_rgb([image])
    if image.ndim == 3:
        return _stack_rgb([image[..., i] for i in range(image.shape[-1])])
    raise ValueError(f"_prepare_rgb: unsupported image shape {image.shape}")


# ===========================================================================
# B1. Label export  (existing ground truth  ->  YOLO format)
# ===========================================================================
# YOLO label files are one ``.txt`` per image, each line:
#     <class_id> <cx> <cy> <w> <h>      (all of cx,cy,w,h normalized to [0,1])
# and images live alongside in a parallel folder. write_data_yaml() ties it
# together.

def nifti_to_yolo_image(nifti_path: str, out_png_path: str,
                        channels: Sequence[int] = DEFAULT_CHANNELS) -> None:
    """Save a case's contrast(s) as an 8-bit 3-channel PNG for YOLO training.

    ``nifti_path`` is the channel-0 file; ``channels`` selects which nnUNet
    contrasts map onto R/G/B (see ``DEFAULT_CHANNELS``). ``(0, 0, 0)`` writes a
    grayscale-as-RGB PNG (version 3); ``(0, 1, 2)`` writes a true multi-contrast
    RGB PNG (version 1).
    """
    from PIL import Image  # lazy; Pillow ships with ultralytics

    arrays = load_channels(nifti_path, channels)
    rgb = _stack_rgb(arrays)
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    Image.fromarray(rgb).save(out_png_path)


def crop_mask_to_yolo_label(crop_mask_path: str,
                            out_txt_path: str,
                            class_id: int = CROP_CLASS_ID) -> str:
    """GT square-crop mask -> a single-line YOLO label (the heart box).

    Reads ``Square_Crop_Mask_Slice_*.nii``, squares it with ``make_square_crop``
    (so training targets match exactly what we evaluate against), and writes the
    normalized YOLO line. Returns the written line.
    """
    mask = load_nifti(crop_mask_path)
    img_h, img_w = mask.shape[:2]
    x_min, x_max, y_min, y_max, _ = make_square_crop(mask)
    cx, cy, w, h = _bbox_repo_to_yolo(x_min, x_max, y_min, y_max, img_h, img_w)
    line = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    with open(out_txt_path, "w") as fh:
        fh.write(line + "\n")
    return line


def combined_mask_to_rvip_label(combined_mask_path: str,
                                out_txt_path: str,
                                box_px: int = 8,
                                anterior_label: int = ANTERIOR_LABEL,
                                inferior_label: int = INFERIOR_LABEL,
                                anterior_class: int = ANTERIOR_CLASS_ID,
                                inferior_class: int = INFERIOR_CLASS_ID,
                                single_class: bool = False,
                                ip_class: int = IP_CLASS_ID) -> list:
    """GT insertion-point mask -> YOLO labels for the two RVIPs.

    Reads the two insertion points from ``anterior_label`` / ``inferior_label``
    (Dataset105: IP1=1 anterior, IP2=2 inferior). Each point is a single annotated
    region; we take its centroid and emit a fixed ``box_px`` x ``box_px`` box around
    it (tiny-bbox formulation). A point absent in the GT is simply skipped.

    ``single_class=True`` writes BOTH points under ``ip_class`` (one detection class)
    instead of separate anterior/inferior classes -- the recommended formulation,
    since anterior/inferior are told apart by geometry at inference, not by YOLO.
    Returns the written lines.
    """
    mask = load_nifti(combined_mask_path)
    img_h, img_w = mask.shape[:2]
    lines = []
    for label, cls in ((anterior_label, anterior_class),
                       (inferior_label, inferior_class)):
        if single_class:
            cls = ip_class
        coords = np.argwhere(mask == label)
        if coords.size == 0:
            continue  # this RVIP not annotated on this slice
        # centroid in repo convention (x = row, y = col)
        cx_repo, cy_repo = coords.mean(axis=0)
        half = box_px / 2.0
        x_min, x_max = cx_repo - half, cx_repo + half
        y_min, y_max = cy_repo - half, cy_repo + half
        cx, cy, w, h = _bbox_repo_to_yolo(x_min, x_max, y_min, y_max, img_h, img_w)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    with open(out_txt_path, "w") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    return lines


def export_dataset(samples: Iterable[dict],
                   out_dir: str,
                   kind: str = "crop",
                   split_key: str = "split",
                   box_px: int = 8,
                   channels: Sequence[int] = DEFAULT_CHANNELS,
                   anterior_label: int = ANTERIOR_LABEL,
                   inferior_label: int = INFERIOR_LABEL,
                   single_class: bool = False) -> dict:
    """Materialize a YOLO dataset on disk from a list of sample dicts.

    Each ``sample`` is a dict::

        {
            "image_path": "<path to DWI_avg .nii>",      # the training image
            "mask_path":  "<path to GT mask .nii>",      # crop mask OR combined mask
            "name":       "Le_002_..._slice_001",        # unique stem
            "split":      "train" | "val" | "test",      # volunteer-level split
        }

    ``kind`` selects the label exporter: ``"crop"`` (whole-heart square box from a
    ``Square_Crop_Mask``) or ``"rvip"`` (two insertion-point boxes from a combined
    mask). Mirrors the existing volunteer-level split rather than a random one.
    ``channels`` selects which nnUNet contrasts feed the R/G/B PNG channels
    (see ``DEFAULT_CHANNELS``): ``(0, 0, 0)`` = version 3, ``(0, 1, 2)`` = version 1.

    Returns a small summary dict with per-split counts.
    """
    assert kind in ("crop", "rvip"), "kind must be 'crop' or 'rvip'"
    samples = list(samples)
    if not samples:
        raise ValueError(
            "export_dataset: got 0 samples, nothing to export. Check that the "
            "images/labels dirs exist and that filenames match '*_0000.nii.gz' "
            "(a common cause is a stale dataset folder name in the config)."
        )
    counts = {}
    for s in samples:
        split = s.get(split_key, "train")
        img_out = os.path.join(out_dir, "images", split, s["name"] + ".png")
        lbl_out = os.path.join(out_dir, "labels", split, s["name"] + ".txt")
        nifti_to_yolo_image(s["image_path"], img_out, channels=channels)
        if kind == "crop":
            crop_mask_to_yolo_label(s["mask_path"], lbl_out)
        else:
            combined_mask_to_rvip_label(s["mask_path"], lbl_out, box_px=box_px,
                                        anterior_label=anterior_label,
                                        inferior_label=inferior_label,
                                        single_class=single_class)
        counts[split] = counts.get(split, 0) + 1
    return counts


def write_data_yaml(out_dir: str, kind: str = "crop",
                    single_class: bool = False) -> str:
    """Write the ``data.yaml`` describing an exported dataset. Returns its path.

    For ``kind='rvip'``, ``single_class=True`` declares one ``insertion_point`` class
    (must match how the labels were exported).
    """
    if kind == "crop":
        names = {CROP_CLASS_ID: "heart"}
    elif single_class:
        names = {IP_CLASS_ID: "insertion_point"}
    else:
        names = {ANTERIOR_CLASS_ID: "anterior_rvip",
                 INFERIOR_CLASS_ID: "inferior_rvip"}
    names_block = "\n".join(f"  {k}: {v}" for k, v in sorted(names.items()))
    os.makedirs(out_dir, exist_ok=True)
    yaml_path = os.path.join(out_dir, "data.yaml")
    content = (
        f"path: {os.path.abspath(out_dir)}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n{names_block}\n"
    )
    with open(yaml_path, "w") as fh:
        fh.write(content)
    return yaml_path


# ===========================================================================
# B2. Train
# ===========================================================================
def train_yolo(data_yaml: str,
               model: str = "yolov8n.pt",
               epochs: int = 100,
               imgsz: int = 256,
               project: str = "yolo_runs",
               name: str = "crop",
               exist_ok: bool = True,
               **kwargs):
    """Thin wrapper over ``ultralytics.YOLO(...).train(...)``.

    Call once for the crop model and once for the RVIP model (with the matching
    ``data.yaml`` and a distinct ``name``). Returns the trained ``YOLO`` object;
    best weights are saved under ``<project>/<name>/weights/best.pt``.

    ``exist_ok=True`` (default) overwrites ``<project>/<name>`` on re-run instead of
    spawning ``name2``, ``name3`` ... so ``weights/best.pt`` stays at a stable path.
    """
    from ultralytics import YOLO

    yolo = YOLO(model)
    yolo.train(data=data_yaml, epochs=epochs, imgsz=imgsz,
               project=project, name=name, exist_ok=exist_ok, **kwargs)
    return yolo


def load_model(weights_path: str):
    """Load a trained YOLO model from a ``.pt`` weights file."""
    from ultralytics import YOLO
    return YOLO(weights_path)


# ===========================================================================
# B3. Inference + adapters  (the notebook-facing API)
# ===========================================================================
def _predict_boxes(model, image, conf: float = 0.25):
    """Run YOLO on an image and return a list of (cls, confidence, cx,cy,w,h).

    ``image`` may be a single 2D array (grayscale-as-RGB), a list/tuple of 2D
    contrast arrays (multi-contrast R/G/B), or an already-prepared (H, W, 3)
    array — see :func:`_prepare_rgb`. Boxes are returned in *normalized YOLO*
    coords; callers convert to repo convention as needed.
    """
    rgb = _prepare_rgb(image)  # (H, W, 3) uint8; idempotent if already prepared
    results = model.predict(rgb, conf=conf, verbose=False)
    out = []
    img_h, img_w = rgb.shape[:2]
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls = int(b.cls.item())
            confidence = float(b.conf.item())
            # ultralytics xywh is in pixels, x=col(horizontal), y=row(vertical)
            x_c, y_c, bw, bh = (float(v) for v in b.xywh[0].tolist())
            out.append((cls, confidence,
                        x_c / img_w, y_c / img_h, bw / img_w, bh / img_h))
    return out


def detect_crop_box(model, image: np.ndarray, conf: float = 0.25):
    """Detect the whole-heart crop box. Drop-in replacement for Stage-1 nnUNet.

    Picks the highest-confidence detection, then *squares* it with the repo's
    ``make_square_crop`` logic so the result is identical in form to the cleaned
    nnUNet crop. Returns ``(x_min, x_max, y_min, y_max, scale_factor)`` in repo
    convention, or ``None`` if nothing was detected.
    """
    rgb = _prepare_rgb(image)
    dets = _predict_boxes(model, rgb, conf=conf)
    dets = [d for d in dets if d[0] == CROP_CLASS_ID]
    if not dets:
        return None
    _, _, cx, cy, w, h = max(dets, key=lambda d: d[1])  # highest confidence
    img_h, img_w = rgb.shape[:2]
    x_min, x_max, y_min, y_max = _bbox_yolo_to_repo(cx, cy, w, h, img_h, img_w)
    # Square it via a binary mask so the logic is byte-identical to the pipeline.
    raw = _box_to_binary_mask(x_min, x_max, y_min, y_max, rgb.shape[:2])
    return make_square_crop(raw)


def crop_box_to_mask(box, shape) -> np.ndarray:
    """Square box -> binary uint8 crop mask (same interface as the nnUNet crop).

    ``box`` may be the 5-tuple from ``detect_crop_box`` or a 4-tuple
    ``(x_min, x_max, y_min, y_max)``. The result is consumable directly by
    ``calculate_scale_factor`` / ``upscale_segmentation_to_original`` and can be
    saved to NIfTI to feed the existing evaluation unchanged.
    """
    x_min, x_max, y_min, y_max = box[0], box[1], box[2], box[3]
    return _box_to_binary_mask(x_min, x_max, y_min, y_max, shape)


def assign_anterior_inferior(points, flip: bool = False) -> dict:
    """Label two detected insertion points as anterior / inferior by geometry.

    Anterior vs inferior differ by *position*, not appearance, so the single-class
    detector returns two anonymous points and we assign them here. Rule: in the
    short-axis crop the **anterior** insertion sits higher than the inferior one, so
    the point with the smaller row (axis-0) is anterior. Set ``flip=True`` if a
    quick check against GT shows the convention is reversed for your data.

    ``points`` is a list of ``(x, y)`` in repo convention (x = row, y = col).
    Returns ``{"anterior": (x, y) | None, "inferior": (x, y) | None}`` -- with 0 or
    1 detections the missing point(s) are ``None`` (a clean failure).
    """
    result = {"anterior": None, "inferior": None}
    pts = list(points)[:2]
    if len(pts) == 1:
        result["anterior"] = pts[0]        # only one point: report it, leave the other None
        return result
    if len(pts) == 2:
        hi, lo = sorted(pts, key=lambda p: p[0])   # smaller row = higher = anterior
        if flip:
            hi, lo = lo, hi
        result["anterior"], result["inferior"] = hi, lo
    return result


def detect_rvips(model, image: np.ndarray, conf_thresh: float = 0.25,
                 single_class: bool = False, flip_assignment: bool = False) -> dict:
    """Detect the two RV insertion points.

    Returns ``{"anterior": (x, y) | None, "inferior": (x, y) | None}`` with points
    in repo convention (x = row/axis-0, y = col/axis-1). A missing / low-confidence
    point becomes ``None`` (a clean failure rather than a far-off outlier).

    Two modes:
    * ``single_class=True`` (recommended): the model has one insertion-point class;
      we take the two highest-confidence detections and label them anterior/inferior
      with :func:`assign_anterior_inferior` (``flip_assignment`` inverts the rule).
    * ``single_class=False``: legacy 2-class model, one box per class.
    """
    rgb = _prepare_rgb(image)
    dets = _predict_boxes(model, rgb, conf=conf_thresh)
    img_h, img_w = rgb.shape[:2]

    def _center(cx, cy):
        return (float(cy * img_h), float(cx * img_w))   # (row, col) in repo convention

    if single_class:
        cands = sorted((d for d in dets if d[0] == IP_CLASS_ID),
                       key=lambda d: d[1], reverse=True)[:2]
        pts = [_center(cx, cy) for _, _, cx, cy, w, h in cands]
        return assign_anterior_inferior(pts, flip=flip_assignment)

    result = {"anterior": None, "inferior": None}
    for key, cls in (("anterior", ANTERIOR_CLASS_ID), ("inferior", INFERIOR_CLASS_ID)):
        cands = [d for d in dets if d[0] == cls]
        if not cands:
            continue
        _, _, cx, cy, w, h = max(cands, key=lambda d: d[1])
        result[key] = _center(cx, cy)
    return result


def detect_crop_box_from_path(model, image_path: str,
                              channels: Sequence[int] = DEFAULT_CHANNELS,
                              conf: float = 0.25):
    """``detect_crop_box`` straight from a channel-0 NIfTI path.

    Loads the ``channels`` contrasts for the case (via :func:`load_channels`)
    and runs detection, so multi-contrast inference needs only the ``_0000``
    path. Use the *same* ``channels`` you trained with.
    """
    return detect_crop_box(model, load_channels(image_path, channels), conf=conf)


def detect_rvips_from_path(model, image_path: str,
                           channels: Sequence[int] = DEFAULT_CHANNELS,
                           conf_thresh: float = 0.25,
                           single_class: bool = False,
                           flip_assignment: bool = False) -> dict:
    """``detect_rvips`` straight from a channel-0 NIfTI path (see above)."""
    return detect_rvips(model, load_channels(image_path, channels),
                        conf_thresh=conf_thresh, single_class=single_class,
                        flip_assignment=flip_assignment)


def rvips_to_mask(points: dict,
                  shape=(256, 256),
                  radius: int = 3,
                  anterior_label: int = ANTERIOR_LABEL,
                  inferior_label: int = INFERIOR_LABEL) -> np.ndarray:
    """Grow a disk around each detected RVIP -> labelled mask (labels 2 & 3).

    Reproduces the circular GT annotation style so DSC / Hausdorff / center-of-mass
    evaluation in ``eval_util.py`` works against the YOLO output with no changes.
    ``points`` is the dict from ``detect_rvips``; ``None`` points are skipped.

    Tip: to compare points directly instead of via masks, just use the centers
    from ``detect_rvips`` and report center-to-center distance.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    rows, cols = np.ogrid[:shape[0], :shape[1]]
    for key, label in (("anterior", anterior_label), ("inferior", inferior_label)):
        pt = points.get(key)
        if pt is None:
            continue
        x, y = pt  # x = row, y = col
        disk = (rows - x) ** 2 + (cols - y) ** 2 <= radius ** 2
        mask[disk] = label
    return mask


# ===========================================================================
# B4. Evaluation helpers (point-distance)
# ===========================================================================
def insertion_points_from_mask(mask_path: str,
                               anterior_label: int = ANTERIOR_LABEL,
                               inferior_label: int = INFERIOR_LABEL) -> dict:
    """GT anterior/inferior insertion-point centroids from a labelled mask.

    Returns ``{"anterior": (x, y) | None, "inferior": (x, y) | None}`` in repo
    convention (x = row/axis-0, y = col/axis-1). A label absent in the mask -> None.
    Mirrors how the labels were exported (Dataset105: IP1=1 anterior, IP2=2 inferior).
    """
    mask = load_nifti(mask_path)
    out = {"anterior": None, "inferior": None}
    for key, label in (("anterior", anterior_label), ("inferior", inferior_label)):
        coords = np.argwhere(mask == label)
        if coords.size:
            cx, cy = coords.mean(axis=0)
            out[key] = (float(cx), float(cy))
    return out


def euclidean(p, q):
    """Euclidean distance between two ``(x, y)`` points; ``None`` if either is None.

    ``None`` propagates so a missed detection is reported as a miss, not a distance.
    """
    if p is None or q is None:
        return None
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


# ===========================================================================
# Convenience: discover GT files for export (optional, layout-aware helper)
# ===========================================================================
def find_crop_samples(main_folder: str,
                      annotator: str,
                      image_glob: str = "*Average_Diffusion_Weighted_Image_Slice_*.nii*",
                      crop_mask_name: str = "Square_Crop_Mask_Slice_{slice}.nii"):
    """Best-effort walker that pairs DWI_avg images with their GT square-crop masks.

    Mirrors the directory layout used in ``eval_util.process_folders``
    (``.../<annotator>/<volunteer>/Distortion_Corrected/<divo>/02_Crop_Masks/...``).
    Returns a list of ``sample`` dicts ready for ``export_dataset(kind='crop')``.
    Adjust the globs/patterns to your exact filenames if they differ -- this is a
    starting point, not a hard dependency.
    """
    samples = []
    pattern = os.path.join(main_folder, annotator, "*", "Distortion_Corrected",
                           "*", "02_Crop_Masks", crop_mask_name.replace("{slice}", "*"))
    for crop_mask_path in sorted(glob.glob(pattern)):
        divo_dir = os.path.dirname(os.path.dirname(crop_mask_path))
        slice_token = os.path.basename(crop_mask_path).split("_")[-1].split(".")[0]
        imgs = glob.glob(os.path.join(divo_dir, "**", image_glob), recursive=True)
        imgs = [p for p in imgs if slice_token in os.path.basename(p)]
        if not imgs:
            continue
        volunteer = os.path.basename(os.path.dirname(os.path.dirname(divo_dir)))
        divo = os.path.basename(divo_dir)
        name = f"{annotator}_{volunteer}_{divo}_slice_{slice_token}"
        samples.append({
            "image_path": imgs[0],
            "mask_path": crop_mask_path,
            "name": name,
            "split": "train",  # set per-volunteer split downstream
        })
    return samples


if __name__ == "__main__":  # pragma: no cover - illustrative, not run on import
    print(__doc__)