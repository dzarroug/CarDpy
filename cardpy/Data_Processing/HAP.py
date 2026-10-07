"""Helix-angle pitch using cubic or LV-centroid radial wall coordinates.

The centroid definition and polar contour interpolation follow the original
centroid-radial formulation in healthy_cdti_analysis/01_Processing/HAP_Demo.
"""
import numpy as np
from scipy.ndimage import label
from skimage.measure import find_contours


def _cavity(mask):
    background, _ = label(~mask)
    edge_labels = np.unique(np.concatenate(
        (background[0], background[-1], background[:, 0], background[:, -1])))
    cavities, count = label((~mask) & ~np.isin(background, edge_labels))
    if count == 0:
        return None
    sizes = np.bincount(cavities.ravel())
    sizes[0] = 0
    return cavities == sizes.argmax()


def _radii(contour, center, spacing, query_angles):
    offsets = contour * spacing - center
    angles = np.arctan2(offsets[:, 0], offsets[:, 1])
    radii = np.linalg.norm(offsets, axis=1)
    order = np.argsort(angles)
    angles, inverse = np.unique(np.round(angles[order], 12), return_inverse=True)
    radii = np.bincount(inverse, weights=radii[order]) / np.bincount(inverse)
    return np.interp(query_angles, np.r_[angles - 2 * np.pi, angles, angles + 2 * np.pi],
                     np.tile(radii, 3))


def centroid_wall_coordinates(myocardium, in_plane_spacing_mm=(1.0, 1.0)):
    """Return normalized depth and signed midwall distance in mm.

    The LV cavity centroid defines radial rays. Subpixel 0.5-level contours
    define endocardial and epicardial radii. Depth runs from 0 (endo) to 1
    (epi); distance is (depth - 0.5) times local radial wall thickness.
    Empty slices or slices without an enclosed cavity return NaN coordinates.
    Spacing follows the first two array axes, including anisotropic pixels.
    """
    mask = np.asarray(myocardium)
    spacing = np.asarray(in_plane_spacing_mm, dtype=float)
    if mask.ndim != 3 or any(size == 0 for size in mask.shape):
        raise ValueError('myocardium must be a nonempty 3D mask.')
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('In-plane spacing must contain two positive finite values.')
    depth = np.full(mask.shape, np.nan)
    distance = np.full(mask.shape, np.nan)
    for slc in range(mask.shape[2]):
        wall = np.isfinite(mask[..., slc]) & (mask[..., slc] > 0)
        cavity = _cavity(wall)
        if cavity is None or not wall.any():
            continue
        center = np.argwhere(cavity).mean(axis=0) * spacing
        contours = [c for c in find_contours(wall.astype(float), 0.5)
                    if len(c) >= 8 and np.allclose(c[0], c[-1])]
        if len(contours) < 2:
            continue
        median = [np.median(np.linalg.norm(c * spacing - center, axis=1)) for c in contours]
        endo, epi = contours[int(np.argmin(median))], contours[int(np.argmax(median))]
        points = np.argwhere(wall)
        offsets = points * spacing - center
        angles = np.arctan2(offsets[:, 0], offsets[:, 1])
        radii = np.linalg.norm(offsets, axis=1)
        endo_r = _radii(endo, center, spacing, angles)
        thickness = _radii(epi, center, spacing, angles) - endo_r
        valid = thickness > 0
        fraction = np.full(len(points), np.nan)
        fraction[valid] = np.clip((radii[valid] - endo_r[valid]) / thickness[valid], 0, 1)
        depth[points[:, 0], points[:, 1], slc] = fraction
        distance[points[:, 0], points[:, 1], slc] = (fraction - 0.5) * thickness
    return depth, distance


def _linear_fit(position, angle):
    finite = np.isfinite(position) & np.isfinite(angle)
    x, y = position[finite], angle[finite]
    if x.size < 4 or np.ptp(x) == 0:
        return np.nan, np.nan, np.nan, x.size
    slope, intercept = np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]
    total = np.sum((y - y.mean()) ** 2)
    r2 = np.nan if total == 0 else 1 - np.sum((y - (intercept + slope * x)) ** 2) / total
    return slope, intercept, r2, x.size


def helix_angle_pitch(helix_angle, myocardium, in_plane_spacing_mm=None,
                      coordinate='cubic', cubic_depth=None):
    """Return per-slice and pooled HAP fits and their coordinate volumes.

    HAP is the signed linear slope in degrees per percent wall thickness
    (the slope against 0..1 depth divided by 100). HAP_mm is degrees/mm,
    fitted independently against centroid-radial signed distance. HAP defaults
    to legacy cubic coordinates; set coordinate='centroid_radial' to use rays.
    Unknown physical spacing yields NaN HAP_mm rather than invented mm units.
    """
    if coordinate not in ('cubic', 'centroid_radial'):
        raise ValueError("coordinate must be 'cubic' or 'centroid_radial'.")
    angles, mask = np.asarray(helix_angle), np.asarray(myocardium)
    if angles.shape != mask.shape or angles.ndim != 3:
        raise ValueError('Helix angles and myocardium must have matching 3D shapes.')
    radial, signed_mm = centroid_wall_coordinates(
        mask, (1.0, 1.0) if in_plane_spacing_mm is None else in_plane_spacing_mm)
    if in_plane_spacing_mm is None:
        signed_mm[:] = np.nan
    if coordinate == 'cubic':
        if cubic_depth is None:
            from cardpy.Data_Processing.cDTI import Endo2Epi_Grid
            cubic_depth = Endo2Epi_Grid(mask)
        depth = np.asarray(cubic_depth, dtype=float)
        if depth.shape != mask.shape:
            raise ValueError('Cubic depth and myocardium must have matching shapes.')
    else:
        depth = radial
    valid_mask = np.isfinite(mask) & (mask > 0)
    depth = np.where(valid_mask, depth, np.nan)
    signed_mm = np.where(valid_mask, signed_mm, np.nan)
    normalized = np.asarray([_linear_fit(depth[..., slc] * 100, angles[..., slc])
                             for slc in range(mask.shape[2])])
    millimeters = np.asarray([_linear_fit(signed_mm[..., slc], angles[..., slc])
                             for slc in range(mask.shape[2])])
    return {
        'coordinate': coordinate, 'HAP_units': 'degrees/percent', 'HAP_mm_units': 'degrees/mm',
        'HAP': normalized[:, 0], 'HAP_intercept': normalized[:, 1],
        'HAP_R2': normalized[:, 2], 'HAP_n': normalized[:, 3].astype(int),
        'HAP_mm': millimeters[:, 0], 'HAP_mm_intercept': millimeters[:, 1],
        'HAP_mm_R2': millimeters[:, 2], 'HAP_mm_n': millimeters[:, 3].astype(int),
        'HAP_global': _linear_fit(depth * 100, angles)[0],
        'HAP_mm_global': _linear_fit(signed_mm, angles)[0],
        'depth': depth, 'centroid_depth': radial, 'distance_mm': signed_mm,
    }
