### cardpy/config.py
### Default configuration for cardpy.process().

DEFAULT_CONFIG = {

    ### ---- Input / data ----
    "data_type":        "DICOM",       # "DICOM" or "NifTis"
    "dicom_reader_info": "ON",
    "operation_type":   "Magnitude",   # "Magnitude" or "Complex"
    "dicom_subpath":    "02_cDTI/SAX/cDTI_SF_b350_RL_71",   # path to DICOM series inside study folder
    "slice_index":      [3],           # list of slice indices, or None for all

    ### ---- Interactive tools ----
    "gui": {
        "saved_configuration_mode": "off",  # off / reuse (no saved GUIs) / review (prefill GUIs)
        "point_size":        8.0,       # v2 marker diameter, in display points (3–20)
        "version":           "legacy",  # "legacy" / "v2" (box crop + responsive contouring)
        "segmentation_review": True,   # show saved contours for confirmation/editing (v2)
        "reuse_existing_segmentation": False,  # discover this output folder's saved JSON
        "segmentation_zoom": True,      # v2: reuse the rejection crop for contouring
        "md_max":            3.0,       # v2 display maximum, in µm^2/ms
    },

    ### ---- Gibbs ringing removal ----
    "gibbs": {
        "enabled": True,
    },

    ### ---- Shot rejection ----
    "rejection": {
        "enabled":        True,
        # Set True to review automatic keep/reject decisions before respiratory sorting.
        # Click thumbnails to toggle (green = keep, red = reject); Apply slice and continue.
        "manual_review":  False,  # requires rejection.enabled=True; use saved_configuration_mode="off" for a fresh run
        "nrmse_threshold": 0.75,
        "nssim_threshold": 0.75,
        "zoom":           "ON",
        "interact_zoom":  "ON",
        "organ":          "Heart",
    },

    ### ---- Respiratory sorting ----
    "respiratory": {
        "enabled":       True,
        "zoom":          "ON",
        "interact_zoom": "ON",
        "organ":         "Stomach",
    },

    ### ---- Registration ----
    "registration": {
        "enabled":            True,
        "algorithm":          "Affine",   # Affine / Rigid / Translation
        "mode":               "2D",       # 2D / 3D (3D requires Rigid)
        "temporary_denoising": "OFF",
    },

    ### ---- ADC filtering & averaging ----
    "adc_filter": {
        "enabled": True,
    },
    "averaging": {
        "enabled": True,
    },

    ### ---- Denoising ----
    "denoising": {
        "enabled":        True,
        "number_of_coils": 20,
        "algorithm":      "LocalPCA",
    },

    ### ---- Interpolation / matrix ----
    "interpolation": {
        "enabled": True,
    },
    "extended_matrix": {
        "enabled": True,
    },

    ### ---- Eigenvector ----
    "e1": {
        "enabled": True,
    },

    ### ---- DTI reconstruction ----
    "dti": {
        "tensor_fit": "NLLS",
        "b0_threshold": 0,  # classify only true b0 as baseline; preserve measured b50 weighting
    },

    ### ---- cDTI analysis (post-contour) ----
    "cdti": {
        "num_interp_points": 200,
        "smoothness_level":  "Low",
        "hap_coordinate":    "cubic",  # cubic / centroid_radial; HAP_mm always radial
        "helix_angle_filter": {
            "linear_outlier_stdev":    1,
            "spatial_wall_depth_factor": 0.25,
            "spatial_kernel_size":     5,
        },
    },

    ### ---- Output ---
    "output": {
        "directory": None,  # None uses study_root/CarDpy_Output.
        "save_diagnostics": True,  
    },
}
