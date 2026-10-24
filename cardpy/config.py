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
        "version":           "v2",  # "legacy" / "v2" (box crop + responsive contouring)
        "segmentation_saved_configuration_mode": None,  # optional off / reuse / review override
        "segmentation_review": True,   # show saved contours for confirmation/editing (v2)
        "reuse_existing_segmentation": False,  # discover this output folder's saved JSON
        "segmentation_zoom": True,      # v2: reuse the rejection crop for contouring
        "md_max":            3.0,       # v2 display maximum, in µm^2/ms
    },

    ### ---- Segmentation (contouring) ----
    "segmentation": {
        "mode":           "manual",  # "manual" (draw contours) / "auto" (nnUNet_cDTI model)
        # Set True to review auto contours before analysis (requires gui.version="v2").
        # Slices the model cannot segment always open the manual GUI.
        "review":         False,
        "weights_dir":    None,      # None = ~/.cache/cardpy/cdti-lv-baseline (auto-downloaded)
        "use_model_mask": True,      # unreviewed slices: True = model mask, False = refill from contours
    },

    ### ---- Gibbs ringing removal ----
    "gibbs": {
        "enabled": True,
    },

    ### ---- Shot rejection ----
    "rejection": {
        "enabled":        True,
        "mode":           "algorithm",  # "algorithm" / "manual" / "ml"
        # Set True to review automatic keep/reject decisions before respiratory sorting.
        # Click thumbnails to toggle (green = keep, red = reject); Apply slice and continue.
        # Works with mode "algorithm" or "ml"; mode "manual" always opens the review.
        "manual_review":  False,  # requires rejection.enabled=True; use saved_configuration_mode="off" for a fresh run
        "ml_model_path":  None,         # path to best.pt (ml mode)
        "ml_threshold":   None,         # None = threshold saved with the model
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

    ### ---- Optional uncertainty (after quantitative results) ----
    "uncertainty": {
        "enabled": False,
        "iterations": 400,
        "n_jobs": 4,
        "seed": 0,
        "desired_averages": None,  # None keeps the original repetition count
        "keep_images": False,
    },

    ### ---- Output ---
    "output": {
        "directory": None,  # None uses study_root/CarDpy_Output.
        "save_diagnostics": True,  
    },
}