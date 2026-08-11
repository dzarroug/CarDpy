### cardpy/config.py
### Default configuration for cardpy.process().

DEFAULT_CONFIG = {

    ### ---- Input / data ----
    "data_type":        "DICOM",       # "DICOM" or "NifTis"
    "dicom_reader_info": "ON",
    "operation_type":   "Magnitude",   # "Magnitude" or "Complex"
    "dicom_subpath":    "02_cDTI/SAX/cDTI_SF_b350_RL_71",   # path to DICOM series inside study folder
    "slice_index":      [3],           # list of slice indices, or None for all

    ### ---- Gibbs ringing removal ----
    "gibbs": {
        "enabled": True,
    },

    ### ---- Shot rejection ----
    "rejection": {
        "enabled":        True,
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
    },

    ### ---- cDTI analysis (post-contour) ----
    "cdti": {
        "num_interp_points": 200,
        "smoothness_level":  "Low",
        "helix_angle_filter": {
            "linear_outlier_stdev":    1,
            "spatial_wall_depth_factor": 0.25,
            "spatial_kernel_size":     5,
        },
    },

    ### ---- Output ---
    "output": {
        "save_diagnostics": True,  
    },
}