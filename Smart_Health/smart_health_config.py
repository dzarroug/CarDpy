"""Local defaults for the Smart Health notebooks.

Processing settings match 01_Blip_Up.ipynb, 02_Blip_Down.ipynb, and
03_TOPUP.ipynb in /Volumes/T7_old/Smart_Health/Scripts. Blip notebooks
select DICOM at import time; TOPUP reads corrected NifTis and resumes at 13.
The v2 GUI choice is retained from the GitHub notebooks.

Edit SMART_HEALTH_CONFIG here to change shared settings. Each notebook gets
an independent deep copy and supplies its own subject and input/output paths.
This module is independent of cardpy.config.DEFAULT_CONFIG.
"""
from copy import deepcopy


SMART_HEALTH_CONFIG = {'gui': {'version': 'v2', 'saved_configuration_mode': 'off'},
 'gibbs': {'enabled': True},
 'rejection': {'enabled': True,
               'mode': 'algorithm',
               'manual_review': False,
               'nrmse_threshold': 0.75,
               'nssim_threshold': 0.75,
               'zoom': 'ON',
               'interact_zoom': 'ON',
               'organ': 'Heart'},
 'respiratory': {'enabled': True,
                 'zoom': 'ON',
                 'interact_zoom': 'ON',
                 'organ': 'Stomach'},
 'registration': {'enabled': True,
                  'algorithm': 'Affine',
                  'mode': '2D',
                  'temporary_denoising': 'OFF'},
 'adc_filter': {'enabled': True},
 'averaging': {'enabled': True},
 'denoising': {'enabled': True, 'number_of_coils': 20, 'algorithm': 'LocalPCA'},
 'interpolation': {'enabled': True},
 'extended_matrix': {'enabled': True},
 'e1': {'enabled': True},
 'dti': {'tensor_fit': 'NLLS', 'b0_threshold': 0},
 'data_type': 'DICOM',
 'dicom_reader_info': 'ON',
 'operation_type': 'Magnitude',
 'slice_index': None,
 'smart_health': {'workflow': 'blip'},
 'dicom_subpath': '',
 'output': {'directory': None}}


def get_config(workflow="blip"):
    """Return independent defaults for the blip or topup workflow."""
    if workflow not in ("blip", "topup"):
        raise ValueError("workflow must be 'blip' or 'topup'")
    config = deepcopy(SMART_HEALTH_CONFIG)
    config["smart_health"]["workflow"] = workflow
    config["data_type"] = "NifTis" if workflow == "topup" else "DICOM"
    return config
