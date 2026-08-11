from cardpy.pipeline import process
from cardpy.config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["data_type"]     = "DICOM"
config["dicom_subpath"] = "02_cDTI/SAX/cDTI_SF_b350_RL_71"
config["slice_index"]   = [3,4]

results = process("Healthy_Volunteer_007", config)