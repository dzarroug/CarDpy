from cardpy.pipeline import process
from cardpy.config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["data_type"]     = "DICOM"
config["dicom_subpath"] = "02_cDTI/SAX/cDTI_SF_b350_RL_71"
config["slice_index"]   = [3]
config["gui"]["version"] = "v2"
config["segmentation"]["mode"] = "auto"
config["segmentation"]["review"] = True
config["rejection"]["mode"] = "algorithm"
# config["rejection"]["ml_model_path"] = None # should use the bundled model

results = process("Healthy_Volunteer_007", config)