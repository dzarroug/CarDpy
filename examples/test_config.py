from cardpy.pipeline import process
from cardpy.config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["data_type"]     = "DICOM"
config["dicom_subpath"] = "02_cDTI/SAX/trap_2x2x8_b400_24_MR"
config["slice_index"]   = [0]
config["gui"]["version"] = "v2"
config["segmentation"]["mode"] = "auto"
# config["segmentation"]["review"] = True
config["rejection"]["mode"] = "manual"
# config["rejection"]["ml_model_path"] = None # should use the bundled model

results = process("/Users/dahniazarroug/Desktop/Dahnia/V001", config)