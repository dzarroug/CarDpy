import os
import numpy as np

from cardpy.Data_Sorting              import stacked2sorted
from cardpy.Data_Import               import DICOM_Reader, NifTi_Reader
from cardpy.Data_Saving               import Save_Diffusion_Image_Data, Save_Primary_Eigenvector_Data
from cardpy.Data_Processing.Gibbs         import unrung
from cardpy.Data_Processing.Registration  import register
from cardpy.Data_Processing.Rejection     import shot_rejection
from cardpy.Data_Processing.Respiratory   import respiratory_sorting
from cardpy.Data_Processing.Diffusivity   import ADC_Filter
from cardpy.Data_Processing.Averaging     import average
from cardpy.Data_Processing.Denoising     import denoise
from cardpy.Data_Processing.Interpolation import interp   # confirm exact name
from cardpy.Data_Processing.DTI           import DTI_recon
from cardpy.Data_Processing.cDTI          import cDTI_recon
from cardpy.Tools.Contours                import Myocardial_Mask_Contour_Filler
from cardpy.GUI_Tools.IntERCOMS_Spline    import New_GUI


def process(study_root, config):
    """
    Run the full CarDpy pipeline end to end in one call.
    Stalls for GUI output then continues. 

    """

    out_path = os.path.join(study_root, 'CarDpy_Output')
    os.makedirs(out_path, exist_ok=True)

    op = config["operation_type"]

    # ----- Load -----
    # (DICOM or NIfTI branch + slice selection — fill from notebook cell 3)
    matrix_stacked, bvals_stacked, bvecs_stacked, Header = _load(study_root, config)
    m, b, v = stacked2sorted(matrix_stacked, bvals_stacked, bvecs_stacked)

    # ----- Gibbs -----
    if config["gibbs"]["enabled"]:
        m, b, v = unrung(m, b, v, operation_type=op)

    # ----- Rejection (register first, per notebook) -----
    if config["rejection"]["enabled"]:
        r = config["rejection"]
        m2, b2, v2 = register(m, b, v, registration_algorithm=config["registration"]["algorithm"],
                              sub_registration=_sub_reg(config), temporary_denoising=config["registration"]["temporary_denoising"],
                              operation_type=op)
        m, b, v, Slice_Coordinates, stats, keep = shot_rejection(
            m2, b2, v2,
            NRMSE_threshold=r["nrmse_threshold"], NSSIM_threshold=r["nssim_threshold"],
            zoom=r["zoom"], IntERACT_zoom=r["interact_zoom"], organ=r["organ"],
            operation_type=op, diagnostics_path=out_path)

    # ----- Respiratory -----
    if config["respiratory"]["enabled"]:
        rs = config["respiratory"]
        m2, b2, v2 = register(m, b, v, registration_algorithm=config["registration"]["algorithm"],
                              sub_registration=_sub_reg(config), temporary_denoising=config["registration"]["temporary_denoising"],
                              operation_type=op)
        m, b, v, _ = respiratory_sorting(m2, b2, v2, zoom=rs["zoom"],
                                         IntERACT_zoom=rs["interact_zoom"], organ=rs["organ"],
                                         operation_type=op)

    # ----- Registration (standalone) -----
    if config["registration"]["enabled"]:
        m, b, v = register(m, b, v, registration_algorithm=config["registration"]["algorithm"],
                           sub_registration=_sub_reg(config), temporary_denoising=config["registration"]["temporary_denoising"],
                           operation_type=op)

    # ----- ADC filter -----
    if config["adc_filter"]["enabled"]:
        m, b, v = ADC_Filter(m, b, v)   # confirm exact args

    # ----- Averaging (+ register after, per notebook) -----
    if config["averaging"]["enabled"]:
        m, b, v = average(m, b, v)      # confirm exact args
        m, b, v = register(m, b, v, registration_algorithm=config["registration"]["algorithm"],
                           sub_registration=_sub_reg(config), temporary_denoising=config["registration"]["temporary_denoising"],
                           operation_type=op)

    # ----- Denoising -----
    if config["denoising"]["enabled"]:
        d = config["denoising"]
        m, b, v = denoise(m, b, v, denoising_algorithm=d["algorithm"],
                          numCoils=d["number_of_coils"], operation_type=op)

    # ----- Interpolation / extended matrix -----
    if config["interpolation"]["enabled"]:
        m, b, v = interp(m, b, v)       # confirm exact name + args
    # (extended matrix step — fill from notebook)

    # ===== DTI recon =====
    _, _, Eigenvectors, Standard_DTI_Metrics = DTI_recon(m, b, v, tensor_fit=config["dti"]["tensor_fit"])

    # ===== CONTOURING — STALLS HERE (per slice) =====
    endo_x, endo_y, epi_x, epi_y, antRVIP, infRVIP = [], [], [], [], [], []
    for slc in range(m.shape[2]):
        avg_diff = np.mean(m[:, :, slc, 1:-1, 0], axis=2)
        MD       = Standard_DTI_Metrics['MD'][:, :, slc]
        E1       = Eigenvectors['E1'][:, :, slc, :]
        ex, ey, px, py, aRV, iRV = New_GUI(avg_diff, MD, E1)   # blocks until window closed
        endo_x.append([ex]); endo_y.append([ey]); epi_x.append([px])
        epi_y.append([py]); antRVIP.append([aRV]); infRVIP.append([iRV])

    # ===== Masks + cDTI analysis =====
    myocardium_mask = np.zeros([matrix_stacked.shape[0], matrix_stacked.shape[1], matrix_stacked.shape[2]])
    for slc in range(myocardium_mask.shape[2]):
        temp = np.zeros([matrix_stacked.shape[0], matrix_stacked.shape[1]])
        myocardium_mask[:, :, slc] = Myocardial_Mask_Contour_Filler(
            temp, np.vstack((endo_x[slc][0], endo_y[slc][0])),
            np.vstack((epi_x[slc][0], epi_y[slc][0])))

    c = config["cdti"]
    filt = {'Linear Filter: Outlier StDev': c["helix_angle_filter"]["linear_outlier_stdev"],
            'Spatial Filter: Wall Depth Factor': c["helix_angle_filter"]["spatial_wall_depth_factor"],
            'Spatial Filter: Kernel Size': c["helix_angle_filter"]["spatial_kernel_size"]}
    Cardiac_DTI_Metrics, Epi, Endo, Mask = cDTI_recon(
        myocardium_mask, Eigenvectors, c["num_interp_points"], c["smoothness_level"], filt)

    return {
        "dti_metrics": Standard_DTI_Metrics,
        "eigenvectors": Eigenvectors,
        "cardiac_metrics": Cardiac_DTI_Metrics,
        "mask": Mask,
        "contours": {"endo_x": endo_x, "endo_y": endo_y, "epi_x": epi_x,
                     "epi_y": epi_y, "antRVIP": antRVIP, "infRVIP": infRVIP},
    }