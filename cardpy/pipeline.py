import os
import warnings
import numpy as np

warnings.filterwarnings('ignore', module='dipy')
warnings.filterwarnings('ignore', module='sklearn')
warnings.filterwarnings('ignore', module='skimage')
warnings.filterwarnings('ignore', message='.*Possible precision loss.*')
np.seterr(divide='ignore', invalid='ignore')

import matplotlib.pyplot as plt
from scipy.io import savemat

from cardpy.Data_Sorting                         import stacked2sorted
from cardpy.Data_Import                           import DICOM_Reader, NifTi_Reader
from cardpy.Data_Saving                           import (Save_Diffusion_Image_Data,
                                                          Save_Primary_Eigenvector_Data,
                                                          Save_NRRD_Segmentation)
from cardpy.Data_Processing.Gibbs                 import unrung
from cardpy.Data_Processing.Registration          import register
from cardpy.Data_Processing.Rejection             import shot_rejection
from cardpy.Data_Processing.Respiratory           import respiratory_sorting
from cardpy.Data_Processing.Diffusivity           import ADC_Filter
from cardpy.Data_Processing.Averaging             import average
from cardpy.Data_Processing.Denoising             import denoise
from cardpy.Data_Processing.Interpolation         import zero_filled
from cardpy.Data_Processing.Segmentation_Matrix_DTI import DTI_segmentation_matrix
from cardpy.Data_Processing.DTI                   import DTI_recon
from cardpy.Data_Processing.cDTI                  import cDTI_recon, Endo2Epi_Grid
from cardpy.Colormaps                             import cDTI_Colormaps_Generator
from cardpy.Tools.Contours                        import Myocardial_Mask_Contour_Filler
from cardpy.GUI_Tools.IntERCOMS_Spline            import New_GUI

from scipy.stats           import gaussian_kde
from sklearn.linear_model  import LinearRegression

cDTI_cmaps = cDTI_Colormaps_Generator()


def _save_quantitative_maps(matrix, Standard_DTI_Metrics, Cardiac_DTI_Metrics,
                            Eigenvectors, mask_smoothed_nan, grid_nan, qr_path):
    """Generate and save the 9-panel quantitative cDTI map figure per slice."""
    x_tick_labels = ['Endo', 'Mid', 'Epi']
    x_tick_values = [0, 0.5, 1.0]
    vmax_image    = 100

    for slc in range(mask_smoothed_nan.shape[2]):
        print('Slice %i of %i' % (slc + 1, mask_smoothed_nan.shape[2]))
        fig = plt.figure(figsize=(20, 16), dpi=100)
        fig.patch.set_facecolor('white')
        fig.suptitle('Quantitative cDTI Maps for Slice %i' % (slc + 1), fontsize=20, fontweight='bold')

        plt.subplot(3, 3, 1)
        plt.imshow(matrix[:, :, slc, 0], vmin=0, vmax=vmax_image, cmap='gray')
        plt.imshow(Standard_DTI_Metrics['MD'][:, :, slc] * mask_smoothed_nan[:, :, slc], vmin=0, vmax=3, cmap=cDTI_cmaps['MD'], interpolation='nearest')
        plt.axis('off'); plt.colorbar(); plt.title('Mean Diffusivity', fontsize=18)

        plt.subplot(3, 3, 2)
        plt.imshow(matrix[:, :, slc, 0], vmin=0, vmax=vmax_image, cmap='gray')
        plt.imshow(Standard_DTI_Metrics['FA'][:, :, slc] * mask_smoothed_nan[:, :, slc], vmin=0, vmax=1, cmap=cDTI_cmaps['FA'], interpolation='nearest')
        plt.axis('off'); plt.colorbar(); plt.title('Fractional Anisotropy', fontsize=18)

        plt.subplot(3, 3, 3)
        plt.imshow(matrix[:, :, slc, 0], vmin=0, vmax=vmax_image, cmap='gray')
        plt.imshow(Standard_DTI_Metrics['MO'][:, :, slc] * mask_smoothed_nan[:, :, slc], vmin=-1, vmax=1, cmap=cDTI_cmaps['MO'], interpolation='nearest')
        plt.axis('off'); plt.colorbar(); plt.title('Mode', fontsize=18)

        plt.subplot(3, 3, 4)
        plt.imshow(matrix[:, :, slc, 0], vmin=0, vmax=vmax_image, cmap='gray')
        plt.colorbar()
        tmp1 = np.expand_dims(mask_smoothed_nan[:, :, slc], axis=2)
        tmp2 = abs(Eigenvectors['E1'][:, :, slc, :])
        tmp3 = np.concatenate((tmp2, tmp1), axis=2)
        plt.imshow(tmp3, interpolation='nearest')
        plt.axis('off'); plt.title('Primary Eigenvector Vector', fontsize=18)

        plt.subplot(3, 3, 5)
        plt.imshow(matrix[:, :, slc, 0], vmin=0, vmax=vmax_image, cmap='gray')
        plt.imshow(Cardiac_DTI_Metrics['HASF'][:, :, slc] * mask_smoothed_nan[:, :, slc], vmin=-90, vmax=90, cmap=cDTI_cmaps['HA'], interpolation='nearest')
        plt.axis('off'); plt.colorbar(); plt.title('Helix Angle', fontsize=18)

        plt.subplot(3, 3, 6)
        E2E_data = grid_nan[:, :, slc].flatten()
        E2E_data = E2E_data[~(np.isnan(grid_nan[:, :, slc].flatten()))]
        x = E2E_data
        HA_data = Cardiac_DTI_Metrics['HASF'][:, :, slc].flatten()
        HA_data = HA_data[~(np.isnan(grid_nan[:, :, slc].flatten()))]
        y = HA_data
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        density = plt.scatter(x, y, c=z, s=50, cmap='hot')
        model = LinearRegression().fit(x[:, np.newaxis], y[:, np.newaxis])
        y_predicted = model.intercept_ + model.coef_ * x[:, np.newaxis]
        plt.plot(x[:, np.newaxis], y_predicted, 'b-', label='Linear Regression')
        plt.legend()
        plt.title('Transmurality of Helix Angle', fontsize=18)
        plt.xticks(x_tick_values, x_tick_labels, fontsize=12); plt.yticks(fontsize=12)
        cbar = plt.colorbar(density); cbar.set_label(label='Density', fontsize=12)
        cbar.ax.tick_params(labelsize=12); plt.clim([0, 0.15])
        plt.ylim([-90, 90]); plt.xlim([0, 1])

        plt.subplot(3, 3, 7)
        plt.imshow(matrix[:, :, slc, 0], vmin=0, vmax=vmax_image, cmap='gray')
        plt.imshow(Cardiac_DTI_Metrics['E2A'][:, :, slc] * mask_smoothed_nan[:, :, slc], vmin=0, vmax=90, cmap=cDTI_cmaps['absE2A'], interpolation='nearest')
        plt.axis('off'); plt.colorbar(); plt.title('Absolute E2 Angle', fontsize=18)

        plt.subplot(3, 3, 8)
        plt.imshow(matrix[:, :, slc, 0], vmin=0, vmax=vmax_image, cmap='gray')
        plt.imshow(Cardiac_DTI_Metrics['TA'][:, :, slc] * mask_smoothed_nan[:, :, slc], vmin=-90, vmax=90, cmap=cDTI_cmaps['TA'], interpolation='nearest')
        plt.axis('off'); plt.colorbar(); plt.title('Transverse Angle', fontsize=18)

        plt.subplot(3, 3, 9)
        x2 = E2E_data
        y2 = Cardiac_DTI_Metrics['TA'][:, :, slc].flatten()
        y2 = y2[~(np.isnan(grid_nan[:, :, slc].flatten()))]
        xy2 = np.vstack([x2, y2])
        z2 = gaussian_kde(xy2)(xy2)
        idx = z2.argsort()
        x2, y2, z2 = x2[idx], y2[idx], z2[idx]
        density = plt.scatter(x2, y2, c=z2, s=50, cmap='hot')
        model = LinearRegression().fit(x2[:, np.newaxis], y2[:, np.newaxis])
        y2_predicted = model.intercept_ + model.coef_ * x2[:, np.newaxis]
        plt.plot(x2[:, np.newaxis], y2_predicted, 'b-', label='Linear Regression')
        plt.legend()
        plt.title('Transmurality of Transverse Angle', fontsize=18)
        plt.xticks(x_tick_values, x_tick_labels, fontsize=12); plt.yticks(fontsize=12)
        cbar = plt.colorbar(density); cbar.set_label(label='Density', fontsize=12)
        cbar.ax.tick_params(labelsize=12); plt.clim([0, 0.15])
        plt.ylim([-90, 90]); plt.xlim([0, 1])

        plt.subplots_adjust(hspace=0.35, wspace=0.35)
        string = 'Quantitative_cDTI_Maps_' + str(slc + 1).zfill(2) + ' Slice.png'
        plt.savefig(os.path.join(qr_path, string))
        plt.close(fig)


def _load(study_root, config):
    """Load DICOM or NIfTI data and apply slice selection."""
    if config["data_type"] == "DICOM":
        dicom_folder = os.path.join(study_root, config["dicom_subpath"])
        matrix, bvals, bvecs, Header = DICOM_Reader(dicom_folder, info=config["dicom_reader_info"])
    else:
        nifti_dir = os.path.join(study_root, "NifTis")
        matrix, bvals, bvecs, Header, _, _ = NifTi_Reader(
            os.path.join(nifti_dir, "data.nii"),
            os.path.join(nifti_dir, "data.bvals"),
            os.path.join(nifti_dir, "data.bvecs"),
            os.path.join(nifti_dir, "data.header"))

    if config["slice_index"] is not None:
        matrix = matrix[:, :, config["slice_index"], :]

    return matrix, bvals, bvecs, Header


def process(study_root, config, save=True):
    """
    Run the full CarDpy pipeline end to end in one call.
    Stalls at the contouring GUI (once per slice) until each window is closed,
    then resumes automatically. Writes stage outputs to CarDpy_Output when
    save=True. Returns a dict of results.
    """
    out_path = os.path.join(study_root, "CarDpy_Output")
    os.makedirs(out_path, exist_ok=True)

    op       = config["operation_type"]
    reg_alg  = config["registration"]["algorithm"]
    reg_den  = config["registration"]["temporary_denoising"]

    counter = {"n": -1}
    def _save_stage(name, mm, bb, vv, Header, folder_name=None):
        if not save:
            return
        counter["n"] += 1
        folder      = str(counter["n"]).zfill(2) + '_' + (folder_name if folder_name is not None else name)
        output_path = os.path.join(out_path, folder)
        os.makedirs(output_path, exist_ok=True)
        Save_Diffusion_Image_Data(output_path, name, Header, mm, bb, vv)

    # ----- Load + sort -----
    matrix_stacked, bvals_stacked, bvecs_stacked, Header = _load(study_root, config)
    print('Loaded matrix shape:', matrix_stacked.shape, '| b-values:', bvals_stacked.shape, '| b-vectors:', bvecs_stacked.shape)
    m, b, v = stacked2sorted(matrix_stacked, bvals_stacked, bvecs_stacked)
    _save_stage('Original', m, b, v, Header)

    # ----- Gibbs -----
    if config["gibbs"]["enabled"]:
        print('Gibbs ringing removal mode is on.')
        m, b, v = unrung(m, b, v, operation_type=op)
        _save_stage('Unrung', m, b, v, Header)

    # ----- Rejection -----
    if config["rejection"]["enabled"]:
        print('Rejection mode is on.')
        print('Re-registering data prior to data rejection.')
        r = config["rejection"]
        m2, b2, v2 = register(m, b, v, registration_algorithm=reg_alg,
                              temporary_denoising=reg_den, operation_type=op)
        m, b, v, Slice_Coordinates, stats, keep = shot_rejection(
            m2, b2, v2,
            NRMSE_threshold=r["nrmse_threshold"], NSSIM_threshold=r["nssim_threshold"],
            zoom=r["zoom"], IntERACT_zoom=r["interact_zoom"], organ=r["organ"],
            operation_type=op, diagnostics_path=out_path)
        _save_stage('Rejected', m, b, v, Header)
        if save:
            diag_dir = os.path.join(out_path, '13_Diagnostics')
            os.makedirs(diag_dir, exist_ok=True)
            with open(os.path.join(diag_dir, 'Crop_Coordinates.txt'), 'w') as f:
                f.write('Heart crop coordinates (per slice)\n')
                f.write('Format: x_start, x_end, y_start, y_end\n\n')
                x_start, x_end, y_start, y_end = Slice_Coordinates
                for s_i in range(len(x_start)):
                    f.write('Slice %d: x_start=%s, x_end=%s, y_start=%s, y_end=%s\n'
                            % (s_i + 1, x_start[s_i], x_end[s_i], y_start[s_i], y_end[s_i]))
            with open(os.path.join(diag_dir, 'Rejection_Statistics.txt'), 'w') as f:
                f.write('Rejection statistics\n')
                f.write('Acceptance rate = %% of averages kept for each slice/direction.\n')
                f.write('Rejected average indices are 0-based within each slice/direction.\n\n')
                n_slc, n_dif, n_avg = keep.shape
                total_rejected = 0
                for s_i in range(n_slc):
                    for dif in range(n_dif):
                        rejected_idx = np.where(keep[s_i, dif, :] == 0)[0]
                        n_rej = int(len(rejected_idx))
                        total_rejected += n_rej
                        rate = stats[s_i, dif]
                        idx_str = ', '.join(str(int(i)) for i in rejected_idx) if n_rej > 0 else 'none'
                        f.write('Slice %d, Direction %d: %.1f%% accepted | %d rejected (indices: %s)\n'
                                % (s_i + 1, dif + 1, rate, n_rej, idx_str))
                f.write('\n--- Summary ---\n')
                f.write('Total averages rejected: %d\n' % total_rejected)
                f.write('Total averages examined: %d\n' % keep.size)
                f.write('Overall mean acceptance rate: %.1f%%\n' % stats.mean())

    # ----- Respiratory -----
    if config["respiratory"]["enabled"]:
        print('Respiratory sorting mode is on.')
        print('Re-registering data prior data reordering.')
        rs = config["respiratory"]
        m2, b2, v2 = register(m, b, v, registration_algorithm=reg_alg,
                              temporary_denoising=reg_den, operation_type=op)
        m, b, v, _ = respiratory_sorting(m2, b2, v2, zoom=rs["zoom"],
                                         IntERACT_zoom=rs["interact_zoom"], organ=rs["organ"],
                                         operation_type=op)
        _save_stage('Respiratory_Ordered', m, b, v, Header, folder_name='Respiratory_Sorted')

    # ----- Registration (standalone) -----
    if config["registration"]["enabled"]:
        print('Registration mode is on.')
        m, b, v = register(m, b, v, registration_algorithm=reg_alg,
                           temporary_denoising=reg_den, operation_type=op)
        _save_stage('Registered', m, b, v, Header)

    # ----- ADC filter -----
    if config["adc_filter"]["enabled"]:
        print('ADC Filter mode is on.')
        m, b, v = ADC_Filter(m, b, v, operation_type=op)
        _save_stage('ADC_Filtered', m, b, v, Header, folder_name='Diffusivity_Filtered')

    # ----- Averaging (register after) -----
    if config["averaging"]["enabled"]:
        print('Averaging mode is on.')
        m, b, v = average(m, b, v, operation_type=op)
        print('Re-registering data after data averaging.')
        m, b, v = register(m, b, v, registration_algorithm=reg_alg,
                           temporary_denoising=reg_den, operation_type=op)
        _save_stage('Averaged', m, b, v, Header)

    # ----- Denoising -----
    if config["denoising"]["enabled"]:
        print('Denoising mode is on.')
        d = config["denoising"]
        m, b, v = denoise(m, b, v, denoising_algorithm=d["algorithm"],
                          numCoils=d["number_of_coils"], operation_type=op)
        _save_stage('Denoised', m, b, v, Header)

    # ----- Interpolation -----
    if config["interpolation"]["enabled"]:
        print('Interpolating mode is on.')
        m, b, v = zero_filled(m, b, v, operation_type=op)
        _save_stage('Interpolated', m, b, v, Header)

    # ----- Extended matrix (save-only branch; does NOT feed DTI/contouring) -----
    if config["extended_matrix"]["enabled"] and save:
        print('Extended matrix mode is on.')
        sm, sb, sv          = DTI_segmentation_matrix(m, b, v)
        ext_m, ext_b, ext_v = stacked2sorted(sm, sb, sv)
        _save_stage('Extended_Matrix', ext_m, ext_b, ext_v, Header)

    # ----- Primary Eigenvector -----
    if config.get("e1", {}).get("enabled", False) and save:
        print('Primary Eigenvector mode is on.')
        counter["n"] += 1
        folder      = str(counter["n"]).zfill(2) + '_Primary_Eigenvector'
        output_path = os.path.join(out_path, folder)
        os.makedirs(output_path, exist_ok=True)
        Save_Primary_Eigenvector_Data(output_path, Header, m, b, v)

    # ===== DTI recon =====
    _, _, Eigenvectors, Standard_DTI_Metrics = DTI_recon(m, b, v, tensor_fit=config["dti"]["tensor_fit"])

    # ===== Contouring =====
    endo_x, endo_y, epi_x, epi_y, antRVIP, infRVIP = [], [], [], [], [], []
    for slc in range(m.shape[2]):
        avg_diff = np.mean(m[:, :, slc, 1:-1, 0], axis=2)
        MD       = Standard_DTI_Metrics['MD'][:, :, slc]
        E1       = Eigenvectors['E1'][:, :, slc, :]
        ex, ey, px, py, aRV, iRV = New_GUI(avg_diff, MD, E1)
        endo_x.append([ex]); endo_y.append([ey]); epi_x.append([px])
        epi_y.append([py]); antRVIP.append([aRV]); infRVIP.append([iRV])

    # ===== Masks =====
    myocardium_mask = np.zeros([m.shape[0], m.shape[1], m.shape[2]])
    for slc in range(myocardium_mask.shape[2]):
        temp = np.zeros([m.shape[0], m.shape[1]])
        myocardium_mask[:, :, slc] = Myocardial_Mask_Contour_Filler(
            temp, np.vstack((endo_x[slc][0], endo_y[slc][0])),
            np.vstack((epi_x[slc][0], epi_y[slc][0])))

    # ----- Save segmentation -----
    if save:
        seg_path = os.path.join(out_path, '11_Segmentation')
        os.makedirs(seg_path, exist_ok=True)
        Save_NRRD_Segmentation(myocardium_mask, Header, seg_path, 'LV_Myocardium')

    # ===== cDTI analysis =====
    c    = config["cdti"]
    filt = {'Linear Filter: Outlier StDev':      c["helix_angle_filter"]["linear_outlier_stdev"],
            'Spatial Filter: Wall Depth Factor':  c["helix_angle_filter"]["spatial_wall_depth_factor"],
            'Spatial Filter: Kernel Size':        c["helix_angle_filter"]["spatial_kernel_size"]}
    Cardiac_DTI_Metrics, Epi, Endo, Mask = cDTI_recon(
        myocardium_mask, Eigenvectors, c["num_interp_points"], c["smoothness_level"], filt)

    # ----- Post-processing for maps -----
    mask_smoothed_nan = np.copy(Mask).astype('float')
    mask_smoothed_nan[mask_smoothed_nan == 0] = np.nan
    grid_nan = np.copy(Endo2Epi_Grid(np.copy(Mask)))
    grid_nan = grid_nan * mask_smoothed_nan
    grid_nan = np.clip(grid_nan, 0.0, 1)

    # ----- Save quantitative results-----
    if save:
        qr_path = os.path.join(out_path, '12_Quantitative_Results')
        os.makedirs(qr_path, exist_ok=True)
        savemat(os.path.join(qr_path, 'Standard_DTI_Metrics.mat'), Standard_DTI_Metrics)
        savemat(os.path.join(qr_path, 'cDTI_Metrics.mat'),         Cardiac_DTI_Metrics)
        savemat(os.path.join(qr_path, 'DTI_Eigenvectors.mat'),     Eigenvectors)
        _save_quantitative_maps(m, Standard_DTI_Metrics, Cardiac_DTI_Metrics,
                                Eigenvectors, mask_smoothed_nan, grid_nan, qr_path)

    return {
        "dti_metrics":     Standard_DTI_Metrics,
        "eigenvectors":    Eigenvectors,
        "cardiac_metrics": Cardiac_DTI_Metrics,
        "mask":            Mask,
        "header":          Header,
        "contours":        {"endo_x": endo_x, "endo_y": endo_y, "epi_x": epi_x,
                            "epi_y": epi_y, "antRVIP": antRVIP, "infRVIP": infRVIP},
    }