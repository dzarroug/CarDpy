import os
import json
import warnings
import numpy as np

warnings.filterwarnings('ignore', module='dipy')
warnings.filterwarnings('ignore', module='sklearn')
warnings.filterwarnings('ignore', module='skimage')
warnings.filterwarnings('ignore', message='.*Possible precision loss.*')
np.seterr(divide='ignore', invalid='ignore')

import matplotlib.pyplot as plt
from scipy.io import savemat

from cardpy.Data_Sorting                         import (dictionary2sorted, stacked2dictionary,
                                                         sorted2dictionary)
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
from cardpy.Data_Processing.HAP                   import helix_angle_pitch
from cardpy.Colormaps                             import cDTI_Colormaps_Generator
from cardpy.Tools.Contours                        import Myocardial_Mask_Contour_Filler
from cardpy.GUI_Tools.IntERCOMS_Spline            import New_GUI

from scipy.stats           import gaussian_kde
from sklearn.linear_model  import LinearRegression

cDTI_cmaps = cDTI_Colormaps_Generator()


def _plot_transmural(position, angles, title):
    finite = np.isfinite(position) & np.isfinite(angles)
    x, y = position[finite], angles[finite]
    if x.size >= 4 and np.ptp(x) > 0:
        try:
            density = gaussian_kde(np.vstack((x, y)))(np.vstack((x, y)))
        except (np.linalg.LinAlgError, ValueError):
            plt.scatter(x, y, s=50, color='steelblue')
        else:
            order = density.argsort()
            points = plt.scatter(x[order], y[order], c=density[order], s=50, cmap='hot')
            plt.colorbar(points).set_label('Density')
            plt.clim([0, 0.15])
        model = LinearRegression().fit(x[:, None], y)
        line = np.linspace(x.min(), x.max(), 100)
        plt.plot(line, model.predict(line[:, None]), 'b-', label='Linear Regression')
        plt.legend()
    else:
        plt.text(.5, .5, 'Insufficient finite wall-depth samples',
                 transform=plt.gca().transAxes, ha='center')
    plt.title(title, fontsize=18)
    plt.xticks([0, .5, 1], ['Endo', 'Mid', 'Epi'], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([-90, 90]); plt.xlim([0, 1])


def _save_quantitative_maps(matrix, Standard_DTI_Metrics, Cardiac_DTI_Metrics,
                            Eigenvectors, mask_smoothed_nan, grid_nan, qr_path):
    """Generate and save the 9-panel quantitative cDTI map figure per slice."""
    vmax_image    = 100

    for slc in range(mask_smoothed_nan.shape[2]):
        print('Slice %i of %i' % (slc + 1, mask_smoothed_nan.shape[2]))
        background = np.nanmean(np.abs(matrix[:, :, slc, :, :]), axis=(2, 3))
        fig = plt.figure(figsize=(20, 16), dpi=100)
        fig.patch.set_facecolor('white')
        fig.suptitle('Quantitative cDTI Maps for Slice %i' % (slc + 1), fontsize=20, fontweight='bold')

        plt.subplot(3, 3, 1)
        plt.imshow(background, vmin=0, vmax=vmax_image, cmap='gray')
        plt.imshow(Standard_DTI_Metrics['MD'][:, :, slc] * mask_smoothed_nan[:, :, slc], vmin=0, vmax=3, cmap=cDTI_cmaps['MD'], interpolation='nearest')
        plt.axis('off'); plt.colorbar(); plt.title('Mean Diffusivity', fontsize=18)

        plt.subplot(3, 3, 2)
        plt.imshow(background, vmin=0, vmax=vmax_image, cmap='gray')
        plt.imshow(Standard_DTI_Metrics['FA'][:, :, slc] * mask_smoothed_nan[:, :, slc], vmin=0, vmax=1, cmap=cDTI_cmaps['FA'], interpolation='nearest')
        plt.axis('off'); plt.colorbar(); plt.title('Fractional Anisotropy', fontsize=18)

        plt.subplot(3, 3, 3)
        plt.imshow(background, vmin=0, vmax=vmax_image, cmap='gray')
        plt.imshow(Standard_DTI_Metrics['MO'][:, :, slc] * mask_smoothed_nan[:, :, slc], vmin=-1, vmax=1, cmap=cDTI_cmaps['MO'], interpolation='nearest')
        plt.axis('off'); plt.colorbar(); plt.title('Mode', fontsize=18)

        plt.subplot(3, 3, 4)
        plt.imshow(background, vmin=0, vmax=vmax_image, cmap='gray')
        plt.colorbar()
        tmp1 = np.expand_dims(mask_smoothed_nan[:, :, slc], axis=2)
        tmp2 = abs(Eigenvectors['E1'][:, :, slc, :])
        tmp3 = np.concatenate((tmp2, tmp1), axis=2)
        plt.imshow(tmp3, interpolation='nearest')
        plt.axis('off'); plt.title('Primary Eigenvector Vector', fontsize=18)

        plt.subplot(3, 3, 5)
        plt.imshow(background, vmin=0, vmax=vmax_image, cmap='gray')
        plt.imshow(Cardiac_DTI_Metrics['HASF'][:, :, slc] * mask_smoothed_nan[:, :, slc], vmin=-90, vmax=90, cmap=cDTI_cmaps['HA'], interpolation='nearest')
        plt.axis('off'); plt.colorbar(); plt.title('Helix Angle', fontsize=18)

        plt.subplot(3, 3, 6)
        _plot_transmural(grid_nan[:, :, slc], Cardiac_DTI_Metrics['HASF'][:, :, slc],
                         'Transmurality of Helix Angle')

        plt.subplot(3, 3, 7)
        plt.imshow(background, vmin=0, vmax=vmax_image, cmap='gray')
        plt.imshow(Cardiac_DTI_Metrics['E2A'][:, :, slc] * mask_smoothed_nan[:, :, slc], vmin=0, vmax=90, cmap=cDTI_cmaps['absE2A'], interpolation='nearest')
        plt.axis('off'); plt.colorbar(); plt.title('Absolute E2 Angle', fontsize=18)

        plt.subplot(3, 3, 8)
        plt.imshow(background, vmin=0, vmax=vmax_image, cmap='gray')
        plt.imshow(Cardiac_DTI_Metrics['TA'][:, :, slc] * mask_smoothed_nan[:, :, slc], vmin=-90, vmax=90, cmap=cDTI_cmaps['TA'], interpolation='nearest')
        plt.axis('off'); plt.colorbar(); plt.title('Transverse Angle', fontsize=18)

        plt.subplot(3, 3, 9)
        _plot_transmural(grid_nan[:, :, slc], Cardiac_DTI_Metrics['TA'][:, :, slc],
                         'Transmurality of Transverse Angle')

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
        indices = np.asarray(config["slice_index"])
        resolved = np.arange(matrix.shape[2])[indices]
        if config["registration"].get("mode", "2D") == '3D' and (
                len(resolved) < 2 or np.any(np.diff(resolved) != 1)):
            raise ValueError('3D registration requires consecutive slices in acquisition order.')
        matrix = matrix[:, :, indices, :]

    return matrix, bvals, bvecs, Header


def process(study_root, config, save=True, segmentation=None):
    """
    Run the full CarDpy pipeline end to end in one call.
    Stalls at the contouring GUI (once per slice) until each window is closed,
    then resumes automatically. Writes stage outputs to CarDpy_Output when
    save=True. Returns a dict of results.
    For stationary phantom scans on the same image grid, pass the first result's
    segmentation dict to reuse its contours and display crops without opening
    contour windows. A saved 11_Segmentation/Segmentation.json can also be loaded
    with json.load and passed here.
    """
    out_path = config.get("output", {}).get("directory") or os.path.join(study_root, "CarDpy_Output")
    out_path = os.path.abspath(os.path.expanduser(os.fspath(out_path)))
    if save:
        os.makedirs(out_path, exist_ok=True)

    op       = config["operation_type"]
    reg_alg  = config["registration"]["algorithm"]
    reg_den  = config["registration"]["temporary_denoising"]
    reg_mode = config["registration"].get("mode", "2D")
    if reg_mode not in ('2D', '3D'):
        raise ValueError("registration.mode must be '2D' or '3D'.")

    if reg_mode == '3D' and reg_alg != 'Rigid':
        raise ValueError('3D registration supports only Rigid; set registration.algorithm="Rigid".')

    gui      = config.get("gui", {})
    gui_version = gui.get("version", "legacy")
    if gui_version not in ('legacy', 'v2'):
        raise ValueError("gui.version must be 'legacy' or 'v2'.")
    segmentation_crop = None
    crop_shape = None

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
    input_encodings = stacked2dictionary(matrix_stacked, bvals_stacked, bvecs_stacked)
    repetition_counts = [encoding['images'].shape[3] for encoding in input_encodings.values()]
    shells_by_vector = {}
    for encoding in input_encodings.values():
        vector_key = tuple(np.round(np.asarray(encoding['bvec'], dtype=float), 5))
        shells_by_vector.setdefault(vector_key, set()).add(round(float(encoding['bval']), 5))
    shared_vectors_across_shells = any(len(shells) > 1 for shells in shells_by_vector.values())
    dictionary_processing = (
        len(set(repetition_counts)) > 1 or shared_vectors_across_shells
    )
    if dictionary_processing:
        reasons = []
        if len(set(repetition_counts)) > 1:
            reasons.append('unequal repetitions')
        if shared_vectors_across_shells:
            reasons.append('b-vectors reused across shells')
        print('Using dictionary-aware processing (%s).' % ', '.join(reasons))
        m, b, v = dictionary2sorted(input_encodings, mode='individual')
    else:
        m, b, v = dictionary2sorted(input_encodings)
    if reg_mode == '3D' and m.shape[2] < 2:
        raise ValueError('3D registration requires multiple contiguous slices; use 2D for a single slice.')
    input_shape = m.shape[:3]
    spacing = tuple(float(Header.get(axis + ' Resolution', 1.0)) for axis in ('X', 'Y', 'Z'))
    b0_threshold = config['dti'].get('b0_threshold', 0)
    register_options = dict(registration_algorithm=reg_alg, temporary_denoising=reg_den,
                            operation_type=op, mode=reg_mode, voxel_size=spacing,
                            b0_threshold=b0_threshold)
    _save_stage('Original', m, b, v, Header)

    aligned = False

    # ----- Gibbs -----
    if config["gibbs"]["enabled"]:
        print('Gibbs ringing removal mode is on.')
        m, b, v = unrung(m, b, v, operation_type=op)
        _save_stage('Unrung', m, b, v, Header)

    # ----- Rejection -----
    if config["rejection"]["enabled"]:
        print('Rejection mode is on.')
        r = config["rejection"]
        print('Aligning input images.')
        m2, b2, v2 = register(m, b, v, **register_options)
        aligned = True
        rejection_input = sorted2dictionary(m2, b2, v2) if dictionary_processing else m2
        m, b, v, Slice_Coordinates, stats, keep = shot_rejection(
            rejection_input, b2, v2,
            NRMSE_threshold=r["nrmse_threshold"], NSSIM_threshold=r["nssim_threshold"],
            zoom=r["zoom"], IntERACT_zoom=r["interact_zoom"], organ=r["organ"],
            operation_type=op, diagnostics_path=out_path, gui_version=gui_version,
            save_diagnostics=save and config.get("output", {}).get("save_diagnostics", True))
        if gui_version == 'v2' and len(Slice_Coordinates) == 4:
            segmentation_crop = Slice_Coordinates
            crop_shape = m2.shape[:2]
        if dictionary_processing:
            m, b, v = dictionary2sorted(m, mode='individual')
        _save_stage('Rejected', m, b, v, Header)
        if save:
            diag_dir = os.path.join(out_path, '13_Diagnostics')
            os.makedirs(diag_dir, exist_ok=True)
            with open(os.path.join(diag_dir, 'Crop_Coordinates.txt'), 'w') as f:
                f.write('Heart crop coordinates (per slice)\n')
                x_start, x_end, y_start, y_end = Slice_Coordinates
                for s_i in range(len(x_start)):
                    f.write('Slice %d: x_start=%s, x_end=%s, y_start=%s, y_end=%s\n'
                            % (s_i + 1, x_start[s_i], x_end[s_i], y_start[s_i], y_end[s_i]))
            with open(os.path.join(diag_dir, 'Rejection_Statistics.txt'), 'w') as f:
                f.write('Rejection statistics\n')
                f.write('Acceptance rate = %% of averages kept for each slice/direction.\n')
                f.write('Rejected average indices are 0-based within each slice/direction.\n\n')
                total_rejected = 0
                if isinstance(keep, dict):
                    total_examined = 0
                    all_rates = []
                    for encoding_index, (encoding_key, encoding_keep) in enumerate(keep.items()):
                        encoding_stats = stats[encoding_key]
                        bval, gx, gy, gz = encoding_key
                        total_examined += encoding_keep.size
                        all_rates.extend(encoding_stats.tolist())
                        for s_i in range(encoding_keep.shape[0]):
                            rejected_idx = np.where(encoding_keep[s_i] == 0)[0]
                            n_rej = int(len(rejected_idx))
                            total_rejected += n_rej
                            idx_str = ', '.join(str(int(i)) for i in rejected_idx) if n_rej > 0 else 'none'
                            f.write('Slice %d, Encoding %d (b=%g, vector=[%g, %g, %g]): '
                                    '%.1f%% accepted | %d rejected (indices: %s)\n'
                                    % (s_i + 1, encoding_index + 1, bval, gx, gy, gz,
                                       encoding_stats[s_i], n_rej, idx_str))
                    mean_acceptance = float(np.mean(all_rates)) if all_rates else float('nan')
                else:
                    n_slc, n_dif, n_avg = keep.shape
                    total_examined = keep.size
                    mean_acceptance = float(stats.mean())
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
                f.write('Total averages examined: %d\n' % total_examined)
                f.write('Overall mean acceptance rate: %.1f%%\n' % mean_acceptance)

    # ----- Respiratory -----
    if config["respiratory"]["enabled"]:
        print('Respiratory sorting mode is on.')
        rs = config["respiratory"]
        if aligned:
            m2, b2, v2 = m, b, v
        else:
            print('Aligning input images.')
            m2, b2, v2 = register(m, b, v, **register_options)
            aligned = True
        if dictionary_processing:
            encodings = sorted2dictionary(m2, b2, v2)
            respiratory_crop = None
            for encoding in encodings.values():
                if encoding['images'].shape[3] < 2:
                    continue
                ordered, _, _, respiratory_crop = respiratory_sorting(
                    encoding['images'][:, :, :, np.newaxis, :],
                    np.asarray([encoding['bval']]), np.asarray([encoding['bvec']]),
                    zoom=rs["zoom"], IntERACT_zoom=rs["interact_zoom"],
                    organ=rs["organ"], operation_type=op, gui_version=gui_version,
                    crop_coordinates=respiratory_crop)
                encoding['images'] = ordered[:, :, :, 0, :]
            m, b, v = dictionary2sorted(encodings, mode='individual')
        else:
            m, b, v, _ = respiratory_sorting(m2, b2, v2, zoom=rs["zoom"],
                                             IntERACT_zoom=rs["interact_zoom"], organ=rs["organ"],
                                             operation_type=op, gui_version=gui_version)
        _save_stage('Respiratory_Ordered', m, b, v, Header, folder_name='Respiratory_Sorted')

    # ----- Registration (standalone) -----
    if config["registration"]["enabled"]:
        print('Registration mode is on.')
        if not aligned:
            m, b, v = register(m, b, v, **register_options)
            aligned = True
        _save_stage('Registered', m, b, v, Header)

    # ----- ADC filter -----
    if config["adc_filter"]["enabled"]:
        print('ADC Filter mode is on.')
        filter_input = sorted2dictionary(m, b, v) if dictionary_processing else m
        m, b, v = ADC_Filter(filter_input, b, v, operation_type=op)
        if dictionary_processing:
            m, b, v = dictionary2sorted(m, mode='individual')
        _save_stage('ADC_Filtered', m, b, v, Header, folder_name='Diffusivity_Filtered')

    # ----- Averaging (register after) -----
    if config["averaging"]["enabled"]:
        print('Averaging mode is on.')
        if dictionary_processing:
            m, b, v = dictionary2sorted(sorted2dictionary(m, b, v),
                                         mode='average', operation_type=op)
        else:
            m, b, v = average(m, b, v, operation_type=op)
        if config["registration"]["enabled"] and not aligned:
            m, b, v = register(m, b, v, **register_options)
            aligned = True
        _save_stage('Averaged', m, b, v, Header)

    # ----- Denoising -----
    if config["denoising"]["enabled"]:
        print('Denoising mode is on.')
        d = config["denoising"]
        m, b, v = denoise(m, b, v, denoising_algorithm=d["algorithm"],
                          numCoils=d["number_of_coils"], operation_type=op, b0_threshold=b0_threshold)
        _save_stage('Denoised', m, b, v, Header)

    # ----- Interpolation -----
    if config["interpolation"]["enabled"]:
        print('Interpolating mode is on.')
        m, b, v = zero_filled(m, b, v, operation_type=op)
        Header = dict(Header)
        for axis, dim in (('X', 0), ('Y', 1)):
            Header[axis + ' Resolution'] = spacing[dim] * input_shape[dim] / m.shape[dim]
        _save_stage('Interpolated', m, b, v, Header)

    # ----- Extended matrix (save-only branch; does NOT feed DTI/contouring) -----
    if config["extended_matrix"]["enabled"] and save:
        print('Extended matrix mode is on.')
        sm, sb, sv          = DTI_segmentation_matrix(m, b, v, tensor_fit=config["dti"]["tensor_fit"], b0_threshold=b0_threshold)
        ext_m, ext_b, ext_v = sm[..., np.newaxis], sb, sv
        _save_stage('Extended_Matrix', ext_m, ext_b, ext_v, Header)

    # ----- Primary Eigenvector -----
    if config.get("e1", {}).get("enabled", False) and save:
        print('Primary Eigenvector mode is on.')
        counter["n"] += 1
        folder      = str(counter["n"]).zfill(2) + '_Primary_Eigenvector'
        output_path = os.path.join(out_path, folder)
        os.makedirs(output_path, exist_ok=True)
        Save_Primary_Eigenvector_Data(output_path, Header, m, b, v,
                                     tensor_fit=config["dti"]["tensor_fit"], b0_threshold=b0_threshold)

    # ===== DTI recon =====
    _, _, Eigenvectors, Standard_DTI_Metrics = DTI_recon(m, b, v, tensor_fit=config["dti"]["tensor_fit"], b0_threshold=b0_threshold)

    # ===== Contouring (optionally reuse a stationary phantom segmentation) =====
    image_shape = list(m.shape[:3])
    voxel_spacing = [float(Header.get(axis + ' Resolution', spacing[index]))
                     for index, axis in enumerate(('X', 'Y', 'Z'))]
    selected_slices = config.get('slice_index')
    if selected_slices is not None:
        selected_slices = [int(index) for index in selected_slices]
    if segmentation is not None:
        if (list(segmentation['image_shape']) != image_shape
                or segmentation['slice_index'] != selected_slices
                or not np.allclose(segmentation['voxel_spacing'], voxel_spacing,
                                   rtol=1e-5, atol=1e-8)):
            raise ValueError('Reused segmentation requires matching image shape, voxel spacing and slice selection. Redraw the phantom mask.')
        from cardpy.GUI_Tools._image_controls import crop_bounds
        contours = segmentation['contours']
        endo_x, endo_y = contours['endo_x'], contours['endo_y']
        epi_x, epi_y = contours['epi_x'], contours['epi_y']
        antRVIP, infRVIP = contours['antRVIP'], contours['infRVIP']
        if any(len(values) != m.shape[2] for values in
               (endo_x, endo_y, epi_x, epi_y, antRVIP, infRVIP, segmentation['crop_bounds'])):
            raise ValueError('Reused segmentation must contain contours and a crop for every slice.')
        display_crops = [crop_bounds(bounds, m.shape[:2]) for bounds in segmentation['crop_bounds']]
        print('Reusing phantom contours and display crops.')
    else:
        display_crops = []
        endo_x, endo_y, epi_x, epi_y, antRVIP, infRVIP = [], [], [], [], [], []
        for slc in range(m.shape[2]):
            diffusion = b > np.min(b)
            avg_diff = np.nanmean(np.abs(m[:, :, slc, diffusion, :]), axis=(2, 3))
            MD       = Standard_DTI_Metrics['MD'][:, :, slc]
            E1       = Eigenvectors['E1'][:, :, slc, :]
            if gui_version == 'v2':
                bounds = None
                if gui.get("segmentation_zoom", True) and segmentation_crop is not None:
                    from cardpy.GUI_Tools._image_controls import scale_crop
                    bounds = scale_crop([coordinates[slc] for coordinates in segmentation_crop],
                                        crop_shape, m.shape[:2])
                contour, bounds = New_GUI(
                    avg_diff, MD, E1, gui_version=gui_version, crop_bounds=bounds,
                    md_max=gui.get("md_max", 2.0), point_size=gui.get("point_size", 8.0),
                    return_crop=True)
                ex, ey, px, py, aRV, iRV = contour
            else:
                ex, ey, px, py, aRV, iRV = New_GUI(avg_diff, MD, E1)
                bounds = [0, m.shape[1], 0, m.shape[0]]
            display_crops.append(bounds)
            endo_x.append([ex]); endo_y.append([ey]); epi_x.append([px])
            epi_y.append([py]); antRVIP.append([aRV]); infRVIP.append([iRV])

    segmentation_data = {
        'image_shape': image_shape, 'voxel_spacing': voxel_spacing,
        'slice_index': selected_slices, 'crop_bounds': display_crops,
        'contours': {key: np.asarray(values).tolist() for key, values in
                     (('endo_x', endo_x), ('endo_y', endo_y), ('epi_x', epi_x),
                      ('epi_y', epi_y), ('antRVIP', antRVIP), ('infRVIP', infRVIP))},
    }

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
        with open(os.path.join(seg_path, 'Segmentation.json'), 'w') as handle:
            json.dump(segmentation_data, handle, indent=2)

    # ===== cDTI analysis =====
    c    = config["cdti"]
    filt = {'Linear Filter: Outlier StDev':      c["helix_angle_filter"]["linear_outlier_stdev"],
            'Spatial Filter: Wall Depth Factor':  c["helix_angle_filter"]["spatial_wall_depth_factor"],
            'Spatial Filter: Kernel Size':        c["helix_angle_filter"]["spatial_kernel_size"]}
    Cardiac_DTI_Metrics, Epi, Endo, Mask = cDTI_recon(
        myocardium_mask, Eigenvectors, c["num_interp_points"], c["smoothness_level"], filt,
        compute_hap=False)

    # ----- Post-processing for maps -----
    mask_smoothed_nan = np.copy(Mask).astype('float')
    mask_smoothed_nan[mask_smoothed_nan == 0] = np.nan
    hap_coordinate = c.get('hap_coordinate', 'cubic')
    in_plane_spacing = (tuple(float(Header[axis + ' Resolution']) for axis in ('X', 'Y'))
                        if all(axis + ' Resolution' in Header for axis in ('X', 'Y')) else None)
    cubic_depth = None
    if hap_coordinate == 'cubic':
        cubic_depth = np.clip(Endo2Epi_Grid(np.copy(Mask)) * mask_smoothed_nan, 0.0, 1.0)
    pitch = helix_angle_pitch(
        Cardiac_DTI_Metrics['HASF'], Mask,
        in_plane_spacing, coordinate=hap_coordinate, cubic_depth=cubic_depth,
    )
    grid_nan = pitch['depth']
    for key in ('HAP', 'HAP_R2', 'HAP_n', 'HAP_mm', 'HAP_mm_R2', 'HAP_mm_n'):
        Cardiac_DTI_Metrics[key] = pitch[key]

    # ----- Save quantitative results-----
    if save:
        qr_path = os.path.join(out_path, '12_Quantitative_Results')
        os.makedirs(qr_path, exist_ok=True)
        savemat(os.path.join(qr_path, 'Standard_DTI_Metrics.mat'), Standard_DTI_Metrics)
        savemat(os.path.join(qr_path, 'cDTI_Metrics.mat'),         Cardiac_DTI_Metrics)
        savemat(os.path.join(qr_path, 'DTI_Eigenvectors.mat'),     Eigenvectors)
        savemat(os.path.join(qr_path, 'HAP.mat'), pitch)
        _save_quantitative_maps(m, Standard_DTI_Metrics, Cardiac_DTI_Metrics,
                                Eigenvectors, mask_smoothed_nan, grid_nan, qr_path)

    return {
        "HAP":             pitch["HAP"],
        "HAP_mm":          pitch["HAP_mm"],
        "hap":             pitch,
        "dti_metrics":     Standard_DTI_Metrics,
        "eigenvectors":    Eigenvectors,
        "cardiac_metrics": Cardiac_DTI_Metrics,
        "mask":            Mask,
        "mask_nan":        mask_smoothed_nan,
        "grid_nan":        grid_nan,
        "matrix":          m,
        "bvals":           b,
        "bvecs":           v,
        "header":          Header,
        "segmentation":    segmentation_data,
        "crop_bounds":     display_crops,
        "contours":        {"endo_x": endo_x, "endo_y": endo_y, "epi_x": epi_x,
                            "epi_y": epi_y, "antRVIP": antRVIP, "infRVIP": infRVIP},
    }
