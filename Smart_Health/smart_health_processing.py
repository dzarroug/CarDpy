"""Single-call Smart Health workflows using the selected GitHub CardPy checkout.

The notebooks select the checkout before importing this module. Processing blocks
are preserved from the originals; TOPUP starts at stage 13 and does not repeat
Gibbs removal, shot rejection, or respiratory sorting.
"""
import os
import numpy as np
from cardpy.Data_Sorting import stacked2sorted
from cardpy.Data_Saving import Save_Diffusion_Image_Data, Save_Primary_Eigenvector_Data
from cardpy.Data_Processing.Gibbs import unrung
from cardpy.Data_Processing.Registration import register
from cardpy.Data_Processing.Rejection import shot_rejection as _cardpy_shot_rejection
from cardpy.Data_Processing.Respiratory import respiratory_sorting
from cardpy.Data_Processing.Diffusivity import ADC_Filter
from cardpy.Data_Processing.Averaging import average
from cardpy.Data_Processing.Denoising import denoise
from cardpy.Data_Processing.Interpolation import zero_filled
from cardpy.Data_Processing.Segmentation_Matrix_DTI import DTI_segmentation_matrix

SETTING_KEYS = ('data_type', 'DICOM_reader_info', 'operation_type', 'Gibbs_Mode', 'Rejection_Mode', 'NRMSE_Threshold', 'NSSIM_Threshold', 'rejection_zoom', 'rejection_IntEACT_zoom', 'rejection_organ', 'Respiratory_Mode', 'respiratory_zoom', 'respiratory_IntERACT_zoom', 'respiratory_organ', 'Registration_Mode', 'registration_algorithm', 'temporary_denoising', 'ADC_Filter_Mode', 'Averaging_Mode', 'Denoising_Mode', 'number_of_coils', 'denoising_algorithm', 'Interpolation_Mode', 'Extended_Matrix_Mode', 'E1_Mode', 'sub_registration')

def _legacy_replacements(matrix, keep):
    """Fill rejected positions with random accepted shots, as in legacy CardPy."""
    matrix = np.asarray(matrix)
    keep = np.asarray(keep)
    if matrix.ndim != 5 or keep.shape != matrix.shape[2:]:
        raise ValueError('Rejection decisions must match slices, encodings, and repetitions.')
    accepted = np.array(matrix, copy=True)
    repetitions = matrix.shape[4]
    for slc in range(matrix.shape[2]):
        for dif in range(matrix.shape[3]):
            bad = np.flatnonzero(keep[slc, dif] == 0)
            if repetitions == 1:
                # Preserve the original singleton branch, including its low-b exception.
                if len(bad) and dif != 0:
                    image = matrix[:, :, slc, dif, 0]
                    accepted[:, :, slc, dif, 0] = image - image.max()
                continue
            good = np.flatnonzero(keep[slc, dif] != 0)
            if len(bad) and not len(good):
                raise ValueError(
                    f'All repetitions rejected for slice {slc + 1}, encoding {dif + 1}; '
                    'legacy replacement requires at least one accepted repetition.'
                )
            for rejected_index in bad:
                replacement = int(np.random.choice(good, 1)[0])
                accepted[:, :, slc, dif, rejected_index] = matrix[:, :, slc, dif, replacement]
    return accepted


def shot_rejection(original_matrix, original_bvals, original_bvecs, **kwargs):
    """Use the new rejection UI/decisions and the original replacement behavior."""
    result = _cardpy_shot_rejection(original_matrix, original_bvals, original_bvecs, **kwargs)
    accepted = _legacy_replacements(original_matrix, result[5])
    return [accepted, result[1], result[2], result[3]]


def _process_blip(input_data, out_path, settings, gui_version="v2"):
    data_type = settings['data_type']
    DICOM_reader_info = settings['DICOM_reader_info']
    operation_type = settings['operation_type']
    Gibbs_Mode = settings['Gibbs_Mode']
    Rejection_Mode = settings['Rejection_Mode']
    NRMSE_Threshold = settings['NRMSE_Threshold']
    NSSIM_Threshold = settings['NSSIM_Threshold']
    rejection_zoom = settings['rejection_zoom']
    rejection_IntEACT_zoom = settings['rejection_IntEACT_zoom']
    rejection_organ = settings['rejection_organ']
    Respiratory_Mode = settings['Respiratory_Mode']
    respiratory_zoom = settings['respiratory_zoom']
    respiratory_IntERACT_zoom = settings['respiratory_IntERACT_zoom']
    respiratory_organ = settings['respiratory_organ']
    Registration_Mode = settings['Registration_Mode']
    registration_algorithm = settings['registration_algorithm']
    temporary_denoising = settings['temporary_denoising']
    ADC_Filter_Mode = settings['ADC_Filter_Mode']
    Averaging_Mode = settings['Averaging_Mode']
    Denoising_Mode = settings['Denoising_Mode']
    number_of_coils = settings['number_of_coils']
    denoising_algorithm = settings['denoising_algorithm']
    Interpolation_Mode = settings['Interpolation_Mode']
    Extended_Matrix_Mode = settings['Extended_Matrix_Mode']
    E1_Mode = settings['E1_Mode']
    sub_registration = settings['sub_registration']
    temporary_matrix_stacked, temporary_bvals_stacked, temporary_bvecs_stacked = input_data[:3]
    Header = dict(input_data[3])
    counter        = 0
    counter_string = str(counter)
    folder         = counter_string.zfill(2) + '_Original'
    [temporary_matrix_sorted,  temporary_bvals_sorted,  temporary_bvecs_sorted]          = stacked2sorted(temporary_matrix_stacked,
                                                                                                          temporary_bvals_stacked,
                                                                                                          temporary_bvecs_stacked)
    output_path = os.path.join(out_path, folder)
    file_name   = 'Original'
    Save_Diffusion_Image_Data(output_path, file_name, Header, temporary_matrix_sorted, temporary_bvals_sorted, temporary_bvecs_sorted)
    del counter_string, folder, output_path
    del temporary_matrix_stacked, temporary_bvals_stacked, temporary_bvecs_stacked
    if Gibbs_Mode == 'ON':
        counter        = counter + 1
        counter_string = str(counter)
        folder         = counter_string.zfill(2) + '_Unrung'
        print('Gibbs ringing removal mode is on.')
        [unrung_matrix_sorted, unrung_bvals_sorted, unrung_bvecs_sorted] = unrung(temporary_matrix_sorted,
                                                                                  temporary_bvals_sorted,
                                                                                  temporary_bvecs_sorted,
                                                                                  operation_type = operation_type)
        output_path = os.path.join(out_path, folder)
        Save_Diffusion_Image_Data(output_path, 'Unrung', Header, unrung_matrix_sorted, unrung_bvals_sorted, unrung_bvecs_sorted)
        temporary_matrix_sorted = unrung_matrix_sorted
        temporary_bvals_sorted  = unrung_bvals_sorted
        temporary_bvecs_sorted  = unrung_bvecs_sorted
        del counter_string, folder, output_path
        del unrung_matrix_sorted, unrung_bvals_sorted, unrung_bvecs_sorted
    if Gibbs_Mode == 'OFF':
        print('Gibbs ringing removal mode is off.')
    if Rejection_Mode == 'ON':
        counter        = counter + 1
        counter_string = str(counter)
        folder         = counter_string.zfill(2) + '_Rejected'
        print('Rejection mode is on.')
        print('Re-registering data prior to data rejection.')
        [registered_matrix_sorted, registered_bvals_sorted, registered_bvecs_sorted] = register(temporary_matrix_sorted,
                                                                                                temporary_bvals_sorted,
                                                                                                temporary_bvecs_sorted,
                                                                                                registration_algorithm = sub_registration,
                                                                                                temporary_denoising    = temporary_denoising,
                                                                                                operation_type         = operation_type)
        [rejected_matrix_sorted, rejected_bvals_sorted, rejected_bvecs_sorted, Slice_Coordinates] = shot_rejection(registered_matrix_sorted,
                                                                                                                   registered_bvals_sorted,
                                                                                                                   registered_bvecs_sorted,
                                                                                                                   NRMSE_threshold = NRMSE_Threshold,
                                                                                                                   NSSIM_threshold = NSSIM_Threshold,
                                                                                                               gui_version = gui_version,
                                                                                                                   zoom            = rejection_zoom,
                                                                                                                   IntERACT_zoom   = rejection_IntEACT_zoom,
                                                                                                                   operation_type  = operation_type)[:4]
        output_path = os.path.join(out_path, folder)
        Save_Diffusion_Image_Data(output_path, 'Rejected', Header, rejected_matrix_sorted, rejected_bvals_sorted, rejected_bvecs_sorted)
        temporary_matrix_sorted = rejected_matrix_sorted
        temporary_bvals_sorted  = rejected_bvals_sorted
        temporary_bvecs_sorted  = rejected_bvecs_sorted
        del counter_string, folder, output_path
        del rejected_matrix_sorted, rejected_bvals_sorted, rejected_bvecs_sorted
        del registered_matrix_sorted, registered_bvals_sorted, registered_bvecs_sorted
    if Rejection_Mode == 'OFF':
        print('Rejection mode is off.')
    if Respiratory_Mode == 'ON':
        counter        = counter + 1
        counter_string = str(counter)
        folder         = counter_string.zfill(2) + '_Respiratory_Sorted'
        print('Respiratory sorting mode is on.')
        print('Re-registering data prior data reordering.')
        [registered_matrix_sorted, registered_bvals_sorted, registered_bvecs_sorted] = register(temporary_matrix_sorted,
                                                                                                temporary_bvals_sorted,
                                                                                                temporary_bvecs_sorted,
                                                                                                registration_algorithm = registration_algorithm,
                                                                                                temporary_denoising    = temporary_denoising,
                                                                                                operation_type         = operation_type)
        [respiratory_matrix_sorted, respiratory_bvals_sorted, respiratory_bvecs_sorted, SLICE_Coordinates] = respiratory_sorting(registered_matrix_sorted,
                                                                                                                                 registered_bvals_sorted,
                                                                                                                                 registered_bvecs_sorted,
                                                                                                                                 zoom           = respiratory_zoom,
                                                                                                                                 IntERACT_zoom  = respiratory_IntERACT_zoom,
                                                                                                                                 organ          = respiratory_organ,
                                                                                                                             gui_version    = gui_version,
                                                                                                                                 operation_type = operation_type)
        output_path = os.path.join(out_path, folder)
        Save_Diffusion_Image_Data(output_path, 'Respiratory_Ordered', Header, respiratory_matrix_sorted, respiratory_bvals_sorted, respiratory_bvecs_sorted)
        temporary_matrix_sorted = respiratory_matrix_sorted
        temporary_bvals_sorted  = respiratory_bvals_sorted
        temporary_bvecs_sorted  = respiratory_bvecs_sorted
        del counter_string, folder, output_path
        del respiratory_matrix_sorted, respiratory_bvals_sorted, respiratory_bvecs_sorted
        del registered_matrix_sorted, registered_bvals_sorted, registered_bvecs_sorted
    if Rejection_Mode == 'OFF':
        print('Rejection mode is off.')
    if Registration_Mode == 'ON':
        counter        = counter + 1
        counter_string = str(counter)
        folder         = counter_string.zfill(2) + '_Registered'
        print('Registration mode is on.')
        [registered_matrix_sorted, registered_bvals_sorted, registered_bvecs_sorted] = register(temporary_matrix_sorted,
                                                                                                temporary_bvals_sorted,
                                                                                                temporary_bvecs_sorted,
                                                                                                registration_algorithm = registration_algorithm,
                                                                                                temporary_denoising    = temporary_denoising,
                                                                                                operation_type         = operation_type)
        output_path = os.path.join(out_path, folder)
        Save_Diffusion_Image_Data(output_path, 'Registered', Header, registered_matrix_sorted, registered_bvals_sorted, registered_bvecs_sorted)
        temporary_matrix_sorted = registered_matrix_sorted
        temporary_bvals_sorted  = registered_bvals_sorted
        temporary_bvecs_sorted  = registered_bvecs_sorted
        del counter_string, folder, output_path
        del registered_matrix_sorted, registered_bvals_sorted, registered_bvecs_sorted
    if Registration_Mode == 'OFF':
        print('Registration mode is off.')
    if ADC_Filter_Mode == 'ON':
        counter        = counter + 1
        counter_string = str(counter)
        folder         = counter_string.zfill(2) + '_Diffusivity_Filtered'
        print('ADC Filter mode is on.')
        [filtered_matrix_sorted, filtered_bvals_sorted, filtered_bvecs_sorted] = ADC_Filter(temporary_matrix_sorted,
                                                                                            temporary_bvals_sorted,
                                                                                            temporary_bvecs_sorted,
                                                                                            operation_type = operation_type)
        output_path = os.path.join(out_path, folder)
        Save_Diffusion_Image_Data(output_path, 'ADC_Filtered', Header, filtered_matrix_sorted, filtered_bvals_sorted, filtered_bvecs_sorted)
        temporary_matrix_sorted = filtered_matrix_sorted
        temporary_bvals_sorted  = filtered_bvals_sorted
        temporary_bvecs_sorted  = filtered_bvecs_sorted
        del counter_string, folder, output_path
        del filtered_matrix_sorted, filtered_bvals_sorted, filtered_bvecs_sorted
    if Averaging_Mode == 'OFF':
        print('Averaging mode is off.')
    if Averaging_Mode == 'ON':
        counter        = counter + 1
        counter_string = str(counter)
        folder         = counter_string.zfill(2) + '_Averaged'
        print('Averaging mode is on.')
        [averaged_matrix_sorted, averaged_bvals_sorted, averaged_bvecs_sorted] = average(temporary_matrix_sorted,
                                                                                         temporary_bvals_sorted,
                                                                                         temporary_bvecs_sorted,
                                                                                         operation_type = operation_type)
        print('Re-registering data after data averaging.')
        [registered_matrix_sorted, registered_bvals_sorted, registered_bvecs_sorted] = register(averaged_matrix_sorted,
                                                                                                averaged_bvals_sorted,
                                                                                                averaged_bvecs_sorted,
                                                                                                registration_algorithm = registration_algorithm,
                                                                                                temporary_denoising    = temporary_denoising,
                                                                                                operation_type         = operation_type)
        output_path = os.path.join(out_path, folder)
        Save_Diffusion_Image_Data(output_path, 'Averaged', Header, registered_matrix_sorted, registered_bvals_sorted, registered_bvecs_sorted)
        temporary_matrix_sorted = registered_matrix_sorted
        temporary_bvals_sorted  = registered_bvals_sorted
        temporary_bvecs_sorted  = registered_bvecs_sorted
        del counter_string, folder, output_path
        del averaged_matrix_sorted, averaged_bvals_sorted, averaged_bvecs_sorted
        del registered_matrix_sorted, registered_bvals_sorted, registered_bvecs_sorted
    if Averaging_Mode == 'OFF':
        print('Averaging mode is off.')
    if Denoising_Mode == 'ON':
        counter        = counter + 1
        counter_string = str(counter)
        folder         = counter_string.zfill(2) + '_Denoised'
        print('Denoising mode is on.')
        [denoised_matrix_sorted, denoised_bvals_sorted, denoised_bvecs_sorted] = denoise(temporary_matrix_sorted,
                                                                                         temporary_bvals_sorted,
                                                                                         temporary_bvecs_sorted,
                                                                                         denoising_algorithm = denoising_algorithm,
                                                                                         operation_type      = operation_type)
        output_path = os.path.join(out_path, folder)
        Save_Diffusion_Image_Data(output_path, 'Denoised', Header, denoised_matrix_sorted, denoised_bvals_sorted, denoised_bvecs_sorted)
        temporary_matrix_sorted = denoised_matrix_sorted
        temporary_bvals_sorted  = denoised_bvals_sorted
        temporary_bvecs_sorted  = denoised_bvecs_sorted
        del counter_string, folder, output_path
        del denoised_matrix_sorted, denoised_bvals_sorted, denoised_bvecs_sorted
    if Denoising_Mode == 'OFF':
        print('Denoising mode is off.')
    if Interpolation_Mode == 'ON':
        counter        = counter + 1
        counter_string = str(counter)
        folder         = counter_string.zfill(2) + '_Interpolated'
        print('Interpolating mode is on.')
        [interpolated_matrix_sorted, interpolated_bvals_sorted, interpolated_bvecs_sorted] = zero_filled(temporary_matrix_sorted,
                                                                                                         temporary_bvals_sorted,
                                                                                                         temporary_bvecs_sorted,
                                                                                                         operation_type = operation_type)
        output_path = os.path.join(out_path, folder)
        Header['X Resolution'] = Header['X Resolution'] / 2
        Header['Y Resolution'] = Header['Y Resolution'] / 2
        Save_Diffusion_Image_Data(output_path, 'Interpolated', Header, interpolated_matrix_sorted, interpolated_bvals_sorted, interpolated_bvecs_sorted)
        temporary_matrix_sorted = interpolated_matrix_sorted
        temporary_bvals_sorted  = interpolated_bvals_sorted
        temporary_bvecs_sorted  = interpolated_bvecs_sorted
        del counter_string, folder, output_path
        del interpolated_matrix_sorted, interpolated_bvals_sorted, interpolated_bvecs_sorted
    if Interpolation_Mode == 'OFF':
        print('Interpolating mode is off.')
    if Extended_Matrix_Mode == 'ON':
        counter        = counter + 1
        counter_string = str(counter)
        folder         = counter_string.zfill(2) + '_Extended_Matrix'
        print('Extended matrix mode is on.')
        [segmentation_matrix, segmentation_bvals, segmentation_bvecs] = DTI_segmentation_matrix(temporary_matrix_sorted,
                                                                                                temporary_bvals_sorted,
                                                                                                temporary_bvecs_sorted)
        [segmentation_matrix, segmentation_bvals, segmentation_bvecs] = stacked2sorted(segmentation_matrix,
                                                                                       segmentation_bvals,
                                                                                       segmentation_bvecs)
        output_path = os.path.join(out_path, folder)
        Save_Diffusion_Image_Data(output_path, 'Extended_Matrix', Header, segmentation_matrix, segmentation_bvals, segmentation_bvecs)
        del counter_string, folder, output_path
        del segmentation_matrix, segmentation_bvals, segmentation_bvecs
    if Extended_Matrix_Mode == 'OFF':
        print('Extended matrix mode is off.')
    if E1_Mode == 'ON':
        counter        = counter + 1
        counter_string = str(counter)
        folder         = counter_string.zfill(2) + '_Primary_Eigenvector'
        output_path    = os.path.join(out_path, folder)
        print('Primary Eigenvector mode is on.')
        Save_Primary_Eigenvector_Data(output_path, Header, temporary_matrix_sorted, temporary_bvals_sorted, temporary_bvecs_sorted)
    if E1_Mode == 'OFF':
        print('Primary eigenvector mode is off.')
    return dict(matrix=temporary_matrix_sorted, bvals=temporary_bvals_sorted, bvecs=temporary_bvecs_sorted, header=Header)

def _process_topup(input_data, out_path, settings, gui_version="v2"):
    data_type = settings['data_type']
    DICOM_reader_info = settings['DICOM_reader_info']
    operation_type = settings['operation_type']
    Gibbs_Mode = settings['Gibbs_Mode']
    Rejection_Mode = settings['Rejection_Mode']
    NRMSE_Threshold = settings['NRMSE_Threshold']
    NSSIM_Threshold = settings['NSSIM_Threshold']
    rejection_zoom = settings['rejection_zoom']
    rejection_IntEACT_zoom = settings['rejection_IntEACT_zoom']
    rejection_organ = settings['rejection_organ']
    Respiratory_Mode = settings['Respiratory_Mode']
    respiratory_zoom = settings['respiratory_zoom']
    respiratory_IntERACT_zoom = settings['respiratory_IntERACT_zoom']
    respiratory_organ = settings['respiratory_organ']
    Registration_Mode = settings['Registration_Mode']
    registration_algorithm = settings['registration_algorithm']
    temporary_denoising = settings['temporary_denoising']
    ADC_Filter_Mode = settings['ADC_Filter_Mode']
    Averaging_Mode = settings['Averaging_Mode']
    Denoising_Mode = settings['Denoising_Mode']
    number_of_coils = settings['number_of_coils']
    denoising_algorithm = settings['denoising_algorithm']
    Interpolation_Mode = settings['Interpolation_Mode']
    Extended_Matrix_Mode = settings['Extended_Matrix_Mode']
    E1_Mode = settings['E1_Mode']
    sub_registration = settings['sub_registration']
    Header = dict(input_data[3])
    temporary_matrix_sorted, temporary_bvals_sorted, temporary_bvecs_sorted = stacked2sorted(*input_data[:3])
    counter = 12
    if Registration_Mode == 'ON':
        counter        = counter + 1
        counter_string = str(counter)
        folder         = counter_string.zfill(2) + '_Registered'
        print('Registration mode is on.')
        [registered_matrix_sorted, registered_bvals_sorted, registered_bvecs_sorted] = register(temporary_matrix_sorted,
                                                                                                temporary_bvals_sorted,
                                                                                                temporary_bvecs_sorted,
                                                                                                registration_algorithm = registration_algorithm,
                                                                                                temporary_denoising    = temporary_denoising,
                                                                                                operation_type         = operation_type)
        output_path = os.path.join(out_path, folder)
        Save_Diffusion_Image_Data(output_path, 'Registered', Header, registered_matrix_sorted, registered_bvals_sorted, registered_bvecs_sorted)
        temporary_matrix_sorted = registered_matrix_sorted
        temporary_bvals_sorted  = registered_bvals_sorted
        temporary_bvecs_sorted  = registered_bvecs_sorted
        del counter_string, folder, output_path
        del registered_matrix_sorted, registered_bvals_sorted, registered_bvecs_sorted
    if Registration_Mode == 'OFF':
        print('Registration mode is off.')
    if ADC_Filter_Mode == 'ON':
        counter        = counter + 1
        counter_string = str(counter)
        folder         = counter_string.zfill(2) + '_Diffusivity_Filtered'
        print('ADC Filter mode is on.')
        [filtered_matrix_sorted, filtered_bvals_sorted, filtered_bvecs_sorted] = ADC_Filter(temporary_matrix_sorted,
                                                                                            temporary_bvals_sorted,
                                                                                            temporary_bvecs_sorted,
                                                                                            operation_type = operation_type)
        output_path = os.path.join(out_path,  folder)
        Save_Diffusion_Image_Data(output_path, 'ADC_Filtered', Header, filtered_matrix_sorted, filtered_bvals_sorted, filtered_bvecs_sorted)
        temporary_matrix_sorted = filtered_matrix_sorted
        temporary_bvals_sorted  = filtered_bvals_sorted
        temporary_bvecs_sorted  = filtered_bvecs_sorted
        del counter_string, folder, output_path
        del filtered_matrix_sorted, filtered_bvals_sorted, filtered_bvecs_sorted
    if Averaging_Mode == 'OFF':
        print('ADC Filter mode is off.')
    if Averaging_Mode == 'ON':
        counter        = counter + 1
        counter_string = str(counter)
        folder         = counter_string.zfill(2) + '_Averaged'
        print('Averaging mode is on.')
        [averaged_matrix_sorted, averaged_bvals_sorted, averaged_bvecs_sorted] = average(temporary_matrix_sorted,
                                                                                         temporary_bvals_sorted,
                                                                                         temporary_bvecs_sorted,
                                                                                         operation_type = operation_type)
        print('Re-registering data after data averaging.')
        [registered_matrix_sorted, registered_bvals_sorted, registered_bvecs_sorted] = register(averaged_matrix_sorted,
                                                                                                averaged_bvals_sorted,
                                                                                                averaged_bvecs_sorted,
                                                                                                registration_algorithm = registration_algorithm,
                                                                                                temporary_denoising    = temporary_denoising,
                                                                                                operation_type         = operation_type)
        output_path = os.path.join(out_path,  folder)
        Save_Diffusion_Image_Data(output_path, 'Averaged', Header, registered_matrix_sorted, registered_bvals_sorted, registered_bvecs_sorted)
        temporary_matrix_sorted = registered_matrix_sorted
        temporary_bvals_sorted  = registered_bvals_sorted
        temporary_bvecs_sorted  = registered_bvecs_sorted
        del counter_string, folder, output_path
        del averaged_matrix_sorted, averaged_bvals_sorted, averaged_bvecs_sorted
        del registered_matrix_sorted, registered_bvals_sorted, registered_bvecs_sorted
    if Averaging_Mode == 'OFF':
        print('Averaging mode is off.')
    if Denoising_Mode == 'ON':
        counter        = counter + 1
        counter_string = str(counter)
        folder         = counter_string.zfill(2) + '_Denoised'
        print('Denoising mode is on.')
        [denoised_matrix_sorted, denoised_bvals_sorted, denoised_bvecs_sorted] = denoise(temporary_matrix_sorted,
                                                                                         temporary_bvals_sorted,
                                                                                         temporary_bvecs_sorted,
                                                                                         denoising_algorithm = denoising_algorithm,
                                                                                         operation_type      = operation_type)
        output_path = os.path.join(out_path,  folder)
        Save_Diffusion_Image_Data(output_path, 'Denoised', Header, denoised_matrix_sorted, denoised_bvals_sorted, denoised_bvecs_sorted)
        temporary_matrix_sorted = denoised_matrix_sorted
        temporary_bvals_sorted  = denoised_bvals_sorted
        temporary_bvecs_sorted  = denoised_bvecs_sorted
        del counter_string, folder, output_path
        del denoised_matrix_sorted, denoised_bvals_sorted, denoised_bvecs_sorted
    if Denoising_Mode == 'OFF':
        print('Denoising mode is off.')
    if Interpolation_Mode == 'ON':
        counter        = counter + 1
        counter_string = str(counter)
        folder         = counter_string.zfill(2) + '_Interpolated'
        print('Interpolating mode is on.')
        [interpolated_matrix_sorted, interpolated_bvals_sorted, interpolated_bvecs_sorted] = zero_filled(temporary_matrix_sorted,
                                                                                                         temporary_bvals_sorted,
                                                                                                         temporary_bvecs_sorted,
                                                                                                         operation_type = operation_type)
        output_path = os.path.join(out_path,  folder)
        Header['X Resolution'] = Header['X Resolution'] / 2
        Header['Y Resolution'] = Header['Y Resolution'] / 2
        Save_Diffusion_Image_Data(output_path, 'Interpolated', Header, interpolated_matrix_sorted, interpolated_bvals_sorted, interpolated_bvecs_sorted)
        temporary_matrix_sorted = interpolated_matrix_sorted
        temporary_bvals_sorted  = interpolated_bvals_sorted
        temporary_bvecs_sorted  = interpolated_bvecs_sorted
        print(interpolated_matrix_sorted.shape)
        del counter_string, folder, output_path
        del interpolated_matrix_sorted, interpolated_bvals_sorted, interpolated_bvecs_sorted
    if Interpolation_Mode == 'OFF':
        print('Interpolating mode is off.')
    if Extended_Matrix_Mode == 'ON':
        counter        = counter + 1
        counter_string = str(counter)
        folder         = counter_string.zfill(2) + '_Extended_Matrix'
        print('Extended matrix mode is on.')
        [segmentation_matrix, segmentation_bvals, segmentation_bvecs] = DTI_segmentation_matrix(temporary_matrix_sorted,
                                                                                                temporary_bvals_sorted,
                                                                                                temporary_bvecs_sorted)
        [segmentation_matrix, segmentation_bvals, segmentation_bvecs] = stacked2sorted(segmentation_matrix,
                                                                                       segmentation_bvals,
                                                                                       segmentation_bvecs)
        output_path = os.path.join(out_path,  folder)
        Save_Diffusion_Image_Data(output_path, 'Extended_Matrix', Header, segmentation_matrix, segmentation_bvals, segmentation_bvecs)
        del counter_string, folder, output_path
        del segmentation_matrix, segmentation_bvals, segmentation_bvecs
    if Extended_Matrix_Mode == 'OFF':
        print('Extended matrix mode is off.')
    if E1_Mode == 'ON':
        counter        = counter + 1
        counter_string = str(counter)
        folder         = counter_string.zfill(2) + '_Primary_Eigenvector'
        output_path    = os.path.join(out_path,  folder)
        print('Primary Eigenvector mode is on.')
        Save_Primary_Eigenvector_Data(output_path, Header, temporary_matrix_sorted, temporary_bvals_sorted, temporary_bvecs_sorted)
    if E1_Mode == 'OFF':
        print('Primary eigenvector mode is off.')
    return dict(matrix=temporary_matrix_sorted, bvals=temporary_bvals_sorted, bvecs=temporary_bvecs_sorted, header=Header)

def process_smart_health(input_data, out_path, settings, *, workflow="blip", gui_version="v2"):
    """Run the original workflow, preserving settings and output folders.

    input_data is (stacked images, b-values, b-vectors, header). This call ends
    after the primary eigenvector output, without adding contouring or analysis.
    """
    if workflow not in ("blip", "topup"):
        raise ValueError("workflow must be 'blip' or 'topup'")
    missing = set(SETTING_KEYS) - settings.keys()
    if missing:
        raise ValueError(f"Missing original settings: {sorted(missing)}")
    runner = _process_blip if workflow == "blip" else _process_topup
    return runner(input_data, os.fspath(out_path), settings, gui_version=gui_version)


def _settings_from_config(config):
    """Translate the CardPy config schema to the original Smart Health settings."""
    def flag(section):
        return 'ON' if config[section]['enabled'] else 'OFF'

    algorithm = config['registration']['algorithm']
    if algorithm not in ('Affine', 'Rigid', 'Translation'):
        raise ValueError('The original workflow supports Affine, Rigid, or Translation.')
    if config.get('slice_index') is not None:
        raise ValueError('Use slice_index=None to preserve the original all-slice workflow.')
    if config['registration'].get('mode', '2D') != '2D':
        raise ValueError('The original workflow requires 2D registration.')
    return {
        'data_type': config['data_type'],
        'DICOM_reader_info': config['dicom_reader_info'],
        'operation_type': config['operation_type'],
        'Gibbs_Mode': flag('gibbs'),
        'Rejection_Mode': flag('rejection'),
        'NRMSE_Threshold': config['rejection']['nrmse_threshold'],
        'NSSIM_Threshold': config['rejection']['nssim_threshold'],
        'rejection_zoom': config['rejection']['zoom'],
        'rejection_IntEACT_zoom': config['rejection']['interact_zoom'],
        'rejection_organ': config['rejection']['organ'],
        'Respiratory_Mode': flag('respiratory'),
        'respiratory_zoom': config['respiratory']['zoom'],
        'respiratory_IntERACT_zoom': config['respiratory']['interact_zoom'],
        'respiratory_organ': config['respiratory']['organ'],
        'Registration_Mode': flag('registration'),
        'registration_algorithm': algorithm,
        'temporary_denoising': config['registration']['temporary_denoising'],
        'ADC_Filter_Mode': flag('adc_filter'),
        'Averaging_Mode': flag('averaging'),
        'Denoising_Mode': flag('denoising'),
        'number_of_coils': config['denoising']['number_of_coils'],
        'denoising_algorithm': config['denoising']['algorithm'],
        'Interpolation_Mode': flag('interpolation'),
        'Extended_Matrix_Mode': flag('extended_matrix'),
        'E1_Mode': flag('e1'),
        'sub_registration': 'Rigid' if algorithm == 'Affine' else 'Translation',
    }


def process(study_root, config):
    """Config/process interface preserving the original Smart Health workflow.

    Uses smart_health_config.get_config() and notebook-specific paths.
    This local adapter intentionally retains the original registration passes
    and stage numbering instead of cardpy.pipeline.process's revised sequence.
    TOPUP reads config['nifti_paths'] relative to study_root and resumes at 13.
    """
    from cardpy.Data_Import import DICOM_Reader, NifTi_Reader

    settings = _settings_from_config(config)
    workflow = config['smart_health']['workflow']
    if workflow not in ('blip', 'topup'):
        raise ValueError("smart_health.workflow must be 'blip' or 'topup'")
    out_path = config['output']['directory']
    if not out_path:
        raise ValueError('Set output.directory to the original output folder.')
    if workflow == 'blip':
        if config['data_type'] != 'DICOM':
            raise ValueError('The original Blip workflow reads DICOM input.')
        input_data = DICOM_Reader(
            os.path.join(study_root, config['dicom_subpath']),
            info=config['dicom_reader_info'],
        )
    else:
        if config['data_type'] != 'NifTis':
            raise ValueError('The original TOPUP workflow reads NifTis input.')
        paths = config['nifti_paths']
        matrix, bvals, bvecs, header, resolution, _ = NifTi_Reader(
            *(os.path.join(study_root, paths[key])
              for key in ('image', 'bvals', 'bvecs', 'header')),
        )
        for index, axis in enumerate(('X', 'Y', 'Z')):
            header[axis + ' Resolution'] = resolution[index]
        input_data = (matrix, bvals, bvecs, header)
    return process_smart_health(
        input_data, out_path, settings, workflow=workflow,
        gui_version=config.get("gui", {}).get("version", "v2"),
    )
