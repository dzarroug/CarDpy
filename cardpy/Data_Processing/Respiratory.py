def respiratory_sorting(original_matrix, original_bvals, original_bvecs, zoom = 'ON', IntERACT_zoom = 'ON', organ = 'Liver', operation_type = 'Magnitude', gui_version = 'legacy', crop_coordinates = None):
    """
    ########## Definition Inputs ##################################################################################################################
    original_matrix         : Sorted diffusion data (5D - [rows, columns, slices, directions, averages]).
    original_bvals          : Sorted b-values.
    original_bvecs          : Sorted b-vectors.
    zoom                    :
    IntERACT_zoom           :
    organ                   :
    operation_type          : Identify the type of operation for original matrix (Optional).
                              Default operation type is magnitude.
                              Operation type options include Magnitude and Complex.
    ########## Definition Outputs #################################################################################################################
    respiratory_matrix      : Sorted respiratory ordered diffusion data (5D - [rows, columns, slices, directions, averages]).
                              Data is sorted in the average dimension from expiration to inspiration.
                              Shots containing nonfinite values (including NaNs from rejection) are excluded
                              from motion estimation and preserved at the end of each slice/encoding.
    respiratory_bvals       : Sorted b-values.
    respiratory_bvecs       : Sorted b-vectors.
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import modules #####################################################################################################################
    import cv2
    import numpy                            as np                                                                                                   # Import numpy module
    from   cardpy.Data_Sorting              import sorted2stacked                                                                   # Import sorted to stacked from CarDpy
    from   cardpy.GUI_Tools.IntERACT        import INTERACT_GUI
    from   cardpy.Data_Processing.Denoising import denoise                                                                                          #
    from   skimage.filters.rank             import entropy                                                                                          #
    from   skimage.morphology               import disk                                                                                             #
    from   skimage.metrics                  import structural_similarity as ssim                                                                    # Import structural similarity index from skimage module
    from   skimage.metrics                  import mean_squared_error as mse                                                                        # Import mean squared error from skimage module
    ########## Extract information about original data matrix #####################################################################################
    directions = original_matrix.shape[3]                                                                                                           # Extract number of directions
    slices     = original_matrix.shape[2]                                                                                                           # Extract number of slices
    averages   = original_matrix.shape[4]                                                                                                           # Extract number of averages
    ########## Initialize respiratory matrix and address data type of original data ###############################################################
    respiratory_matrix = np.zeros(original_matrix.shape)                                                                                            # Initialize respiratory matrix
    if np.iscomplexobj(original_matrix):                                                                                                       # If data type is complex ...
        if operation_type == 'Complex':                                                                                                                 # If operation type is complex ...
            respiratory_matrix = respiratory_matrix.astype(np.complex128)                                                                                  # Cast respiratory matrix for complex data
        if operation_type == 'Magnitude':                                                                                                               # If operation type is magnitude ...
            respiratory_matrix = respiratory_matrix.astype(np.float64)                                                                                      # Cast respiratory matrix for magnitude data
            original_matrix    = np.abs(original_matrix)                                                                                                    # Convert complex original matrix to magnitude data
            print('Input data type is complex, but magnitude is being executed.')                                                                           # Print warning
        temporary_matrix = np.abs(original_matrix)                                                                                                      # Create temporary matrix variable using magnitude data
    else:                                                                                                                                           # Otherwise ...
        respiratory_matrix = respiratory_matrix.astype(np.float64)                                                                                      # Cast interpolated matrix for magnitude data
        temporary_matrix   = original_matrix                                                                                                            # Create temporary matrix variable
    [temporary_matrix_stacked, original_bvals_stacked, original_bvecs_stacked] = sorted2stacked(temporary_matrix, original_bvals, original_bvecs)   #
    ########## ROI Cropping for Image #############################################################################################################
    Slice_Crop_Coordinates = []                                                                                                                     #
    if crop_coordinates is not None:
        from cardpy.GUI_Tools._image_controls import crop_bounds
        if len(crop_coordinates) != 4 or any(len(values) != slices for values in crop_coordinates):
            raise ValueError('Respiratory crop coordinates must contain four lists, one value per slice.')
        bounds = [crop_bounds([values[slc] for values in crop_coordinates], original_matrix.shape)
                  for slc in range(slices)]
        Slice_Crop_Coordinates = [list(values) for values in zip(*bounds)]
    elif zoom == 'ON':                                                                                                                                #
        if IntERACT_zoom == 'ON':                                                                                                                       #
            [x_start, x_end, y_start, y_end] = INTERACT_GUI(temporary_matrix_stacked, organ, gui_version=gui_version)                                                            #
            Slice_Crop_Coordinates = [x_start, x_end, y_start, y_end]                                                                                   #
        if IntERACT_zoom == 'OFF':                                                                                                                  #
            print('Interact is off')                                                                                                                    #
            x_start = []                                                                                                                                #
            x_end   = []                                                                                                                                #
            y_start = []                                                                                                                                #
            y_end   = []                                                                                                                                #
            temporary_matrix_list = []                                                                                                                  #
            for slc in range(slices):                                                                                                                   # Iterate through slices
                image               = np.max(temporary_matrix_stacked[:, :, slc, :], axis = 2)                                                              #
                image_normalization = image / image.max()                                                                                                   #
                title_pre = 'Select area for slice ' + str(slc + 1)                                                                                         #
                r = cv2.selectROI(title_pre, image_normalization, False, False)
                cv2.startWindowThread()
                x_start.append(int(r[0]))
                x_end.append(int(r[0]+r[2]))
                y_start.append(int(r[1]))
                y_end.append(int(r[1]+r[3]))
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
            Slice_Crop_Coordinates = [x_start, x_end, y_start, y_end]
    elif zoom == 'OFF':
        x_start = []
        x_end   = []
        y_start = []
        y_end   = []
        temporary_matrix_list = []
        for slc in range(temporary_matrix_stacked.shape[2]):
            x_start.append(int(0))
            x_end.append(int(temporary_matrix_stacked.shape[1]))
            y_start.append(int(0))
            y_end.append(int(temporary_matrix_stacked.shape[0]))
        Slice_Crop_Coordinates = [x_start, x_end, y_start, y_end]

    ### Extract Crop Coordinates
    x_start = Slice_Crop_Coordinates[0]
    x_end   = Slice_Crop_Coordinates[1]
    y_start = Slice_Crop_Coordinates[2]
    y_end   = Slice_Crop_Coordinates[3]

    # Rejection represents unavailable shots as NaN. Never pass these shots
    # into PCA or entropy, and retain their original values at the output tail.
    available = np.isfinite(temporary_matrix).all(axis=(0, 1))
    for slc in range(slices):
        if x_end[slc] - x_start[slc] < 7 or y_end[slc] - y_start[slc] < 7:
            raise ValueError('Respiratory sorting requires a crop at least 7 x 7 pixels per slice.')
    if available.all():
        denoised_matrix, _, _ = denoise(temporary_matrix, original_bvals, original_bvecs,
                                        denoising_algorithm='LocalPCA')
    else:
        denoised_matrix = np.zeros_like(temporary_matrix, dtype=float)
        for slc in range(slices):
            # Keep all available encodings together for PCA, with their actual
            # b-values and vectors, even when repetition counts differ.
            indices = np.argwhere(available[slc])
            if not len(indices):
                continue
            images = np.stack([temporary_matrix[:, :, slc, dif, avg]
                               for dif, avg in indices], axis=2)
            if len(indices) == 1:
                cleaned = images
            else:
                cleaned, _, _ = denoise(
                    images[:, :, None, :, None],
                    np.asarray(original_bvals)[indices[:, 0]],
                    np.asarray(original_bvecs)[indices[:, 0]],
                    denoising_algorithm='LocalPCA')
                cleaned = cleaned[:, :, 0, :, 0]
            for index, (dif, avg) in enumerate(indices):
                denoised_matrix[:, :, slc, dif, avg] = cleaned[:, :, index]

    from skimage.util import img_as_ubyte
    entropy_matrix = np.zeros(temporary_matrix.shape)
    for slc, dif, avg in np.argwhere(available):
        image = denoised_matrix[:, :, slc, dif, avg]
        high = image.max()
        normalized = np.clip(image / high, 0, 1) if high > 0 else np.zeros_like(image)
        entropy_matrix[:, :, slc, dif, avg] = entropy(img_as_ubyte(normalized), disk(7))

    for dif in range(directions):
        for slc in range(slices):
            good = np.flatnonzero(available[slc, dif])
            missing = np.flatnonzero(~available[slc, dif])
            order = good
            if len(good) > 1:
                roi = np.s_[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc]]
                features = entropy_matrix[:, :, slc, dif, :][roi][:, :, good]
                similarity = np.eye(len(good))
                errors = np.zeros_like(similarity)
                for i in range(len(good)):
                    for j in range(i + 1, len(good)):
                        similarity[i, j] = similarity[j, i] = np.round(
                            ssim(features[:, :, i], features[:, :, j], data_range=2.0), 2)
                        errors[i, j] = errors[j, i] = np.round(
                            np.sqrt(mse(features[:, :, i], features[:, :, j])), 2)
                # Choose distinct endpoints. With identical/blank images the
                # old tie handling could select a diagonal or index past a pair.
                left, right = np.triu_indices(len(good), k=1)
                scores = similarity[left, right]
                candidates = np.flatnonzero(scores == scores.min())
                pair = candidates[np.argmax(errors[left[candidates], right[candidates]])]
                first, second = left[pair], right[pair]
                sums = [np.sum(temporary_matrix[:, :, slc, dif, good[i]][roi])
                        for i in (first, second)]
                expiration = first if sums[0] <= sums[1] else second
                order = good[np.argsort(-similarity[expiration], kind='stable')]
            for avg, source in enumerate(np.concatenate((order, missing))):
                respiratory_matrix[:, :, slc, dif, avg] = original_matrix[:, :, slc, dif, source]
    ########## Tie-up-loose-ends ... ##############################################################################################################
    respiratory_bvals = original_bvals                                                                                                              # Store b-value information
    respiratory_bvecs = original_bvecs                                                                                                              # Store b-vector information
    return [respiratory_matrix, respiratory_bvals, respiratory_bvecs, Slice_Crop_Coordinates]
