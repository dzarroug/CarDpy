def _apply_dictionary_keep(dictionary_items, keep_matrix_stacked):
    """Map stacked rejection decisions back to variable-length encodings."""
    import numpy as np

    accepted_encodings = {}
    stats_by_encoding = {}
    keep_by_encoding = {}
    offset = 0
    for key, encoding, repetitions in dictionary_items:
        encoding_keep = keep_matrix_stacked[:, offset:offset + repetitions].copy()
        accepted_images = np.array(encoding['images'], dtype=np.complex128 if np.iscomplexobj(encoding['images']) else float, copy=True)
        stats_by_encoding[key] = np.mean(encoding_keep, axis=1) * 100
        keep_by_encoding[key] = encoding_keep
        for slc in range(encoding_keep.shape[0]):
            good = np.flatnonzero(encoding_keep[slc] == 1)
            bad = np.flatnonzero(encoding_keep[slc] == 0)
            if bad.size == 0:
                continue
            # Preserve one fallback shot when everything was rejected. Otherwise
            # mark rejected repetitions missing so each survivor counts once.
            if good.size == 0:
                bad = bad[bad != 0]
            accepted_images[:, :, slc, bad] = np.nan
        accepted_encodings[key] = {
            'bval': float(encoding['bval']),
            'bvec': np.asarray(encoding['bvec']).copy(),
            'images': accepted_images,
        }
        offset += repetitions
    if offset != keep_matrix_stacked.shape[1]:
        raise ValueError('Dictionary repetitions do not match the stacked rejection decisions.')
    return accepted_encodings, stats_by_encoding, keep_by_encoding


def _apply_sorted_keep(matrix, keep):
    """Keep each accepted repetition once, with repetition 0 as fallback."""
    import numpy as np
    accepted = np.array(matrix, dtype=np.complex128 if np.iscomplexobj(matrix) else float, copy=True)
    if keep.shape != matrix.shape[2:]:
        raise ValueError('Rejection decisions must match slices, encodings, and repetitions.')
    for slc, dif in np.ndindex(keep.shape[:2]):
        bad = np.flatnonzero(keep[slc, dif] == 0)
        if not np.any(keep[slc, dif]):
            print('All repetitions were rejected for slice %i, direction %i; retaining repetition 0.' % (slc + 1, dif + 1))
            bad = bad[bad != 0]
        accepted[:, :, slc, dif, bad] = np.nan
    return accepted


def shot_rejection(original_matrix, original_bvals, original_bvecs, NRMSE_threshold = 0.75, NSSIM_threshold = 0.75, zoom = 'ON',IntERACT_zoom = 'ON', organ = 'Heart', operation_type = 'Magnitude', diagnostics_path = None, gui_version = 'legacy', save_diagnostics = True, manual_review = False, saved_configuration_mode = "off", rejection_mode = 'algorithm', ml_model_path = None, ml_threshold = None, ml_input = None): 
    """
    ########## Definition Inputs ##################################################################################################################
    original_matrix       : Sorted diffusion data (5D).
    original_bvals        : Sorted b-values.
    original_bvecs        : Sorted b vectors.
    zoom                  :
    IntERACT_zoom         :
    organ                 :
    operation_type        : Identify the type of operation for original matrix (Optional).
                            Default operation type is magnitude.
                            Operation type options include Magnitude and Complex.
    ########## Definition Outputs #################################################################################################################
    accepted_matrix       : Accepted matrix (5D).
    accepted_bvals        : Accepted b-values.
    accepted_bvecs        : Accepted b vectors.
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import Modules #####################################################################################################################
    from   cardpy.Data_Sorting       import sorted2stacked                                                          # Import sorted to stacked from CarDpy
    from   cardpy.GUI_Tools.IntERACT import INTERACT_GUI                    #
    import numpy                     as     np                                                                                      #
    import cv2                                                                                                                      #
    from   skimage.metrics           import structural_similarity as ssim                                                           # Import strutural similarity metric from skimage module
    from   skimage.metrics           import mean_squared_error    as mse                                                            #
    from   kneed                     import KneeLocator                                                                         
    import matplotlib.style                                                                                                         #
    import matplotlib                as mpl                                                                                         #
    import matplotlib.pyplot         as plt                                                                                         #
    from   sklearn.cluster           import KMeans                                                                                  # Import K means clustering from sklearn module
    import itertools                                                                                                                #
    import math                                                                                                                     #
    mpl.style.use('default')      

    import os
    if diagnostics_path is None:
        diagnostics_path = os.getcwd()
    _diag_dir = os.path.join(diagnostics_path, '13_Diagnostics')

    def diagnostic_file(name):
        if not save_diagnostics:
            return None
        os.makedirs(_diag_dir, exist_ok=True)
        return os.path.join(_diag_dir, name)
    
    def _elbow_k(X, k_max, plot_title = None, save_path = None):
        # Replaces yellowbrick's KElbowVisualizer (default: distortion metric +
        # KneeLocator, convex/decreasing). Does the same as yellowbrick (fit bare KMeans() with
        # sklearn's default n_init). Returns None if no elbow. 
        from   sklearn.metrics       import pairwise_distances
        from   sklearn.preprocessing import LabelEncoder
        ks     = list(range(1, k_max))
        scores = []
        for k in ks:
            km   = KMeans(n_clusters = k).fit(X)
            le   = LabelEncoder(); le.fit(km.labels_)
            dist = 0.0
            for lab in le.classes_:
                inst   = X[km.labels_ == lab]
                center = np.array([inst.mean(axis = 0)])
                dist  += (pairwise_distances(inst, center) ** 2).sum()
            scores.append(dist)
        if len(ks) < 2:
            return None
        elbow = KneeLocator(ks, scores, curve = 'convex', direction = 'decreasing').elbow
        
        # Reproduce yellowbrick's elbow plot: distortion score vs k, elbow marked.
        fig = plt.figure(figsize = (10, 5))
        fig.patch.set_facecolor('white')
        plt.plot(ks, scores, marker = 'o', label = 'distortion score')
        if elbow is not None:
            plt.axvline(elbow, color = 'k', linestyle = '--', label = 'elbow at k = %d' % elbow)
        if plot_title is not None:
            plt.title(plot_title)
        plt.xlabel('k')
        plt.ylabel('distortion score')
        plt.legend(loc = 1)
        if save_path is not None:
            fig.savefig(save_path, dpi = 150, bbox_inches = 'tight')
        plt.close(fig)
        return elbow                                                                                       #

    ########## Initialize accepted matrix and address data type of stacked matrix #################################################################
    dictionary_input = isinstance(original_matrix, dict)
    dictionary_items = []
    if dictionary_input:
        stacked_images = []
        stacked_bvals = []
        stacked_bvecs = []
        for key, encoding in original_matrix.items():
            images = np.asarray(encoding['images'])
            if images.ndim != 4:
                raise ValueError('Dictionary encoding images must have shape [rows, columns, slices, repetitions].')
            repetitions = images.shape[3]
            dictionary_items.append((key, encoding, repetitions))
            stacked_images.append(images)
            stacked_bvals.extend([float(encoding['bval'])] * repetitions)
            stacked_bvecs.extend([np.asarray(encoding['bvec'], dtype=float)] * repetitions)
        if not stacked_images:
            raise ValueError('The encoding dictionary is empty.')
        original_matrix_stacked = np.concatenate(stacked_images, axis=3)
        original_bvals_stacked = np.asarray(stacked_bvals, dtype=float)
        original_bvecs_stacked = np.asarray(stacked_bvecs, dtype=float)
        accepted_matrix = None
    else:
        [original_matrix_stacked, original_bvals_stacked, original_bvecs_stacked] = sorted2stacked(original_matrix,
                                                                                                   original_bvals,
                                                                                                   original_bvecs)
        accepted_matrix = np.zeros_like(original_matrix)                                                # Initialize accepted matrix
    if len(np.unique(original_bvals_stacked)) != 2:
        raise ValueError('Shot rejection currently requires exactly two b-value shells; process each shell pair separately or disable rejection.')
    if np.iscomplexobj(original_matrix_stacked):                                                                                        # If data type is complex ...
        if operation_type == 'Complex':                                                                                                 # If operation type is complex ...
            if accepted_matrix is not None:
                accepted_matrix = accepted_matrix.astype(np.complex128)                                                                         # Cast accepted matrix for complex data
        if operation_type == 'Magnitude':                                                                                               # If operation type is magnitude ...
            if accepted_matrix is not None:
                accepted_matrix = accepted_matrix.astype(np.float64)                                                                            # Cast accepted matrix for magnitude data
            if not dictionary_input:
                original_matrix = np.abs(original_matrix)                                                                                           # Convert complex original matrix to magnitude data
            print('Input data type is complex, but magnitude is being executed.')                                                           # Print warning
        temporary_matrix = abs(original_matrix_stacked)                                                                                          # Create temporary matrix variable using magnitude data
    else:                                                                                                                          # Otherwise ...
        if accepted_matrix is not None:
            accepted_matrix  = accepted_matrix.astype(np.float64)                                                                           # Cast accepted matrix for magnitude data
        temporary_matrix = original_matrix_stacked                                                                                              # Create temporary matrix variable
    if saved_configuration_mode not in ('off', 'reuse', 'review'):
        raise ValueError('saved_configuration_mode must be off, reuse, or review.')
    previous = None
    if saved_configuration_mode != 'off':
        previous = _load_rejection_configuration(
            _diag_dir, original_matrix_stacked, original_bvals_stacked,
            original_bvecs_stacked, operation_type)
    if previous is not None:
        coordinates, automatic_keep, final_keep = previous
        if saved_configuration_mode == 'review':
            coordinates = INTERACT_GUI(original_matrix_stacked, organ, gui_version=gui_version,
                                       initial_coordinates=coordinates)
            final_keep = _manual_rejection_review(
                original_matrix_stacked, automatic_keep, initial_keep=final_keep,
                diagnostics_path=diagnostics_path, bvals=original_bvals_stacked,
                bvecs=original_bvecs_stacked, save_decisions=save_diagnostics)
        if save_diagnostics:
            _save_rejection_configuration(_diag_dir, original_matrix_stacked,
                                          original_bvals_stacked, original_bvecs_stacked,
                                          coordinates, automatic_keep, final_keep, operation_type)
        return _rejection_result(original_matrix, original_bvals, original_bvecs,
                                 dictionary_items, original_bvecs_stacked,
                                 coordinates, final_keep, operation_type)
    
       ########## Rejection mode: manual / ml (algorithm mode continues below unchanged) ##########
    if rejection_mode not in ('algorithm', 'manual', 'ml'):
        raise ValueError("rejection_mode must be 'algorithm', 'manual', or 'ml'.")
    if rejection_mode != 'algorithm':
        if dictionary_input:
            raise NotImplementedError("rejection_mode '%s' does not support dictionary input yet." % rejection_mode)
        n_rows, n_cols, n_slc, n_vol = original_matrix_stacked.shape
        Slice_Crop_Coordinates = [[0] * n_slc, [n_cols] * n_slc, [0] * n_slc, [n_rows] * n_slc]
        if rejection_mode == 'ml':
            from cardpy.Data_Processing.ML_Rejection import ml_keep_matrix
            ml_source = temporary_matrix
            if ml_input is not None:   # unregistered images, matching training data
                ml_source = np.abs(sorted2stacked(ml_input, original_bvals, original_bvecs)[0])
            automatic_keep, _ = ml_keep_matrix(ml_source, original_bvals_stacked, original_bvecs_stacked,
                                               model_path=ml_model_path, threshold=ml_threshold,
                                               save_dir=_diag_dir if save_diagnostics else None)
        else:   # manual: every image starts as KEEP, user decides
            automatic_keep = np.ones((n_slc, n_vol), dtype=np.uint8)
        keep_matrix_stacked = automatic_keep.copy()
        if rejection_mode == 'manual' or manual_review:
            keep_matrix_stacked = _manual_rejection_review(
                original_matrix_stacked, automatic_keep, diagnostics_path=diagnostics_path,
                bvals=original_bvals_stacked, bvecs=original_bvecs_stacked,
                save_decisions=save_diagnostics)
        if save_diagnostics:
            _save_rejection_configuration(_diag_dir, original_matrix_stacked,
                                          original_bvals_stacked, original_bvecs_stacked,
                                          Slice_Crop_Coordinates, automatic_keep,
                                          keep_matrix_stacked, operation_type)
        return _rejection_result(original_matrix, original_bvals, original_bvecs,
                                 dictionary_items, original_bvecs_stacked,
                                 Slice_Crop_Coordinates, keep_matrix_stacked, operation_type)
    
    ########## ROI Cropping for Image #############################################################################################################
    slices                 = original_matrix_stacked.shape[2]                                                                      #
    Slice_Crop_Coordinates = []                                                                                             #
    if zoom == 'ON':                                                                                                               #
        if IntERACT_zoom == 'ON':
            [x_start, x_end, y_start, y_end] = INTERACT_GUI(original_matrix_stacked, organ, gui_version=gui_version)
            Slice_Crop_Coordinates = [x_start, x_end, y_start, y_end]
            print('Slice Crop Coordinates: %s' % Slice_Crop_Coordinates)
        if IntERACT_zoom == 'OFF':
            print('Interact is off')
            x_start = []                                                                                                                   #
            x_end   = []                                                                                                                   #
            y_start = []                                                                                                                   #
            y_end   = []                                                                                                                   #
            temporary_matrix_list = []                                                                                                     #
            for slc in range(slices):                                                                            # Iterate through slices
                print(slc)
                image               = np.max(original_matrix_stacked[:, :, slc, :], axis = 2)                                                  #
                image_normalization = image / image.max()                                                                                      #
                # Select ROI
                title_pre = 'Select area for slice ' + str(slc + 1)                                                                            #
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
    if zoom == 'OFF':
        x_start = []
        x_end   = []
        y_start = []
        y_end   = []
        temporary_matrix_list = []
        for slc in range(original_matrix_stacked.shape[2]):
            x_start.append(int(0))
            x_end.append(int(original_matrix_stacked.shape[1]))
            y_start.append(int(0))
            y_end.append(int(original_matrix_stacked.shape[0]))
        Slice_Crop_Coordinates = [x_start, x_end, y_start, y_end]

    ### Extract Crop Coordinates
    x_start = Slice_Crop_Coordinates[0]
    x_end   = Slice_Crop_Coordinates[1]
    y_start = Slice_Crop_Coordinates[2]
    y_end   = Slice_Crop_Coordinates[3]
    ########## Identify high and low b-values #####################################################################################################
    (unique, index, counts) = np.unique(original_bvals_stacked, return_counts = True, return_index = True, axis = 0)                                                        # Extract number of different diffusion directions
    numBvals                = len(unique)                                                                                                                           # Identify number of b-values in stacked b-values
    bval_low                = min(unique)                                                                                                                           # Identify lower b-value (typically b = 0)
    bval_low_indicies       = np.where(original_bvals_stacked == bval_low)[0]                                                                                               # Identify index of lower b-value in stacked b-values
    bval_high               = max(unique)                                                                                                                           # Identify higher b-value
    bval_high_indicies      = np.where(original_bvals_stacked == bval_high)[0]                                                                                              # Identify indices of higher b-values in stacked b-values
    ########## Low b-value rejection ##############################################################################################################
    SSIM_bvl_v_bvl  = np.zeros([len(bval_low_indicies), len(bval_low_indicies), slices])                                                                                # Initialize structure similarity index measure (SSIM) for low b-value comparison matrix
    NSSIM_bvl_v_bvl = np.zeros([len(bval_low_indicies), len(bval_low_indicies), slices])                                                                                # Initialize normalized SSIM (NSSIM) for low b-value comparison matrix
    RMSE_bvl_v_bvl  = np.zeros([len(bval_low_indicies), len(bval_low_indicies), slices])                                                                                # Initialize root mean squared error (RMSE) for low b-value comparison matrix
    NRMSE_bvl_v_bvl = np.zeros([len(bval_low_indicies), len(bval_low_indicies), slices])                                                                                # Initialize normalized root mean squared error (NRMSE) for low b-value
    ##### SSIM low b-value across averages #####
    for slc in range(slices):                                                                                                                                       # Iterate through slices
        for avg_i in range(len(bval_low_indicies)):                                                                                                                                   # Iterate through averages (ith average)
            for avg_j in range(len(bval_low_indicies)):                                                                                                                                   # Iterate through averages (jth average)
                SSIM_bvl_v_bvl[avg_i, avg_j, slc] = ssim(temporary_matrix[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, bval_low_indicies[avg_i]],
                                                         temporary_matrix[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, bval_low_indicies[avg_j]], data_range=2.0)                           # Compute SSIM between ith and jth low b-value averages
                RMSE_bvl_v_bvl[avg_i, avg_j, slc] = np.sqrt(mse(temporary_matrix[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, bval_low_indicies[avg_i]],
                                                                temporary_matrix[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, bval_low_indicies[avg_j]]))                   # Compute RMSE between ith and jth low b-value averages
        NSSIM_bvl_v_bvl[:, :, slc] = SSIM_bvl_v_bvl[:, :, slc] / SSIM_bvl_v_bvl[:, :, slc].max()                                          # Normalize SSIM across all averages
        NRMSE_bvl_v_bvl[:, :, slc] = RMSE_bvl_v_bvl[:, :, slc] / RMSE_bvl_v_bvl[:, :, slc].max()                                          # Normalize RMSE across all averages
        ###### ######
    NSSIM_AoA_pre = np.zeros((len(bval_low_indicies), slices))                                                                                            # Initialize NSSIM assesment of averages (AoA) for pre automatic acquisition rejection
    NRMSE_AoA_pre = np.zeros((len(bval_low_indicies), slices))                                                                                            # Initialize NRMSE assesment of averages (AoA) for pre automatic acquisition rejection
    for slc in range(slices):                                                                                                                                       # Iterate through slices
        #for baseline in range(len(bval_low_indicies)):                                                                                                                  # Iterate through low b-values
        NSSIM_AoA_pre[:, slc] = np.sum(NSSIM_bvl_v_bvl[:, :, slc], axis = 0) / np.sum(NSSIM_bvl_v_bvl[:, :, slc], axis = 0).max()         # Calcualte NSSIM AoA for pre automatic acquisition rejection
        NRMSE_AoA_pre[:, slc] = np.sum(NRMSE_bvl_v_bvl[:, :, slc], axis = 0) / np.sum(NRMSE_bvl_v_bvl[:, :, slc], axis = 0).max()         # Calcualte NRMSE AoA for pre automatic acquisition rejection
        #-------------------------------------
    for slc in range(slices):                                                                                                                                       # Iterate through slices
            NSSIM_AoA_pre_list = NSSIM_AoA_pre[:, slc].tolist()                                                                                                   # Convert NSSIM AoA for pre automatic acquisition rejection to a list
            NRMSE_AoA_pre_list = NRMSE_AoA_pre[:, slc].tolist()                                                                                                   # Convert NRMSE AoA for pre automatic acquisition rejection to a list
            if  all(x >= NSSIM_threshold for x in NSSIM_AoA_pre_list) and all(x >= NRMSE_threshold for x in NRMSE_AoA_pre_list):                                            # If both lists are greater than or equal to respective thresholds ...
                continue                                                                                                                                                        # Continue on to next block of code
            else:                                                                                                                                                           # Otherwise ...
                for avg in range(len(NSSIM_AoA_pre_list)):                                                                                                                      # Iterate through averages
                    if NRMSE_AoA_pre_list[avg] > NRMSE_threshold and NSSIM_AoA_pre_list[avg] < NSSIM_threshold:                                                                     # If NRMSE AoA pre is greater than NRMSE threshold and NSSIM AoA pre is less than NSSIM threshold ...
                        NSSIM_bvl_v_bvl[avg,   :, slc] = 0                                                                                                                    # Set current average columns of NSSIM for low b-value comparison matrix to zero (automatically reject acquisition)
                        NSSIM_bvl_v_bvl[  :, avg, slc] = 0                                                                                                                    # Set current average rows of NSSIM for low b-value comparison matrix to zero (automatically reject acquisition)
                        NRMSE_bvl_v_bvl[avg,   :, slc] = 0                                                                                                                    # Set current average columns of NRMSE for low b-value comparison matrix to zero (automatically reject acquisition)
                        NRMSE_bvl_v_bvl[  :, avg, slc] = 0                                                                                                                    # Set current average rows of NRMSE for low b-value comparison matrix to zero (automatically reject acquisition)
    NSSIM_AoA_post = np.zeros((len(bval_low_indicies), slices))                                                                                           # Initialize NSSIM assesment of averages (AoA) for post automatic acquisition rejection
    NRMSE_AoA_post = np.zeros((len(bval_low_indicies), slices))                                                                                           # Initialize NRMSE assesment of averages (AoA) for post automatic acquisition rejection
    for slc in range(slices):                                                                                                                                       # Iterate through slices
        NSSIM_AoA_post[:, slc] = np.sum(NSSIM_bvl_v_bvl[:, :, slc], axis = 0) / np.sum(NSSIM_bvl_v_bvl[:, :, slc], axis = 0).max()        # Calcualte NSSIM AoA for post automatic acquisition rejection
        NRMSE_AoA_post[:, slc] = np.sum(NRMSE_bvl_v_bvl[:, :, slc], axis = 0) / np.sum(NRMSE_bvl_v_bvl[:, :, slc], axis = 0).max()        # Calcualte NRMSE AoA for post automatic acquisition rejection
        if np.isnan(NRMSE_AoA_post[:, slc]).any():
            NRMSE_AoA_post[:, slc] = 1
    kmeans_labels   = np.zeros(NSSIM_AoA_post.shape)                                                                                                                # Initialize k means labels matrix
    keep_bvl_matrix = np.zeros(NSSIM_AoA_post.shape)

    k_means_cluster = []
    for slc in range(slices):                                                                                                                                       # Iterate through slices
        NSSIM_AoA_post_list = NSSIM_AoA_post[:, slc].tolist()                                                                                                 # Convert NSSIM AoA for post automatic acquisition rejection to a list
        NRMSE_AoA_post_list = NRMSE_AoA_post[:, slc].tolist()                                                                                                 # Convert NRMSE AoA for post automatic acquisition rejection to a list
        data = np.hstack((NSSIM_AoA_post[:, slc, np.newaxis], NRMSE_AoA_post[:, slc, np.newaxis]))                                                  # Combine NSSIM AoA post and NRMSE AoA post into an [avg,2] shape
        tile_size = int(np.floor(30 / data.shape[0]))
        data_estimate = np.tile(data, (tile_size, 1))
        if data.shape[0] > 1:
            _elbow_value = _elbow_k(data_estimate, data_estimate.shape[0] + 1, plot_title = 'K-Means Estimate (Low $\\it{b-value}$) for Slice %i' % int(slc + 1), save_path = diagnostic_file('KMeans_Estimate_LowB_Slice_%02d.png' % int(slc + 1)))
            k_means_cluster.append(_elbow_value)
            NSSIM_AoA_post_list_temp = [i for i in NSSIM_AoA_post_list if i != 0]
            NRMSE_AoA_post_list_temp = [i for i in NRMSE_AoA_post_list if i != 0]
            if all(x >= NSSIM_threshold for x in NSSIM_AoA_post_list_temp) and all(x >= NRMSE_threshold for x in NRMSE_AoA_post_list_temp):                                           # If both lists are greater than or equal to respective thresholds ...
                if all(x >= NSSIM_threshold for x in NSSIM_AoA_post_list) and all(x >= NRMSE_threshold for x in NRMSE_AoA_post_list):
                    k_mean_initialization        = np.array(([1.0, 1.0]))                                                                                                           # Initalize only one cluster centroid
                    k_means                      = KMeans(n_clusters = 1, algorithm = 'elkan', n_init = 1).fit(data)                                                                # Run k-means algorithm
                    k_means_labels               = k_means.labels_                                                                                                                  # Extract k-means labels
                    kmeans_labels[:, slc]   = k_means.labels_                                                                                                                  # Store k-means labels
                    keep_bvl_matrix[:, slc] = (kmeans_labels[:, slc] == 0).astype('int')                                                                                  # Store keep low b-values based off k-means labels
                else:
                    k_mean_initialization        = np.array(([0, 0], [1.0, 1.0]))                                                                                                           # Initalize only one cluster centroid
                    k_means                      = KMeans(n_clusters = 2, algorithm = 'elkan', init = k_mean_initialization, n_init = 1).fit(data)                                                                # Run k-means algorithm
                    k_means_labels               = k_means.labels_                                                                                                                  # Extract k-means labels
                    kmeans_labels[:, slc]   = k_means.labels_                                                                                                                  # Store k-means labels
                    keep_bvl_matrix[:, slc] = (kmeans_labels[:, slc] >= 1).astype('int')                                                                         # Store keep low b-values based off k-means labels
            else:                                                                                                                                                           # Otherwise ...
                distance_max = 0
                for ii in range(len(NSSIM_AoA_post_list_temp)):
                    for jj in range(len(NRMSE_AoA_post_list_temp)):
                        distance = math.dist([NSSIM_AoA_post_list_temp[ii], NRMSE_AoA_post_list_temp[ii]], [NSSIM_AoA_post_list_temp[jj], NRMSE_AoA_post_list_temp[jj]])
                        if distance > distance_max:
                            distance_max = distance
                            point_1_temp = [NSSIM_AoA_post_list_temp[ii], NRMSE_AoA_post_list_temp[ii]]
                            point_2_temp = [NSSIM_AoA_post_list_temp[jj], NRMSE_AoA_post_list_temp[jj]]
                if point_1_temp[0] >  point_2_temp[0]:
                    point_1 = point_2_temp
                    point_2 = point_1_temp
                else:
                    point_1 = point_1_temp
                    point_2 = point_2_temp
                X = [point_1[0], point_2[0]]
                Y = [point_1[1], point_2[1]]
                a, b = np.polyfit(X, Y, 1)                                                                                                                                      # Calculate linear fit of X and Y data
                X_vals = np.linspace(min(X), max(X), k_means_cluster[slc])
                Y_vals = a * X_vals + b
                Fit_Centers = np.hstack((X_vals[np.newaxis, :].T, Y_vals[np.newaxis, :].T))
                if (0 in NSSIM_AoA_post_list and 0 in NRMSE_AoA_post_list):                                                                                                   # Else if automatic acquisition rejection is detected ...
                    Fit_Centers = np.insert(Fit_Centers, 0, [0, 0], axis = 0)
                    k_mean_initialization        = Fit_Centers                                                    # Initialize four cluster centroids (one for automatic acquisiton rejection)
                    k_means                      = KMeans(n_clusters = k_means_cluster[slc] + 1, algorithm = 'elkan', init = k_mean_initialization, n_init = 1).fit(data)                                  # Run k-means algorithm
                    k_means_labels               = k_means.labels_                                                                                                                  # Extract k-means labels
                    kmeans_labels[:, slc]        = k_means.labels_                                                                                                                  # Store k-means labels
                    if len(np.unique(k_means_labels)) == 2:
                        keep_bvl_matrix[:, slc] = (kmeans_labels[:, slc] >= 1).astype('int')                                                                                  # Store keep low b-values based off k-means labels
                    else:
                        keep_bvl_matrix[:, slc] = (kmeans_labels[:, slc] >= 2).astype('int')
                else:                                                                                                                                                           # Otherwise ...
                    k_mean_initialization        = Fit_Centers                                                              # Initialize three cluster centroids
                    k_means                      = KMeans(n_clusters = k_means_cluster[slc], algorithm = 'elkan', init = k_mean_initialization, n_init = 1).fit(data)                                  # Run k-means algorithm
                    k_means_labels               = k_means.labels_                                                                                                                  # Extract k-means labels
                    kmeans_labels[:, slc]        = k_means.labels_                                                                                                                  # Store k-means labels
                    keep_bvl_matrix[:, slc]      = (kmeans_labels[:, slc] >= 1).astype('int')                                                                                  # Store keep low b-values based off k-means labels
        else:
            k_means_cluster.append(1)
            k_mean_initialization             = np.array(([1.0, 1.0]))                                                                                                      # Initalize only one cluster centroid
            k_means                           = KMeans(n_clusters = 1, algorithm = 'elkan', n_init = 1).fit(data)                                                           # Run k-means algorithm
            k_means_labels                    = k_means.labels_                                                                                                             # Extract k-means labels
            kmeans_labels[:, slc]   = k_means.labels_                                                                                                             # Store k-means labels
            keep_bvl_matrix[:, slc] = (kmeans_labels[:, slc] == 0).astype('int')
        fig = plt.figure(figsize=(10, 5))
        fig.patch.set_facecolor('white')
        plt.scatter(NSSIM_AoA_post[:, slc], NRMSE_AoA_post[:, slc], c = kmeans_labels[:, slc], marker = 'o')

        plt.axhline(NRMSE_threshold, color = 'red', label = 'NRMSE Threshold')
        plt.axvline(NSSIM_threshold, color = 'blue', label = 'NSSIM Threshold')
        plt.title(r'K-Means Cluster Assignemnts (Low $\it{b-value}$) for Slice %i' %int(slc + 1))
        plt.plot(k_means.cluster_centers_[:,0], k_means.cluster_centers_[:,1], 'gx', label = 'Cluster Centers')
        plt.legend(loc = 8)
        plt.xlabel('NSSIM')
        plt.ylabel('NRMSE')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        if save_diagnostics:
            fig.savefig(diagnostic_file('KMeans_Clusters_LowB_Slice_%02d.png' % int(slc + 1)), dpi = 150, bbox_inches = 'tight')
        plt.close(fig)
    ########## High b-value rejection ##############################################################################################################
    SSIM_bvh_v_bvh  = np.zeros([len(bval_high_indicies), len(bval_high_indicies), slices])                                                                                # Initialize structure similarity index measure (SSIM) for low b-value comparison matrix
    NSSIM_bvh_v_bvh = np.zeros([len(bval_high_indicies), len(bval_high_indicies), slices])                                                                                # Initialize normalized SSIM (NSSIM) for low b-value comparison matrix
    RMSE_bvh_v_bvh  = np.zeros([len(bval_high_indicies), len(bval_high_indicies), slices])                                                                                # Initialize root mean squared error (RMSE) for low b-value comparison matrix
    NRMSE_bvh_v_bvh = np.zeros([len(bval_high_indicies), len(bval_high_indicies), slices])                                                                                # Initialize normalized root mean squared error (NRMSE) for low b-value
    ##### SSIM high b-value across all diffusion data #####
    for slc in range(slices):                                                                                                                                       # Iterate through slices
        for avg_i in range(len(bval_high_indicies)):                                                                                                                                   # Iterate through averages (ith average)
            for avg_j in range(len(bval_high_indicies)):                                                                                                                                   # Iterate through averages (jth average)
                SSIM_bvh_v_bvh[avg_i, avg_j, slc] = ssim(temporary_matrix[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, bval_high_indicies[avg_i]],
                                                         temporary_matrix[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, bval_high_indicies[avg_j]], data_range=2.0)                           # Compute SSIM between ith and jth low b-value averages
                RMSE_bvh_v_bvh[avg_i, avg_j, slc] = np.sqrt(mse(temporary_matrix[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, bval_high_indicies[avg_i]],
                                                                temporary_matrix[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, bval_high_indicies[avg_j]]))                   # Compute RMSE between ith and jth low b-value averages
        NSSIM_bvh_v_bvh[:, :, slc] = SSIM_bvh_v_bvh[:, :, slc] / SSIM_bvh_v_bvh[:, :, slc].max()                                          # Normalize SSIM across all averages
        NRMSE_bvh_v_bvh[:, :, slc] = RMSE_bvh_v_bvh[:, :, slc] / RMSE_bvh_v_bvh[:, :, slc].max()                                          # Normalize RMSE across all averages
    ###### ######
    NSSIM_AoA_pre = np.zeros((len(bval_high_indicies), slices))                                                                                            # Initialize NSSIM assesment of averages (AoA) for pre automatic acquisition rejection
    NRMSE_AoA_pre = np.zeros((len(bval_high_indicies), slices))                                                                                            # Initialize NRMSE assesment of averages (AoA) for pre automatic acquisition rejection
    for slc in range(slices):                                                                                                                                       # Iterate through slices
        NSSIM_AoA_pre[:, slc] = np.sum(NSSIM_bvh_v_bvh[:, :, slc], axis = 0) / np.sum(NSSIM_bvh_v_bvh[:, :, slc], axis = 0).max()         # Calcualte NSSIM AoA for pre automatic acquisition rejection
        NRMSE_AoA_pre[:, slc] = np.sum(NRMSE_bvh_v_bvh[:, :, slc], axis = 0) / np.sum(NRMSE_bvh_v_bvh[:, :, slc], axis = 0).max()         # Calcualte NRMSE AoA for pre automatic acquisition rejection
        #---------------------
    for slc in range(slices):                                                                                                                                       # Iterate through slices
        NSSIM_AoA_pre_list = NSSIM_AoA_pre[:, slc].tolist()                                                                                                      # Convert NSSIM AoA for pre automatic acquisition rejection to a list
        NRMSE_AoA_pre_list = NRMSE_AoA_pre[:, slc].tolist()                                                                                                      # Convert NSSIM AoA for pre automatic acquisition rejection to a list
        if  all(x >= NSSIM_threshold for x in NSSIM_AoA_pre_list) and all(x < NRMSE_threshold for x in NRMSE_AoA_pre_list):                                                # If both lists are greater than or equal to respective thresholds ...
            continue                                                                                                                                                        # Continue on to next block of code
        else:                                                                                                                                                           # Otherwise ...
            for avg in range(len(NSSIM_AoA_pre_list)):                                                                                                                      # Iterate through averages
                if NRMSE_AoA_pre_list[avg] >= NRMSE_threshold and NSSIM_AoA_pre_list[avg] < NSSIM_threshold:                                                                         # If NMI AoA pre is greater than NMI threshold and NSSIM AoA pre is less than NSSIM threshold ...
                    NSSIM_bvh_v_bvh[avg,   :, slc] = np.nan                                                                                                                         # Set current average columns of NSSIM for high b-value comparison matrix to zero (automatically reject acquisition)
                    NSSIM_bvh_v_bvh[  :, avg, slc] = np.nan                                                                                                                         # Set current average rows of NSSIM for high b-value comparison matrix to zero (automatically reject acquisition)
                    NRMSE_bvh_v_bvh[avg,   :, slc] = np.nan                                                                                                                         # Set current average columns of NRMSE for high b-value comparison matrix to zero (automatically reject acquisition)
                    NRMSE_bvh_v_bvh[  :, avg, slc] = np.nan                                                                                                                         # Set current average rows of NRMSE for high b-value comparison matrix to zero (automatically reject acquisition)
            NRMSE_AoA_pre_list = NRMSE_bvh_v_bvh[:, :, slc].tolist()
            NRMSE_AoA_pre_list = list(itertools.chain.from_iterable(NRMSE_AoA_pre_list))
            if  sum(NRMSE_AoA_pre_list) == 0:
                for avg in range(len(NSSIM_AoA_pre_list)):
                    if  NSSIM_bvh_v_bvh[avg, avg, dif, slc] == 1:
                        NRMSE_bvh_v_bvh[avg, avg, dif, slc] = 1
    NSSIM_AoA_post = np.zeros((len(bval_high_indicies), slices))                                                                                          # Initialize NSSIM AoA for post automatic acquisition rejection
    NRMSE_AoA_post = np.zeros((len(bval_high_indicies), slices))                                                                                          # Initialize NRMSE AoA for post automatic acquisition rejection
    for slc in range(slices):                                                                                                                                       # Iterate through slices
        for dif in range(len(bval_high_indicies)):                                                                                                                      # Iterate through high b-values
            NSSIM_AoA_post[:, slc] = np.nansum(NSSIM_bvh_v_bvh[:, :, slc], axis = 0) / np.nansum(NSSIM_bvh_v_bvh[:, :, slc], axis = 0).max()                       # Calcualte NSSIM AoA for post automatic acquisition rejection
            NRMSE_AoA_post[:, slc] = np.nansum(NRMSE_bvh_v_bvh[:, :, slc], axis = 0) / np.nansum(NRMSE_bvh_v_bvh[:, :, slc], axis = 0).max()                       # Calcualte NRMSE AoA for post automatic acquisition rejection
    kmeans_labels   = np.zeros(NSSIM_AoA_post.shape)                                                                                                                # Initialize k means labels matrix
    keep_bvh_matrix = np.zeros(NSSIM_AoA_post.shape)
    k_means_cluster = []

    for slc in range(slices):                                                                                                                                       # Iterate through slices
        NSSIM_AoA_post_list = NSSIM_AoA_post[:, slc].tolist()                                                                                                 # Convert NSSIM AoA for post automatic acquisition rejection to a list
        NRMSE_AoA_post_list = NRMSE_AoA_post[:, slc].tolist()                                                                                                 # Convert NRMSE AoA for post automatic acquisition rejection to a list
        data = np.hstack((NSSIM_AoA_post[:, slc, np.newaxis], NRMSE_AoA_post[:, slc, np.newaxis]))                                                  # Combine NSSIM AoA post and NRMSE AoA post into an [avg,2] shape
        _elbow_value = _elbow_k(data, len(NSSIM_AoA_post_list) + 1, plot_title = 'K-Means Estimate (High $\\it{b-value}$) for Slice %i' % int(slc + 1), save_path = diagnostic_file('KMeans_Estimate_HighB_Slice_%02d.png' % int(slc + 1)))
        # handles no elbow case now as well for security (can change if unwanted)
        if _elbow_value is None or _elbow_value < 6:
            k_means_cluster.append(6)
            print('Overridding estimated number of clusters to be 5 clusters.')
        else:
            k_means_cluster.append(_elbow_value)
        NSSIM_AoA_post_list_temp = [i for i in NSSIM_AoA_post_list if i != 0]
        NRMSE_AoA_post_list_temp = [i for i in NRMSE_AoA_post_list if i != 0]
        if all(x >= NSSIM_threshold for x in NSSIM_AoA_post_list_temp) and all(x >= NRMSE_threshold for x in NRMSE_AoA_post_list_temp):                                           # If both lists are greater than or equal to respective thresholds ...
            if all(x >= NSSIM_threshold for x in NSSIM_AoA_post_list) and all(x >= NRMSE_threshold for x in NRMSE_AoA_post_list):
                k_mean_initialization        = np.array(([1.0, 1.0]))                                                                                                           # Initalize only one cluster centroid
                k_means                      = KMeans(n_clusters = 1, algorithm = 'elkan', n_init = 1).fit(data)                                                                # Run k-means algorithm
                k_means_labels               = k_means.labels_                                                                                                                  # Extract k-means labels
                kmeans_labels[:, slc]   = k_means.labels_                                                                                                                  # Store k-means labels
                keep_bvh_matrix[:, slc] = (kmeans_labels[:, slc] == 0).astype('int')                                                                                  # Store keep low b-values based off k-means labels
            else:
                k_mean_initialization        = np.array(([0, 0], [1.0, 1.0]))                                                                                                           # Initalize only one cluster centroid
                k_means                      = KMeans(n_clusters = 2, algorithm = 'elkan', init = k_mean_initialization, n_init = 1).fit(data)                                                                # Run k-means algorithm
                k_means_labels               = k_means.labels_                                                                                                                  # Extract k-means labels
                kmeans_labels[:, slc]   = k_means.labels_                                                                                                                  # Store k-means labels
                keep_bvh_matrix[:, slc] = (kmeans_labels[:, slc] >= 1).astype('int')                                                                                  # Store keep low b-values based off k-means labels
        else:                                                                                                                                                           # Otherwise ...
            distance_max = 0
            for ii in range(len(NSSIM_AoA_post_list_temp)):
                for jj in range(len(NRMSE_AoA_post_list_temp)):
                    distance = math.dist([NSSIM_AoA_post_list_temp[ii], NRMSE_AoA_post_list_temp[ii]], [NSSIM_AoA_post_list_temp[jj], NRMSE_AoA_post_list_temp[jj]])
                    if distance > distance_max:
                        distance_max = distance
                        point_1_temp = [NSSIM_AoA_post_list_temp[ii], NRMSE_AoA_post_list_temp[ii]]
                        point_2_temp = [NSSIM_AoA_post_list_temp[jj], NRMSE_AoA_post_list_temp[jj]]
            if point_1_temp[0] >  point_2_temp[0]:
                point_1 = point_2_temp
                point_2 = point_1_temp
            else:
                point_1 = point_1_temp
                point_2 = point_2_temp
            X = [point_1[0], point_2[0]]
            Y = [point_1[1], point_2[1]]
            a, b = np.polyfit(X, Y, 1)                                                                                                                                      # Calculate linear fit of X and Y data
            X_vals = np.linspace(min(X), max(X), k_means_cluster[slc])
            Y_vals = a * X_vals + b
            Fit_Centers = np.hstack((X_vals[np.newaxis, :].T, Y_vals[np.newaxis, :].T))
            if (0 in NSSIM_AoA_post_list and 0 in NRMSE_AoA_post_list):                                                                                                   # Else if automatic acquisition rejection is detected ...
                Fit_Centers = np.insert(Fit_Centers, 0, [0, 0], axis = 0)
                k_mean_initialization        = Fit_Centers                                                  # Initialize four cluster centroids (one for automatic acquisiton rejection)
                k_means                      = KMeans(n_clusters = k_means_cluster[slc] + 1, algorithm = 'elkan', init = k_mean_initialization, n_init = 1, verbose = 1).fit(data)                                  # Run k-means algorithm
                k_means_labels               = k_means.labels_                                                                                                                  # Extract k-means labels
                kmeans_labels[:, slc]   = k_means.labels_                                                                                                                  # Store k-means labels
                if len(np.unique(k_means_labels)) == 2:
                    keep_bvh_matrix[:, slc] = (kmeans_labels[:, slc] >= 1).astype('int')                                                                                  # Store keep low b-values based off k-means labels
                else:
                    keep_bvh_matrix[:, slc] = (kmeans_labels[:, slc] >= 2).astype('int')
            else:                                                                                                                                                           # Otherwise ...
                k_mean_initialization        = Fit_Centers                                                                   # Initialize three cluster centroids
                k_means                      = KMeans(n_clusters = k_means_cluster[slc], algorithm = 'elkan', init = k_mean_initialization, n_init = 1, verbose = 1).fit(data)                                  # Run k-means algorithm
                k_means_labels               = k_means.labels_                                                                                                                  # Extract k-means labels
                kmeans_labels[:, slc]   = k_means.labels_                                                                                                                  # Store k-means labels
                keep_bvh_matrix[:, slc] = (kmeans_labels[:, slc] >= 1).astype('int')                                                                                  # Store keep low b-values based off k-means labels
        fig = plt.figure(figsize=(10, 5))
        fig.patch.set_facecolor('white')
        plt.scatter(NSSIM_AoA_post[:, slc], NRMSE_AoA_post[:, slc], c = kmeans_labels[:, slc], marker = 'o')
        plt.axhline(NRMSE_threshold, color = 'red', label = 'NRMSE Threshold')
        plt.axvline(NSSIM_threshold, color = 'blue', label = 'NSSIM Threshold')
        plt.title(r'K-Means Cluster Assignemnts (High $\it{b-value}$) for Slice %i' %int(slc + 1))
        plt.plot(k_means.cluster_centers_[:,0], k_means.cluster_centers_[:,1], 'gx', label = 'Cluster Centers')
        plt.legend(loc=8)
        plt.xlabel('NSSIM')
        plt.ylabel('NRMSE')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        if save_diagnostics:
            fig.savefig(diagnostic_file('KMeans_Clusters_HighB_Slice_%02d.png' % int(slc + 1)), dpi = 150, bbox_inches = 'tight')
        plt.close(fig)
    keep_matrix_stacked = np.zeros([original_matrix_stacked.shape[2], original_matrix_stacked.shape[3]])
    for slc in range(keep_bvl_matrix.shape[1]):
        for acquisition in range(keep_bvl_matrix.shape[0]):
            keep_matrix_stacked[slc, bval_low_indicies[acquisition]] = keep_bvl_matrix[acquisition, slc]
    for slc in range(keep_bvh_matrix.shape[1]):
        for acquisition in range(keep_bvh_matrix.shape[0]):
            keep_matrix_stacked[slc, bval_high_indicies[acquisition]] = keep_bvh_matrix[acquisition, slc]
    automatic_keep = keep_matrix_stacked.copy()
    if manual_review:
        keep_matrix_stacked = _manual_rejection_review(
            original_matrix_stacked, keep_matrix_stacked, diagnostics_path=diagnostics_path,
            bvals=original_bvals_stacked, bvecs=original_bvecs_stacked,
            save_decisions=save_diagnostics)
    if save_diagnostics:
        _save_rejection_configuration(_diag_dir, original_matrix_stacked,
                                      original_bvals_stacked, original_bvecs_stacked,
                                      Slice_Crop_Coordinates, automatic_keep,
                                      keep_matrix_stacked, operation_type)
    return _rejection_result(original_matrix, original_bvals, original_bvecs,
                             dictionary_items, original_bvecs_stacked,
                             Slice_Crop_Coordinates, keep_matrix_stacked, operation_type)


def _rejection_result(original_matrix, original_bvals, original_bvecs, dictionary_items,
                      original_bvecs_stacked, Slice_Crop_Coordinates,
                      keep_matrix_stacked, operation_type):
    import numpy as np
    dictionary_input = isinstance(original_matrix, dict)
    if dictionary_input:
        accepted_encodings, stats_by_encoding, keep_by_encoding = _apply_dictionary_keep(
            dictionary_items, keep_matrix_stacked)
        if operation_type == 'Magnitude':
            for encoding in accepted_encodings.values():
                if np.iscomplexobj(encoding['images']):
                    encoding['images'] = np.abs(encoding['images'])
        for key, encoding, _ in dictionary_items:
            for slc, rate in enumerate(stats_by_encoding[key]):
                print('Acceptance rate for slice %i, b=%g, vector=%s: %.1f%%'
                      % (slc + 1, encoding['bval'], np.asarray(encoding['bvec']), rate))
                if not np.any(keep_by_encoding[key][slc]):
                    print('All repetitions were rejected for slice %i, b=%g, vector=%s; '
                          'retaining repetition 0.'
                          % (slc + 1, encoding['bval'], np.asarray(encoding['bvec'])))
        accepted_bvals = np.asarray([encoding['bval'] for encoding in accepted_encodings.values()])
        accepted_bvecs = np.asarray([encoding['bvec'] for encoding in accepted_encodings.values()])
        return [accepted_encodings, accepted_bvals, accepted_bvecs,
                Slice_Crop_Coordinates, stats_by_encoding, keep_by_encoding]
    ########## Identify new matrix dimensions ######################################################################################################
    numRow                  = original_matrix.shape[0]                                                                   # Define number of rows (x)
    numCol                  = original_matrix.shape[1]                                                                   # Define number of columns (y)
    numSlc                  = keep_matrix_stacked.shape[0]                                                                   # Define number of slices (z)
    (unique, index, counts) = np.unique(original_bvecs_stacked, return_counts = True, return_index = True, axis = 0)             # Extract number of different diffusion directions
    unique                  = unique[np.argsort(index)]                                                                 # Retain stacked diffusion direction order
    numDif                  = len(unique)                                                                               # Define number of diffusion directions
    numAvg                  = max(counts)                                                                               # Define number of averages
    keep_matrix_sorted = np.zeros([numSlc, numDif, numAvg])
    stacked_index_list = []                                                                                        # Initialize stacked index list
    for dif in range(numDif):                                                                                           # Iterate through diffusion directions
        idxAvg = -1                                                                                                         # Initialize average index
        for idx in range(keep_matrix_stacked.shape[1]):                                                                          # Iterate through stacked diffusion dimension
            current_search = unique[dif]                                                                                        # Define current diffusion direction
            if np.array_equal(current_search, original_bvecs_stacked[idx, :]) == True:                                                   # Check if current diff dir matches desired diff dir ...
                idxAvg                                 = idxAvg + 1                                                                 # Index current average
                idxDif                                 = np.where(np.all(unique == original_bvecs_stacked[idx, :],axis = 1))[0][0]           # Index current diffuion direction
                keep_matrix_sorted[:, idxDif, idxAvg]  = keep_matrix_stacked[:, idx]                                               # Store sorted matrix from current index of
    stats = np.zeros([keep_matrix_sorted.shape[0], keep_matrix_sorted.shape[1]])
    for slc in range(stats.shape[0]):
        for dif in range(stats.shape[1]):
            stats[slc, dif] = (np.sum(keep_matrix_sorted[slc, dif, :]) / keep_matrix_sorted.shape[2]) * 100
            print('Acceptance rate for slice %i, direction %i: %.1f%%' %(int(slc + 1), int(dif + 1), stats[slc, dif]))
    accepted_matrix = _apply_sorted_keep(original_matrix, keep_matrix_sorted)
    accepted_bvals = original_bvals
    accepted_bvecs = original_bvecs
    return[accepted_matrix, accepted_bvals, accepted_bvecs, Slice_Crop_Coordinates, stats, keep_matrix_sorted]


def _manual_rejection_review(images, keep, diagnostics_path=None, bvals=None,
                             bvecs=None, save_decisions=True, initial_keep=None,
                             show_automatic=True, slice_labels=None, allow_cancel=False):
    """Open a native Tk window for each slice; click a tile to toggle its decision.

    Zoom: the slider zooms toward the image center; 'Choose zoom region' lets the
    user drag a box on an enlarged overview so every tile shows that region.
    The chosen region carries over to later slices until reset.
    """
    import base64
    import math
    import json
    import os
    import struct
    import zlib
    import numpy as np
    import tkinter as tk
    from tkinter import ttk
    from cardpy.GUI_Tools._runtime import create_window, wait_window

    automatic_keep = np.asarray(keep, dtype=np.uint8).copy()
    keep = automatic_keep.copy() if initial_keep is None else np.asarray(initial_keep, dtype=np.uint8).copy()
    n_slices, n_images = keep.shape
    columns = min(6, max(1, n_images))
    zoom_box = {'bounds': None}   # (y0, y1, x0, x1) in image pixels; persists across slices
    for slc in range(n_slices):
        slice_label = slice_labels[slc] if slice_labels is not None else f'{slc + 1}/{n_slices}'
        window = create_window(f'CarDpy: Manual shot rejection — slice {slice_label}')
        window.geometry('1120x820')
        window.minsize(720, 500)
        window.columnconfigure(0, weight=1)
        window.columnconfigure(1, weight=0)
        window.rowconfigure(2, weight=1)
        ttk.Label(window, text=f'Slice {slice_label}: click an image to toggle keep/reject',
                  font=('Verdana', 12, 'bold'), anchor='center').grid(
                      row=0, column=0, sticky='ew', padx=10, pady=8)

        zoom_bar = ttk.Frame(window, padding=(10, 0, 10, 6))
        zoom_bar.grid(row=1, column=0, sticky='ew')
        ttk.Label(zoom_bar, text='Center zoom').pack(side='left', padx=(0, 8))
        ttk.Button(zoom_bar, text='Choose zoom region…',
                   command=lambda: choose_region()).pack(side='left', padx=4)
        ttk.Button(zoom_bar, text='Reset zoom',
                   command=lambda: reset_zoom()).pack(side='left', padx=4)
        zoom_value = tk.DoubleVar(value=1.0)
        zoom_label = ttk.Label(zoom_bar, text='100%', width=10, anchor='e')
        zoom_label.pack(side='right', padx=(8, 0))
        zoom_scale = ttk.Scale(zoom_bar, from_=1.0, to=4.0, variable=zoom_value,
                               orient='horizontal', length=300)
        zoom_scale.pack(side='right')

        container = ttk.Frame(window)
        container.grid(row=2, column=0, sticky='nsew', padx=8)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')

        intensity_panel = ttk.Frame(window, padding=(4, 12, 10, 12))
        intensity_panel.grid(row=2, column=1, sticky='ns')
        for column_index in range(2):
            intensity_panel.columnconfigure(column_index, weight=1)
        ttk.Label(intensity_panel, text='Intensity\n0 → Vmax').grid(
            row=0, column=0, pady=(0, 6))
        ttk.Label(intensity_panel, text='Vmax').grid(row=0, column=1, pady=(0, 6))
        colorbar = tk.Canvas(intensity_panel, width=44, height=260, highlightthickness=0,
                            cursor='hand2')
        colorbar.grid(row=1, column=0, sticky='ns', padx=(4, 8))
        tiles = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=tiles, anchor='nw')
        tiles.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda event: canvas.itemconfigure(canvas_window, width=event.width))
        def scroll_canvas(event):
            step = int(-event.delta / 120) or (-1 if event.delta > 0 else 1)
            canvas.yview_scroll(step, 'units')
        window.bind_all('<MouseWheel>', scroll_canvas)

        image_refs = [None] * n_images
        magnitude_images = []
        tile_widgets = []
        for index in range(n_images):
            magnitude = np.abs(images[:, :, slc, index])
            magnitude_images.append(np.asarray(magnitude, dtype=np.float32))
            tile = tk.Frame(tiles, borderwidth=5, relief='solid', padx=3, pady=3)
            tile.grid(row=index // columns, column=index % columns, padx=5, pady=5, sticky='n')
            picture = tk.Label(tile, cursor='hand2')
            picture.pack()
            status = tk.Label(tile, font=('Verdana', 9, 'bold'), cursor='hand2')
            status.pack(fill='x')
            tile_widgets.append((tile, picture, status, index))

        finite_values = [im[np.isfinite(im)] for im in magnitude_images if np.isfinite(im).any()]
        finite_values = np.concatenate(finite_values) if finite_values else np.asarray([0.0, 1.0])
        intensity_max = float(np.percentile(finite_values, 99.5))
        data_max = float(np.max(finite_values))
        if not np.isfinite(intensity_max) or intensity_max <= 0:
            intensity_max = max(data_max, 1.0)
        if not np.isfinite(data_max) or data_max <= 0:
            data_max = intensity_max
        # Keep the initial display robust to outliers while allowing Vmax to
        # extend well above every pixel in the current slice.
        vmax_control_max = max(data_max * 2.0, intensity_max * 4.0, 1.0)
        maximum_value = tk.DoubleVar(value=intensity_max)
        maximum_scale = tk.Scale(intensity_panel, from_=vmax_control_max, to=0.0,
                                 variable=maximum_value, orient='vertical', length=260,
                                 width=26, sliderlength=30, showvalue=False,
                                 resolution=max(vmax_control_max / 500, 1e-6),
                                 highlightthickness=0, bd=0, troughcolor='#666666')
        maximum_scale.grid(row=1, column=1, sticky='ns')
        minimum_text = ttk.Label(intensity_panel, text='0', width=8, anchor='center')
        maximum_text = ttk.Label(intensity_panel, text=f'{intensity_max:.3g}', width=8, anchor='center')
        minimum_text.grid(row=2, column=0, pady=(6, 0))
        maximum_text.grid(row=2, column=1, pady=(6, 0))

        def draw_colorbar():
            colorbar.delete('all')
            height = int(colorbar.cget('height'))
            width = int(colorbar.cget('width'))
            for y in range(height):
                gray = round(255 * (1 - y / max(1, height - 1)))
                colorbar.create_line(0, y, width, y, fill=f'#{gray:02x}{gray:02x}{gray:02x}')
            colorbar.create_line(0, 0, width, 0, fill='#ffca28', width=3)
            colorbar.create_line(0, height - 1, width, height - 1,
                                fill='#e53935', width=3)

        def update_intensity(value):
            high = min(vmax_control_max, max(float(value), np.finfo(float).eps))
            maximum_value.set(high)
            maximum_text.configure(text=f'{high:.3g}')
            draw_colorbar()
            refresh_tiles()

        maximum_scale.configure(command=update_intensity)

        def set_vmax_from_colorbar(event):
            fraction = 1.0 - min(1.0, max(0.0, event.y / max(1, int(colorbar.cget('height')) - 1)))
            value = fraction * vmax_control_max
            maximum_scale.set(value)
            update_intensity(value)

        # The color bar itself is an easy, wide click-and-drag target; the
        # adjacent native Scale remains available for precise adjustment.
        colorbar.bind('<Button-1>', set_vmax_from_colorbar)
        colorbar.bind('<B1-Motion>', set_vmax_from_colorbar)

        def png_photo(gray):
            out_h, out_w = gray.shape
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
            raw_png = b''.join(b'\x00' + row.tobytes() for row in rgb)
            def png_chunk(kind, payload):
                body = kind + payload
                return struct.pack('>I', len(payload)) + body + struct.pack('>I', zlib.crc32(body) & 0xffffffff)
            png = (b'\x89PNG\r\n\x1a\n'
                   + png_chunk(b'IHDR', struct.pack('>IIBBBBB', out_w, out_h, 8, 2, 0, 0, 0))
                   + png_chunk(b'IDAT', zlib.compress(raw_png))
                   + png_chunk(b'IEND', b''))
            return tk.PhotoImage(data=base64.b64encode(png).decode('ascii'), format='png')

        def to_gray(array):
            high = max(maximum_value.get(), np.finfo(float).eps)
            shown = np.clip(array / high, 0, 1)
            return np.asarray(np.round(np.nan_to_num(shown, nan=0.0) * 255), dtype=np.uint8)

        def zoom_window(shape):
            """Return (y0, y1, x0, x1) of the region every tile shows."""
            h, w = shape
            box = zoom_box['bounds']
            if box is not None:
                y0, y1, x0, x1 = box
                y0, x0 = max(0, min(y0, h - 1)), max(0, min(x0, w - 1))
                y1, x1 = max(y0 + 1, min(y1, h)), max(x0 + 1, min(x1, w))
                return y0, y1, x0, x1
            zoom = float(zoom_value.get())
            crop_h = max(1, int(round(h / zoom)))
            crop_w = max(1, int(round(w / zoom)))
            y0 = (h - crop_h) // 2
            x0 = (w - crop_w) // 2
            return y0, y0 + crop_h, x0, x0 + crop_w

        def refresh_tiles(value=None):
            if zoom_box['bounds'] is None:
                zoom_label.configure(text=f'{float(zoom_value.get()):.0%}')
            else:
                y0, y1, x0, x1 = zoom_box['bounds']
                zoom_label.configure(text=f'region {x1 - x0}×{y1 - y0}')
            for index, (tile, picture, status, _) in enumerate(tile_widgets):
                scaled = magnitude_images[index]
                y0, y1, x0, x1 = zoom_window(scaled.shape)
                crop = scaled[y0:y1, x0:x1]
                # Fit the region into a 160 x 132 tile, keeping its aspect ratio.
                fit = min(160 / crop.shape[1], 132 / crop.shape[0])
                out_h = max(1, int(round(crop.shape[0] * fit)))
                out_w = max(1, int(round(crop.shape[1] * fit)))
                ys = np.linspace(0, crop.shape[0] - 1, out_h).astype(int)
                xs = np.linspace(0, crop.shape[1] - 1, out_w).astype(int)
                image_refs[index] = png_photo(to_gray(crop[np.ix_(ys, xs)]))
                picture.configure(image=image_refs[index])

        def slider_zoom(value=None):
            zoom_box['bounds'] = None          # moving the slider returns to center zoom
            refresh_tiles()

        zoom_scale.configure(command=slider_zoom)

        def reset_zoom():
            zoom_box['bounds'] = None
            zoom_value.set(1.0)
            refresh_tiles()

        def choose_region():
            """Enlarged overview of this slice; drag a box to set the zoom region."""
            stack = np.stack(magnitude_images, axis=2)
            overview = np.max(np.where(np.isfinite(stack), stack, 0), axis=2)
            h, w = overview.shape
            k = max(1, 512 // max(h, w))                      # integer upscale for display
            big = np.repeat(np.repeat(to_gray(overview), k, axis=0), k, axis=1)

            top = tk.Toplevel(window)
            top.title(f'Choose zoom region — slice {slc + 1}')
            top.transient(window)
            ttk.Label(top, text='Drag a box around the region to zoom into, then release.',
                      padding=8).pack()
            board = tk.Canvas(top, width=w * k, height=h * k, highlightthickness=0, cursor='crosshair')
            board.pack(padx=8, pady=(0, 8))
            photo = png_photo(big)
            board.create_image(0, 0, image=photo, anchor='nw')
            board.image = photo
            if zoom_box['bounds'] is not None:
                y0, y1, x0, x1 = zoom_box['bounds']
                board.create_rectangle(x0 * k, y0 * k, x1 * k, y1 * k, outline='#61ba86', dash=(4, 2))
            drag = {'start': None, 'rect': None}

            def press(event):
                drag['start'] = (event.x, event.y)
                if drag['rect'] is not None:
                    board.delete(drag['rect'])
                drag['rect'] = board.create_rectangle(event.x, event.y, event.x, event.y,
                                                      outline='#61ba86', width=2)

            def motion(event):
                if drag['start'] is not None:
                    sx, sy = drag['start']
                    board.coords(drag['rect'], sx, sy, event.x, event.y)

            def release(event):
                if drag['start'] is None:
                    return
                sx, sy = drag['start']
                x0, x1 = sorted((int(sx // k), int(math.ceil(event.x / k))))
                y0, y1 = sorted((int(sy // k), int(math.ceil(event.y / k))))
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                drag['start'] = None
                if x1 - x0 < 4 or y1 - y0 < 4:      # ignore accidental clicks
                    return
                zoom_box['bounds'] = (y0, y1, x0, x1)
                top.destroy()
                refresh_tiles()

            board.bind('<ButtonPress-1>', press)
            board.bind('<B1-Motion>', motion)
            board.bind('<ButtonRelease-1>', release)
            ttk.Button(top, text='Cancel', command=top.destroy).pack(pady=(0, 8))
            top.grab_set()

        def redraw():
            for tile, picture, status, index in tile_widgets:
                rejected = keep[slc, index] == 0
                color = '#d7191c' if rejected else '#238b45'
                origin = 'auto' if keep[slc, index] == automatic_keep[slc, index] else 'manual'
                decision = 'REJECT' if rejected else 'KEEP'
                tile.configure(background=color, highlightbackground=color)
                picture.configure(background=color)
                label = f'{index + 1}: {decision}'
                if show_automatic:
                    label += f' ({origin})'
                status.configure(text=label,
                                 foreground='white' if rejected else '#102d19', background=color)

        def apply_slice(event=None):
            window.unbind_all('<MouseWheel>')
            window.destroy()

        for tile, picture, status, index in tile_widgets:
            for widget in (tile, picture, status):
                widget.bind('<Button-1>', lambda event, i=index: toggle(i))

        def toggle(index):
            keep[slc, index] = 1 - keep[slc, index]
            redraw()

        footer = ttk.Frame(window, padding=8)
        footer.grid(row=3, column=0, sticky='ew')
        legend = 'Green = keep   Red = reject'
        if show_automatic:
            legend += '   (auto/manual shows decision source)'
        ttk.Label(footer, text=legend).pack(side='left')
        ttk.Button(footer, text='Apply slice and continue', command=apply_slice).pack(side='right')
        cancelled = [False]

        def cancel_review():
            cancelled[0] = True
            apply_slice()

        if allow_cancel:
            ttk.Button(footer, text='Stop (discard this slice)', command=cancel_review).pack(side='right')
        window.bind('<Return>', apply_slice)
        window.protocol('WM_DELETE_WINDOW', cancel_review if allow_cancel else apply_slice)
        window.update_idletasks()
        refresh_tiles()
        update_intensity(maximum_value.get())
        redraw()
        wait_window(window)
        if cancelled[0]:
            raise InterruptedError('Manual review stopped; the current slice was not saved.')
    if save_decisions and diagnostics_path:
        decision_dir = os.path.join(diagnostics_path, '13_Diagnostics')
        os.makedirs(decision_dir, exist_ok=True)
        bvals = np.asarray(bvals) if bvals is not None else np.full(n_images, np.nan)
        bvecs = np.asarray(bvecs) if bvecs is not None else np.full((n_images, 3), np.nan)
        decisions = []
        for slc in range(n_slices):
            for image_index in range(n_images):
                auto_keep = bool(automatic_keep[slc, image_index])
                final_keep = bool(keep[slc, image_index])
                decisions.append({
                    'slice_index': slc,
                    'stacked_image_index': image_index,
                    'b_value': float(bvals[image_index]),
                    'b_vector': [float(v) for v in bvecs[image_index]],
                    'automatic_decision': 'keep' if auto_keep else 'reject',
                    'manual_decision': 'keep' if final_keep else 'reject',
                    'manually_changed': auto_keep != final_keep,
                })
        with open(os.path.join(decision_dir, 'Rejection_Decisions.json'), 'w', encoding='utf-8') as stream:
            json.dump({'automatic_keep': automatic_keep.tolist(),
                       'final_keep': keep.tolist(), 'decisions': decisions}, stream, indent=2)
    return keep


def _save_rejection_configuration(directory, images, bvals, bvecs, coordinates,
                                  automatic_keep, final_keep, operation_type):
    import json
    import os
    import numpy as np
    os.makedirs(directory, exist_ok=True)
    payload = dict(image_shape=list(images.shape), bvals=np.asarray(bvals).tolist(),
                   bvecs=np.asarray(bvecs).tolist(), crop_coordinates=np.asarray(coordinates).tolist(),
                   automatic_keep=np.asarray(automatic_keep).tolist(),
                   final_keep=np.asarray(final_keep).tolist(), operation_type=operation_type)
    path = os.path.join(directory, 'Rejection_Configuration.json')
    with open(path + '.tmp', 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2)
    os.replace(path + '.tmp', path)


def _load_rejection_configuration(directory, images, bvals, bvecs, operation_type):
    """Restore decisions only for matching geometry and ordered encodings.

    Older outputs store crop bounds and reviewed decisions in separate files.
    Missing decisions mean rejection is unfinished and must run interactively.
    """
    import json
    import re
    from pathlib import Path
    import numpy as np
    directory = Path(directory)
    path = directory / 'Rejection_Configuration.json'
    if path.exists():
        payload = json.loads(path.read_text())
        if (payload['image_shape'] != list(images.shape)
                or payload['operation_type'] != operation_type
                or not np.array_equal(payload['bvals'], bvals)
                or not np.array_equal(payload['bvecs'], bvecs)):
            raise ValueError('Saved rejection configuration does not match the input geometry/encodings; use mode off to redo it.')
        coordinates = payload['crop_coordinates']
    else:
        crop = directory / 'Crop_Coordinates.txt'
        decisions = directory / 'Rejection_Decisions.json'
        if not crop.exists() or not decisions.exists():
            return None
        payload = json.loads(decisions.read_text())
        rows = re.findall(r'Slice (\d+): x_start=(\d+), x_end=(\d+), y_start=(\d+), y_end=(\d+)', crop.read_text())
        if [int(row[0]) for row in rows] != list(range(1, images.shape[2] + 1)):
            raise ValueError('Saved crop slice count does not match the input.')
        coordinates = [list(values) for values in zip(*[list(map(int, row[1:])) for row in rows])]
        for decision in payload['decisions']:
            idx = decision['stacked_image_index']
            if (not 0 <= idx < len(bvals) or not np.isclose(decision['b_value'], bvals[idx])
                    or not np.allclose(decision['b_vector'], bvecs[idx])):
                raise ValueError('Saved rejection decisions do not match the ordered input encodings.')
    bounds = np.asarray(coordinates)
    if bounds.shape != (4, images.shape[2]) or not np.isfinite(bounds).all() or not np.equal(bounds, np.floor(bounds)).all():
        raise ValueError('Invalid saved crop coordinates.')
    x0, x1, y0, y1 = bounds
    if not ((0 <= x0).all() and (x0 < x1).all() and (x1 <= images.shape[1]).all()
            and (0 <= y0).all() and (y0 < y1).all() and (y1 <= images.shape[0]).all()):
        raise ValueError('Saved crop lies outside the input image.')
    automatic = np.asarray(payload['automatic_keep'])
    final = np.asarray(payload['final_keep'])
    for keep in (automatic, final):
        if keep.shape != images.shape[2:] or not np.isin(keep, [0, 1]).all():
            raise ValueError('Saved rejection decisions do not match slices/acquisitions.')
    return bounds.astype(int).tolist(), automatic, final
