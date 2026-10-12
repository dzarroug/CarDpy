def _apply_dictionary_keep(dictionary_items, keep_matrix_stacked):
    """Map stacked rejection decisions back to variable-length encodings."""
    import numpy as np

    dictionary_items = list(dictionary_items)
    keep_matrix_stacked = np.asarray(keep_matrix_stacked)
    if (keep_matrix_stacked.ndim != 2
            or keep_matrix_stacked.shape[1] != sum(item[2] for item in dictionary_items)
            or not np.isin(keep_matrix_stacked, [0, 1]).all()):
        raise ValueError('Binary rejection decisions must match the dictionary images.')
    for _, encoding, repetitions in dictionary_items:
        images = np.asarray(encoding['images'])
        if (images.ndim != 4 or images.shape[2:] != (keep_matrix_stacked.shape[0], repetitions)
                or repetitions == 0):
            raise ValueError('Dictionary images must match the decision slices and repetitions.')
    accepted_encodings = {}
    stats_by_encoding = {}
    keep_by_encoding = {}
    offset = 0
    for key, encoding, repetitions in dictionary_items:
        encoding_keep = keep_matrix_stacked[:, offset:offset + repetitions].copy()
        accepted_images = np.asarray(encoding['images']).copy()
        stats_by_encoding[key] = np.mean(encoding_keep, axis=1) * 100
        keep_by_encoding[key] = encoding_keep
        for slc in range(encoding_keep.shape[0]):
            good = np.flatnonzero(encoding_keep[slc] == 1)
            bad = np.flatnonzero(encoding_keep[slc] == 0)
            if bad.size == 0:
                continue
            if good.size == 0:
                good = np.asarray([0])
            for rejected in bad:
                replacement = int(np.random.choice(good, 1)[0])
                accepted_images[:, :, slc, rejected] = accepted_images[:, :, slc, replacement]
        accepted_encodings[key] = {
            'bval': float(encoding['bval']),
            'bvec': np.asarray(encoding['bvec']).copy(),
            'images': accepted_images,
        }
        offset += repetitions
    if offset != keep_matrix_stacked.shape[1]:
        raise ValueError('Dictionary repetitions do not match the stacked rejection decisions.')
    return accepted_encodings, stats_by_encoding, keep_by_encoding

def _apply_sorted_keep(matrix, keep_matrix_stacked):
    """Undo repetition-major stacking without regrouping diffusion encodings."""
    import numpy as np

    matrix = np.asarray(matrix)
    keep = np.asarray(keep_matrix_stacked)
    if matrix.ndim != 5 or any(size == 0 for size in matrix.shape):
        raise ValueError('Sorted images must be a nonempty 5D array.')
    expected = (matrix.shape[2], matrix.shape[3] * matrix.shape[4])
    if keep.shape != expected or not np.isin(keep, [0, 1]).all():
        raise ValueError('Binary rejection decisions must match the sorted images.')
    keep_sorted = keep.reshape(matrix.shape[2], matrix.shape[4], matrix.shape[3]).swapaxes(1, 2).copy()
    accepted = matrix.copy()
    for slc in range(matrix.shape[2]):
        for encoding in range(matrix.shape[3]):
            good = np.flatnonzero(keep_sorted[slc, encoding] == 1)
            bad = np.flatnonzero(keep_sorted[slc, encoding] == 0)
            if good.size == 0:
                print('All repetitions were rejected for slice %i, encoding %i; retaining repetition 0.'
                      % (slc + 1, encoding + 1))
                good = np.asarray([0])
            for repetition in bad:
                replacement = int(np.random.choice(good, 1)[0])
                accepted[:, :, slc, encoding, repetition] = matrix[:, :, slc, encoding, replacement]
    return accepted, np.mean(keep_sorted, axis=2) * 100, keep_sorted


def shot_rejection(original_matrix, original_bvals, original_bvecs, NRMSE_threshold = 0.75, NSSIM_threshold = 0.75, zoom = 'ON', IntERACT_zoom = 'ON', organ = 'Heart', operation_type = 'Magnitude', diagnostics_path = None, gui_version = 'legacy', save_diagnostics = True, manual_review = False):
    """
    ########## Definition Inputs ##################################################################################################################
    original_matrix       : Sorted diffusion data (5D).
    original_bvals        : Sorted b-values.
    original_bvecs        : Sorted b vectors.
    manual_review         : Open an optional image grid to edit automatic keep/reject decisions.
                            Closing or cancelling the review aborts rejection.
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
    if manual_review:
        from cardpy.GUI_Tools.Rejection_Review import review_rejection
        keep_matrix_stacked = review_rejection(
            original_matrix_stacked, original_bvals_stacked,
            original_bvecs_stacked, keep_matrix_stacked)
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
    accepted_matrix, stats, keep_matrix_sorted = _apply_sorted_keep(original_matrix, keep_matrix_stacked)
    for slc in range(stats.shape[0]):
        for encoding in range(stats.shape[1]):
            print('Acceptance rate for slice %i, encoding %i: %.1f%%'
                  % (slc + 1, encoding + 1, stats[slc, encoding]))
    return [accepted_matrix, original_bvals, original_bvecs,
            Slice_Crop_Coordinates, stats, keep_matrix_sorted]
