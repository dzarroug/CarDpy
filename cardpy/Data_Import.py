########## CarDpy Data Import ######################################################################################################################
### Written by Tyler E. Cork, tyler.e.cork@gmail.com
### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
###
### Reader notes:
###   * Sorting is driven by DICOM header content only. Files may be handed over in any order; file names are never trusted.
###   * Slices are indexed by sorted slice position, diffusion directions by first appearance in acquisition order, and
###     averages by a per (slice, direction) repetition counter, so a skipped or corrupt image leaves a NaN hole instead
###     of shifting every image after it.
###   * Enhanced multi-frame, classic single-frame and Siemens mosaic layouts all flow through one code path.
###   * Output contract is unchanged: [matrix, b_vals, b_vecs, Header] with matrix (rows, columns, slices, volumes) and
###     volumes ordered as (direction index) + (average index * number of directions).
####################################################################################################################################################


def _to_float(value):
    """
    ########## Definition Inputs ##################################################################################################################
    value         : Any DICOM value to be coerced to a float.
    ########## Definition Outputs #################################################################################################################
    number        : Float representation of value, or NaN if the value cannot be coerced.
    """
    import numpy as np
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(np.nan)


def _tag_value(dcm, *path, default = None):
    """
    ########## Definition Inputs ##################################################################################################################
    dcm           : DICOM dataset or sequence item.
    path          : Sequence of tags / sequence indices to walk, e.g. (0x52009230, 0, 0x00189117, 0, 0x00189087).
    default       : Value returned if any step of the path is missing. Default is None.
    ########## Definition Outputs #################################################################################################################
    value         : Value found at the end of the path, or default if the path does not exist.
    """
    node = dcm                                                                              # Start at the supplied dataset
    try:                                                                                    # Attempt to walk the path ...
        for step in path:                                                                       # Iterate through path steps (tags and sequence indices both index with [])
            node = node[step]                                                                       # Step into the tag or sequence item
        return node.value if hasattr(node, 'value') else node                                   # Return the value if present
    except (KeyError, IndexError, TypeError, AttributeError):                               # If any step is missing ...
        return default                                                                          # Return the default


def _phoenix_b_value(dcm):
    """Read the nominal b value from a top-level or nested Siemens Phoenix protocol field."""
    import re
    values = []                                                                             # Initialize configured Phoenix b values
    for element in dcm:                                                                     # Walk every top-level element
        if element.VR == 'SQ':                                                                 # If this element contains nested datasets ...
            for item in element.value:                                                            # Iterate through sequence items
                value = _phoenix_b_value(item)                                                       # Search the nested item recursively
                if value is not None:                                                                # If a b value was found ...
                    values.append(value)                                                                 # Retain it
        elif element.tag == (0x0021, 0x1019):                                              # Siemens Phoenix protocol data (direct or nested)
            payload = element.value                                                          # Extract the protocol payload
            if isinstance(payload, bytes):                                                   # If stored as OB bytes ...
                payload = payload.decode('latin-1', errors = 'ignore')                           # Decode without losing scanner text
            matches = re.findall(r'sDiffusion\.alBValue\[\d+\]\s*=\s*'
                                 r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))', str(payload), re.IGNORECASE)  # Find configured diffusion shells
            values.extend(float(match) for match in matches)                                 # Store every configured value
    unique = sorted(set(values))                                                            # Collapse repeated protocol copies
    if len(unique) == 1:                                                                    # If the protocol identifies one shell ...
        return unique[0]                                                                        # Return it
    nonzero = [value for value in unique if value != 0]                                     # Ignore a baseline when one diffusion shell exists
    if len(nonzero) == 1:                                                                   # If exactly one nonzero shell remains ...
        return nonzero[0]                                                                       # Return its nominal value
    return None                                                                             # Multiple shells cannot identify the current image


def _phoenix_diffusion_vectors(dcm):
    """Read and normalize the free diffusion direction table from Siemens Phoenix protocol data."""
    import re
    import numpy as np
    components = dict()                                                                    # Collect vector components by Phoenix table index
    for element in dcm:                                                                     # Walk every top-level element
        if element.VR == 'SQ':                                                                 # If this element contains nested datasets ...
            for item in element.value:                                                            # Iterate through sequence items
                for idx, vector in enumerate(_phoenix_diffusion_vectors(item)):                         # Search the nested item recursively
                    components.setdefault(idx, dict(zip(('Sag', 'Cor', 'Tra'), vector)))                   # Retain vectors not already found
        elif element.tag == (0x0021, 0x1019):                                              # Siemens Phoenix protocol data (direct or nested)
            payload = element.value                                                          # Extract the protocol payload
            if isinstance(payload, bytes):                                                   # If stored as OB bytes ...
                payload = payload.decode('latin-1', errors = 'ignore')                           # Decode without losing scanner text
            matches = re.findall(r'sDiffusion\.sFreeDiffusionData\.asDiffDirVector\[(\d+)\]\.d(Sag|Cor|Tra)\s*=\s*'
                                 r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))', str(payload), re.IGNORECASE)  # Find every free direction component
            for idx, axis, value in matches:                                                # Store every component by vector number
                components.setdefault(int(idx), {})[axis.capitalize()] = float(value)          # Preserve scanner Sag / Cor / Tra order
    vectors = []                                                                           # Initialize normalized direction table
    for idx in sorted(components):                                                         # Return directions in protocol order
        component = components[idx]                                                           # Extract this vector's components
        if not all(axis in component for axis in ('Sag', 'Cor', 'Tra')):                       # If the vector is incomplete ...
            continue                                                                               # Do not publish a partial direction
        vector = np.array([component['Sag'], component['Cor'], component['Tra']], dtype = float)  # Construct scanner-coordinate vector
        length = np.linalg.norm(vector)                                                        # Calculate its magnitude
        if length > 0:                                                                         # If this is a valid direction ...
            vectors.append((vector / length).tolist())                                             # Normalize it as a DICOM b vector
    return vectors                                                                          # Return every complete Phoenix direction


def _phoenix_gradient(dcm, acquisition_order = 1):
    """Select the Phoenix direction associated with an acquisition, repeating the protocol table as needed."""
    import numpy as np
    vectors = _phoenix_diffusion_vectors(dcm)                                               # Read the ordered protocol direction table
    if len(vectors) == 0:                                                                   # If no free direction table is present ...
        return None                                                                             # Report that no fallback exists
    order = _to_float(acquisition_order)                                                    # Coerce the acquisition order to a number
    if np.isnan(order):                                                                     # If no usable order was supplied ...
        order = 1                                                                               # Select the first protocol direction
    index = max(int(round(order)) - 1, 0) % len(vectors)                                   # XA61 repeats directions cyclically across averages
    return vectors[index]                                                                   # Return the direction for this acquisition


def _diffusion_b_value(dcm, frame = None, default = None):
    """Read a b value from standard, Siemens private, Phoenix, or series metadata storage."""
    import re
    paths = []                                                                              # Initialize ordered b-value locations
    if frame is not None:                                                                   # If an enhanced frame was supplied ...
        paths.append((0x52009230, frame, 0x00189117, 0, 0x00189087))                            # Prefer its per-frame standard value
    paths.extend([(0x00189087,),                                                            # Standard classic Diffusion B Value
                  (0x0019100c,)])                                                           # Siemens classic private b value
    for path in paths:                                                                      # Search explicit DICOM fields first
        value = _tag_value(dcm, *path)                                                         # Read the candidate value
        if value not in (None, ''):                                                            # If the scanner populated it ...
            return _to_float(value)                                                                 # Trust the explicit value, including a real b0
    value = _phoenix_b_value(dcm)                                                           # Search direct and enhanced-nested Phoenix protocol fields
    if value is not None:                                                                   # If the protocol identifies one diffusion shell ...
        return value                                                                            # Use its nominal b value
    metadata = [str(_tag_value(dcm, tag, default = '') or '')
                for tag in (0x00180024,                                                     # Sequence Name
                            0x0008103e,                                                     # Series Description
                            0x00181030)]                                                    # Protocol Name
    for text in metadata:                                                                   # Search stable public metadata next
        match = re.search(r'(?:^|[^a-z0-9])b[_ -]?(\d+)(?:[^0-9]|$)', text, re.IGNORECASE)  # Match b50, ep_b400, b-500, etc.
        if match is not None:                                                                  # If the nominal b value is encoded in the name ...
            return float(match.group(1))                                                           # Use that nominal value
    sequence_names = [text.strip().lower().lstrip('*') for text in metadata]                 # Normalize scanner naming variants
    if any(text.endswith('ep2d_diff') for text in sequence_names):                           # If this known single-shell sequence omits its b-value tag ...
        return 50.0                                                                            # Its acquisition is the b50 series
    return default                                                                           # No b value could be inferred


def _reorient_gradient(orientation, gradient):
    """
    ########## Definition Inputs ##################################################################################################################
    orientation   : Image Orientation (Patient) - six element sequence.
    gradient      : Diffusion gradient direction - three element sequence.
    ########## Definition Outputs #################################################################################################################
    b_vec         : Diffusion gradient direction rotated into the image frame.
    """
    import numpy as np
    Im_1          = np.expand_dims(np.array(orientation[0:3], dtype = float), axis = 1)      # Extract image orientation 1
    Im_2          = np.expand_dims(np.array(orientation[3:6], dtype = float), axis = 1)      # Extract image orientation 2
    Im_3          = np.expand_dims(np.cross(np.array(orientation[0:3], dtype = float),
                                            np.array(orientation[3:6], dtype = float)), axis = 1)  # Cross product image orientation 1 and 2
    ReOr_Diff_Dir = np.dot(np.hstack((Im_1, Im_2, Im_3)).T, np.array(gradient, dtype = float))      # Correct diffusion direction based on patient orientation
    return [float(ReOr_Diff_Dir[0]), float(ReOr_Diff_Dir[1]), float(ReOr_Diff_Dir[2])]


def _direction_key(pe_pol, b_val, b_vec, ndigits = 4):
    """
    ########## Definition Inputs ##################################################################################################################
    pe_pol        : Phase encode polarity.
    b_val         : B value.
    b_vec         : Sequence of three b vector components.
    ndigits       : Number of decimal places used to quantize the key. Default is 4.
    ########## Definition Outputs #################################################################################################################
    key           : Hashable tuple identifying a unique diffusion encoding, tolerant to floating point noise.
    """
    import numpy as np
    parts = [_to_float(pe_pol), _to_float(b_val)] + [_to_float(v) for v in b_vec]           # Coerce every component to a float
    key   = []                                                                              # Initialize key
    for part in parts:                                                                      # Iterate through components
        if np.isnan(part):                                                                      # If the component is missing ...
            key.append('nan')                                                                       # Use a hashable sentinel so NaNs group together
        else:                                                                                   # Otherwise ...
            key.append(round(part + 0.0, ndigits))                                                  # Quantize to collapse noise (+0.0 folds -0.0 into 0.0)
    return tuple(key)


def _acquisition_order(dcm, file_index):
    """
    ########## Definition Inputs ##################################################################################################################
    dcm           : DICOM dataset.
    file_index    : Position of the DICOM in the file list, used as a final tie break.
    ########## Definition Outputs #################################################################################################################
    order         : Float used to reconstruct acquisition order without trusting file names.
    """
    import numpy as np
    for path in [(0x00200013,),                                                             # Instance Number
                 (0x00080032,),                                                             # Acquisition Time
                 (0x00080033,),                                                             # Content Time
                 (0x00181060,)]:                                                            # Trigger Time
        order = _to_float(_tag_value(dcm, *path))                                               # Attempt to read the ordering tag
        if not np.isnan(order):                                                                 # If the tag exists ...
            return order                                                                            # Use it as the acquisition order
    return float(file_index)                                                                # Otherwise fall back to file position


def _frame_diffusion(dcm, frame, Header, acquisition_order = 1):
    """
    ########## Definition Inputs ##################################################################################################################
    dcm           : DICOM dataset.
    frame         : Frame index inside a Per-frame Functional Groups Sequence.
    Header        : Header dictionary from VendorHeaders, used as a fall back for missing per-frame tags.
    ########## Definition Outputs #################################################################################################################
    pe_pol        : Phase encode polarity for this frame.
    b_val         : B value for this frame.
    b_vec         : Reoriented b vector for this frame.
    ########## Definition Information #############################################################################################################
    ### Read diffusion encoding per frame rather than assuming one file holds exactly one direction. Falls back to the
    ### frame 0 values held in Header whenever the per-frame tags are absent.
    """
    import numpy as np
    ########## Phase encode polarity ###############################################################################################################
    pe_raw = _tag_value(dcm, 0x52009230, frame, 0x002111fe, 0, 0x0021111c)                   # Siemens private per-frame phase encode polarity
    if pe_raw is None:                                                                      # If the per-frame tag is absent ...
        pe_pol = Header.get('Phase Encode Polarity', np.nan)                                    # Fall back to the header value
    else:                                                                                   # Otherwise ...
        pe_pol = -1 if int(pe_raw) == 0 else 1                                                  # Map 0 to negative and 1 to positive polarity
    ########## B value #############################################################################################################################
    b_val = _diffusion_b_value(dcm, frame = frame)                                          # Read per-frame, classic, private or nominal b value
    if b_val is None:                                                                       # If every DICOM source is absent ...
        b_val = Header.get('B Value', np.nan)                                                   # Fall back to the header value
    b_val = _to_float(b_val)                                                                # Coerce to a float
    ########## B vector ############################################################################################################################
    if b_val == 0:                                                                          # If b value equals 0 ...
        return pe_pol, b_val, [0.0, 0.0, 0.0]                                                   # Set b vector to zero
    gradient = _tag_value(dcm, 0x52009230, frame, 0x00189117, 0, 0x00189076, 0, 0x00189089)  # Per-frame diffusion gradient direction
    if gradient is None:                                                                    # If the standard per-frame direction is absent ...
        gradient = _tag_value(dcm, 0x00189089)                                                  # Try the standard classic direction
    if gradient is None:                                                                    # If the standard direction is absent ...
        gradient = _tag_value(dcm, 0x0019100e)                                                  # Try the Siemens classic private direction
    if gradient is None:                                                                    # If XA61 omitted all per-image direction fields ...
        gradient = _phoenix_gradient(dcm, acquisition_order)                                    # Select its direction from the Phoenix protocol table
    orient   = _tag_value(dcm, 0x52009230, frame, 0x00209116, 0, 0x00200037)                 # Per-frame image orientation
    if orient is None:                                                                      # If the per-frame orientation is absent ...
        orient = _tag_value(dcm, 0x00200037)                                                    # Try classic Image Orientation (Patient)
    if gradient is None or orient is None:                                                  # If either per-frame tag is absent ...
        return pe_pol, b_val, [_to_float(Header.get('B Vector 1')),
                               _to_float(Header.get('B Vector 2')),
                               _to_float(Header.get('B Vector 3'))]                             # Fall back to the header values
    return pe_pol, b_val, _reorient_gradient(orient, gradient)                               # Reorient the per-frame gradient


def _build_index(records, slice_order = 'ascending', averages = 'mode'):
    """
    ########## Definition Inputs ##################################################################################################################
    records       : List of image records. Each record is a dict with at least 'slc_key', 'dir_key' and 'order_key'.
    slice_order   : 'ascending' sorts slices by slice position; 'file' preserves order of appearance. Default is 'ascending'.
    averages      : 'mode' sizes to the modal repetition count, so a missing image leaves a NaN hole while a stray extra
                    repetition does not inflate the matrix; 'max' keeps every repetition; 'min' truncates to the fewest.
                    Default is 'mode'.
    ########## Definition Outputs #################################################################################################################
    index         : Dictionary describing the sorted layout:
                    'slice_values' : Ordered slice positions, position in list is the slice index.
                    'dir_keys'     : Ordered diffusion keys, position in list is the direction index.
                    'numSlc'       : Number of slices.
                    'numDir'       : Number of diffusion directions.
                    'numAvg'       : Number of averages the matrix is sized for.
                    'assignments'  : List of (record, idxSlc, idxDir, idxAvg) for records that fit the matrix.
                    'dropped'      : List of (record, idxSlc, idxDir, idxAvg) for repetitions beyond numAvg.
    ########## Definition Information #############################################################################################################
    ### Sorting is driven entirely by DICOM header content, never by file order. Files may be supplied in any order and
    ### repetitions may be missing; a missing image leaves a NaN hole rather than shifting every later image.
    """
    ########## Order records by acquisition, not by file name ######################################################################################
    ordered = sorted(records, key = lambda r: r['order_key'])                                # Reconstruct acquisition order from header content
    ########## Index slices by spatial position ####################################################################################################
    if slice_order == 'file':                                                               # If legacy behaviour is requested ...
        slice_values = []                                                                       # Initialize slice value list
        for record in ordered:                                                                  # Iterate through records
            if record['slc_key'] not in slice_values:                                               # If the slice position is new ...
                slice_values.append(record['slc_key'])                                                  # Append in order of appearance
    else:                                                                                   # Otherwise ...
        slice_values = sorted(set(record['slc_key'] for record in ordered))                     # Sort unique slice positions ascending
    ########## Index diffusion directions by first appearance in acquisition order #################################################################
    dir_keys = []                                                                           # Initialize diffusion key list
    for record in ordered:                                                                  # Iterate through records
        if record['dir_key'] not in dir_keys:                                                   # If the diffusion encoding is new ...
            dir_keys.append(record['dir_key'])                                                      # Append in order of appearance
    ########## Count repetitions per slice and direction ###########################################################################################
    counters = dict()                                                                       # Initialize per (slice, direction) repetition counter
    staged   = []                                                                           # Initialize staged assignment list
    for record in ordered:                                                                  # Iterate through records
        idxSlc         = slice_values.index(record['slc_key'])                                  # Index current slice
        idxDir         = dir_keys.index(record['dir_key'])                                      # Index current diffusion direction
        pair           = (idxSlc, idxDir)                                                       # Identify the (slice, direction) pair
        idxAvg         = counters.get(pair, 0)                                                  # Index current average within this pair
        counters[pair] = idxAvg + 1                                                             # Advance the repetition counter for this pair
        staged.append((record, idxSlc, idxDir, idxAvg))                                         # Stage the assignment
    ########## Size the matrix #####################################################################################################################
    if len(counters) == 0:                                                                  # If no records were supplied ...
        numAvg = 0                                                                              # No averages
    elif averages == 'min':                                                                 # If legacy truncation is requested ...
        numAvg = min(counters.values())                                                         # Size to the fewest repetitions
    elif averages == 'max':                                                                 # If every repetition must be kept ...
        numAvg = max(counters.values())                                                         # Size to the most repetitions found
    else:                                                                                   # Otherwise size to the modal repetition count ...
        tally  = dict()                                                                         # Initialize repetition count tally
        for count in counters.values():                                                         # Iterate through per (slice, direction) counts
            tally[count] = tally.get(count, 0) + 1                                                  # Tally how many pairs share this count
        most   = max(tally.values())                                                            # Identify the most common tally
        numAvg = max(count for count, hits in tally.items() if hits == most)                    # Size to the modal count, preferring the larger on a tie
    ########## Split assignments that fit from repetitions that do not #############################################################################
    assignments = [item for item in staged if item[3] < numAvg]                              # Keep assignments inside the matrix
    dropped     = [item for item in staged if item[3] >= numAvg]                             # Record repetitions beyond numAvg
    return {'slice_values' : slice_values,
            'dir_keys'     : dir_keys,
            'numSlc'       : len(slice_values),
            'numDir'       : len(dir_keys),
            'numAvg'       : numAvg,
            'assignments'  : assignments,
            'dropped'      : dropped}


def DICOM_Reader(dcm_path, info = 'ON', slice_order = 'ascending', averages = 'mode'):
    """
    ########## Definition Inputs ##################################################################################################################
    dcm_path      : Path leading to DICOMs.
    info          : Information flag to show DICOM related information. Default is set to on.
    slice_order   : 'ascending' indexes slices by sorted slice position; 'file' reproduces legacy order of appearance.
                    Default is 'ascending'.
    averages      : 'mode' sizes the matrix to the modal repetition count, so a missing image leaves a NaN hole while
                    a stray extra repetition does not inflate the matrix; 'max' keeps every repetition found; 'min'
                    reproduces legacy truncation to the fewest. Default is 'mode'.
    ########## Definition Outputs #################################################################################################################
    matrix        : 4D Matrix (Rows, Columns, Slices, Diffusion Directions) from DICOM folder.
    b_vals        : List containing all b values from DICOM folder.
    b_vecs        : List containing all b vectors from DICOM folder.
    Header        : Dictionary containing header information.
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ###
    ### Volume ordering is preserved from earlier versions: volume index = direction index + (average index * numDir),
    ### so b_vals, b_vecs and cardpy.Data_Sorting.stacked2sorted all behave exactly as before. Slices are indexed by
    ### sorted slice position and diffusion directions by first appearance in acquisition order, both read from the
    ### DICOM headers rather than from file order. B value and b vector grouping is quantized so floating point
    ### variation between repetitions of the same direction cannot split it into two directions.
    ###
    ### Incomplete data: a missing image leaves a NaN volume and every other image keeps its slot with the correct
    ### b_val / b_vec, but a direction missing from the start of the series shifts later direction indices relative
    ### to a complete acquisition. Fitting is unaffected because b_vals and b_vecs travel with the matrix.
    ########## Import Modules ######################################################################################################################
    import glob
    import os
    import numpy              as     np
    import pydicom
    from   cardpy.Data_Import import VendorHeaders
    ########## Find DICOMs in Input Directory ######################################################################################################
    dcmPath = []                                                                            # Initialize DICOM path list
    dcmEXT  = '*.dcm'                                                                       # DICOM extension
    for name in glob.glob(os.path.join(dcm_path, dcmEXT)):                                  # Search for .dcm files in DICOM folder path
        dcmPath.append(name)                                                                    # Append .dcm files to DICOM path list
    if dcmPath == []:                                                                       # if DICOM path list is empty ...
        if info == 'ON':                                                                        # If information flag is turned on ...
            print('No .dcm found: Trying .IMA')                                                     # Print DCM not found
        imaEXT = '*.IMA'                                                                        # IMA (Siemens) extension
        for name in glob.glob(os.path.join(dcm_path, imaEXT)):                                  # Search for .IMA files in DICOM folder path
            dcmPath.append(name)                                                                    # Append .IMA files to DICOM path list
    if dcmPath == []:                                                                       # If no DICOMs were found at all ...
        raise FileNotFoundError('No .dcm or .IMA files found in: ' + str(dcm_path))              # Report the empty directory
    dcmPath.sort()                                                                          # Sort DICOM path list (tie break only - ordering comes from headers)
    ########## Pass One: Read Headers Only and Build Image Records ##################################################################################
    records    = []                                                                         # Initialize image record list
    Header     = None                                                                       # Initialize header dictionary
    skipped    = []                                                                         # Initialize unreadable file list
    pixel_rows = None                                                                       # Initialize stored number of rows
    pixel_cols = None                                                                       # Initialize stored number of columns
    for ii in range(len(dcmPath)):                                                          # Iterate through DICOMs
        try:                                                                                    # Attempt to read the header ...
            ds = pydicom.dcmread(dcmPath[ii], stop_before_pixels = True)                            # Load Nth (ii) DICOM header without pixel data
            H  = VendorHeaders(ds)                                                                  # Extract all meaningful DICOM tags for DTI
    
        except Exception as error:                                                              # If the DICOM cannot be read ...
            skipped.append((dcmPath[ii], str(error)))                                               # Record the unreadable file
            continue                                                                                # Skip to the next DICOM
        if Header is None:                                                                      # If this is the first readable DICOM ...
            Header = H                                                                              # Keep its header for the caller
        if pixel_rows is None:                                                                  # If the pixel dimensions are not known yet ...
            pixel_rows = _tag_value(ds, 0x00280010, default = H.get('Total Rows'))                   # Number of rows as stored in the pixel data
            pixel_cols = _tag_value(ds, 0x00280011, default = H.get('Total Columns'))                # Number of columns as stored in the pixel data
        acq_order = _acquisition_order(ds, ii)                                                  # Reconstruct acquisition order from the header
        mosaic    = H.get('Mosaic', None)                                                       # Extract mosaic slice count, if any
        slc_locs  = H.get('Slice Location', [])                                                 # Extract slice location list
        if mosaic is not None:                                                                  # If there are multiple slices (mosaic) per matrix ...
            for grid in range(int(mosaic)):                                                          # Iterate through mosaic tiles
                pe_pol, b_val, b_vec = (H.get('Phase Encode Polarity', np.nan),
                                        _to_float(H.get('B Value')),
                                        [_to_float(H.get('B Vector 1')),
                                         _to_float(H.get('B Vector 2')),
                                         _to_float(H.get('B Vector 3'))])                                # Mosaic tiles share one diffusion encoding
                records.append({'file_index' : ii,
                                'frame'      : grid,
                                'layout'     : 'mosaic',
                                'slc_key'    : float(grid),
                                'dir_key'    : _direction_key(pe_pol, b_val, b_vec),
                                'order_key'  : (acq_order, float(grid), float(ii)),
                                'b_val'      : b_val,
                                'b_vec'      : b_vec})                                                   # Append one record per mosaic tile
        elif len(slc_locs) > 1:                                                                 # Or if the DICOM is enhanced multi-frame ...
            for frame in range(len(slc_locs)):                                                      # Iterate through frames
                pe_pol, b_val, b_vec = _frame_diffusion(ds, frame, H, acquisition_order = acq_order)    # Read diffusion encoding for this frame
                records.append({'file_index' : ii,
                                'frame'      : frame,
                                'layout'     : 'multiframe',
                                'slc_key'    : _to_float(slc_locs[frame]),
                                'dir_key'    : _direction_key(pe_pol, b_val, b_vec),
                                'order_key'  : (acq_order, float(frame), float(ii)),
                                'b_val'      : b_val,
                                'b_vec'      : b_vec})                                                   # Append one record per frame
        else:                                                                                   # Or if there is one slice per matrix ...
            pe_pol, b_val, b_vec = _frame_diffusion(ds, 0, H, acquisition_order = acq_order)       # Read per-file diffusion encoding, including XA61 Phoenix fallback
            records.append({'file_index' : ii,
                            'frame'      : 0,
                            'layout'     : 'single',
                            'slc_key'    : _to_float(slc_locs[0]) if len(slc_locs) else 0.0,
                            'dir_key'    : _direction_key(pe_pol, b_val, b_vec),
                            'order_key'  : (acq_order, 0.0, float(ii)),
                            'b_val'      : b_val,
                            'b_vec'      : b_vec})                                                   # Append one record per DICOM
    if Header is None:                                                                      # If no DICOM could be read ...
        detail = ''                                                                             # Initialize failure detail
        if len(skipped) > 0:                                                                    # If files were skipped ...
            detail = '\n  ' + '\n  '.join(os.path.basename(p) + ': ' + r for p, r in skipped[:5])   # Report why the first few failed
        raise RuntimeError('No readable DICOMs in: ' + str(dcm_path) + detail)                   # Report the unreadable directory with cause
    ########## Determine Matrix Dimensions #########################################################################################################
    layout = records[0]['layout']                                                           # Identify the layout of the series
    if layout == 'mosaic':                                                                  # If there are multiple slices (mosaic) per matrix ...
        tot_rows = Header['Total Rows']                                                         # Extract total number of rows in matrix
        tot_cols = Header['Total Columns']                                                      # Extract total number of columns in matrix
        acq_rows = Header['Acquisition Rows']                                                   # Extract number of rows in acquisition
        acq_cols = Header['Acquisition Columns']                                                # Extract number of columns in acquisition
        numRows  = int(acq_rows)                                                                # Re-define number of rows as the mosaic tile size
        numCols  = int(acq_cols)                                                                # Re-define number of columns as the mosaic tile size
        col_ims  = tot_cols / acq_cols                                                          # Number of sub slices per column
    else:                                                                                   # Otherwise ...
        numRows  = int(pixel_rows or 0)                                                         # Number of rows as stored in the pixel data
        numCols  = int(pixel_cols or 0)                                                         # Number of columns as stored in the pixel data
        col_ims  = None                                                                         # Mosaic grid is not used
    if numRows == 0 or numCols == 0:                                                        # If the series carries no image dimensions ...
        raise RuntimeError('No image dimensions in: ' + str(dcm_path) +
                           '\n  Rows (0028,0010) and Columns (0028,0011) are absent, so this is not an image series'
                           '\n  (Siemens TENSOR / Raw Data objects have no pixel data).')        # Report rather than return an empty matrix
    ########## Sort Records into Slices, Directions and Averages ###################################################################################
    index  = _build_index(records, slice_order = slice_order, averages = averages)           # Build the sorted layout from header content
    numSlc = index['numSlc']                                                                # Extract number of slices
    numDir = index['numDir']                                                                # Extract number of diffusion directions
    numAvg = index['numAvg']                                                                # Extract number of averages
    numVol = numDir * numAvg                                                                # Extract total number of volumes
    ########## Initialize Outputs as NaN so Missing Images Stay Identifiable #######################################################################
    matrix = np.full([numRows, numCols, numSlc, numVol], np.nan)                             # Initialize image matrix
    b_vals = np.full([numVol], np.nan)                                                       # Initialize b value array
    b_vecs = np.full([numVol, 3], np.nan)                                                    # Initialize b vector array
    ########## Pass Two: Read Pixel Data and Fill the Matrix #######################################################################################
    by_file = dict()                                                                        # Initialize file to assignment map
    for record, idxSlc, idxDir, idxAvg in index['assignments']:                             # Iterate through assignments
        by_file.setdefault(record['file_index'], []).append((record, idxSlc, idxDir, idxAvg))    # Group assignments so each DICOM is read once
    filled = 0                                                                              # Initialize filled image counter
    for file_index in sorted(by_file):                                                      # Iterate through DICOMs holding assigned images
        try:                                                                                    # Attempt to read the pixel data ...
            pixels = pydicom.dcmread(dcmPath[file_index]).pixel_array                              # Load Nth (file_index) DICOM pixel data
        except Exception as error:                                                              # If the pixel data cannot be read ...
            skipped.append((dcmPath[file_index], str(error)))                                       # Record the unreadable file
            continue                                                                                # Leave these images as NaN
        for record, idxSlc, idxDir, idxAvg in by_file[file_index]:                              # Iterate through this DICOM's assignments
            idxVol = idxDir + (numDir * idxAvg)                                                     # Index current volume, preserving legacy ordering
            if record['layout'] == 'mosaic':                                                        # If there are multiple slices (mosaic) per matrix ...
                grid_x  = int(np.floor(record['frame'] / col_ims))                                      # Identify slice's location in x grid of mosaic matrix
                grid_y  = int(record['frame'] - (col_ims * grid_x))                                     # Identify slice's location in y grid of mosaic matrix
                x_start = int(0 + numRows * grid_x)                                                     # Define row (x) start point of mosaic matrix
                x_stop  = int(numRows + numRows * grid_x)                                               # Define row (x) stop point of mosaic matrix
                y_start = int(0 + numCols * grid_y)                                                     # Define column (y) start point of mosaic matrix
                y_stop  = int(numCols + numCols * grid_y)                                               # Define column (y) stop point of mosaic matrix
                if x_stop > pixels.shape[0] or y_stop > pixels.shape[1]:                                # If the tile falls outside the mosaic ...
                    continue                                                                               # Leave this image as NaN
                image = pixels[x_start:x_stop, y_start:y_stop]                                          # Extract Nth (idxSlc) slice from mosaic matrix
            elif pixels.ndim == 3:                                                                  # Or if the DICOM holds multiple frames ...
                if record['frame'] >= pixels.shape[0]:                                                  # If the frame is missing from the pixel data ...
                    continue                                                                               # Leave this image as NaN
                image = pixels[record['frame'], :, :]                                                   # Extract Nth (frame) slice from DICOM
            else:                                                                                   # Or if the DICOM holds one slice ...
                image = pixels                                                                          # Extract the only slice from DICOM
            if image.shape != (numRows, numCols):                                               # If the image does not match the matrix ...
                skipped.append((dcmPath[file_index], 'image shape ' + str(image.shape) +
                                ' does not match matrix ' + str((numRows, numCols))))                   # Record the mismatch
                continue                                                                            # Leave this image as NaN
            matrix[:, :, idxSlc, idxVol] = image                                                 # Store the image
            b_vals[idxVol]               = record['b_val']                                       # Store b value information for this volume
            b_vecs[idxVol, :]            = record['b_vec']                                       # Store b vector information for this volume
            filled = filled + 1                                                                  # Advance the filled image counter
    ########## Backfill b Values and b Vectors for Volumes with No Image ###########################################################################
    ### A ragged protocol (for example a b0 acquired fewer times than the diffusion weighted directions) leaves volume
    ### slots with no image. Their b value and b vector are still known from the direction they belong to, so fill them
    ### from the direction key. The gradient table therefore never contains NaN and the empty volumes are identified by
    ### NaN in the matrix instead.
    for idxDir in range(numDir):                                                            # Iterate through diffusion directions
        key = index['dir_keys'][idxDir]                                                          # Extract the diffusion key for this direction
        for idxAvg in range(numAvg):                                                            # Iterate through averages
            idxVol = idxDir + (numDir * idxAvg)                                                     # Index current volume
            if np.isnan(b_vals[idxVol]):                                                            # If the volume was never filled ...
                b_vals[idxVol]    = _to_float(key[1])                                                   # Store b value from the direction key
                b_vecs[idxVol, :] = [_to_float(key[2]), _to_float(key[3]), _to_float(key[4])]           # Store b vector from the direction key
    ########## Rescale Phase Data ##################################################################################################################
    if Header.get('Image Type', 'Magnitude') == 'Phase':                                    # If image type is phase ...
        matrix = ((matrix / abs(Header['Rescale Intercept'])) - 0.5) * (Header['Rescale Slope'] * np.pi)  # Rescale and recenter for phase data
    ########## Report ##############################################################################################################################
    if info == 'ON':                                                                        # If information flag is turned on ...
        print('Number of Diffusion Directions:', str(numDir))                                   # Print dimension 4 (Diffusion Directions)
        print('Number of Slices:', str(numSlc))                                                 # Print dimension 3 (Slices)
        print('Number of Columns:', str(numCols))                                               # Print dimension 2 (Columns)
        print('Number of Rows:', str(numRows))                                                  # Print dimension 1 (Rows)
        print('Number of Averages:', str(numAvg))                                               # Print number of averages
        expected = numSlc * numVol                                                              # Expected number of images
        if filled != expected:                                                                  # If the matrix is not completely filled ...
            print('Incomplete data:', str(expected - filled), 'of', str(expected),
                  'images missing (left as NaN)')                                                   # Report the NaN holes
            print('  Note: b_vals / b_vecs still match the image held in every volume, but a')      # Direction indices are assigned by first
            print('        direction absent from the start of the series shifts later direction')   # appearance in acquisition order, so an
            print('        indices relative to a complete acquisition.')                            # incomplete series can permute indices
        if len(index['dropped']) > 0:                                                           # If repetitions were discarded ...
            print('Discarded', str(len(index['dropped'])),
                  'extra repetition(s) beyond numAvg =', str(numAvg))                               # Report the discarded repetitions
        for path, reason in skipped:                                                            # Iterate through unreadable files
            print('Skipped', os.path.basename(path) + ':', reason)                                  # Report the unreadable file
    return [matrix, b_vals, b_vecs, Header]


def VendorHeaders(dcm):
    """
    ########## Definition Inputs ##################################################################################################################
    # dcm           : DICOM
    ########## Definition Outputs #################################################################################################################
    # HeadersDict   : Dictionary containing header information.
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import Modules ######################################################################################################################
    import numpy as np                                                                                                                              # Import numpy module
    HeadersDict = dict()                                                                                                                            # Initialize header dictionary
    Vendor = dcm[0x00080070].value                                                                                                                  # Extract manufacturer from DICOM header
    ########## SIEMENS HEADER EXTRACTION ###########################################################################################################
    if Vendor =='GE MEDICAL SYSTEMS':
        HeadersDict['Manufacturer']            = dcm[0x00080070].value                                                                                    # Extract manufacturer from DICOM header
        HeadersDict['Scanner Model']           = dcm[0x00081090].value                                                                                    # Extract scanner model from DICOM header
        HeadersDict['Magnet Strength']         = dcm[0x00180087].value                                                                                    # Extract magnet strength from DICOM header
        if dcm[0x00100010].value == '':
            HeadersDict['Patient ID'] = ''
        else:
            HeadersDict['Patient ID']          = dcm[0x00100010].value                                                                                    # Extract patient ID from DICOM header
        ### Need to fix this for Months
#         if dcm[0x00101010].value == '':
#             HeadersDict['Patient Age'] = ''
#         else:
#             HeadersDict['Patient Age']             = [int(x) for x in dcm[0x00101010].value.split('Y') if x.isdigit()][0]                                 # Extract patient age from DICOM header
        HeadersDict['Patient Sex']              = dcm[0x00100040].value                                                                                    # Extract patient sex from DICOM header
        HeadersDict['Body Part']                = dcm[0x00180015].value                                                                                    # Extract examination region from DICOM header
        HeadersDict['X Resolution']             = dcm[0x00280030][0]
        HeadersDict['Y Resolution']             = dcm[0x00280030][1]
        HeadersDict['Z Resolution']             = dcm[0x00180050].value
        HeadersDict['Echo Time']                = dcm[0x00180081].value
        HeadersDict['Repetition Time']          = dcm[0x00180080].value
        HeadersDict['Scanner Orientation']      = dcm[0x00185100].value
        HeadersDict['Pixel Bandwidth']          = dcm[0x00180095].value
#         HeadersDict['Parallel Imaging Factor']  =
#         HeadersDict['Parallel Imaging Type']    =
        HeadersDict['Phase Encoding Direction'] = dcm[0x00181312].value
        if (((0x00281052) in dcm) == True):                                                                                                           # Check if data contains phase data ...
            HeadersDict['Image Type']        = 'Phase'                                                                                                  # If true, set data as phase
            HeadersDict['Rescale Intercept'] = dcm[0x00281052].value                                                                                    # Extract rescale intercept information from DICOM header
            HeadersDict['Rescale Slope']     = dcm[0x00281053].value                                                                                    # Extract rescale slope information from DICOM header
        else:                                                                                                                                         # Otherwise ...
            HeadersDict['Image Type']        = 'Magnitude'
        if HeadersDict['Phase Encoding Direction'] == 'ROW':
            HeadersDict['Total Rows']    = dcm[0x00280010].value
            HeadersDict['Total Columns'] = dcm[0x00280011].value
        else:
            HeadersDict['Total Columns'] = dcm[0x00280010].value
            HeadersDict['Total Rows']    = dcm[0x00280011].value

        HeadersDict['Acquisition Matrix'] = dcm[0x00181310].value                                                                                     # Extract acquisition matrix from DICOM Header
        HeadersDict['Acquisition Matrix'] = [x for x in HeadersDict['Acquisition Matrix'] if x != 0]                                                  # Remove zeros from acquisition matrix
        HeadersDict['Acquisition Rows']   = int(HeadersDict['Acquisition Matrix'][0])                                                                 # Extract acquisition rows from DICOM Header
        HeadersDict['Acquisition Columns']   = int(HeadersDict['Acquisition Matrix'][1])                                                                 # Extract acquisition columns from DICOM Header

        if (HeadersDict['Acquisition Rows'] > HeadersDict['Acquisition Columns'] and HeadersDict['Total Rows'] < HeadersDict['Total Columns']):             #
            HeadersDict['Acquisition Rows'] = int(HeadersDict['Acquisition Matrix'][1])                                                               #
            HeadersDict['Acquisition Columns'] = int(HeadersDict['Acquisition Matrix'][0])                                                               #

        HeadersDict['Mosaic'] = None                                                                                                                  # GE data is never mosaic format

        if dcm[0x00189034].value == 'LINEAR':                                                    # If phase encoding direction is negative ...
            HeadersDict['Phase Encode Polarity'] = -1                                                                                                   # Set phase encode polarity to -1
        if dcm[0x00189034].value == 'REVERSE_LINEAR':                                                    # If phase encoding direction is positive ...
            HeadersDict['Phase Encode Polarity'] = 1
#         try:
#             HeadersDict['Trigger Time']        = dcm[0x52009230][0][0x00189118][0][0x00209153].value
#         except KeyError:
#             HeadersDict['Trigger Time']        = 'N/A'

#         HeadersDict['Echo Train Length']       = dcm[0x52009229][0][0x00189112][0][0x00189241].value
#         HeadersDict['Phase Encode Steps']      = dcm[0x52009229][0][0x00189125][0][0x00189231].value
#         if (HeadersDict['Partial Fourier'] > 3.5 / 8 and HeadersDict['Partial Fourier'] < 4.5 / 8):                                                 # If Partial Fourier is between 3.5/8 and 4.5/8 ...
#             HeadersDict['Partial Fourier'] = '4/8'                                                                                                    # Set Partial Fourier to 4/8
#         elif (HeadersDict['Partial Fourier'] > 4.5 / 8 and HeadersDict['Partial Fourier'] < 5.5 / 8):                                               # If Partial Fourier is between 4.5/8 and 5.5/8 ...
#             HeadersDict['Partial Fourier'] = '5/8'                                                                                                    # Set Partial Fourier to 5/8
#         elif (HeadersDict['Partial Fourier'] > 5.5 / 8 and HeadersDict['Partial Fourier'] < 6.5 / 8):                                               # If Partial Fourier is between 5.5/8 and 6.5/8 ...
#             HeadersDict['Partial Fourier'] = '6/8'                                                                                                    # Set Partial Fourier to 6/8
#         elif (HeadersDict['Partial Fourier'] > 6.5 / 8 and HeadersDict['Partial Fourier'] < 7.5 / 8):                                               # If Partial Fourier is between 6.5/8 and 7.5/8 ...
#             HeadersDict['Partial Fourier'] = '7/8'                                                                                                    # Set Partial Fourier to 7/8
#         elif (HeadersDict['Partial Fourier'] > 7.5 / 8 and HeadersDict['Partial Fourier'] < 8.5 / 8):                                               # If Partial Fourier is between 7.5/8 and 8.5/8 ...
#             HeadersDict['Partial Fourier'] = '8/8'                                                                                                    # Set Partial Fourier to 8/8
#         else:                                                                                                                                       # Otherwise ...
#             HeadersDict['Partial Fourier'] = 'N/A'
        HeadersDict['Slice Location']  = [float(dcm[0x00201041].value)]                                                                               # Extract slice location from DICOM header

        try:
            HeadersDict['B Value']        = int(dcm[0x00189087].value)
        except KeyError:
            HeadersDict['B Value']        = int(0)
#         HeadersDict['B Value']         =   int(dcm[0x52009230][0][0x00189117][0][0x00189087].value)
        if HeadersDict['B Value'] == 0:                                                                                                               # If b value equals 0 ...
            HeadersDict['B Vector 1'] = 0.0                                                                                                             # Set b vector 1 to 0
            HeadersDict['B Vector 2'] = 0.0                                                                                                             # Set b vector 2 to 0
            HeadersDict['B Vector 3'] = 0.0
        else:
            Orientation               = dcm[0x00200037].value                                                                                           # Extract patient orientation from DICOM Header
            Im_1                      = np.array(Orientation[0:3])                                                                                      # Extract image orientation 1
            Im_2                      = np.array(Orientation[3:6])                                                                                      # Extract image orientation 2
            Im_3                      = np.cross(Im_1, Im_2)                                                                                            # Cross product image orientaiton 1 and 2
            B_vec_x                   = dcm[0x001910BB].value
            B_vec_y                   = dcm[0x001910BC].value
            B_vec_z                   = dcm[0x001910BD].value
            Diff_Dir                  = np.array([B_vec_x, B_vec_y, B_vec_z])
            #             Diff_Dir                  = dcm[0x52009230][0][0x00189117][0][0x00189076][0][0x00189089].value                                                                                           # Extract diffusion direction from DICOM Header
            Im_1                      = np.expand_dims(Im_1, axis = 1)                                                                                  # Expand dimension of image orientation 1
            Im_2                      = np.expand_dims(Im_2, axis = 1)                                                                                  # Expand dimension of image orientation 2
            Im_3                      = np.expand_dims(Im_3, axis = 1)                                                                                  # Expand dimension of image orientation 3
            ReOr_Diff_Dir             = np.dot(np.hstack((Im_1, Im_2, Im_3)).T, Diff_Dir)                                                               # Correct diffusion directions based of patient orientation
            HeadersDict['B Vector 1'] = float(ReOr_Diff_Dir[0])                                                                                         # Extract b vector 1
            HeadersDict['B Vector 2'] = float(ReOr_Diff_Dir[1])                                                                                         # Extract b vector 2
            HeadersDict['B Vector 3'] = float(ReOr_Diff_Dir[2])
    if (Vendor =='SIEMENS' or Vendor =='Siemens' or Vendor == 'Siemens Healthineers'):                                                                                                                          # If the vendor is Siemens...
        ### Software Version XA50 ### TEC Feb.20.2025
        #print(dcm[0x00181020].value)
        if dcm[0x00181020].value == 'syngo MR XA50':
            HeadersDict['Manufacturer']            = dcm[0x00080070].value                                                                                    # Extract manufacturer from DICOM header
            HeadersDict['Scanner Model']           = dcm[0x00081090].value                                                                                    # Extract scanner model from DICOM header
            HeadersDict['Magnet Strength']         = dcm[0x00180087].value                                                                                    # Extract magnet strength from DICOM header
            if dcm[0x00100010].value == '':
                HeadersDict['Patient ID'] = ''
            else:
                HeadersDict['Patient ID']          = dcm[0x00100010].value                                                                                    # Extract patient ID from DICOM header
            if dcm[0x00101010].value == '':
                HeadersDict['Patient Age'] = ''
            else:
                HeadersDict['Patient Age']             = [int(x) for x in dcm[0x00101010].value.split('Y') if x.isdigit()][0]                                 # Extract patient age from DICOM header
            HeadersDict['Patient Sex']             = dcm[0x00100040].value                                                                                    # Extract patient sex from DICOM header
            HeadersDict['Body Part']               = dcm[0x00180015].value                                                                                    # Extract examination region from DICOM header
            HeadersDict['X Resolution']            = dcm[0x52009230][0][0x00289110][0][0x00280030][0]                                                                                       # Extract x resolution from DICOM header
            HeadersDict['Y Resolution']            = dcm[0x52009230][0][0x00289110][0][0x00280030][1]                                                                                        # Extract y resolution from DICOM header
            HeadersDict['Z Resolution']            = dcm[0x52009230][0][0x00289110][0][0x00180050].value                                                                                    # Extract z resolution from DICOM header
            HeadersDict['Echo Time']               = dcm[0x52009230][0][0x00189114][0][0x00189082].value                                                                                   # Extract echo time (TE) from DICOM header
            HeadersDict['Repetition Time']         = dcm[0x52009229][0][0x00189112][0][0x00180080].value                                                                                    # Extract repetition time (TR) from DICOM header
            HeadersDict['Scanner Orientation']     = dcm[0x52009229][0][0x002110fe][0][0x0021100c].value                                                                                  # Extract scanner orientation
            HeadersDict['Pixel Bandwidth']         = dcm[0x52009229][0][0x00189006][0][0x00180095].value
            HeadersDict['Parallel Imaging Factor'] = dcm[0x52009229][0][0x00189115][0][0x00189069].value
            HeadersDict['Parallel Imaging Type']   = dcm[0x52009229][0][0x00189115][0][0x00189078].value
            HeadersDict['Phase Encoding Direction'] = dcm[0x52009229][0][0x00189125][0][0x00181312].value
            if HeadersDict['Phase Encoding Direction'] == 'ROW':
                HeadersDict['Total Rows']    = dcm[0x52009229][0][0x00189125][0][0x00189058].value
                HeadersDict['Total Columns'] = dcm[0x52009229][0][0x00189125][0][0x00189231].value
            else:
                HeadersDict['Total Columns'] = dcm[0x52009229][0][0x00189125][0][0x00189058].value
                HeadersDict['Total Rows']    = dcm[0x52009229][0][0x00189125][0][0x00189231].value

            if dcm[0x52009230][0][0x002111fe][0][0x0021111c].value == 0:                                                    # If phase encoding direction is negative ...
                HeadersDict['Phase Encode Polarity'] = -1                                                                                                   # Set phase encode polarity to -1
            if dcm[0x52009230][0][0x002111fe][0][0x0021111c].value == 1:                                                    # If phase encoding direction is positive ...
                HeadersDict['Phase Encode Polarity'] = 1
            try:
                HeadersDict['Trigger Time']        = dcm[0x52009230][0][0x00189118][0][0x00209153].value
            except KeyError:
                HeadersDict['Trigger Time']        = 'N/A'

            HeadersDict['Echo Train Length']       = dcm[0x52009229][0][0x00189112][0][0x00189241].value
            HeadersDict['Phase Encode Steps']      = dcm[0x52009229][0][0x00189125][0][0x00189231].value
            HeadersDict['Partial Fourier']         = (HeadersDict['Echo Train Length'] * HeadersDict['Parallel Imaging Factor']) / HeadersDict['Phase Encode Steps']  # Extract Partial Fourier infromation from DICOM header
            if (HeadersDict['Partial Fourier'] > 3.5 / 8 and HeadersDict['Partial Fourier'] < 4.5 / 8):                                                 # If Partial Fourier is between 3.5/8 and 4.5/8 ...
                HeadersDict['Partial Fourier'] = '4/8'                                                                                                    # Set Partial Fourier to 4/8
            elif (HeadersDict['Partial Fourier'] > 4.5 / 8 and HeadersDict['Partial Fourier'] < 5.5 / 8):                                               # If Partial Fourier is between 4.5/8 and 5.5/8 ...
                HeadersDict['Partial Fourier'] = '5/8'                                                                                                    # Set Partial Fourier to 5/8
            elif (HeadersDict['Partial Fourier'] > 5.5 / 8 and HeadersDict['Partial Fourier'] < 6.5 / 8):                                               # If Partial Fourier is between 5.5/8 and 6.5/8 ...
                HeadersDict['Partial Fourier'] = '6/8'                                                                                                    # Set Partial Fourier to 6/8
            elif (HeadersDict['Partial Fourier'] > 6.5 / 8 and HeadersDict['Partial Fourier'] < 7.5 / 8):                                               # If Partial Fourier is between 6.5/8 and 7.5/8 ...
                HeadersDict['Partial Fourier'] = '7/8'                                                                                                    # Set Partial Fourier to 7/8
            elif (HeadersDict['Partial Fourier'] > 7.5 / 8 and HeadersDict['Partial Fourier'] < 8.5 / 8):                                               # If Partial Fourier is between 7.5/8 and 8.5/8 ...
                HeadersDict['Partial Fourier'] = '8/8'                                                                                                    # Set Partial Fourier to 8/8
            else:                                                                                                                                       # Otherwise ...
                HeadersDict['Partial Fourier'] = 'N/A'

            Slice_Position = []
            for slc in range(len(dcm[0x52009230].value)):
                Slice_Position.append(int(dcm[0x52009230][slc][0x002111fe][0][0x00211188].value))
                # print(dcm[0x52009230][slc][0x002111fe][0][0x00211188].value)
            # print(Slice_Position)
            HeadersDict['Slice Location'] = Slice_Position
            # print(Slice_Position)
            # print(HeadersDict['Slice Location'])
            HeadersDict['B Value']         =   int(dcm[0x52009230][0][0x00189117][0][0x00189087].value)
            if HeadersDict['B Value'] == 0:                                                                                                               # If b value equals 0 ...
                HeadersDict['B Vector 1'] = 0.0                                                                                                             # Set b vector 1 to 0
                HeadersDict['B Vector 2'] = 0.0                                                                                                             # Set b vector 2 to 0
                HeadersDict['B Vector 3'] = 0.0
            else:
                Orientation               = dcm[0x52009230][0][0x00209116][0][0x00200037].value                                                                                           # Extract patient orientation from DICOM Header
                Im_1                      = np.array(Orientation[0:3])                                                                                      # Extract image orientation 1
                Im_2                      = np.array(Orientation[3:6])                                                                                      # Extract image orientation 2
                Im_3                      = np.cross(Im_1, Im_2)                                                                                            # Cross product image orientaiton 1 and 2
                Diff_Dir                  = dcm[0x52009230][0][0x00189117][0][0x00189076][0][0x00189089].value                                                                                           # Extract diffusion direction from DICOM Header
                Im_1                      = np.expand_dims(Im_1, axis = 1)                                                                                  # Expand dimension of image orientation 1
                Im_2                      = np.expand_dims(Im_2, axis = 1)                                                                                  # Expand dimension of image orientation 2
                Im_3                      = np.expand_dims(Im_3, axis = 1)                                                                                  # Expand dimension of image orientation 3
                ReOr_Diff_Dir             = np.dot(np.hstack((Im_1, Im_2, Im_3)).T, Diff_Dir)                                                               # Correct diffusion directions based of patient orientation
                HeadersDict['B Vector 1'] = float(ReOr_Diff_Dir[0])                                                                                         # Extract b vector 1
                HeadersDict['B Vector 2'] = float(ReOr_Diff_Dir[1])                                                                                         # Extract b vector 2
                HeadersDict['B Vector 3'] = float(ReOr_Diff_Dir[2])
        ### Software Version XA61 ###
        ### XA61 does not populate the shared and per-frame functional groups consistently. Four layouts are seen in one
        ### exam: a full diffusion series with everything present; b50 / b100 series with no MR Diffusion Sequence
        ### (0018,9117); derived ADC / FA / ColFA maps with no Diffusion B Value (0018,9087); and TENSOR maps with no
        ### functional groups at all. Every read below therefore falls back to the classic top level tag or a default,
        ### so the branch returns whatever the file actually contains instead of raising KeyError.
        if dcm[0x00181020].value == 'syngo MR XA61':
            HeadersDict['Manufacturer']             = dcm[0x00080070].value                                                                                   # Extract manufacturer from DICOM header
            HeadersDict['Scanner Model']            = _tag_value(dcm, 0x00081090, default = '')                                                               # Extract scanner model from DICOM header
            HeadersDict['Magnet Strength']          = _tag_value(dcm, 0x00180087, default = 'N/A')                                                            # Extract magnet strength (absent on TENSOR maps)
            HeadersDict['Patient ID']               = _tag_value(dcm, 0x00100010, default = '') or ''                                                          # Extract patient ID from DICOM header
            if _tag_value(dcm, 0x00101010) in (None, ''):
                HeadersDict['Patient Age'] = ''
            else:
                HeadersDict['Patient Age']          = [int(x) for x in dcm[0x00101010].value.split('Y') if x.isdigit()][0]                                    # Extract patient age from DICOM header
            HeadersDict['Patient Sex']              = _tag_value(dcm, 0x00100040, default = '')                                                               # Extract patient sex from DICOM header
            HeadersDict['Body Part']                = _tag_value(dcm, 0x00180015, default = '')                                                               # Extract examination region from DICOM header
            ########## Resolution - per frame group first, then classic tags ############################################################################
            pix_spacing = _tag_value(dcm, 0x52009230, 0, 0x00289110, 0, 0x00280030)                                                                           # Per-frame pixel spacing
            if pix_spacing is None:                                                                                                                           # If the functional group is absent ...
                pix_spacing = _tag_value(dcm, 0x00280030, default = [1.0, 1.0])                                                                                   # Fall back to the classic tag
            HeadersDict['X Resolution']             = pix_spacing[0]                                                                                          # Extract x resolution from DICOM header
            HeadersDict['Y Resolution']             = pix_spacing[1]                                                                                          # Extract y resolution from DICOM header
            z_res = _tag_value(dcm, 0x52009230, 0, 0x00289110, 0, 0x00180050)                                                                                  # Per-frame slice thickness
            HeadersDict['Z Resolution']             = _tag_value(dcm, 0x00180050, default = z_res) if z_res is None else z_res                                # Extract z resolution from DICOM header
            ########## Timing - per frame group first, then classic tags ################################################################################
            te = _tag_value(dcm, 0x52009230, 0, 0x00189114, 0, 0x00189082)                                                                                    # Per-frame echo time
            HeadersDict['Echo Time']                = _tag_value(dcm, 0x00180081, default = te) if te is None else te                                         # Extract echo time (TE) from DICOM header
            tr = _tag_value(dcm, 0x52009229, 0, 0x00189112, 0, 0x00180080)                                                                                    # Shared repetition time
            HeadersDict['Repetition Time']          = _tag_value(dcm, 0x00180080, default = tr) if tr is None else tr                                         # Extract repetition time (TR) from DICOM header
            HeadersDict['Trigger Time']             = _tag_value(dcm, 0x52009230, 0, 0x00189118, 0, 0x00209153,
                                                                 default = _tag_value(dcm, 0x00181060, default = 'N/A'))                                      # Extract trigger time from DICOM header
            HeadersDict['Scanner Orientation']      = _tag_value(dcm, 0x52009229, 0, 0x002110fe, 0, 0x0021100c, default = 'N/A')                               # Extract scanner orientation
            HeadersDict['Pixel Bandwidth']          = _tag_value(dcm, 0x52009229, 0, 0x00189006, 0, 0x00180095,
                                                                 default = _tag_value(dcm, 0x00180095, default = 'N/A'))                                      # Extract pixel bandwidth
            HeadersDict['Parallel Imaging Factor']  = _tag_value(dcm, 0x52009229, 0, 0x00189115, 0, 0x00189069, default = 1)                                   # Extract parallel imaging factor (1 if absent)
            HeadersDict['Parallel Imaging Type']    = _tag_value(dcm, 0x52009229, 0, 0x00189115, 0, 0x00189078, default = 'NONE')                              # Extract parallel imaging type
            ########## Matrix Size - per frame group first, then classic tags ###########################################################################
            HeadersDict['Phase Encoding Direction'] = _tag_value(dcm, 0x52009229, 0, 0x00189125, 0, 0x00181312,
                                                                 default = _tag_value(dcm, 0x00181312, default = 'COL'))                                      # Extract phase encoding direction
            freq_steps = _tag_value(dcm, 0x52009229, 0, 0x00189125, 0, 0x00189058)                                                                            # Samples in frequency direction
            phase_steps = _tag_value(dcm, 0x52009229, 0, 0x00189125, 0, 0x00189231)                                                                           # Steps in phase direction
            if freq_steps is None or phase_steps is None:                                                                                                     # If either functional group is absent ...
                HeadersDict['Total Rows']    = int(_tag_value(dcm, 0x00280010, default = 0))                                                                       # Fall back to stored rows
                HeadersDict['Total Columns'] = int(_tag_value(dcm, 0x00280011, default = 0))                                                                       # Fall back to stored columns
            elif HeadersDict['Phase Encoding Direction'] == 'ROW':
                HeadersDict['Total Rows']    = freq_steps
                HeadersDict['Total Columns'] = phase_steps
            else:
                HeadersDict['Total Columns'] = freq_steps
                HeadersDict['Total Rows']    = phase_steps
            HeadersDict['Echo Train Length']        = _tag_value(dcm, 0x52009229, 0, 0x00189112, 0, 0x00189241, default = None)                                # Extract echo train length
            HeadersDict['Phase Encode Steps']       = phase_steps                                                                                             # Extract phase encode steps
            ########## Partial Fourier - only computable when every term is present #####################################################################
            if (HeadersDict['Echo Train Length'] in (None, 0) or HeadersDict['Phase Encode Steps'] in (None, 0)):                                             # If any term is missing ...
                HeadersDict['Partial Fourier'] = 'N/A'                                                                                                            # Set Partial Fourier to N/A
            else:                                                                                                                                             # Otherwise ...
                HeadersDict['Partial Fourier'] = (HeadersDict['Echo Train Length'] * HeadersDict['Parallel Imaging Factor']) / HeadersDict['Phase Encode Steps']   # Extract Partial Fourier information
                if (HeadersDict['Partial Fourier'] > 3.5 / 8 and HeadersDict['Partial Fourier'] < 4.5 / 8):
                    HeadersDict['Partial Fourier'] = '4/8'
                elif (HeadersDict['Partial Fourier'] > 4.5 / 8 and HeadersDict['Partial Fourier'] < 5.5 / 8):
                    HeadersDict['Partial Fourier'] = '5/8'
                elif (HeadersDict['Partial Fourier'] > 5.5 / 8 and HeadersDict['Partial Fourier'] < 6.5 / 8):
                    HeadersDict['Partial Fourier'] = '6/8'
                elif (HeadersDict['Partial Fourier'] > 6.5 / 8 and HeadersDict['Partial Fourier'] < 7.5 / 8):
                    HeadersDict['Partial Fourier'] = '7/8'
                elif (HeadersDict['Partial Fourier'] > 7.5 / 8 and HeadersDict['Partial Fourier'] < 8.5 / 8):
                    HeadersDict['Partial Fourier'] = '8/8'
                else:
                    HeadersDict['Partial Fourier'] = 'N/A'
            ########## Phase Encode Polarity ###########################################################################################################
            pe_raw = _tag_value(dcm, 0x52009230, 0, 0x002111fe, 0, 0x0021111c)                                                                                # Siemens private phase encode polarity
            if pe_raw is not None:                                                                                                                           # If the private tag is present ...
                HeadersDict['Phase Encode Polarity'] = -1 if int(pe_raw) == 0 else 1                                                                              # Map 0 to negative and 1 to positive polarity
            else:                                                                                                                                             # Otherwise ...
                HeadersDict['Phase Encode Polarity'] = -1                                                                                                         # Default to negative polarity
            ########## Image Type ######################################################################################################################
            image_component = str(_tag_value(dcm, 0x00089208, default = '')).upper()                                                                            # Read Complex Image Component when present
            image_tokens    = [str(item).upper() for item in _tag_value(dcm, 0x00080008, default = [])]                                                         # Read Image Type tokens
            if image_component == 'PHASE' or 'PHASE' in image_tokens:                                                                                           # If the DICOM explicitly identifies phase data ...
                HeadersDict['Image Type'] = 'Phase'                                                                                                                # Set data as phase
            else:                                                                                                                                             # Otherwise ...
                HeadersDict['Image Type'] = 'Magnitude'                                                                                                            # Set data as magnitude
            HeadersDict['Rescale Intercept'] = _tag_value(dcm, 0x00281052, default = 0.0)                                                                      # Publish a stable rescale intercept key for every XA61 export
            HeadersDict['Rescale Slope']     = _tag_value(dcm, 0x00281053, default = 1.0)                                                                      # Publish a stable rescale slope key for every XA61 export
            HeadersDict['Mosaic'] = None                                                                                                                      # XA61 data is enhanced DICOM, never mosaic
            ########## Slice Location - private per frame tag, then classic tags ########################################################################
            Slice_Position = []                                                                                                                               # Initialize slice position list
            frames = _tag_value(dcm, 0x52009230, default = None)                                                                                              # Per-frame functional groups
            if frames is not None:                                                                                                                            # If the per-frame group is present ...
                for slc in range(len(frames)):                                                                                                                    # Iterate through frames
                    pos = _tag_value(dcm, 0x52009230, slc, 0x002111fe, 0, 0x00211188)                                                                                 # Siemens private slice position
                    if pos is None:                                                                                                                                # If the private tag is absent ...
                        pos = _tag_value(dcm, 0x52009230, slc, 0x00209113, 0, 0x00200032)                                                                                 # Fall back to Image Position (Patient)
                        pos = pos[2] if pos is not None else slc                                                                                                          # Use the through plane component
                    Slice_Position.append(int(pos) if isinstance(pos, (int, float)) else int(float(pos)))                                                          # Append slice position
            if Slice_Position == []:                                                                                                                          # If no per-frame position was found ...
                Slice_Position = [int(float(_tag_value(dcm, 0x00201041, default = 0)))]                                                                           # Fall back to classic Slice Location
            HeadersDict['Slice Location'] = Slice_Position                                                                                                     # Extract slice location from DICOM header
            ########## Diffusion Encoding ##############################################################################################################
            b_value = _diffusion_b_value(dcm, frame = 0, default = 0)                                                                                         # Read standard/private b value or infer it from Cima series metadata
            HeadersDict['B Value'] = int(_to_float(b_value)) if not np.isnan(_to_float(b_value)) else 0                                                        # Extract b value from DICOM header
            Diff_Dir = _tag_value(dcm, 0x52009230, 0, 0x00189117, 0, 0x00189076, 0, 0x00189089)                                                                # Per-frame diffusion gradient direction
            if Diff_Dir is None:                                                                                                                             # If the interoperable export omitted the standard diffusion sequence ...
                Diff_Dir = _tag_value(dcm, 0x0019100e)                                                                                                            # Read the Siemens classic private diffusion direction
            if Diff_Dir is None:                                                                                                                             # If XA61 omitted every per-image direction field ...
                Diff_Dir = _phoenix_gradient(dcm, _tag_value(dcm, 0x00200013, default = 1))                                                                      # Select the direction from its Phoenix protocol table
            Orientation = _tag_value(dcm, 0x52009230, 0, 0x00209116, 0, 0x00200037,
                                     default = _tag_value(dcm, 0x00200037))                                                                                   # Image orientation, per frame then classic
            if HeadersDict['B Value'] == 0 or Diff_Dir is None or Orientation is None:                                                                         # If b value equals 0 or the gradient is absent ...
                HeadersDict['B Vector 1'] = 0.0                                                                                                                   # Set b vector 1 to 0
                HeadersDict['B Vector 2'] = 0.0                                                                                                                   # Set b vector 2 to 0
                HeadersDict['B Vector 3'] = 0.0                                                                                                                   # Set b vector 3 to 0
            else:                                                                                                                                             # Otherwise ...
                b_vec = _reorient_gradient(Orientation, Diff_Dir)                                                                                                 # Correct diffusion direction based on patient orientation
                HeadersDict['B Vector 1'] = b_vec[0]                                                                                                              # Extract b vector 1
                HeadersDict['B Vector 2'] = b_vec[1]                                                                                                              # Extract b vector 2
                HeadersDict['B Vector 3'] = b_vec[2]                                                                                                              # Extract b vector 3
        ### Software Version XA20 ###
        if dcm[0x00181020].value == 'syngo MR XA20':
            HeadersDict['Manufacturer']            = dcm[0x00080070].value                                                                                    # Extract manufacturer from DICOM header
            HeadersDict['Scanner Model']           = dcm[0x00081090].value                                                                                    # Extract scanner model from DICOM header
            HeadersDict['Magnet Strength']         = dcm[0x00180087].value                                                                                    # Extract magnet strength from DICOM header
#            if dcm[0x00100020].value == '':
#                HeadersDict['Patient ID'] = ''
#            else:
#                HeadersDict['Patient ID']          = dcm[0x00100020].value                                                                                    # Extract patient ID from DICOM header
            if dcm[0x00100010].value == '':
                HeadersDict['Patient ID'] = ''
            else:
                HeadersDict['Patient ID']          = dcm[0x00100010].value                                                                                    # Extract patient ID from DICOM header
            if dcm[0x00101010].value == '':
                HeadersDict['Patient Age'] = ''
            else:
                HeadersDict['Patient Age']             = [int(x) for x in dcm[0x00101010].value.split('Y') if x.isdigit()][0]                                 # Extract patient age from DICOM header
            HeadersDict['Patient Sex']             = dcm[0x00100040].value                                                                                    # Extract patient sex from DICOM header
#            if dcm[0x00101020].value == None:
#                HeadersDict['Patient Height'] = ''
#            else:
#                HeadersDict['Patient Height']          = float(round(dcm[0x00101020].value, 2))                                                               # Extract patient height from DICOM header
#            if dcm[0x00101030].value == None:
#                HeadersDict['Patient Weight'] = ''
#            else:
#                HeadersDict['Patient Weight']          = float(round(dcm[0x00101030].value, 2))                                                               # Extract patient weight from DICOM header
            HeadersDict['Body Part']               = dcm[0x00180015].value                                                                                    # Extract examination region from DICOM header
            HeadersDict['X Resolution']            = dcm[0x52009230][0][0x00289110][0][0x00280030][0]                                                                                       # Extract x resolution from DICOM header
            HeadersDict['Y Resolution']            = dcm[0x52009230][0][0x00289110][0][0x00280030][1]                                                                                        # Extract y resolution from DICOM header
            HeadersDict['Z Resolution']            = dcm[0x52009230][0][0x00289110][0][0x00180050].value                                                                                    # Extract z resolution from DICOM header
            HeadersDict['Echo Time']               = dcm[0x52009230][0][0x00189114][0][0x00189082].value                                                                                   # Extract echo time (TE) from DICOM header
            HeadersDict['Repetition Time']         = dcm[0x52009229][0][0x00189112][0][0x00180080].value                                                                                    # Extract repetition time (TR) from DICOM header
            HeadersDict['Scanner Orientation']     = dcm[0x52009229][0][0x002110fe][0][0x0021100c].value                                                                                  # Extract scanner orientation
            HeadersDict['Pixel Bandwidth']         = dcm[0x52009229][0][0x00189006][0][0x00180095].value
            HeadersDict['Parallel Imaging Factor'] = dcm[0x52009229][0][0x00189115][0][0x00189069].value
            HeadersDict['Parallel Imaging Type']   = dcm[0x52009229][0][0x00189115][0][0x00189078].value
            HeadersDict['Phase Encoding Direction'] = dcm[0x52009229][0][0x00189125][0][0x00181312].value
            if HeadersDict['Phase Encoding Direction'] == 'ROW':
                HeadersDict['Total Rows']    = dcm[0x52009229][0][0x00189125][0][0x00189058].value
                HeadersDict['Total Columns'] = dcm[0x52009229][0][0x00189125][0][0x00189231].value
            else:
                HeadersDict['Total Columns'] = dcm[0x52009229][0][0x00189125][0][0x00189058].value
                HeadersDict['Total Rows']    = dcm[0x52009229][0][0x00189125][0][0x00189231].value
            
            if dcm[0x52009230][0][0x002111fe][0][0x0021111c].value == 0:                                                    # If phase encoding direction is negative ...
                HeadersDict['Phase Encode Polarity'] = -1                                                                                                   # Set phase encode polarity to -1
            if dcm[0x52009230][0][0x002111fe][0][0x0021111c].value == 1:                                                    # If phase encoding direction is positive ...
                HeadersDict['Phase Encode Polarity'] = 1
            try:
                HeadersDict['Trigger Time']        = dcm[0x52009230][0][0x00189118][0][0x00209153].value
            except KeyError:
                HeadersDict['Trigger Time']        = 'N/A'

            HeadersDict['Echo Train Length']       = dcm[0x52009229][0][0x00189112][0][0x00189241].value
            HeadersDict['Phase Encode Steps']      = dcm[0x52009229][0][0x00189125][0][0x00189231].value
            HeadersDict['Partial Fourier']         = (HeadersDict['Echo Train Length'] * HeadersDict['Parallel Imaging Factor']) / HeadersDict['Phase Encode Steps']  # Extract Partial Fourier infromation from DICOM header
            if (HeadersDict['Partial Fourier'] > 3.5 / 8 and HeadersDict['Partial Fourier'] < 4.5 / 8):                                                 # If Partial Fourier is between 3.5/8 and 4.5/8 ...
                HeadersDict['Partial Fourier'] = '4/8'                                                                                                    # Set Partial Fourier to 4/8
            elif (HeadersDict['Partial Fourier'] > 4.5 / 8 and HeadersDict['Partial Fourier'] < 5.5 / 8):                                               # If Partial Fourier is between 4.5/8 and 5.5/8 ...
                HeadersDict['Partial Fourier'] = '5/8'                                                                                                    # Set Partial Fourier to 5/8
            elif (HeadersDict['Partial Fourier'] > 5.5 / 8 and HeadersDict['Partial Fourier'] < 6.5 / 8):                                               # If Partial Fourier is between 5.5/8 and 6.5/8 ...
                HeadersDict['Partial Fourier'] = '6/8'                                                                                                    # Set Partial Fourier to 6/8
            elif (HeadersDict['Partial Fourier'] > 6.5 / 8 and HeadersDict['Partial Fourier'] < 7.5 / 8):                                               # If Partial Fourier is between 6.5/8 and 7.5/8 ...
                HeadersDict['Partial Fourier'] = '7/8'                                                                                                    # Set Partial Fourier to 7/8
            elif (HeadersDict['Partial Fourier'] > 7.5 / 8 and HeadersDict['Partial Fourier'] < 8.5 / 8):                                               # If Partial Fourier is between 7.5/8 and 8.5/8 ...
                HeadersDict['Partial Fourier'] = '8/8'                                                                                                    # Set Partial Fourier to 8/8
            else:                                                                                                                                       # Otherwise ...
                HeadersDict['Partial Fourier'] = 'N/A'

            Slice_Position = []
            for slc in range(len(dcm[0x52009230].value)):
                Slice_Position.append(dcm[0x52009230][slc][0x002111fe][0][0x00211188].value)
            HeadersDict['Slice Location'] = Slice_Position
            HeadersDict['B Value']         =   int(dcm[0x52009230][0][0x00189117][0][0x00189087].value)
            if HeadersDict['B Value'] == 0:                                                                                                               # If b value equals 0 ...
                HeadersDict['B Vector 1'] = 0.0                                                                                                             # Set b vector 1 to 0
                HeadersDict['B Vector 2'] = 0.0                                                                                                             # Set b vector 2 to 0
                HeadersDict['B Vector 3'] = 0.0
            else:
                Orientation               = dcm[0x52009230][0][0x00209116][0][0x00200037].value                                                                                           # Extract patient orientation from DICOM Header
                Im_1                      = np.array(Orientation[0:3])                                                                                      # Extract image orientation 1
                Im_2                      = np.array(Orientation[3:6])                                                                                      # Extract image orientation 2
                Im_3                      = np.cross(Im_1, Im_2)                                                                                            # Cross product image orientaiton 1 and 2
                Diff_Dir                  = dcm[0x52009230][0][0x00189117][0][0x00189076][0][0x00189089].value                                                                                           # Extract diffusion direction from DICOM Header
                Im_1                      = np.expand_dims(Im_1, axis = 1)                                                                                  # Expand dimension of image orientation 1
                Im_2                      = np.expand_dims(Im_2, axis = 1)                                                                                  # Expand dimension of image orientation 2
                Im_3                      = np.expand_dims(Im_3, axis = 1)                                                                                  # Expand dimension of image orientation 3
                ReOr_Diff_Dir             = np.dot(np.hstack((Im_1, Im_2, Im_3)).T, Diff_Dir)                                                               # Correct diffusion directions based of patient orientation
                HeadersDict['B Vector 1'] = float(ReOr_Diff_Dir[0])                                                                                         # Extract b vector 1
                HeadersDict['B Vector 2'] = float(ReOr_Diff_Dir[1])                                                                                         # Extract b vector 2
                HeadersDict['B Vector 3'] = float(ReOr_Diff_Dir[2])

        ### Software Version VE11E ###
        if dcm[0x00181020].value == 'syngo MR E11':
            HeadersDict['Manufacturer']        = dcm[0x00080070].value                                                                                    # Extract manufacturer from DICOM header
            HeadersDict['Scanner Model']       = dcm[0x00081090].value                                                                                    # Extract scanner model from DICOM header
            HeadersDict['Magnet Strength']     = dcm[0x00180087].value                                                                                    # Extract magnet strength from DICOM header
            HeadersDict['Patient ID']          = dcm[0x00100020].value                                                                                    # Extract patient ID from DICOM header
            HeadersDict['Patient Age']         = [int(x) for x in dcm[0x00101010].value.split('Y') if x.isdigit()][0]                                     # Extract patient age from DICOM header
            HeadersDict['Patient Sex']         = dcm[0x00100040].value                                                                                    # Extract patient sex from DICOM header
            HeadersDict['Patient Height']      = float(round(dcm[0x00101020].value, 2))                                                                   # Extract patient height from DICOM header
            HeadersDict['Patient Weight']      = float(round(dcm[0x00101030].value, 2))                                                                   # Extract patient weight from DICOM header
            HeadersDict['Body Part']           = dcm[0x00180015].value                                                                                 # Extract examination region from DICOM header
            HeadersDict['X Resolution']        = dcm[0x00280030][0]                                                                                       # Extract x resolution from DICOM header
            HeadersDict['Y Resolution']        = dcm[0x00280030][1]                                                                                       # Extract y resolution from DICOM header
            HeadersDict['Z Resolution']        = dcm[0x00180050].value                                                                                    # Extract z resolution from DICOM header
            HeadersDict['Echo Time']           = dcm[0x00180081].value                                                                                    # Extract echo time (TE) from DICOM header
            HeadersDict['Repetition Time']     = dcm[0x00180080].value                                                                                    # Extract repetition time (TR) from DICOM header
            HeadersDict['Scanner Orientation'] = dcm[0x00511013].value                                                                                    # Extract scanner orientation

            if (((0x00281052) in dcm) == True):                                                                                                           # Check if data contains phase data ...
                HeadersDict['Image Type']        = 'Phase'                                                                                                  # If true, set data as phase
                HeadersDict['Rescale Intercept'] = dcm[0x00281052].value                                                                                    # Extract rescale intercept information from DICOM header
                HeadersDict['Rescale Slope']     = dcm[0x00281053].value                                                                                    # Extract rescale slope information from DICOM header
            else:                                                                                                                                         # Otherwise ...
                HeadersDict['Image Type']        = 'Magnitude'                                                                                              # Set data as magnitude

            if (((0x00180088) in dcm) == True):                                                                                                           # Check if data contains slice spacing ...
                HeadersDict['Slice Spacing']       = dcm[0x00180088].value                                                                                  # If true, extract slice spacing from DICOM header
            else:                                                                                                                                         # Otherwise ...
                HeadersDict['Slice Spacing']       = 'N/A'                                                                                                  # Set slice spacing to N/A

            if (((0x00181060) in dcm) == True):                                                                                                           # Check if DICOM header contains trigger time ...
                HeadersDict['Trigger Time'] = dcm[0x00181060].value                                                                                         # If true, extract trigger time from DICOM header
            else:                                                                                                                                         # Otherwise ...
                HeadersDict['Trigger Time'] = 'N/A'                                                                                                         # Set trigger time to N/A

            import nibabel.nicom.csareader as csareader                                                                                                   # Import csareader (Siemens) module
            Siemens_CSA_Private_Header1    = csareader.read(dcm[0x00291010].value)                                                                        # Read CSA tag in DICOM Header
            if Siemens_CSA_Private_Header1['tags']['PhaseEncodingDirectionPositive']['items'][0] == 0:                                                    # If phase encoding direction is negative ...
                HeadersDict['Phase Encode Polarity'] = -1                                                                                                   # Set phase encode polarity to -1
            if Siemens_CSA_Private_Header1['tags']['PhaseEncodingDirectionPositive']['items'][0] == 1:                                                    # If phase encoding direction is positive ...
                HeadersDict['Phase Encode Polarity'] = 1                                                                                                    # Set phase encode polarity to 1

            HeadersDict['Slice Location']  = [float(dcm[0x00201041].value)]                                                                               # Extract slice location from DICOM header
            HeadersDict['B Value']         =   int(dcm[0x0019100c].value)                                                                                 # Extract b value from DICOM header

            if HeadersDict['B Value'] == 0:                                                                                                               # If b value equals 0 ...
                HeadersDict['B Vector 1'] = 0.0                                                                                                             # Set b vector 1 to 0
                HeadersDict['B Vector 2'] = 0.0                                                                                                             # Set b vector 2 to 0
                HeadersDict['B Vector 3'] = 0.0                                                                                                             # Set b vector 3 to 0
            else:                                                                                                                                         # Otherwise...
                Orientation               = dcm[0x00200037].value                                                                                           # Extract patient orientation from DICOM Header
                Im_1                      = np.array(Orientation[0:3])                                                                                      # Extract image orientation 1
                Im_2                      = np.array(Orientation[3:6])                                                                                      # Extract image orientation 2
                Im_3                      = np.cross(Im_1, Im_2)                                                                                            # Cross product image orientaiton 1 and 2
                Diff_Dir                  = dcm[0x0019100e].value                                                                                           # Extract diffusion direction from DICOM Header
                Im_1                      = np.expand_dims(Im_1, axis = 1)                                                                                  # Expand dimension of image orientation 1
                Im_2                      = np.expand_dims(Im_2, axis = 1)                                                                                  # Expand dimension of image orientation 2
                Im_3                      = np.expand_dims(Im_3, axis = 1)                                                                                  # Expand dimension of image orientation 3
                ReOr_Diff_Dir             = np.dot(np.hstack((Im_1, Im_2, Im_3)).T, Diff_Dir)                                                               # Correct diffusion directions based of patient orientation
                HeadersDict['B Vector 1'] = float(ReOr_Diff_Dir[0])                                                                                         # Extract b vector 1
                HeadersDict['B Vector 2'] = float(ReOr_Diff_Dir[1])                                                                                         # Extract b vector 2
                HeadersDict['B Vector 3'] = float(ReOr_Diff_Dir[2])                                                                                         # Extract b vector 3

            HeadersDict['Total Rows'] = int(dcm[0x00280010].value)                                                                                        # Extract number of rows from DICOM Header
            HeadersDict['Total Columns'] = int(dcm[0x00280011].value)                                                                                        # Extract number of columns from DICOM Header

            HeadersDict['Acquisition Matrix'] = dcm[0x00181310].value                                                                                     # Extract acquisition matrix from DICOM Header
            HeadersDict['Acquisition Matrix'] = [x for x in HeadersDict['Acquisition Matrix'] if x != 0]                                                  # Remove zeros from acquisition matrix
            HeadersDict['Acquisition Rows']   = int(HeadersDict['Acquisition Matrix'][0])                                                                 # Extract acquisition rows from DICOM Header
            HeadersDict['Acquisition Columns']   = int(HeadersDict['Acquisition Matrix'][1])                                                                 # Extract acquisition columns from DICOM Header

            if (HeadersDict['Acquisition Rows'] > HeadersDict['Acquisition Columns'] and HeadersDict['Total Rows'] < HeadersDict['Total Columns']):             #
                HeadersDict['Acquisition Rows'] = int(HeadersDict['Acquisition Matrix'][1])                                                               #
                HeadersDict['Acquisition Columns'] = int(HeadersDict['Acquisition Matrix'][0])                                                               #

            if ((HeadersDict['Total Rows'] * HeadersDict['Total Columns']) != (HeadersDict['Acquisition Rows'] * HeadersDict['Acquisition Columns'])):          # Check if data is mosaic format ...
                HeadersDict['Mosaic'] = int(dcm[0x0019100a].value)                                                                                          # If so, extract number of mosaic slices from DICOM Header
            else:                                                                                                                                         # Otherwise ...
                if (HeadersDict['Total Rows'] == HeadersDict['Acquisition Columns'] and HeadersDict['Total Columns'] == HeadersDict['Acquisition Rows']):         # Check if single slice rows and columns are correct ...
                    HeadersDict['Acquisition Rows'] = int(HeadersDict['Acquisition Matrix'][1])                                                               # If not, switch rows ...
                    HeadersDict['Acquisition Columns'] = int(HeadersDict['Acquisition Matrix'][0])                                                               # and switch columns ...
                HeadersDict['Mosaic'] = None                                                                                                                # Set Mosaic to none

            if (((0x00511011) in dcm) == True):                                                                                                           # Check if data uses parallel imaging ...
                HeadersDict['Parallel Imaging'] = [int(x) for x in dcm[0x00511011].value.split('p') if x.isdigit()][0]                                      #
            else:                                                                                                                                         # Otherwise ...
                HeadersDict['Parallel Imaging'] = 1                                                                                                         # Set parallel imaging to 1

            HeadersDict['Partial Fourier'] = float((dcm[0x00180091].value * HeadersDict['Parallel Imaging']) / min(HeadersDict['Acquisition Matrix']))  # Extract Partial Fourier infromation from DICOM header
            if (HeadersDict['Partial Fourier'] > 3.5 / 8 and HeadersDict['Partial Fourier'] < 4.5 / 8):                                                 # If Partial Fourier is between 3.5/8 and 4.5/8 ...
                HeadersDict['Partial Fourier'] = '4/8'                                                                                                    # Set Partial Fourier to 4/8
            elif (HeadersDict['Partial Fourier'] > 4.5 / 8 and HeadersDict['Partial Fourier'] < 5.5 / 8):                                               # If Partial Fourier is between 4.5/8 and 5.5/8 ...
                HeadersDict['Partial Fourier'] = '5/8'                                                                                                    # Set Partial Fourier to 5/8
            elif (HeadersDict['Partial Fourier'] > 5.5 / 8 and HeadersDict['Partial Fourier'] < 6.5 / 8):                                               # If Partial Fourier is between 5.5/8 and 6.5/8 ...
                HeadersDict['Partial Fourier'] = '6/8'                                                                                                    # Set Partial Fourier to 6/8
            elif (HeadersDict['Partial Fourier'] > 6.5 / 8 and HeadersDict['Partial Fourier'] < 7.5 / 8):                                               # If Partial Fourier is between 6.5/8 and 7.5/8 ...
                HeadersDict['Partial Fourier'] = '7/8'                                                                                                    # Set Partial Fourier to 7/8
            elif (HeadersDict['Partial Fourier'] > 7.5 / 8 and HeadersDict['Partial Fourier'] < 8.5 / 8):                                               # If Partial Fourier is between 7.5/8 and 8.5/8 ...
                HeadersDict['Partial Fourier'] = '8/8'                                                                                                    # Set Partial Fourier to 8/8
            else:                                                                                                                                       # Otherwise ...
                HeadersDict['Partial Fourier'] = 'N/A'                                                                                                    # Set Patrial Fourier to N/A
    ########## Alias Column Keys ###################################################################################################################
    ### Earlier releases used 'Total Cols' / 'Acquisition Cols' while later ones used 'Total Columns' / 'Acquisition Columns'.
    ### Publish both spellings so either naming convention resolves.
    for short, long in [('Total Cols', 'Total Columns'), ('Acquisition Cols', 'Acquisition Columns')]:                                                 # Iterate through key pairs
        if long in HeadersDict and short not in HeadersDict:                                                                                              # If only the long spelling exists ...
            HeadersDict[short] = HeadersDict[long]                                                                                                            # Publish the short spelling
        elif short in HeadersDict and long not in HeadersDict:                                                                                            # Or if only the short spelling exists ...
            HeadersDict[long] = HeadersDict[short]                                                                                                            # Publish the long spelling
    return HeadersDict
    
def NifTi_Reader(NifTi_path, b_values_path = None, b_vectors_path = None, header_path = None, info = 'ON'):
    from dipy.io.image      import load_nifti
    from dipy.io.gradients  import read_bvals_bvecs
    from cardpy.Data_Import import Header_Reader

    matrix, affine_matrix, voxel_resolution = load_nifti(NifTi_path, return_voxsize = True)
    Header                                  = Header_Reader(header_path)
    Header['X Resolution']                  = voxel_resolution[0]
    Header['Y Resolution']                  = voxel_resolution[1]
    Header['Z Resolution']                  = voxel_resolution[2]
    b_vals, b_vecs                          = read_bvals_bvecs(b_values_path, b_vectors_path)
    
    return [matrix, b_vals, b_vecs, Header, voxel_resolution, affine_matrix]
    
def Header_Reader(Header_Path):
    """
    ########## Definition Inputs ##################################################################################################################
    Header_Path   : Path CarDpy header (*.header) file.
    ########## Definition Outputs #################################################################################################################
    Headers_Dict  : Dictionary containing header information from CarDpy header (*.header) file.
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import Modules ######################################################################################################################
    with open(Header_Path) as f:
        lines = f.readlines()
    HeadersDict = dict()
    for idx in range(len(lines)):
        key_word              = lines[idx].split(':')[0]
        if key_word != 'Scanner Model':
            key_value             = lines[idx].split(':')[1]
            key_value             = key_value.strip()
            key_value             = key_value.split(' ')[0]
        else:
            key_value             = lines[idx].split(':')[1]
            key_value             = key_value.strip()
        HeadersDict[key_word] = key_value
    return HeadersDict
