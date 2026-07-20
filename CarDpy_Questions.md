# CarDpy — Questions

- `registration_algorithm = 'Affine'`
- `operation_type = 'Magnitude'`
- single b = 0 volume (`Healthy_Volunteer_007`)


### 1. Elastic registration — undefined loop variable
- **Line:** `Registration.py:211`
- **Issue:** `moving_elastic = affine.transform(temporary_matrix[:, :, slc, dif, avg])` uses `dif`, but the loop here only iterates `avg` and `slc`. The `for dif` loop doesn't start until line 215, so `dif` is either undefined (`NameError`) or a leftover from the first registration stage. may silently register the wrong direction. only surfaces when `registration_algorithm = 'Elastic'`.
- **Possible fix:** set `dif = 0` (matches the `static`/`moving` selection just above). Changes which image is registered, so it's a methods decision.

### 2. Elastic registration — mismatched similarity metrics
- **Lines:** `Registration.py:139` (`EMMetric(2)`) vs `:209` (`SSDMetric(2)`)
- **Issue:** the two elastic stages use different metrics. Could be intentional??. Only relevant `registration_algorithm = 'Elastic'`.
- **Possible fix:** make both the same metric, if unintended.

### 3. Diffusivity filter — int compared to array
- **Line:** `Diffusivity.py:38`
- **Issue:** `if dif == bval_low_indicies:` compares an int loop counter to a NumPy array of all b=0 indices. Works with one b=0 volume; raises `ValueError: truth value of an array... is ambiguous` with two or more.
- **Possible fix:** `if dif in bval_low_indicies:`. But this forces a decision the original never made — what should ADC filtering do with multiple b=0 volumes (skip all, or average before computing `S_0`)? Line 38 already hardcodes `bval_low_indicies[0]` as `S_0`, so intended behavior is unclear.

---

## Data-handling note
`Data_Sorting.stacked2sorted()` infers averages as `max(counts)` (64 image volumes vs 80 declared in bvals/bvecs → all-zero average → crash in `transform_centers_of_mass`). A shape-consistency check at import??