# CarDpy — Questions

Three defects found in the CarDpy processing code during the July 2026
packaging audit. All three were **left unfixed on purpose**, because fixing any
of them means deciding what a calculation *should* do — a methods decision, not
a packaging one.

**None of these affect current results.** All are latent under the settings in
use today:

- `registration_algorithm = 'Affine'`
- `operation_type = 'Magnitude'`
- single b = 0 volume (`Healthy_Volunteer_007`)

They surface only when one of those settings changes.

---

## 1. Elastic registration uses an undefined loop variable

**File:** `cardpy/Data_Processing/Registration.py`, line 211
**Surfaces when:** `registration_algorithm = 'Elastic'`

```python
moving_elastic = affine.transform(temporary_matrix[:, :, slc, dif, avg])
```

This line is in the second registration stage, which registers each average's
b = 0 image to the first average's b = 0 image. That loop iterates over `avg`
and `slc` only. **`dif` is not defined in this scope** — the
`for dif in range(directions)` loop does not begin until line 215, four lines
later.

So `dif` either raises `NameError`, or silently holds a leftover value from the
first registration stage — in which case the code runs without complaint but
registers against the wrong diffusion direction.

**The question:** the two lines just above select `static` and `moving` at
diffusion index `0`, which suggests `dif = 0` was intended here. Is that right?

I did not change it, because the code currently produces output, and setting
the index changes which image gets registered — and therefore changes the
resulting numbers. That is a registration behavior decision.

*(For reference: the same expression at line 141 is correct. It sits inside the
first registration stage, where `dif` is a live loop variable.)*

---

## 2. The two elastic registration stages use different similarity metrics

**File:** `cardpy/Data_Processing/Registration.py`, lines 139 and 209
**Surfaces when:** `registration_algorithm = 'Elastic'`

| Stage | Line | Metric |
|---|---|---|
| Diffusion directions → b = 0 | 139 | `EMMetric(2)` |
| Averages → first average | 209 | `SSDMetric(2)` |

**The question:** is this deliberate? The two stages register different things,
so different metrics could be well justified. It could equally be a copy-paste
artifact.

Either way, changing a similarity metric is a methods choice, so I left it
alone.

---

## 3. Diffusivity filter breaks with more than one b = 0 volume

**File:** `cardpy/Data_Processing/Diffusivity.py`, line 38
**Surfaces when:** a dataset has two or more b = 0 volumes

```python
if dif == bval_low_indicies:
```

`dif` is an integer loop counter. `bval_low_indicies` is a NumPy array holding
*every* index where b = b_low. Comparing them gives a one-element boolean array,
which Python accepts in an `if` — but only while there is exactly one b = 0
volume. With two or more it raises:

> `ValueError: The truth value of an array with more than one element is ambiguous.`

**The question:** what should ADC filtering do when there are several b = 0
volumes — skip all of them, or average them before computing `S_0`?

The mechanical fix (`if dif in bval_low_indicies:`) is obvious, but it forces an
answer the original code never gave. Line 38 already hardcodes
`bval_low_indicies[0]` as the `S_0` source, so the intended multi-b0 behavior is
genuinely unclear.

---

## Context: what *was* changed

For completeness, so it is clear these three are the only open items. The
packaging work was kept strictly behavior-preserving — nothing below can alter
a computed value in an output map:

- Resolved a `Colormaps.py` / `Colormaps/` module-vs-package name collision
  (the directory was unreachable dead code)
- Replaced fragile `__file__.split(...)` path resolution with
  `os.path.dirname(os.path.abspath(__file__))`
- Closed seven leaked file handles in `Colormaps.py`
- Added a missing `Tools/__init__.py` (would have been dropped from a wheel)
- Made the `GUI_Tools` import lazy, so `import cardpy` no longer requires
  tkinter and works on headless machines
- Removed `Sample_Data/` and the stale `Sample_Notebooks/` copies
- Rewired both notebooks to an explicit `STUDY_ROOT` config block, writing
  outputs to the study directory instead of inside the installed package

---

## One data-handling note worth raising

`Data_Sorting.stacked2sorted()` infers the number of averages as `max(counts)`
over unique b-vectors. If a dataset's averages are not uniform across
directions, it **silently zero-pads** the missing volumes rather than raising an
error.

This is what broke the old bundled sample data: the image file held 64 volumes
(4 averages × 16 directions) while its `.bvals`/`.bvecs` declared 80
(5 averages × 16). The result was an all-zero average that crashed
`transform_centers_of_mass` during registration.

Not a bug in the math, but a shape-consistency check at import would turn a
confusing downstream crash into a clear error message. Worth considering as a
future addition.
