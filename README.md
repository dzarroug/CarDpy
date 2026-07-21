# CarDpy

A Python toolbox for processing cardiac diffusion tensor imaging (cDTI) data.

# Getting Started

### Environment Setup

Create the Conda environment from the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate cardpy
```

If the environment already exists, remove it and recreate it:

```bash
conda deactivate
conda env remove -n cardpy
conda env create -f environment.yml
conda activate cardpy
```

Alternatively, you can update an existing environment in place:

```bash
conda env update -f environment.yml --prune
```

### Run the pipeline! 
Start with 01_Data_Processing then run 02_Post-Processing Using Healthy_Volunteer_07 Dataset 


## Associated Publication

**Evaluating the Effect of Post-processing Steps When Analyzing Cardiac Diffusion Tensor Data**

> Cork, T.E., Hannum, A.J., Loecher, M., Perotti, L.E., Ennis, D.B. (2025). *Evaluating the Effect of Post-processing Steps When Analyzing Cardiac Diffusion Tensor Data*. In: Chabiniok, R., Zou, Q., Hussain, T., Nguyen, H.H., Zaha, V.G., Gusseva, M. (eds) **Functional Imaging and Modeling of the Heart (FIMH 2025)**. Lecture Notes in Computer Science, vol. 15673. Springer, Cham.

- DOI: https://doi.org/10.1007/978-3-031-94562-5_13
- Link: https://link.springer.com/chapter/10.1007/978-3-031-94562-5_13

---

## Completed Tasks

- Rebuilt Yellowbrick's `KElbowVisualizer` using `scikit-learn` and the `kneed` library.
- Updated all SSIM `data_range` values to `2.0`.
- Added shape consistency checks to `Data_Sorting.py` (lines 64–71).

---

## Ongoing Tasks

- Add `kneed` as an installation dependency or switch up package
- Update so warnings are mitigated 
- Add in output folder Diagnostics sub-folder that includes
    - store crop heart coordinates
        - call the GUI an additional time to do a "whole-heart crop" which maybe useful for segmenntation
    - intermediate k-means clustering and rejection images that currently pop up (instead can be saved as a .png as diagnostic)
    - data on the # images that were rejected (not just the percentages)
    - store index of rejected images 
- move cardpy to pip! 


# Future Tasks
- Add automated segmentation GUI
- GUI updates for usability 
- manual rejection tool to see all the images and click on one to reject
- version of cardpy calls that is not jupyter notebook --> have a config file and use that to determine what steps to use
    