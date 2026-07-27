# CarDpy

A Python toolbox for processing cardiac diffusion tensor imaging (cDTI) data.

# Getting Started

### Environment Setup

Create the Conda environment:

```bash
conda create -n cardpy-test python=3.12 -y, then
conda activate cardpy-test
```

### Installation

Install **CarDpy** with:

```bash
pip install "cardpy[macos] @ git+https://github.com/dzarroug/CarDpy.git@dev"
```

### Run the pipeline! 
Start with 01_Data_Processing then run 02_Post-Processing Using Healthy_Volunteer_07 Dataset 


## Associated Publication

**Evaluating the Effect of Post-processing Steps When Analyzing Cardiac Diffusion Tensor Data**

> Cork, T.E., Hannum, A.J., Loecher, M., Perotti, L.E., Ennis, D.B. (2025). *Evaluating the Effect of Post-processing Steps When Analyzing Cardiac Diffusion Tensor Data*. In: Chabiniok, R., Zou, Q., Hussain, T., Nguyen, H.H., Zaha, V.G., Gusseva, M. (eds) **Functional Imaging and Modeling of the Heart (FIMH 2025)**. Lecture Notes in Computer Science, vol. 15673. Springer, Cham.

DOI: https://doi.org/10.1007/978-3-031-94562-5_13

---

## Completed Tasks

- preliminary pip install
- GUI compatibility on Python 3.12 and 3.14

---

## Ongoing Tasks
- Update so warnings are mitigated 
- Add in output folder Diagnostics sub-folder that includes
    - store crop heart coordinates
        - call the GUI an additional time to do a "whole-heart crop" which maybe useful for segmenntation
    - intermediate k-means clustering and rejection images that currently pop up (instead can be saved as a .png as diagnostic)
    - data on the # images that were rejected (not just the percentages)
    - store index of rejected images
- refine pip install set-up
- Add automated segmentation GUI with Sascha's model
- GUI updates for usability 
- manual rejection tool to see all the images and click on one to reject
- version of cardpy calls that is not jupyter notebook --> have a config file and use that to determine what steps to use
- windows testing 


## Proposed Merges with Tyler
- New registration WIP
- New Gibb's Ringing Correction 
- New data loading algorithm