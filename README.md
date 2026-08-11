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
Delete [macos] if not using Mac. 

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
When possible:
- Add automated segmentation GUI with Sascha's model
- windows testing

Action Tasks:
4 - call the GUI an additional time to do a "whole-heart crop" which maybe useful for segmenntation
    -  Zooming, drag to resize
5 - GUI updates for usability 
3 - manual rejection tool to see all the images and click on one to reject
Done - stop pop ups for plots
2 - version of cardpy calls that is not jupyter notebook --> have a config file and use that to determine what steps to use
    - refine pip install set-up (pypi)
 


## Proposed Merges with Tyler
- New registration WIP
- New Gibb's Ringing Correction 
- New data loading algorithm