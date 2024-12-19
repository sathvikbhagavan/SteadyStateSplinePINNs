<h1 align=center>Steady State Spline PINNs</h1>
<h4 align=center>Sathvik Bhagavan, Ananya Gupta, Tamar Alphaidze</h4>

## Introduction

This project implements Spline Physics Informed Neural Networks (Spline PINNs) based on [this paper](https://arxiv.org/pdf/2109.07143) for solving Steady State Navier Stokes PDE in a 3D domain.

## Reproducibility

To run our best model, we provide the following [`run.py`](./run.py) script. You can run it with the following command:

- The `src/run.py` script can be used for best steady state spline pinn model by running the following command: `cd src/; python3 run.py --model sssplinepinn`
- The `src/run.py` script can also be used for best baseline pinn model by running the following command: `cd src/; python3 run.py --model pinn`
- This will run inference using the trained model and generate plots storing them in `run/`. The plots include the visualizing different velocity fields, pressure and temperature as well as their difference from the ground truth data.

## Layout of the repository

```markdown
.
├── README.md                  # The following README :)
├── best_models/               # The directory where all the best models will be stored
├── run/                       # The directory where all the plots will be stored after inference
├── src/
    ├── preProceessedData/     # The data folder which is processed from CFD simulations
    ├── constants.py           # All the constants required for training
    ├── hermite_spline.py      # Functions for defining hermite spline kernels
    ├── pinn_run.py            # Run script for training baseline pinn models
    ├── pinn.py                # Classes and functions to define pinn models and its loss functions
    ├── run.batch              # Batch script to run on HPC
    ├── run.sh                 # Wrapper over `run.batch` for ease
    ├── spline_pinn_run.py     # Run script for training spline pinn models
    ├── spline_pinn.py         # Functions to define loss functions for spline pinns
    ├── unet.py                # Class for defining UNET architecture for spline pinns
    ├── utils.py               # Helper functions
├── .gitignore                 # Git ignore file
├── requirements.txt           # The necessary packages to run the project
├── run.py                     # Inference script to reproduce results
```
