#!/bin/bash

# Ensure the script stops on errors
set -e

# Submit the job to the batch script
sbatch ./spline_pinn.batch

# Git add, commit, and push
git add .
git commit -m "Submitting HPC job"
git push

