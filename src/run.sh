#!/bin/bash

# Ensure the script stops on errors
set -e

# Submit the job to the batch script
sbatch ./run.batch
