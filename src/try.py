import trimesh
import numpy as np
import random
import torch
from sample import *
from hermite_spline import *
from unet import *
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau
from spline_pinn import *
import time
import wandb
import os
from git import Repo
from inference import *

# Path to the parent directory of the `src/` folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Initialize the repository at the parent directory level
repo = Repo(parent_dir)

if repo.is_dirty(untracked_files=True):
    print("Repository has changes, preparing to commit.")

    # Stage all changes in the parent directory
    repo.git.add(A=True)  # Stages all changes

    # Commit the changes
    commit_message = f"trying git push logs and previous run with ssh key"
    repo.index.commit(commit_message)
    print(f"Committed changes with message: {commit_message}")

    # Push changes
    origin = repo.remote(name="origin")
    origin.push()
    print("Pushed changes to the remote repository.")
else:
    print("No changes to commit.")