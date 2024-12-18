import trimesh
import numpy as np
import random
import torch

torch.set_default_dtype(torch.float64)
from utils import *
from hermite_spline import *
from unet import *
from torch.optim import LBFGS
from spline_pinn import *
import time
import wandb
import os
from git import Repo
from utils import *
from constants import *

folder = "dp11"
Project_name = (
    "Spline-PINNs_with_heat"  # Full_Project_name will be {Project_name}_{folder}
)
device = "cuda"  # Turn this to "cpu" if you are debugging the flow on the CPU
debug = False  # Turn this to "True" if you are debugging the flow and don't want to send logs to Wandb

data_folder = "./preProcessedData/with_T/" + folder + "/"
Full_Project_name = Project_name + "_" + folder

# Model Hyperparams
epochs = 100

seed = 42

# Path to the parent directory of the `src/` folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Initialize the repository at the parent directory level
repo = Repo(parent_dir)

if not debug:
    run = wandb.init(
        # set the wandb project where this run will be logged
        project=Full_Project_name,
        entity="sathvikbhagavan-epfl",
        # track hyperparameters and run metadata
        config={
            "optimizer": "LBFGS",
            "architecture": "Unet",
            "epochs": epochs,
            "seed": seed,
        },
    )

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.set_default_device(device)
print(f"Using device: {device}")

inlet = np.load(data_folder + "vel_x_inlet.npy")
inlet_points = torch.tensor(inlet[:, 0:3] * 1000.0)
vx_inlet_data = torch.tensor(np.load(data_folder + "vel_x_inlet.npy")[:, 3])
vy_inlet_data = torch.tensor(np.load(data_folder + "vel_y_inlet.npy")[:, 3])
vz_inlet_data = torch.tensor(np.load(data_folder + "vel_z_inlet.npy")[:, 3])

data_points = torch.tensor(np.load(data_folder + "vel_x.npy")[:, 0:3] * 1000.0)
vx_data = torch.tensor(np.load(data_folder + "vel_x.npy")[:, 3])
vy_data = torch.tensor(np.load(data_folder + "vel_y.npy")[:, 3])
vz_data = torch.tensor(np.load(data_folder + "vel_z.npy")[:, 3])
p_data = torch.tensor(np.load(data_folder + "press.npy")[:, 3])
temp_data = torch.tensor(np.load(data_folder + "temp.npy")[:, 3])

# Use data for supervised loss
num_samples = 50000
indices = torch.randint(0, data_points.shape[0], (num_samples,))
sampled_points = data_points[indices]

# Get the corresponding fields
vx_sampled_data = vx_data[indices]
vy_sampled_data = vy_data[indices]
vz_sampled_data = vz_data[indices]
p_sampled_data = p_data[indices] / 10**5
temp_sampled_data = temp_data[indices]

# Load the domain
obj = trimesh.load("./Baseline_ML4Science.stl")

grid_resolution = np.array([512, 64, 16])
binary_mask = get_binary_mask(obj, grid_resolution)
step = obj.bounding_box.extents / (grid_resolution - 1)

# Instantiate the neural network
unet_model = UNet3D().to(device)
unet_model.apply(initialize_weights)
unet_model = unet_model.double()
print(
    f"Number of parameters in the model is: {sum(p.numel() for p in unet_model.parameters())}"
)

# Define optimizer
optimizer = LBFGS(unet_model.parameters(), line_search_fn="strong_wolfe")

start_time = time.time()
training_loss_track = []
validation_loss_track = []

# Sample validation points
validation_points, validation_labels = sample_points(obj, 10000, 3000, 10000)
unet_input = prepare_mesh_for_unet(binary_mask).to(device)

for epoch in range(epochs):
    print(f"{epoch+1}/{epochs}")
    # Sample train points
    train_points, train_labels = sample_points(obj, 10000, 3000, 10000)

    def closure():
        # Get hermite spline coefficients from the model
        spline_coeff = unet_model(unet_input)[0]

        # Calculating various field terms using coefficients
        (
            _,
            _,
            _,
            _,
            _,
            loss_divergence,
            loss_momentum_x,
            loss_momentum_y,
            loss_momentum_z,
            loss_outlet_boundary,
            loss_other_boundary,
            loss_heat,
            loss_t_wall_boundary,
        ) = get_fields_and_losses(
            spline_coeff, train_points, train_labels, step, grid_resolution
        )

        inlet_fields = get_fields(spline_coeff, inlet_points, step, grid_resolution)

        loss_inlet_boundary = (
            torch.mean((inlet_fields[0] - vx_inlet_data) ** 2)
            + torch.mean((inlet_fields[1] - vy_inlet_data) ** 2)
            + torch.mean((inlet_fields[2] - vz_inlet_data) ** 2)
        )

        loss_inlet_temp_boundary = torch.mean((inlet_fields[4] - T_inlet) ** 2) / 10**6

        vx_supervised, vy_supervised, vz_supervised, p_supervised, t_supervised = (
            get_fields(spline_coeff, sampled_points, step, grid_resolution)
        )
        supervised_loss = (
            torch.mean((vx_supervised - vx_sampled_data) ** 2)
            + torch.mean((vy_supervised - vy_sampled_data) ** 2)
            + torch.mean((vz_supervised - vz_sampled_data) ** 2)
            + torch.mean((p_supervised - p_sampled_data) ** 2)
            + torch.mean((t_supervised - temp_sampled_data) ** 2) / 10**6
        )

        loss_total = (
            0.1*loss_divergence
            + loss_momentum_x
            + 0.1*loss_momentum_y
            + 0.1*loss_momentum_z
            + loss_inlet_boundary
            + 0.1*loss_outlet_boundary
            + 0.1*loss_other_boundary
            + supervised_loss
            + 0.1*loss_heat
            + 0.1*loss_inlet_temp_boundary
            + 10*loss_t_wall_boundary
        )

        if not debug:
            wandb.log(
                {
                    "Divergence Loss": np.log10(loss_divergence.item()),
                    "X Momentum Loss": np.log10(loss_momentum_x.item()),
                    "Y Momentum Loss": np.log10(loss_momentum_y.item()),
                    "Z Momentum Loss": np.log10(loss_momentum_z.item()),
                    "Inlet Boundary Loss": np.log10(loss_inlet_boundary.item()),
                    "Outlet Boundary Loss": np.log10(loss_outlet_boundary.item()),
                    "Other Boundary Loss": np.log10(loss_other_boundary.item()),
                    "Supervised Loss": np.log10(supervised_loss.item()),
                    "Heat Loss": np.log10(loss_heat.item()),
                    "Inlet Temperature Boundary Loss": np.log10(
                        loss_inlet_temp_boundary.item()
                    ),
                    "Surface Temperature Boundary Loss": np.log10(
                        loss_t_wall_boundary.item()
                    ),
                    "Total Loss": np.log10(loss_total.item()),
                }
            )

        training_loss_track.append(loss_total.item())
        print(
            f"Divergence Loss: {loss_divergence.item()}, "
            f"X Momentum Loss: {loss_momentum_x.item()}, "
            f"Y Momentum Loss: {loss_momentum_y.item()}, "
            f"Z Momentum Loss: {loss_momentum_z.item()}, "
            f"Inlet Boundary Loss: {loss_inlet_boundary.item()}, "
            f"Outlet Boundary Loss: {loss_outlet_boundary.item()}, "
            f"Other Boundary Loss: {loss_other_boundary.item()}, "
            f"Supervised Loss: {supervised_loss.item()}",
            f"Heat Loss: {loss_heat.item()}",
            f"Inlet Temperature Boundary Loss: {loss_inlet_temp_boundary.item()}",
            f"Surface Temperature Boundary Loss: {loss_t_wall_boundary.item()}",
            f"Total Loss: {loss_total.item()}",
        )

        # Using LBFGS optimizer
        optimizer.zero_grad()
        loss_total.backward()
        return loss_total

    # Validation
    # Switch model to evaluation mode
    unet_model.eval()
    with torch.no_grad():
        spline_coeff = unet_model(unet_input)[0]
        (
            validation_vx,
            validation_vy,
            validation_vz,
            validation_p,
            validation_T,
            validation_loss_divergence,
            validation_loss_momentum_x,
            validation_loss_momentum_y,
            validation_loss_momentum_z,
            validation_loss_outlet_boundary,
            validation_loss_other_boundary,
            validation_loss_heat,
            validation_loss_t_wall_boundary,
        ) = get_fields_and_losses(
            spline_coeff, validation_points, validation_labels, step, grid_resolution
        )

        validation_loss_total = (
            validation_loss_divergence
            + validation_loss_momentum_x
            + validation_loss_momentum_y
            + validation_loss_momentum_z
            + validation_loss_outlet_boundary
            + validation_loss_other_boundary
            + validation_loss_heat
            + validation_loss_t_wall_boundary
        )

        fields = [
            ("vx", validation_vx),
            ("vy", validation_vy),
            ("vz", validation_vz),
            ("p", validation_p),
            ("temp", validation_T),
        ]
        plot_fields(fields, validation_points)

        if not debug:
            wandb.log(
                {
                    "Validation Divergence Loss": np.log10(
                        validation_loss_divergence.item()
                    ),
                    "Validation X Momentum Loss": np.log10(
                        validation_loss_momentum_x.item()
                    ),
                    "Validation Y Momentum Loss": np.log10(
                        validation_loss_momentum_y.item()
                    ),
                    "Validation Z Momentum Loss": np.log10(
                        validation_loss_momentum_z.item()
                    ),
                    "Validation outlet Boundary Loss": np.log10(
                        validation_loss_outlet_boundary.item()
                    ),
                    "Validation Other Boundary Loss": np.log10(
                        validation_loss_other_boundary.item()
                    ),
                    "Validation Heat Loss": np.log10(validation_loss_heat.item()),
                    "Validation Surface Temperature Boundary Loss": np.log10(
                        validation_loss_t_wall_boundary.item()
                    ),
                    "Validation Total Loss": np.log10(validation_loss_total.item()),
                }
            )

        validation_loss_track.append(validation_loss_total.item())
        print(
            f"Validation Divergence Loss: {validation_loss_divergence.item()}, "
            f"Validation X Momentum Loss: {validation_loss_momentum_x.item()}, "
            f"Validation Y Momentum Loss: {validation_loss_momentum_y.item()}, "
            f"Validation Z Momentum Loss: {validation_loss_momentum_z.item()}, "
            f"Validation Outlet Boundary Loss: {validation_loss_outlet_boundary.item()}, "
            f"Validation Other Boundary Loss: {validation_loss_other_boundary.item()}, "
            f"Validation Heat Loss: {validation_loss_heat.item()}, "
            f"Validation Surface Temperature Boundary Loss: {validation_loss_t_wall_boundary.item()}, "
            f"Validation Total Loss: {validation_loss_total.item()}"
        )

    unet_model.train()
    optimizer.step(closure)

stop_time = time.time()
print(f"Time taken for training is: {stop_time - start_time}")
torch.save(unet_model.state_dict(), "../run/unet_model.pt")

## Plotting
unet_model.eval()
unet_input = prepare_mesh_for_unet(binary_mask).to(device)
with torch.no_grad():
    spline_coeff = unet_model(unet_input)[0]
    (
        validation_vx,
        validation_vy,
        validation_vz,
        validation_p,
        validation_T,
        validation_loss_divergence,
        validation_loss_momentum_x,
        validation_loss_momentum_y,
        validation_loss_momentum_z,
        validation_loss_outlet_boundary,
        validation_loss_other_boundary,
        validation_loss_heat,
        validation_loss_t_wall_boundary,
    ) = get_fields_and_losses(
        spline_coeff, validation_points, validation_labels, step, grid_resolution
    )

fields = [
    ("vx", validation_vx),
    ("vy", validation_vy),
    ("vz", validation_vz),
    ("p", validation_p),
    ("temp", validation_T),
]
plot_fields(fields, validation_points)

######## Inference

device = "cpu"
torch.set_default_device(device)
unet_model = UNet3D().to(device)
unet_model.load_state_dict(
    torch.load(
        "../run/unet_model.pt", weights_only=True, map_location=torch.device("cpu")
    )
)
unet_input = prepare_mesh_for_unet(binary_mask).to(device)
spline_coeff = unet_model(unet_input)[0]

all_points = torch.tensor(
    np.concatenate(
        (
            np.load(os.path.join(data_folder, "vel_x_inlet.npy"))[:, :3],
            np.load(os.path.join(data_folder, "vel_x.npy"))[:, :3],
        )
    )
    * 1000
)

x, y, z, x_supports, y_supports, z_supports = get_support_points(
    all_points, step, grid_resolution
)
vx_pred, vy_pred, vz_pred, p_pred, T_pred = get_fields(
    spline_coeff, all_points, step, grid_resolution
)
vx_pred = vx_pred.cpu().detach().numpy()
vy_pred = vy_pred.cpu().detach().numpy()
vz_pred = vz_pred.cpu().detach().numpy()
p_pred = p_pred.cpu().detach().numpy() * 10**5
T_pred = T_pred.cpu().detach().numpy()

plot_aginast_data(data_folder, vx_pred, vy_pred, vz_pred, p_pred, T_pred)

time.sleep(120)
if repo.is_dirty(untracked_files=True):
    print("Repository has changes, preparing to commit.")

    # Stage all changes in the parent directory
    repo.git.add(A=True)  # Stages all changes

    # Commit the changes
    commit_message = f"Running job with run name: {run.name}, url: {run.url}"
    repo.index.commit(commit_message)
    print(f"Committed changes with message: {commit_message}")

    # Push changes
    origin = repo.remote(name="origin")
    origin.push()
    print("Pushed changes to the remote repository.")
else:
    print("No changes to commit.")
