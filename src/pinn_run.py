import trimesh
import numpy as np
import random
import torch

torch.set_default_dtype(torch.float64)
from utils import *
from unet import *
from torch.optim import LBFGS
from pinn import *
import time
import wandb
import os
from git import Repo
from utils import *
from constants import *

folder = "dp11"
Project_name = (
    "PINNs-baseline_with_heat"  # Full_Project_name will be {Project_name}_{folder}
)
device = "cuda"  # Turn this to "cpu" if you are debugging the flow on the CPU
debug = False  # Turn this to "True" if you are debugging the flow and don't want to send logs to Wandb

data_folder = "./preProcessedData/with_T/" + folder + "/"
Full_Project_name = Project_name + "_" + folder

# Model Hyperparams
epochs = 100
hidden_dim = 128
num_layer = 4
seed = 42

# Path to the parent directory of the `src/` folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Initialize the repository at the parent directory level
repo = Repo(parent_dir)

if not debug:
    run = wandb.init(
        # set the wandb project where this run will be logged
        project=Full_Project_name,
        # track hyperparameters and run metadata
        config={
            # "learning_rate": lr,
            "optimizer": "LBFGS",
            "architecture": "FFNN",
            "epochs": epochs,
            "seed": seed,
            "hidden_dim": hidden_dim,
            "num_layers": num_layer,
        },
    )

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.set_default_device(device)
print(f"Using device: {device}")

inlet = np.load(data_folder + "vel_x_inlet.npy")
inlet_points = torch.tensor(inlet[:, 0:3])
vx_inlet_data = torch.tensor(np.load(data_folder + "vel_x_inlet.npy")[:, 3])
vy_inlet_data = torch.tensor(np.load(data_folder + "vel_y_inlet.npy")[:, 3])
vz_inlet_data = torch.tensor(np.load(data_folder + "vel_z_inlet.npy")[:, 3])

data_points = torch.tensor(np.load(data_folder + "vel_x.npy")[:, 0:3])
vx_data = torch.tensor(np.load(data_folder + "vel_x.npy")[:, 3])
vy_data = torch.tensor(np.load(data_folder + "vel_y.npy")[:, 3])
vz_data = torch.tensor(np.load(data_folder + "vel_z.npy")[:, 3])
p_data = torch.tensor(np.load(data_folder + "press.npy")[:, 3])
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

# Define the model
pinn_model = PINNs(in_dim=3, hidden_dim=hidden_dim, out_dim=5, num_layer=num_layer).to(
    device
)
pinn_model.apply(init_weights)
pinn_model = pinn_model.double()

# Define optimizer
optimizer = LBFGS(pinn_model.parameters(), line_search_fn="strong_wolfe")

start_time = time.time()
training_loss_track = []
validation_loss_track = []

# Sample validation points
validation_points, validation_labels = sample_points(obj, 30000, 3000, 20000)
validation_points = validation_points / 1000

for epoch in range(epochs):
    print(f"{epoch+1}/{epochs}")
    # Sample train points
    train_points, train_labels = sample_points(obj, 30000, 3000, 20000)
    train_points = train_points / 1000
    train_points.requires_grad_(True)

    def closure():
        train_fields = pinn_model(train_points)
        (
            vx,
            vy,
            vz,
            p,
            T,
            vx_x,
            vx_y,
            vx_z,
            vx_xx,
            vx_yy,
            vx_zz,
            vy_x,
            vy_y,
            vy_z,
            vy_xx,
            vy_yy,
            vy_zz,
            vz_x,
            vz_y,
            vz_z,
            vz_xx,
            vz_yy,
            vz_zz,
            p_x,
            p_y,
            p_z,
            T_x,
            T_y,
            T_z,
            T_xx,
            T_yy,
            T_zz,
        ) = get_values_and_derivatives(train_fields, train_points)

        (
            loss_divergence,
            loss_momentum_x,
            loss_momentum_y,
            loss_momentum_z,
            loss_outlet_boundary,
            loss_other_boundary,
            loss_heat,
            loss_t_wall_boundary,
        ) = get_loss(
            vx,
            vy,
            vz,
            p,
            T,
            vx_x,
            vx_y,
            vx_z,
            vx_xx,
            vx_yy,
            vx_zz,
            vy_x,
            vy_y,
            vy_z,
            vy_xx,
            vy_yy,
            vy_zz,
            vz_x,
            vz_y,
            vz_z,
            vz_xx,
            vz_yy,
            vz_zz,
            p_x,
            p_y,
            p_z,
            T_x,
            T_y,
            T_z,
            T_xx,
            T_yy,
            T_zz,
            train_labels,
            p_outlet,
        )

        inlet_fields = pinn_model(inlet_points)

        loss_inlet_boundary = (
            torch.mean((inlet_fields[:, 0] - vx_inlet_data) ** 2)
            + torch.mean((inlet_fields[:, 1] - vy_inlet_data) ** 2)
            + torch.mean((inlet_fields[:, 2] - vz_inlet_data) ** 2)
        )

        loss_inlet_temp_boundary = (
            torch.mean((inlet_fields[:, 4] * 1000 - T_inlet) ** 2) / 10**6
        )

        fields_supervised = pinn_model(sampled_points)
        vx_supervised = fields_supervised[:, 0]
        vy_supervised = fields_supervised[:, 1]
        vz_supervised = fields_supervised[:, 2]
        p_supervised = fields_supervised[:, 3]
        t_supervised = fields_supervised[:, 4] * 1000

        supervised_loss = (
            torch.mean((vx_supervised - vx_sampled_data) ** 2)
            + torch.mean((vy_supervised - vy_sampled_data) ** 2)
            + torch.mean((vz_supervised - vz_sampled_data) ** 2)
            + torch.mean((p_supervised - p_sampled_data) ** 2)
            + torch.mean((t_supervised - temp_sampled_data) ** 2) / 10**6
        )

        loss_total = (
            0.1 * loss_divergence
            + loss_momentum_x
            + 0.1 * loss_momentum_y
            + 0.1 * loss_momentum_z
            + loss_inlet_boundary
            + 0.1 * loss_outlet_boundary
            + 0.1 * loss_other_boundary
            + supervised_loss
            + 0.1 * loss_heat
            + 0.1 * loss_inlet_temp_boundary
            + 10 * loss_t_wall_boundary
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
    pinn_model.eval()
    optimizer.zero_grad()
    validation_points.requires_grad_(True)
    validation_fields = pinn_model(validation_points)

    (
        vx,
        vy,
        vz,
        p,
        T,
        vx_x,
        vx_y,
        vx_z,
        vx_xx,
        vx_yy,
        vx_zz,
        vy_x,
        vy_y,
        vy_z,
        vy_xx,
        vy_yy,
        vy_zz,
        vz_x,
        vz_y,
        vz_z,
        vz_xx,
        vz_yy,
        vz_zz,
        p_x,
        p_y,
        p_z,
        T_x,
        T_y,
        T_z,
        T_xx,
        T_yy,
        T_zz,
    ) = get_values_and_derivatives(validation_fields, validation_points)

    (
        validation_loss_divergence,
        validation_loss_momentum_x,
        validation_loss_momentum_y,
        validation_loss_momentum_z,
        validation_loss_outlet_boundary,
        validation_loss_other_boundary,
        validation_loss_heat,
        validation_loss_t_wall_boundary,
    ) = get_loss(
        vx,
        vy,
        vz,
        p,
        T,
        vx_x,
        vx_y,
        vx_z,
        vx_xx,
        vx_yy,
        vx_zz,
        vy_x,
        vy_y,
        vy_z,
        vy_xx,
        vy_yy,
        vy_zz,
        vz_x,
        vz_y,
        vz_z,
        vz_xx,
        vz_yy,
        vz_zz,
        p_x,
        p_y,
        p_z,
        T_x,
        T_y,
        T_z,
        T_xx,
        T_yy,
        T_zz,
        validation_labels,
        p_outlet,
    )

    validation_loss_total = (
        validation_loss_divergence
        + validation_loss_momentum_x
        + validation_loss_momentum_y
        + validation_loss_momentum_z
        + validation_loss_outlet_boundary
        + validation_loss_other_boundary
        + validation_loss_heat
        + 20 * validation_loss_t_wall_boundary
    )

    validation_fields = [
        ("vx", validation_fields[:, 0].cpu().detach().numpy()),
        ("vy", validation_fields[:, 1].cpu().detach().numpy()),
        ("vz", validation_fields[:, 2].cpu().detach().numpy()),
        ("p", validation_fields[:, 3].cpu().detach().numpy()),
        ("T", validation_fields[:, 4].cpu().detach().numpy() * 1000),
    ]
    plot_fields(validation_fields, validation_points)

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
    optimizer.zero_grad(True)

    pinn_model.train()
    optimizer.step(closure)

stop_time = time.time()
print(f"Time taken for training is: {stop_time - start_time}")
torch.save(pinn_model.state_dict(), "../run/pinn_model.pt")

## Plotting
pinn_model.eval()
validation_points.requires_grad_(True)
validation_fields = pinn_model(validation_points)
(
    vx,
    vy,
    vz,
    p,
    T,
    vx_x,
    vx_y,
    vx_z,
    vx_xx,
    vx_yy,
    vx_zz,
    vy_x,
    vy_y,
    vy_z,
    vy_xx,
    vy_yy,
    vy_zz,
    vz_x,
    vz_y,
    vz_z,
    vz_xx,
    vz_yy,
    vz_zz,
    p_x,
    p_y,
    p_z,
    T_x,
    T_y,
    T_z,
    T_xx,
    T_yy,
    T_zz,
) = get_values_and_derivatives(validation_fields, validation_points)
(
    validation_loss_divergence,
    validation_loss_momentum_x,
    validation_loss_momentum_y,
    validation_loss_momentum_z,
    validation_loss_outlet_boundary,
    validation_loss_other_boundary,
    validation_loss_heat,
    validation_loss_t_wall_boundary,
) = get_loss(
    vx,
    vy,
    vz,
    p,
    T,
    vx_x,
    vx_y,
    vx_z,
    vx_xx,
    vx_yy,
    vx_zz,
    vy_x,
    vy_y,
    vy_z,
    vy_xx,
    vy_yy,
    vy_zz,
    vz_x,
    vz_y,
    vz_z,
    vz_xx,
    vz_yy,
    vz_zz,
    p_x,
    p_y,
    p_z,
    T_x,
    T_y,
    T_z,
    T_xx,
    T_yy,
    T_zz,
    validation_labels,
    p_outlet,
)

fields = [
    ("vx", validation_fields[:, 0].cpu().detach().numpy()),
    ("vy", validation_fields[:, 1].cpu().detach().numpy()),
    ("vz", validation_fields[:, 2].cpu().detach().numpy()),
    ("p", validation_fields[:, 3].cpu().detach().numpy()),
    ("T", validation_fields[:, 4].cpu().detach().numpy() * 1000),
]
plot_fields(fields, validation_points)

######## Inference

device = "cpu"
torch.set_default_device(device)
pinn_model = PINNs(in_dim=3, hidden_dim=hidden_dim, out_dim=5, num_layer=num_layer).to(
    device
)
pinn_model.load_state_dict(
    torch.load(
        "../run/pinn_model.pt", weights_only=True, map_location=torch.device("cpu")
    )
)

all_points = torch.tensor(
    np.concatenate(
        (
            np.load(os.path.join(data_folder, "vel_x_inlet.npy"))[:, :3],
            np.load(os.path.join(data_folder, "vel_x.npy"))[:, :3],
        )
    )
)

all_fields = pinn_model(all_points)

vx_pred = all_fields[:, 0].cpu().detach().numpy()
vy_pred = all_fields[:, 1].cpu().detach().numpy()
vz_pred = all_fields[:, 2].cpu().detach().numpy()
p_pred = all_fields[:, 3].cpu().detach().numpy()
T_pred = (all_fields[:, 4] * 1000).cpu().detach().numpy()

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
