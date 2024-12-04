import trimesh
import numpy as np
import random
import torch
from sample import *
from hermite_spline import *
from unet import *
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from tqdm import tqdm
import time
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from git import Repo

# Path to the parent directory of the `src/` folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Initialize the repository at the parent directory level
repo = Repo(parent_dir)

seed = 42

# Model Hyperparams
epochs = 100

# Physics Constants
inlet_velocity = 0.5
rho = 1.010427
mu = 2.02e-5

run = wandb.init(
    # set the wandb project where this run will be logged
    project="Spline-PINNs_with_validation",
    # track hyperparameters and run metadata
    config={
        "optimizer": "LBFGS",
        "architecture": "Unet",
        "epochs": epochs,
        "seed": seed,
        "inlet_velocity": inlet_velocity,
    },
)

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Check for Metal (MPS) device
device = 'cuda'
torch.set_default_device(device)
print(f"Using device: {device}")


def get_supoort_points(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    x_floor = (x // step[0]).long()
    y_floor = (y // step[1]).long()
    z_floor = (z // step[2]).long()
    x_supports_indices = torch.vstack(
        (x_floor, torch.clamp(x_floor + 1, max=grid_resolution[0] - 1))
    )
    y_support_indices = torch.vstack(
        (y_floor, torch.clamp(y_floor + 1, max=grid_resolution[1] - 1))
    )
    z_support_indices = torch.vstack(
        (z_floor, torch.clamp(z_floor + 1, max=grid_resolution[2] - 1))
    )
    return x, y, z, x_supports_indices, y_support_indices, z_support_indices


def f(
    spline_coeff,
    channel,
    x,
    y,
    z,
    x_supports,
    y_supports,
    z_supports,
    der_x=0,
    der_y=0,
    der_z=0,
):
    conv_sum = 0
    for type in range(spline_coeff[channel].shape[0]):
        i, j, k = binary_array(type)
        spline_coeff_ijk = spline_coeff[channel][type]
        for x_supports_ind in x_supports:
            for y_support_ind in y_supports:
                for z_support_ind in z_supports:
                    # One of the 8 grid support points(enclosing cube vertices) for each sample point.
                    support_point_ind = torch.vstack(
                        (x_supports_ind, y_support_ind, z_support_ind)
                    ).T

                    x_indices = support_point_ind[:, 0]
                    y_indices = support_point_ind[:, 1]
                    z_indices = support_point_ind[:, 2]

                    x_input = (x - x_indices * step[0]) / step[0]
                    y_input = (y - y_indices * step[1]) / step[1]
                    z_input = (z - z_indices * step[2]) / step[2]

                    conv_sum += (
                        spline_coeff_ijk[x_indices, y_indices, z_indices]
                    ) * hermite_kernel_3d(
                        i, j, k, x_input, y_input, z_input, der_x, der_y, der_z
                    )
    return conv_sum


def sample_points(
    num_volume_points, num_inlet_surface_points, num_other_surface_points, shuffle=True
):
    # Prepare the sample points
    inlet_surface_points, inlet_surface_labels = get_inlet_surface_points(
        obj, num_inlet_surface_points
    )
    other_surface_points, other_surface_labels = get_other_surface_points(
        obj, num_other_surface_points
    )
    volume_points, volume_labels = get_volume_points(obj, num_volume_points)

    # Combine points and labels
    all_points = torch.cat(
        [inlet_surface_points, other_surface_points, volume_points], dim=0
    )
    all_labels = torch.cat(
        [inlet_surface_labels, other_surface_labels, volume_labels], dim=0
    )

    if shuffle:
        permutation = torch.randperm(all_points.size(0))
        return all_points[permutation], all_labels[permutation]
    return all_points, all_labels


# Calculating various field terms using coefficients
def get_fields_and_losses(spline_coeff, points, labels):
    x, y, z, x_supports, y_supports, z_supports = get_supoort_points(points)
    vx = f(spline_coeff, 0, x, y, z, x_supports, y_supports, z_supports, 0, 0, 0)
    vy = f(spline_coeff, 1, x, y, z, x_supports, y_supports, z_supports, 0, 0, 0)
    vz = f(spline_coeff, 2, x, y, z, x_supports, y_supports, z_supports, 0, 0, 0)
    p = f(spline_coeff, 3, x, y, z, x_supports, y_supports, z_supports, 0, 0, 0)

    vx_x = f(spline_coeff, 0, x, y, z, x_supports, y_supports, z_supports, 1, 0, 0)
    vx_y = f(spline_coeff, 0, x, y, z, x_supports, y_supports, z_supports, 0, 1, 0)
    vx_z = f(spline_coeff, 0, x, y, z, x_supports, y_supports, z_supports, 0, 0, 1)

    vy_x = f(spline_coeff, 1, x, y, z, x_supports, y_supports, z_supports, 1, 0, 0)
    vy_y = f(spline_coeff, 1, x, y, z, x_supports, y_supports, z_supports, 0, 1, 0)
    vy_z = f(spline_coeff, 1, x, y, z, x_supports, y_supports, z_supports, 0, 0, 1)

    vz_x = f(spline_coeff, 2, x, y, z, x_supports, y_supports, z_supports, 1, 0, 0)
    vz_y = f(spline_coeff, 2, x, y, z, x_supports, y_supports, z_supports, 0, 1, 0)
    vz_z = f(spline_coeff, 2, x, y, z, x_supports, y_supports, z_supports, 0, 0, 1)

    p_x = f(spline_coeff, 3, x, y, z, x_supports, y_supports, z_supports, 1, 0, 0)
    p_y = f(spline_coeff, 3, x, y, z, x_supports, y_supports, z_supports, 0, 1, 0)
    p_z = f(spline_coeff, 3, x, y, z, x_supports, y_supports, z_supports, 0, 0, 1)

    vx_xx = f(spline_coeff, 0, x, y, z, x_supports, y_supports, z_supports, 2, 0, 0)
    vx_yy = f(spline_coeff, 0, x, y, z, x_supports, y_supports, z_supports, 0, 2, 0)
    vx_zz = f(spline_coeff, 0, x, y, z, x_supports, y_supports, z_supports, 0, 0, 2)

    vy_xx = f(spline_coeff, 1, x, y, z, x_supports, y_supports, z_supports, 2, 0, 0)
    vy_yy = f(spline_coeff, 1, x, y, z, x_supports, y_supports, z_supports, 0, 2, 0)
    vy_zz = f(spline_coeff, 1, x, y, z, x_supports, y_supports, z_supports, 0, 0, 2)

    vz_xx = f(spline_coeff, 2, x, y, z, x_supports, y_supports, z_supports, 2, 0, 0)
    vz_yy = f(spline_coeff, 2, x, y, z, x_supports, y_supports, z_supports, 0, 2, 0)
    vz_zz = f(spline_coeff, 2, x, y, z, x_supports, y_supports, z_supports, 0, 0, 2)

    # calculate losses
    loss_divergence = torch.mean((vx_x + vy_y + vz_z) ** 2)
    loss_momentum_x = torch.mean(
        (
            (vx * vx_x + vy * vx_y + vz * vx_z)
            + (1 / rho) * p_x
            - (mu / rho) * (vx_xx + vx_yy + vx_zz)
        )
        ** 2
    )
    loss_momentum_y = torch.mean(
        (
            (vx * vy_x + vy * vy_y + vz * vy_z)
            + (1 / rho) * p_y
            - (mu / rho) * (vy_xx + vy_yy + vy_zz)
        )
        ** 2
    )
    loss_momentum_z = torch.mean(
        (
            (vx * vz_x + vy * vz_y + vz * vz_z)
            + (1 / rho) * p_z
            - (mu / rho) * (vz_xx + vz_yy + vz_zz)
        )
        ** 2
    )
    loss_inlet_boundary = (
        torch.mean((vx[labels == 1] - inlet_velocity) ** 2)
        + torch.mean((vy[labels == 1]) ** 2)
        + torch.mean((vz[labels == 1]) ** 2)
    )
    loss_other_boundary = (
        torch.mean((vx[labels == 2]) ** 2)
        + torch.mean((vy[labels == 2]) ** 2)
        + torch.mean((vz[labels == 2]) ** 2)
    )

    return (
        vx,
        vy,
        vz,
        p,
        loss_divergence,
        loss_momentum_x,
        loss_momentum_y,
        loss_momentum_z,
        loss_inlet_boundary,
        loss_other_boundary,
    )


##################################################################################################################################

obj = trimesh.load("./Baseline_ML4Science.stl")

grid_resolution = np.array([64, 32, 16])
binary_mask = get_binary_mask(obj, grid_resolution)
step = obj.bounding_box.extents / (grid_resolution - 1)

# Instantiate the neural network
unet_model = UNet3D().to(device)
optimizer = LBFGS(unet_model.parameters(), line_search_fn='strong_wolfe')
unet_model.apply(initialize_weights)

start_time = time.time()
training_loss_track = []
validation_loss_track = []

validation_points, validation_labels = sample_points(10000, 3000, 8000)


for epoch in range(epochs):
    print(f'{epoch+1}/{epochs}')
    def closure():
        train_points, train_labels = sample_points(10000, 3000, 8000)

        # Ensure training points allow gradient computation
        train_points.requires_grad_(True)

        # Get Hermite Spline coefficients from the Unet
        unet_input = prepare_mesh_for_unet(binary_mask).to(device)
        spline_coeff = unet_model(unet_input)[0]

        # Calculating various field terms using coefficients
        (
            vx,
            vy,
            vz,
            p,
            loss_divergence,
            loss_momentum_x,
            loss_momentum_y,
            loss_momentum_z,
            loss_inlet_boundary,
            loss_other_boundary,
        ) = get_fields_and_losses(spline_coeff, train_points, train_labels)

        loss_total = (
            loss_divergence
            + loss_momentum_x
            + loss_momentum_y
            + loss_momentum_z
            + 100 * loss_inlet_boundary
            + 100 * loss_other_boundary
        )

        wandb.log(
            {
                "Divergence Loss": np.log10(loss_divergence.item()),
                "X Momentum Loss": np.log10(loss_momentum_x.item()),
                "Y Momentum Loss": np.log10(loss_momentum_y.item()),
                "Z Momentum Loss": np.log10(loss_momentum_z.item()),
                "Inlet Boundary Loss": np.log10(loss_inlet_boundary.item()),
                "Other Boundary Loss": np.log10(loss_other_boundary.item()),
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
            f"Other Boundary Loss: {loss_other_boundary.item()}, "
            f"Total Loss: {loss_total.item()}"
        )

        # Using LBFGS optimizer
        optimizer.zero_grad()
        loss_total.backward()
        return loss_total

    # Validation
    # Switch model to evaluation mode
    unet_model.eval()
    unet_input = prepare_mesh_for_unet(binary_mask).to(device)
    spline_coeff = unet_model(unet_input)[0]
    with torch.no_grad():
        (
            validation_vx,
            validation_vy,
            validation_vz,
            validation_p,
            validation_loss_divergence,
            validation_loss_momentum_x,
            validation_loss_momentum_y,
            validation_loss_momentum_z,
            validation_loss_inlet_boundary,
            validation_loss_other_boundary,
        ) = get_fields_and_losses(spline_coeff, validation_points, validation_labels)

        validation_loss_total = (
            validation_loss_divergence
            + validation_loss_momentum_x
            + validation_loss_momentum_y
            + validation_loss_momentum_z
            + 100 * validation_loss_inlet_boundary
            + 100 * validation_loss_other_boundary
        )

        wandb.log(
            {
                "Validation Divergence Loss": np.log10(validation_loss_divergence.item()),
                "Validation X Momentum Loss": np.log10(validation_loss_momentum_x.item()),
                "Validation Y Momentum Loss": np.log10(validation_loss_momentum_y.item()),
                "Validation Z Momentum Loss": np.log10(validation_loss_momentum_z.item()),
                "Validation Inlet Boundary Loss": np.log10(
                    validation_loss_inlet_boundary.item()
                ),
                "Validation Other Boundary Loss": np.log10(
                    validation_loss_other_boundary.item()
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
            f"Validation Inlet Boundary Loss: {validation_loss_inlet_boundary.item()}, "
            f"Validation Other Boundary Loss: {validation_loss_other_boundary.item()}, "
            f"Validation Total Loss: {validation_loss_total.item()}"
        )

    unet_model.train()
    optimizer.step(closure)

stop_time = time.time()
print(f"Time taken for training is: {stop_time - start_time}")
torch.save(unet_model.state_dict(), '../run/unet_model.pt')

## Plotting
unet_input = prepare_mesh_for_unet(binary_mask).to(device)
spline_coeff = unet_model(unet_input)[0]
unet_model.eval()
unet_input = prepare_mesh_for_unet(binary_mask).to(device)
spline_coeff = unet_model(unet_input)[0]
with torch.no_grad():
    (
        validation_vx,
        validation_vy,
        validation_vz,
        validation_p,
        validation_loss_divergence,
        validation_loss_momentum_x,
        validation_loss_momentum_y,
        validation_loss_momentum_z,
        validation_loss_inlet_boundary,
        validation_loss_other_boundary,
    ) = get_fields_and_losses(spline_coeff, validation_points, validation_labels)

fields = [
    ("vx", validation_vx),
    ("vy", validation_vy),
    ("vz", validation_vz),
    ("p", validation_p),
]

for field in fields:
    # Convert to numpy for plotting
    points = validation_points.cpu().detach().numpy()
    scalar_field = field[1].cpu().detach().numpy()

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], c=scalar_field, cmap="viridis", s=10
    )

    # Add color bar and labels
    plt.colorbar(sc, label=f"{field[0]}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title(f"{field[0]}")
    plt.savefig(f"../run/{field[0]}.png", dpi=300, bbox_inches="tight")

time.sleep(60)
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