import torch
from unet import *
from sample import *
import trimesh
import os
import numpy as np
from spline_pinn import *

device = "cpu"
obj = trimesh.load("./Baseline_ML4Science.stl")
grid_resolution = np.array([512, 64, 16])
binary_mask = get_binary_mask(obj, grid_resolution)
step = obj.bounding_box.extents / (grid_resolution - 1)

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
            np.load(os.path.join("./dp0", "vel_x_inlet.npy"))[:, :3],
            np.load(os.path.join("./dp0", "vel_x.npy"))[:, :3],
        )
    )
    * 1000
)

x, y, z, x_supports, y_supports, z_supports = get_support_points(
    all_points, step, grid_resolution
)
vx_pred, vy_pred, vz_pred, p_pred = get_fields(
    spline_coeff, all_points, step, grid_resolution
)
vx_pred = vx_pred.detach().numpy()
vy_pred = vy_pred.detach().numpy()
vz_pred = vz_pred.detach().numpy()
p_pred = p_pred.detach().numpy()


def plot_data(folder_path, vx_pred, vy_pred, vz_pred, p_pred):
    vx = np.load(os.path.join(folder_path, "vel_x.npy"))
    vx_inlet = np.load(os.path.join(folder_path, "vel_x_inlet.npy"))
    vy = np.load(os.path.join(folder_path, "vel_y.npy"))
    vy_inlet = np.load(os.path.join(folder_path, "vel_y_inlet.npy"))
    vz = np.load(os.path.join(folder_path, "vel_z.npy"))
    vz_inlet = np.load(os.path.join(folder_path, "vel_z_inlet.npy"))
    p = np.load(os.path.join(folder_path, "press.npy"))
    all_points = np.concatenate((vx_inlet[:, :3], vx[:, :3]))
    fields = [
        [vx_inlet, vx, vx_pred, "vx"],
        [vy_inlet, vy, vy_pred, "vy"],
        [vz_inlet, vz, vz_pred, "vz"],
    ]
    for field in fields:
        scalar_field = np.concatenate((field[0][:, 3], field[1][:, 3]))
        index = np.random.choice(all_points.shape[0], 100000, replace=False)
        selected_points = all_points[index]
        selected_scalar_field = scalar_field[index]
        selected_scalar_pred = field[2][index]
        # Create a 3D scatter plot using Plotly
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=selected_points[:, 0],
                    y=selected_points[:, 1],
                    z=selected_points[:, 2],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=selected_scalar_field
                        - selected_scalar_pred,  # Color by scalar field
                        colorscale="Viridis",  # Color map
                        colorbar=dict(title=f"{field[3]}"),  # Colorbar with title
                    ),
                )
            ]
        )
        # Update layout with aspectmode='data' to preserve aspect ratio
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",  # Ensures aspect ratio is based on data's extents
            ),
            title=f"Diff in true and pred for {field[3]}",
        )
        fig.write_html(f"../run/{field[3]}_diff.html")
        print(
            f"The mean of {field[3]} differences is: {np.mean(scalar_field - field[2])}"
        )

    scalar_field = p[:, 3]
    index = np.random.choice(vx[:, :3].shape[0], 50000, replace=False)
    selected_points = vx[:, :3][index]
    selected_scalar_field = scalar_field[index]
    selected_scalar_pred = p_pred[vx_inlet.shape[0] :][index] * 10**5

    # Create a 3D scatter plot using Plotly
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=selected_points[:, 0],
                y=selected_points[:, 1],
                z=selected_points[:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=selected_scalar_field
                    - selected_scalar_pred,  # Color by scalar field
                    colorscale="Viridis",  # Color map
                    colorbar=dict(title="p"),  # Colorbar with title
                ),
            )
        ]
    )

    # Update layout with aspectmode='data' to preserve aspect ratio
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",  # Ensures aspect ratio is based on data's extents
        ),
        title="Diff in true and pred for pressure",
    )
    fig.write_html(f"../run/p_diff.html")
    print(
        f"The mean of pressure differences is: {np.mean(scalar_field - p_pred[vx_inlet.shape[0]:]*10**5)}"
    )
    return


plot_data("./dp0", vx_pred, vy_pred, vz_pred, p_pred)
