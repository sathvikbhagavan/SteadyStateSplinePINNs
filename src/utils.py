import trimesh
import numpy as np
import torch
import os
import plotly.graph_objects as go


def get_binary_mask(model, grid_resolution):
    bounds = model.bounds
    min_bound, max_bound = bounds[0], bounds[1]
    x = np.linspace(min_bound[0], max_bound[0], grid_resolution[0])
    y = np.linspace(min_bound[1], max_bound[1], grid_resolution[1])
    z = np.linspace(min_bound[2], max_bound[2], grid_resolution[2])
    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    inside_points = grid_points[model.contains(grid_points)]
    binary_volume = np.zeros(grid_resolution, dtype=np.uint8)
    for point in inside_points:
        # Compute the index of the point in the grid
        index_x = np.searchsorted(x, point[0])
        index_y = np.searchsorted(y, point[1])
        index_z = np.searchsorted(z, point[2])
        # Set the corresponding position in the binary volume to 1
        if (
            0 <= index_x < grid_resolution[0]
            and 0 <= index_y < grid_resolution[1]
            and 0 <= index_z < grid_resolution[2]
        ):
            binary_volume[index_x, index_y, index_z] = 1
    return binary_volume


def get_support_points(points, step, grid_resolution):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    x_floor = (x // step[0]).long()
    y_floor = (y // step[1]).long()
    z_floor = (z // step[2]).long()
    x_support_indices = torch.vstack(
        (x_floor, torch.clamp(x_floor + 1, max=grid_resolution[0] - 1))
    )
    y_support_indices = torch.vstack(
        (y_floor, torch.clamp(y_floor + 1, max=grid_resolution[1] - 1))
    )
    z_support_indices = torch.vstack(
        (z_floor, torch.clamp(z_floor + 1, max=grid_resolution[2] - 1))
    )
    x_support_indices[0, x_support_indices[0] == x_support_indices[1]] -= 1
    y_support_indices[0, y_support_indices[0] == y_support_indices[1]] -= 1
    z_support_indices[0, z_support_indices[0] == z_support_indices[1]] -= 1
    return x, y, z, x_support_indices, y_support_indices, z_support_indices


def sample_points(
    obj,
    num_volume_points,
    num_outlet_surface_points,
    num_other_surface_points,
    shuffle=True,
):
    # Prepare the sample points
    # inlet_surface_points, inlet_surface_labels = get_inlet_surface_points(
    #     obj, num_inlet_surface_points
    # )
    outlet_surface_points, outlet_surface_labels = get_outlet_surface_points(
        obj, num_outlet_surface_points
    )
    other_surface_points, other_surface_labels = get_other_surface_points(
        obj, num_other_surface_points
    )
    volume_points, volume_labels = get_volume_points(obj, num_volume_points)
    # Combine points and labels
    all_points = torch.cat(
        [
            # inlet_surface_points,
            outlet_surface_points,
            other_surface_points,
            volume_points,
        ],
        dim=0,
    )
    all_labels = torch.cat(
        [
            # inlet_surface_labels,
            outlet_surface_labels,
            other_surface_labels,
            volume_labels,
        ],
        dim=0,
    )
    print(
        f"Number of Outlet surface points: {outlet_surface_points.size()[0]}, volume points: {volume_points.size()[0]}, surface points: {other_surface_points.size()[0]}"
    )
    if shuffle:
        permutation = torch.randperm(all_points.size(0))
        return all_points[permutation], all_labels[permutation]
    return all_points, all_labels


def get_inlet_surface_points(obj, num_points):
    threshold = 1e-5
    faces_x_zero = [
        i
        for i, face in enumerate(obj.faces)
        if np.all(
            np.abs(obj.vertices[face, 0]) < threshold
        )  # Check if all vertices' x-coordinates are 0
    ]
    subset_mesh = obj.submesh([faces_x_zero], only_watertight=False)[0]
    points, _ = trimesh.sample.sample_surface(subset_mesh, count=num_points)
    inlet_surface_points = torch.tensor(points, dtype=torch.float64)
    inlet_surface_labels = torch.ones(inlet_surface_points.size(0), dtype=torch.int64)
    return inlet_surface_points, inlet_surface_labels


def get_outlet_surface_points(obj, num_points):
    threshold = 1e-5
    x_max = obj.vertices[:, 0].max()
    faces_x_max = [
        i
        for i, face in enumerate(obj.faces)
        if np.all(
            np.abs(obj.vertices[face, 0] - x_max) < threshold
        )  # Check if all vertices' x-coordinates are x_max
    ]
    subset_mesh = obj.submesh([faces_x_max], only_watertight=False)[0]
    points, _ = trimesh.sample.sample_surface(subset_mesh, count=num_points)
    outlet_surface_points = torch.tensor(points, dtype=torch.float64)
    outlet_surface_labels = (
        torch.ones(outlet_surface_points.size(0), dtype=torch.int64) * 3
    )
    return outlet_surface_points, outlet_surface_labels


def get_other_surface_points(obj, num_points):
    threshold = 1e-5
    x_max = obj.vertices[:, 0].max()
    points, _ = trimesh.sample.sample_surface(obj, count=num_points)
    filtered_points = points[
        (np.abs(points[:, 0]) > threshold)
        & ~((x_max - threshold <= points[:, 0]) & (points[:, 0] <= x_max + threshold))
    ]
    other_surface_points = torch.tensor(filtered_points, dtype=torch.float64)
    other_surface_labels = 2 * torch.ones(
        other_surface_points.size(0), dtype=torch.int64
    )
    return other_surface_points, other_surface_labels


def get_volume_points(obj, num_points):
    volume_points = torch.tensor(
        trimesh.sample.volume_mesh(obj, num_points), dtype=torch.float64
    )
    volume_labels = torch.zeros(volume_points.size(0), dtype=torch.int64)
    return volume_points, volume_labels


def dynamic_viscosity(T, mu_ref=1.716e-5, T_ref=273.15, S=110.4):
    """
    Calculate dynamic viscosity using Sutherland's law with robust error handling.

    Args:
    T: Temperature field (tensor)
    mu_ref: Reference viscosity (default: 1.716e-5 Pa.s)
    T_ref: Reference temperature (default: 273.15 K)
    S: Sutherland constant (default: 110.4 K)

    Returns:
    Tensor of dynamic viscosity values
    """
    try:
        # Ensure T is a floating point tensor
        # T = T.float()

        # Clamp temperature to prevent extreme values
        # T_clamped = torch.clamp(T, min=100, max=2000)

        # Compute viscosity with safe computation
        mu = (
            mu_ref
            * ((torch.abs(T) / T_ref) ** 1.5)
            * ((T_ref + S) / (torch.abs(T) + S))
        )

        # Replace any remaining NaNs or infs with reference viscosity
        # mu = torch.nan_to_num(mu, nan=mu_ref, posinf=mu_ref, neginf=mu_ref)

        return mu

    except Exception as e:
        print(f"Viscosity calculation error: {e}")
        print(f"Problematic temperature tensor: {T}")
        return torch.full_like(T, mu_ref, dtype=T.dtype)


def plot_fields(fields, validation_points, train=False):
    for field in fields:
        # Convert to numpy for plotting
        points = validation_points
        scalar_field = field[1]

        # Create a 3D scatter plot using Plotly
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=scalar_field,  # Color by scalar field
                        colorscale="Viridis",  # Color map
                        colorbar=dict(title=f"{field[0]}"),  # Colorbar with title
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
            title=f"{field[0]}",
        )

        # Save as HTML (interactive)
        if train:
            fig.write_html(f"../run/{field[0]}_train.html")
        else:
            fig.write_html(f"../run/{field[0]}.html")


def plot_aginast_data(folder_path, vx_pred, vy_pred, vz_pred, p_pred, T_pred):
    vx = np.load(os.path.join(folder_path, "vel_x.npy"))
    vx_inlet = np.load(os.path.join(folder_path, "vel_x_inlet.npy"))
    vy = np.load(os.path.join(folder_path, "vel_y.npy"))
    vy_inlet = np.load(os.path.join(folder_path, "vel_y_inlet.npy"))
    vz = np.load(os.path.join(folder_path, "vel_z.npy"))
    vz_inlet = np.load(os.path.join(folder_path, "vel_z_inlet.npy"))
    p = np.load(os.path.join(folder_path, "press.npy"))
    T = np.load(os.path.join(folder_path, "temp.npy"))
    all_points = np.concatenate((vx_inlet[:, :3], vx[:, :3]))
    _fields = [
        [vx_inlet, vx, vx_pred, "vx"],
        [vy_inlet, vy, vy_pred, "vy"],
        [vz_inlet, vz, vz_pred, "vz"],
    ]
    all_indices = torch.arange(vx.shape[0])
    supervised_indices = torch.tensor(
        np.load(os.path.join("indices.npy")), dtype=torch.long
    )
    test_indices = torch.tensor(
        list(set(all_indices.tolist()) - set(supervised_indices.tolist()))
    )
    for _field in _fields:
        scalar_field = np.concatenate((_field[0][:, 3], _field[1][:, 3]))
        index = np.random.choice(all_points.shape[0], 100000, replace=False)
        selected_points = all_points[index]
        selected_scalar_field = scalar_field[index]
        selected_scalar_pred = _field[2][index]
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
                        colorbar=dict(title=f"{_field[3]}"),  # Colorbar with title
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
            title=f"Diff in true and pred for {_field[3]}",
        )
        fig.write_html(f"../run/{_field[3]}_diff.html")
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=selected_points[:, 0],
                    y=selected_points[:, 1],
                    z=selected_points[:, 2],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=selected_scalar_field,
                        colorscale="Viridis",  # Color map
                        colorbar=dict(title=f"{_field[3]}"),  # Colorbar with title
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
            title=f"Data for {_field[3]}",
        )
        fig.write_html(f"../run/{_field[3]}_data.html")
        supervised_rms_error_velocity = np.sqrt(
            np.mean(
                (_field[1][:, 3][supervised_indices] - _field[2][supervised_indices])
                ** 2
            )
        )
        test_rms_error_velocity = np.sqrt(
            np.mean((_field[1][:, 3][test_indices] - _field[2][test_indices]) ** 2)
        )
        print(
            f"{_field[3]} - Train RMSE: {supervised_rms_error_velocity}, Test RMSE: {test_rms_error_velocity}"
        )

    scalar_field = p[:, 3]
    index = np.random.choice(vx[:, :3].shape[0], 50000, replace=False)
    selected_points = vx[:, :3][index]
    selected_scalar_field = scalar_field[index]
    selected_scalar_pred = p_pred[vx_inlet.shape[0] :][index]

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
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=selected_points[:, 0],
                y=selected_points[:, 1],
                z=selected_points[:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=selected_scalar_field,
                    colorscale="Viridis",  # Color map
                    colorbar=dict(title="p"),  # Colorbar with title
                ),
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",  # Ensures aspect ratio is based on data's extents
        ),
        title=f"Data for p",
    )
    fig.write_html(f"../run/p_data.html")
    supervised_rms_error_pressure = np.sqrt(
        np.mean(
            (
                scalar_field[supervised_indices]
                - p_pred[vx_inlet.shape[0] :][supervised_indices]
            )
            ** 2
        )
    )
    test_rms_error_pressure = np.sqrt(
        np.mean(
            (scalar_field[test_indices] - p_pred[vx_inlet.shape[0] :][test_indices])
            ** 2
        )
    )
    print(
        f"Pressure - Train RMSE: {supervised_rms_error_pressure}, Test RMSE: {test_rms_error_pressure}"
    )

    # Temperature plotting
    scalar_field = T[:, 3]
    index = np.random.choice(vx[:, :3].shape[0], 50000, replace=False)
    selected_points = vx[:, :3][index]
    selected_scalar_field = scalar_field[index]
    selected_scalar_pred = T_pred[vx_inlet.shape[0] :][index]

    # Temperature difference plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=selected_points[:, 0],
                y=selected_points[:, 1],
                z=selected_points[:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=selected_scalar_field - selected_scalar_pred,
                    colorscale="Viridis",
                    colorbar=dict(title="T"),
                ),
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        title="Diff in true and pred for temperature",
    )
    fig.write_html(f"../run/T_diff.html")
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=selected_points[:, 0],
                y=selected_points[:, 1],
                z=selected_points[:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=selected_scalar_field,
                    colorscale="Viridis",  # Color map
                    colorbar=dict(title="T"),  # Colorbar with title
                ),
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",  # Ensures aspect ratio is based on data's extents
        ),
        title=f"Data for T",
    )
    fig.write_html(f"../run/T_data.html")
    supervised_rms_error_temp = np.sqrt(
        np.mean(
            (
                scalar_field[supervised_indices]
                - T_pred[vx_inlet.shape[0] :][supervised_indices]
            )
            ** 2
        )
    )
    test_rms_error_temp = np.sqrt(
        np.mean(
            (scalar_field[test_indices] - T_pred[vx_inlet.shape[0] :][test_indices])
            ** 2
        )
    )
    print(
        f"Temperature - Train RMSE: {supervised_rms_error_temp}, Test RMSE: {test_rms_error_temp}"
    )
    return
