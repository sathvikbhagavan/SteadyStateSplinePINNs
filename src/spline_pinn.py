import torch
from sample import *
from hermite_spline import *
from unet import *
import plotly.graph_objects as go

def dynamic_viscosity(T, mu_ref=1.716e-5, T_ref=273.15, S=110.4):
    """
    Calculate dynamic viscosity using Sutherland's law.
    
    Args:
        T: Temperature field (tensor).
        mu_ref: Reference viscosity (default: 1.716e-5 Pa.s).
        T_ref: Reference temperature (default: 273.15 K).
        S: Sutherland constant (default: 110.4 K).
    
    Returns:
        Tensor of dynamic viscosity values.
    """
    return mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)

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


def f(
    step,
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
        for x_support_ind in x_supports:
            for y_support_ind in y_supports:
                for z_support_ind in z_supports:
                    # One of the 8 grid support points(enclosing cube vertices) for each sample point.
                    support_point_ind = torch.vstack(
                        (x_support_ind, y_support_ind, z_support_ind)
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


def get_fields(spline_coeff, points, step, grid_resolution):
    x, y, z, x_supports, y_supports, z_supports = get_support_points(
        points, step, grid_resolution
    )
    vx = f(step, spline_coeff, 0, x, y, z, x_supports, y_supports, z_supports, 0, 0, 0)
    vy = f(step, spline_coeff, 1, x, y, z, x_supports, y_supports, z_supports, 0, 0, 0)
    vz = f(step, spline_coeff, 2, x, y, z, x_supports, y_supports, z_supports, 0, 0, 0)
    p = f(step, spline_coeff, 3, x, y, z, x_supports, y_supports, z_supports, 0, 0, 0)
    T = f(step, spline_coeff, 4, x, y, z, x_supports, y_supports, z_supports, 0, 0, 0)

    return vx, vy, vz, p, T


# Calculating various field terms using coefficients
def get_fields_and_losses(
    spline_coeff, points, labels, step, grid_resolution, T, rho, p_outlet, thermal_conductivity, density, specific_heat, T_wall
):
    x, y, z, x_supports, y_supports, z_supports = get_support_points(
        points, step, grid_resolution
    )
    vx = f(step, spline_coeff, 0, x, y, z, x_supports, y_supports, z_supports, 0, 0, 0)
    vy = f(step, spline_coeff, 1, x, y, z, x_supports, y_supports, z_supports, 0, 0, 0)
    vz = f(step, spline_coeff, 2, x, y, z, x_supports, y_supports, z_supports, 0, 0, 0)
    p = f(step, spline_coeff, 3, x, y, z, x_supports, y_supports, z_supports, 0, 0, 0)
    T = f(step, spline_coeff, 4, x, y, z, x_supports, y_supports, z_supports, 0, 0, 0)
    vx_x = f(
        step, spline_coeff, 0, x, y, z, x_supports, y_supports, z_supports, 1, 0, 0
    )
    vx_y = f(
        step, spline_coeff, 0, x, y, z, x_supports, y_supports, z_supports, 0, 1, 0
    )
    vx_z = f(
        step, spline_coeff, 0, x, y, z, x_supports, y_supports, z_supports, 0, 0, 1
    )
    vy_x = f(
        step, spline_coeff, 1, x, y, z, x_supports, y_supports, z_supports, 1, 0, 0
    )
    vy_y = f(
        step, spline_coeff, 1, x, y, z, x_supports, y_supports, z_supports, 0, 1, 0
    )
    vy_z = f(
        step, spline_coeff, 1, x, y, z, x_supports, y_supports, z_supports, 0, 0, 1
    )
    vz_x = f(
        step, spline_coeff, 2, x, y, z, x_supports, y_supports, z_supports, 1, 0, 0
    )
    vz_y = f(
        step, spline_coeff, 2, x, y, z, x_supports, y_supports, z_supports, 0, 1, 0
    )
    vz_z = f(
        step, spline_coeff, 2, x, y, z, x_supports, y_supports, z_supports, 0, 0, 1
    )
    p_x = f(step, spline_coeff, 3, x, y, z, x_supports, y_supports, z_supports, 1, 0, 0)
    p_y = f(step, spline_coeff, 3, x, y, z, x_supports, y_supports, z_supports, 0, 1, 0)
    p_z = f(step, spline_coeff, 3, x, y, z, x_supports, y_supports, z_supports, 0, 0, 1)
    vx_xx = f(
        step, spline_coeff, 0, x, y, z, x_supports, y_supports, z_supports, 2, 0, 0
    )
    vx_yy = f(
        step, spline_coeff, 0, x, y, z, x_supports, y_supports, z_supports, 0, 2, 0
    )
    vx_zz = f(
        step, spline_coeff, 0, x, y, z, x_supports, y_supports, z_supports, 0, 0, 2
    )
    vy_xx = f(
        step, spline_coeff, 1, x, y, z, x_supports, y_supports, z_supports, 2, 0, 0
    )
    vy_yy = f(
        step, spline_coeff, 1, x, y, z, x_supports, y_supports, z_supports, 0, 2, 0
    )
    vy_zz = f(
        step, spline_coeff, 1, x, y, z, x_supports, y_supports, z_supports, 0, 0, 2
    )
    vz_xx = f(
        step, spline_coeff, 2, x, y, z, x_supports, y_supports, z_supports, 2, 0, 0
    )
    vz_yy = f(
        step, spline_coeff, 2, x, y, z, x_supports, y_supports, z_supports, 0, 2, 0
    )
    vz_zz = f(
        step, spline_coeff, 2, x, y, z, x_supports, y_supports, z_supports, 0, 0, 2
    )
    T_x =  f(step, spline_coeff, 4, x, y, z, x_supports, y_supports, z_supports, 1, 0, 0)
    T_y = f(step, spline_coeff, 4, x, y, z, x_supports, y_supports, z_supports, 0, 1, 0)
    T_z = f(step, spline_coeff, 4, x, y, z, x_supports, y_supports, z_supports, 0, 0, 1)
    T_xx = f(step, spline_coeff, 4, x, y, z, x_supports, y_supports, z_supports, 2, 0, 0)
    T_yy = f(step, spline_coeff, 4, x, y, z, x_supports, y_supports, z_supports, 0, 2, 0)
    T_zz = f(step, spline_coeff, 4, x, y, z, x_supports, y_supports, z_supports, 0, 0, 2)

    # calculate losses
    # Divergence-free condition
    loss_divergence = torch.mean(
        (vx_x[labels == 0] + vy_y[labels == 0] + vz_z[labels == 0]) ** 2
    )

    # Momentum equations (including temperature-dependent viscosity)
    mu = dynamic_viscosity(T)

    loss_momentum_x = torch.mean(
        (
            (
                vx[labels == 0] * vx_x[labels == 0]
                + vy[labels == 0] * vx_y[labels == 0]
                + vz[labels == 0] * vx_z[labels == 0]
            )
            + (1 / rho) * p_x[labels == 0]
            - (mu[labels == 0] / rho)
            * (vx_xx[labels == 0] + vx_yy[labels == 0] + vx_zz[labels == 0])
        )
        ** 2
    )
    loss_momentum_y = torch.mean(
        (
            (
                vx[labels == 0] * vy_x[labels == 0]
                + vy[labels == 0] * vy_y[labels == 0]
                + vz[labels == 0] * vy_z[labels == 0]
            )
            + (1 / rho) * p_y[labels == 0]
            - (mu[labels == 0] / rho)
            * (vy_xx[labels == 0] + vy_yy[labels == 0] + vy_zz[labels == 0])
        )
        ** 2
    )
    loss_momentum_z = torch.mean(
        (
            (
                vx[labels == 0] * vz_x[labels == 0]
                + vy[labels == 0] * vz_y[labels == 0]
                + vz[labels == 0] * vz_z[labels == 0]
            )
            + (1 / rho) * p_z[labels == 0]
            - (mu[labels == 0] / rho)
            * (vz_xx[labels == 0] + vz_yy[labels == 0] + vz_zz[labels == 0])
        )
        ** 2
    )

    # Calculate thermal diffusivity using specific heat at constant pressure
    alpha = thermal_conductivity / (density * specific_heat)

    # Steady-state loss function (no time derivative)
    loss_heat = torch.mean(
        (alpha * (T_xx[labels == 0] + T_yy[labels == 0] + T_zz[labels == 0])  # Diffusion term (nabla^2 T)
        + vx[labels == 0] * T_x[labels == 0] + vy[labels == 0] * T_y[labels == 0] + vz[labels == 0] * T_z[labels == 0]  # Advection term (v · nabla T)
        ) ** 2
    )

    # loss_inlet_boundary = (
    #     torch.mean((vx[labels == 1] - inlet_vx) ** 2)
    #     + torch.mean((vy[labels == 1] - inlet_vy) ** 2)
    #     + torch.mean((vz[labels == 1] - inlet_vz) ** 2)
    # )

    loss_outlet_boundary = torch.mean((p[labels == 3] - p_outlet) ** 2)
    loss_other_boundary = (
    torch.mean((vx[labels == 2]) ** 2)
    + torch.mean((vy[labels == 2]) ** 2)
    + torch.mean((vz[labels == 2]) ** 2)
)

    loss_t_wall_boundary = torch.mean((T[labels == 2] - T_wall) ** 2)


    return (
        vx,
        vy,
        vz,
        p,
        loss_divergence,
        loss_momentum_x,
        loss_momentum_y,
        loss_momentum_z,
        # loss_inlet_boundary,
        loss_outlet_boundary,
        loss_other_boundary,
        loss_heat,
        loss_t_wall_boundary
    )


def plot_fields(fields, validation_points, train=False):
    for field in fields:
        # Convert to numpy for plotting
        points = validation_points.cpu().detach().numpy()
        scalar_field = field[1].cpu().detach().numpy()

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
