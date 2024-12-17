import numpy as np
import torch
import torch.nn as nn
import random
from torch.optim import Adam, LBFGS
import trimesh
import time
import wandb
from constants import *
from utils import *


class PINNs(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        super(PINNs, self).__init__()

        layers = []
        for i in range(num_layer - 1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())
            else:
                layers.append(
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
                )
                layers.append(nn.Tanh())

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

        self.linear = nn.Sequential(*layers)

    def forward(self, train_points):
        # src = torch.cat((x,y,z), dim=-1)
        return self.linear(train_points)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def get_values_and_derivatives(fields, points):
    vx = fields[:, 0:1]
    vy = fields[:, 1:2]
    vz = fields[:, 2:3]
    p = fields[:, 3:4]
    T = fields[:, 4:5] * 1000

    # Compute all gradients of vx with respect to points in one pass
    grad_vx = torch.autograd.grad(
        outputs=vx,
        inputs=points,
        grad_outputs=torch.ones_like(vx),
        retain_graph=True,
        create_graph=True,
    )[0]

    # Extract first derivatives
    vx_x = grad_vx[:, 0:1]  # Gradient w.r.t x
    vx_y = grad_vx[:, 1:2]  # Gradient w.r.t y
    vx_z = grad_vx[:, 2:3]  # Gradient w.r.t z

    # Second derivatives
    second_grad_vx = torch.autograd.grad(
        outputs=grad_vx,
        inputs=points,
        grad_outputs=torch.ones_like(grad_vx),
        retain_graph=True,
        create_graph=True,
    )[0]

    # Extract second derivatives
    vx_xx = second_grad_vx[:, 0:1]  # Second derivative w.r.t x
    vx_yy = second_grad_vx[:, 1:2]  # Second derivative w.r.t y
    vx_zz = second_grad_vx[:, 2:3]  # Second derivative w.r.t z

    # Compute all gradients of vy with respect to points in one pass
    grad_vy = torch.autograd.grad(
        outputs=vy,
        inputs=points,
        grad_outputs=torch.ones_like(vy),
        retain_graph=True,
        create_graph=True,
    )[0]

    # Extract components
    vy_x = grad_vy[:, 0:1]  # Gradient w.r.t x
    vy_y = grad_vy[:, 1:2]  # Gradient w.r.t y
    vy_z = grad_vy[:, 2:3]  # Gradient w.r.t z

    # Second derivatives
    second_grad_vy = torch.autograd.grad(
        outputs=grad_vy,
        inputs=points,
        grad_outputs=torch.ones_like(grad_vy),
        retain_graph=True,
        create_graph=True,
    )[0]

    # Extract second derivatives
    vy_xx = second_grad_vy[:, 0:1]  # Second derivative w.r.t x
    vy_yy = second_grad_vy[:, 1:2]  # Second derivative w.r.t y
    vy_zz = second_grad_vy[:, 2:3]  # Second derivative w.r.t z

    # Compute all gradients of vz with respect to points in one pass
    grad_vz = torch.autograd.grad(
        outputs=vz,
        inputs=points,
        grad_outputs=torch.ones_like(vz),
        retain_graph=True,
        create_graph=True,
    )[0]

    # Extract components
    vz_x = grad_vz[:, 0:1]  # Gradient w.r.t x
    vz_y = grad_vz[:, 1:2]  # Gradient w.r.t y
    vz_z = grad_vz[:, 2:3]  # Gradient w.r.t z

    # Second derivatives
    second_grad_vz = torch.autograd.grad(
        outputs=grad_vz,
        inputs=points,
        grad_outputs=torch.ones_like(grad_vz),
        retain_graph=True,
        create_graph=True,
    )[0]

    # Extract second derivatives
    vz_xx = second_grad_vz[:, 0:1]  # Second derivative w.r.t x
    vz_yy = second_grad_vz[:, 1:2]  # Second derivative w.r.t y
    vz_zz = second_grad_vz[:, 2:3]  # Second derivative w.r.t z

    # Compute all gradients of p with respect to points in one pass
    grad_p = torch.autograd.grad(
        outputs=p,
        inputs=points,
        grad_outputs=torch.ones_like(p),
        retain_graph=True,
        create_graph=True,
    )[0]

    # Extract first derivatives
    p_x = grad_p[:, 0:1]  # Gradient w.r.t x
    p_y = grad_p[:, 1:2]  # Gradient w.r.t y
    p_z = grad_p[:, 2:3]  # Gradient w.r.t z

    # Compute all gradients of T with respect to points in one pass
    grad_T = torch.autograd.grad(
        outputs=T,
        inputs=points,
        grad_outputs=torch.ones_like(T),
        retain_graph=True,
        create_graph=True,
    )[0]

    # Extract first derivatives
    T_x = grad_T[:, 0:1] * 1000
    T_y = grad_T[:, 1:2] * 1000
    T_z = grad_T[:, 2:3] * 1000

    # Second derivatives
    second_grad_T = torch.autograd.grad(
        outputs=grad_T,
        inputs=points,
        grad_outputs=torch.ones_like(grad_T),
        retain_graph=True,
        create_graph=True,
    )[0]

    T_xx = second_grad_T[:, 0:1] * 1000
    T_yy = second_grad_T[:, 1:2] * 1000
    T_zz = second_grad_T[:, 2:3] * 1000

    return (
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
    )


def get_loss(
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
    labels,
    p_outlet,
):
    loss_divergence = torch.mean((vx_x + vy_y + vz_z) ** 2)

    mu = dynamic_viscosity(T)

    loss_momentum_x = torch.mean(
        (
            vx[labels == 0] * vx_x[labels == 0]
            + vy[labels == 0] * vx_y[labels == 0]
            + vz[labels == 0] * vx_z[labels == 0]
            + (1 / rho) * p_x[labels == 0]
            - (mu[labels == 0] / rho)
            * (vx_xx[labels == 0] + vx_yy[labels == 0] + vx_zz[labels == 0])
        )
        ** 2
    )
    loss_momentum_y = torch.mean(
        (
            vx[labels == 0] * vy_x[labels == 0]
            + vy[labels == 0] * vy_y[labels == 0]
            + vz[labels == 0] * vy_z[labels == 0]
            + (1 / rho) * p_y[labels == 0]
            - (mu[labels == 0] / rho)
            * (vy_xx[labels == 0] + vy_yy[labels == 0] + vy_zz[labels == 0])
        )
        ** 2
    )
    loss_momentum_z = torch.mean(
        (
            vx[labels == 0] * vz_x[labels == 0]
            + vy[labels == 0] * vz_y[labels == 0]
            + vz[labels == 0] * vz_z[labels == 0]
            + (1 / rho) * p_z[labels == 0]
            - (mu[labels == 0] / rho)
            * (vz_xx[labels == 0] + vz_yy[labels == 0] + vz_zz[labels == 0])
        )
        ** 2
    )
    # inlet_boundary_loss = (
    #     torch.mean((vx[labels == 1] - inlet_velocity) ** 2)
    #     + torch.mean((vy[labels == 1]) ** 2)
    #     + torch.mean((vz[labels == 1]) ** 2)
    # )
    loss_other_boundary = (
        torch.mean((vx[labels == 2]) ** 2)
        + torch.mean((vy[labels == 2]) ** 2)
        + torch.mean((vz[labels == 2]) ** 2)
    )

    alpha = thermal_conductivity / (density * specific_heat)
    # Steady-state loss function (no time derivative)
    loss_heat = (
        torch.mean(
            (
                alpha
                * (
                    T_xx[labels == 0] + T_yy[labels == 0] + T_zz[labels == 0]
                )  # Diffusion term (nabla^2 T)
                + vx[labels == 0] * T_x[labels == 0]
                + vy[labels == 0] * T_y[labels == 0]
                + vz[labels == 0] * T_z[labels == 0]  # Advection term (v · nabla T)
            )
            ** 2
        )
        / 10**6
    )

    loss_t_wall_boundary = torch.mean((T[labels == 2] - T_wall) ** 2) / 10**6

    loss_outlet_boundary = torch.mean((p[labels == 3] - p_outlet) ** 2)
    return (
        loss_divergence,
        loss_momentum_x,
        loss_momentum_y,
        loss_momentum_z,
        # loss_inlet_boundary,
        loss_outlet_boundary,
        loss_other_boundary,
        loss_heat,
        loss_t_wall_boundary,
    )
