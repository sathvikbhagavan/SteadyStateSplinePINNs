# Keeping the older version

import trimesh
import numpy as np
import torch
from sample import *
from hermite_spline import *
from unet import *

def f(channel,x,y,z):
        x_supports = [np.floor(x).dtype(int), np.ceil(x).dtype(int)]
        y_supports = [np.floor(y).dtype(int), np.ceil(y).dtype(int)]
        z_supports = [np.floor(z).dtype(int), np.ceil(z).dtype(int)]

        conv_sum = 0

        for type in range(spline_coeff[channel].shape[0]):
            i, j, k = binary_array(type)
            spline_coeff_ijk = spline_coeff[channel][type]
            for x_support in x_supports:
                for y_support in y_supports:
                    for z_support in z_supports:
                        conv_sum += (binary_mask[i][j][k] * spline_coeff_ijk[x_support][y_support][z_support].item()) * h(i,j,k,x-x_support,y-y_support,z-z_support)

        return conv_sum

def df(channel,x,y,z):
    x_supports = [np.floor(x).dtype(int), np.ceil(x).dtype(int)]
    y_supports = [np.floor(y).dtype(int), np.ceil(y).dtype(int)]
    z_supports = [np.floor(z).dtype(int), np.ceil(z).dtype(int)]
    
    conv_sum = torch.zeros((3,x.shape[0]))

    for type in range(spline_coeff[channel].shape[0]):
        i, j, k = binary_array(type)
        spline_coeff_ijk = spline_coeff[channel][type]
        for x_support in x_supports:
            for y_support in y_supports:
                for z_support in z_supports:
                    conv_sum += (binary_mask[i][j][k] * spline_coeff_ijk[x_support][y_support][z_support].item()) * dh(i,j,k,x-x_support,y-y_support,z-z_support) 

    return conv_sum

def d2f(channel,x,y,z):
    x_supports = [np.floor(x).dtype(int), np.ceil(x).dtype(int)]
    y_supports = [np.floor(y).dtype(int), np.ceil(y).dtype(int)]
    z_supports = [np.floor(z).dtype(int), np.ceil(z).dtype(int)]

    conv_sum = torch.zeros((3,x.shape[0]))

    for type in range(spline_coeff[channel].shape[0]):
        i, j, k = binary_array(type)
        spline_coeff_ijk = spline_coeff[channel][type]
        for x_support in x_supports:
            for y_support in y_supports:
                for z_support in z_supports:
                    conv_sum += (binary_mask[i][j][k] * spline_coeff_ijk[x_support][y_support][z_support].item()) * d2h(i,j,k,x-x_support,y-y_support,z-z_support)

    return conv_sum

# Output fields Vx, Vy, Vz and P
def v_x(x,y,z):
    return f(0,x,y,z)

def v_y(x,y,z):
    return f(1,x,y,z)

def v_z(x,y,z):
    return f(2,x,y,z)

def p(x,y,z):
    return f(3,x,y,z)

# Derivatives for the Output fields Vx, Vy, Vz and P
def dv_x(x,y,z):
    return df(0,x,y,z)

def dv_y(x,y,z):
    return df(1,x,y,z)

def dv_z(x,y,z):
    return df(2,x,y,z)

def dp(x,y,z):
    return df(3,x,y,z)

# Double Derivatives for the Output fields Vx, Vy, Vz
def d2v_x(x,y,z):
    return d2f(0,x,y,z)

def d2v_y(x,y,z):
    return d2f(1,x,y,z)

def d2v_z(x,y,z):
    return d2f(2,x,y,z)


model_path = "src/Baseline_ML4Science.stl"

grid_resolution = np.array([20,20,20])

model = trimesh.load("src/Baseline_ML4Science.stl")

binary_mask = get_binary_mask(model, grid_resolution)

# spline_coeff = process_mesh(model_path)
spline_coeff = torch.rand((4,8,20,20,20))

rho = 1

x = torch.tensor([1,1])
y = torch.tensor([1,1])
z = torch.tensor([1,1])

print(d2f(0,x,y,z).shape)

def domain_loss(x,y,z):
    divergence_loss = torch.mean(dv_x(x,y,z)[0] + dv_y(x,y,z)[1] + dv_z(x,y,z)[2])
    momentum_loss = torch.mean(v_x(x,y,z)*dv_x(x,y,z) + v_y(x,y,z)*dv_y(x,y,z) + v_z(x,y,z)*dv_z(x,y,z) + (1/rho) * dp(x,y,z) - (1/rho) * (d2v_x(x,y,z) + d2v_y(x,y,z) + d2v_z(x,y,z)))
    return divergence_loss + momentum_loss

def boundary_loss(x,y,z):
    return torch.mean(v_x(x,y,z)**2 + v_y(x,y,z)**2 + v_z(x,y,z)**2)

surface_points = torch.tensor(trimesh.sample.sample_surface(model, 100)[0])
volume_points = torch.tensor(trimesh.sample.volume_mesh(model, 100))
surface_labels = torch.ones(surface_points.size(0), dtype=torch.int64)
volume_labels = torch.zeros(volume_points.size(0), dtype=torch.int64)

# Combine points and labels
all_points = torch.cat([surface_points, volume_points], dim = 0)
all_labels = torch.cat([surface_labels, volume_labels], dim = 0)

# Shuffle points and labels together
permutation = torch.randperm(all_points.size(0))
train_points = all_points[permutation]
train_points.requires_grad_(True)
train_labels = all_labels[permutation]

print(surface_points)

print(domain_loss(.5,.5,.5))


# # Ensure support_points contains valid indices
# assert torch.all(support_points[:, 0] < grid_resolution[0])
# assert torch.all(support_points[:, 1] < grid_resolution[1])
# assert torch.all(support_points[:, 2] < grid_resolution[2])