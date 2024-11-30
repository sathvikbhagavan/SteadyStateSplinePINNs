import trimesh
import numpy as np
import torch
from sample import *
from hermite_spline import *
from unet import *
from torch.optim import Adam
from tqdm import tqdm

def get_supoort_points(points):
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    x_floor = (x//step[0]).long()
    y_floor = (y//step[1]).long()
    z_floor = (z//step[2]).long()
    x_support_indices = torch.vstack((x_floor, torch.clamp(x_floor+1,max=grid_resolution[0]-1)))
    y_support_indices = torch.vstack((y_floor, torch.clamp(y_floor+1,max=grid_resolution[1]-1)))
    z_support_indices = torch.vstack((z_floor, torch.clamp(z_floor+1,max=grid_resolution[2]-1)))

    return x,y,z,x_support_indices,y_support_indices,z_support_indices

def f(spline_coeff, channel,points,func=h):
    x,y,z,x_support_indices,y_supports_indices,z_supports_indices = get_supoort_points(points)

    conv_sum = 0

    for type in range(spline_coeff[channel].shape[0]):
        i, j, k = binary_array(type)
        spline_coeff_ijk = spline_coeff[channel][type]
        for x_support_ind in x_support_indices:
            for y_support_ind in y_supports_indices:
                for z_support_ind in z_supports_indices:
                    # One of the 8 grid support points(enclosing cube vertices) for each sample point.
                    support_point_ind = torch.vstack((x_support_ind,y_support_ind,z_support_ind)).T

                    x_indices = support_point_ind[:, 0]
                    y_indices = support_point_ind[:, 1]
                    z_indices = support_point_ind[:, 2]

                    x_input = (x-x_indices*step[0])/step[0]
                    y_input = (y-y_indices*step[1])/step[1]
                    z_input = (z-y_indices*step[2])/step[2]
                    
                    conv_sum += (spline_coeff_ijk[x_indices, y_indices, z_indices]) * func(i,j,k,x_input,y_input,z_input)
    return conv_sum

def df(spline_coeff,channel,points):
    return f(spline_coeff,channel,points,dh).T

def d2f(spline_coeff,channel,points):
    return f(spline_coeff,channel,points,d2h).T

# Output fields Vx, Vy, Vz and P
def v_x(spline_coeff,points):
    return f(spline_coeff,0,points)

def v_y(spline_coeff,points):
    return f(spline_coeff,1,points)

def v_z(spline_coeff,points):
    return f(spline_coeff,2,points)

def p(spline_coeff,points):
    return f(spline_coeff,3,points)

# Derivatives for the Output fields Vx, Vy, Vz and P
def dv_x(spline_coeff,points):
    return df(spline_coeff,0,points)

def dv_y(spline_coeff,points):
    return df(spline_coeff,1,points)

def dv_z(spline_coeff,points):
    return df(spline_coeff,2,points)

def dp(spline_coeff,points):
    return df(spline_coeff,3,points)

# Double Derivatives for the Output fields Vx, Vy, Vz
def d2v_x(spline_coeff,points):
    return d2f(spline_coeff,0,points)

def d2v_y(spline_coeff,points):
    return d2f(spline_coeff,1,points)

def d2v_z(spline_coeff,points):
    return d2f(spline_coeff,2,points)

########################################################################################################################

# Check for Metal (MPS) device
device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

grid_resolution = np.array([20,20,20])
model = trimesh.load("src/Baseline_ML4Science.stl")
binary_mask = get_binary_mask(model, grid_resolution)
step = model.bounding_box.extents/(grid_resolution-1)

# Instantiate the neural network
unet_model = UNet3D().to(device)
optimizer = Adam(unet_model.parameters(), lr = 1e-3)

loss_track = []

for epoch in tqdm(range(2)):
    # Prepare for UNet
    unet_input = prepare_mesh_for_unet(binary_mask)

    # Process through UNet
    # with torch.no_grad():  # If just inferencing
    spline_coeff = unet_model(unet_input)[0]
    # print(spline_coeff.shape)

    # spline_coeff = torch.rand((4,8,51,21,11))
    # points = torch.tensor([[1,1,3],[1,4,5],[4,6,8],[2,1,5]])
    rho = 1
    mu = 0.00002

    def domain_losses(points):
        # 3D Loss Terms(with 3 dimensions corresponding to x, y, z)
        v_x_term = v_x(spline_coeff,points).unsqueeze(1)*dv_x(spline_coeff,points)
        v_y_term = v_y(spline_coeff,points).unsqueeze(1)*dv_y(spline_coeff,points)
        v_z_term = v_z(spline_coeff,points).unsqueeze(1)*dv_z(spline_coeff,points)
        p_term = (1/rho) * dp(spline_coeff,points)
        d2v_x_term = (mu/rho) * d2v_x(spline_coeff,points)
        d2v_y_term = (mu/rho) * d2v_y(spline_coeff,points)
        d2v_z_term = (mu/rho) * d2v_z(spline_coeff,points)

        momentum_term = v_x_term + v_y_term + v_z_term + p_term - (d2v_x_term + d2v_y_term + d2v_z_term)

        divergence_losses = torch.mean(dv_x(spline_coeff,points)[0] + dv_y(spline_coeff,points)[1] + dv_z(spline_coeff,points)[2])
        x_momentum_losses = momentum_term[:,0]
        y_momentum_losses = momentum_term[:,1]
        z_momentum_losses = momentum_term[:,2]

        domain_losses = divergence_losses**2 + x_momentum_losses**2 + y_momentum_losses**2 + z_momentum_losses**2
        return domain_losses

    def boundary_losses(points):
        return v_x(spline_coeff,points)**2 + v_y(spline_coeff,points)**2 + v_z(spline_coeff,points)**2

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

    losses = domain_losses(train_points)*train_labels + boundary_losses(train_points)*(1-train_labels)
    loss = torch.mean(losses)
    loss_track.append(loss.item())
    print(f'Loss: {loss.item()}')
    # print(f'v_x of point1: {v_x(spline_coeff,train_points)[0]}')

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()