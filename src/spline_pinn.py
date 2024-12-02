import trimesh
import numpy as np
import random
import torch
from sample import *
from hermite_spline import *
from unet import *
from torch.optim import Adam
from tqdm import tqdm
import time
import wandb

seed = 42

# Model Params
epochs = 500
lr = 1e-4

# Physics Constants
inlet_velocity = 0.5
rho = 1.010427 
mu = 2.02e-5

wandb.init(
    # set the wandb project where this run will be logged
    project="Spline-PINNs_with_validation",

    # track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "optimizer": "Adam",
        "architecture": "Unet",
        "epochs": epochs,
        "seed": seed,
        "inlet_velocity":inlet_velocity
    }
)

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Check for Metal (MPS) device
device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
torch.set_default_device(device)
print(f"Using device: {device}")

def get_supoort_points(points):
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    x_floor = (x//step[0]).long()
    y_floor = (y//step[1]).long()
    z_floor = (z//step[2]).long()
    x_supports_indices = torch.vstack((x_floor, torch.clamp(x_floor+1,max=grid_resolution[0]-1)))
    y_support_indices = torch.vstack((y_floor, torch.clamp(y_floor+1,max=grid_resolution[1]-1)))
    z_support_indices = torch.vstack((z_floor, torch.clamp(z_floor+1,max=grid_resolution[2]-1)))
    return x,y,z,x_supports_indices,y_support_indices,z_support_indices

def f(spline_coeff,channel,x,y,z,x_supports,y_supports,z_supports,der_x=0,der_y=0,der_z=0):
    conv_sum = 0
    for type in range(spline_coeff[channel].shape[0]):
        i, j, k = binary_array(type)
        spline_coeff_ijk = spline_coeff[channel][type]
        for x_supports_ind in x_supports:
            for y_support_ind in y_supports:
                for z_support_ind in z_supports:
                    # One of the 8 grid support points(enclosing cube vertices) for each sample point.
                    support_point_ind = torch.vstack((x_supports_ind,y_support_ind,z_support_ind)).T

                    x_indices = support_point_ind[:, 0]
                    y_indices = support_point_ind[:, 1]
                    z_indices = support_point_ind[:, 2]

                    x_input = (x-x_indices*step[0])/step[0]
                    y_input = (y-y_indices*step[1])/step[1]
                    z_input = (z-z_indices*step[2])/step[2]
                    
                    conv_sum += (spline_coeff_ijk[x_indices, y_indices, z_indices]) * hermite_kernel_3d(i,j,k,x_input,y_input,z_input,der_x,der_y,der_z)
    return conv_sum

# Calculating various field terms using coefficients
def get_fields_and_losses(spline_coeff, points, labels):
    x,y,z,x_supports,y_supports,z_supports = get_supoort_points(points)
    vx = f(spline_coeff,0,x,y,z,x_supports,y_supports,z_supports,0,0,0)
    vy = f(spline_coeff,1,x,y,z,x_supports,y_supports,z_supports,0,0,0)
    vz = f(spline_coeff,2,x,y,z,x_supports,y_supports,z_supports,0,0,0)
    p = f(spline_coeff,3,x,y,z,x_supports,y_supports,z_supports,0,0,0)

    vx_x = f(spline_coeff,0,x,y,z,x_supports,y_supports,z_supports,1,0,0)
    vx_y = f(spline_coeff,0,x,y,z,x_supports,y_supports,z_supports,0,1,0)
    vx_z = f(spline_coeff,0,x,y,z,x_supports,y_supports,z_supports,0,0,1)

    vy_x = f(spline_coeff,1,x,y,z,x_supports,y_supports,z_supports,1,0,0)
    vy_y = f(spline_coeff,1,x,y,z,x_supports,y_supports,z_supports,0,1,0)
    vy_z = f(spline_coeff,1,x,y,z,x_supports,y_supports,z_supports,0,0,1)

    vz_x = f(spline_coeff,2,x,y,z,x_supports,y_supports,z_supports,1,0,0)
    vz_y = f(spline_coeff,2,x,y,z,x_supports,y_supports,z_supports,0,1,0)
    vz_z = f(spline_coeff,2,x,y,z,x_supports,y_supports,z_supports,0,0,1)
 
    p_x = f(spline_coeff,3,x,y,z,x_supports,y_supports,z_supports,1,0,0)
    p_y = f(spline_coeff,3,x,y,z,x_supports,y_supports,z_supports,0,1,0)
    p_z = f(spline_coeff,3,x,y,z,x_supports,y_supports,z_supports,0,0,1)

    vx_xx = f(spline_coeff,0,x,y,z,x_supports,y_supports,z_supports,2,0,0)
    vx_yy = f(spline_coeff,0,x,y,z,x_supports,y_supports,z_supports,0,2,0)
    vx_zz = f(spline_coeff,0,x,y,z,x_supports,y_supports,z_supports,0,0,2)

    vy_xx = f(spline_coeff,1,x,y,z,x_supports,y_supports,z_supports,2,0,0)
    vy_yy = f(spline_coeff,1,x,y,z,x_supports,y_supports,z_supports,0,2,0)
    vy_zz = f(spline_coeff,1,x,y,z,x_supports,y_supports,z_supports,0,0,2)

    vz_xx = f(spline_coeff,2,x,y,z,x_supports,y_supports,z_supports,2,0,0)
    vz_yy = f(spline_coeff,2,x,y,z,x_supports,y_supports,z_supports,0,2,0)
    vz_zz = f(spline_coeff,2,x,y,z,x_supports,y_supports,z_supports,0,0,2)
    
    # calculate losses
    loss_divergence = torch.mean((vx_x + vy_y + vz_z)**2)
    loss_momentum_x = torch.mean(((vx*vx_x + vy*vx_y + vz*vx_z) + (1/rho) * p_x - 1000 * (mu/rho) * (vx_xx + vx_yy + vx_zz))**2)
    loss_momentum_y = torch.mean(((vx*vy_x + vy*vy_x + vz*vy_x) + (1/rho) * p_y - 1000 * (mu/rho) * (vy_xx + vy_yy + vy_zz))**2)
    loss_momentum_z = torch.mean(((vx*vz_x + vy*vz_x + vz*vz_x) + (1/rho) * p_z - 1000 * (mu/rho) * (vz_xx + vz_yy + vz_zz))**2)
    loss_inlet_boundary = torch.mean((vx[labels == 1] - inlet_velocity)**2) + torch.mean((vy[labels == 1])**2) + torch.mean((vz[labels == 1])**2)
    loss_other_boundary = torch.mean((vx[labels == 2])**2) + torch.mean((vy[labels == 2])**2) + torch.mean((vz[labels == 2])**2)

    return vx,vy,vz,p,loss_divergence,loss_momentum_x,loss_momentum_y,loss_momentum_z,loss_inlet_boundary,loss_other_boundary

##################################################################################################################################

obj = trimesh.load("src/Baseline_ML4Science.stl")

grid_resolution = np.array([20,20,20])
binary_mask = get_binary_mask(obj, grid_resolution)
step = obj.bounding_box.extents/(grid_resolution-1)

# Instantiate the neural network
unet_model = UNet3D().to(device)
optimizer = Adam(unet_model.parameters(), lr = lr)
unet_model.apply(initialize_weights)

start_time = time.time()
training_loss_track = []
validation_loss_track = []
for epoch in tqdm(range(epochs)):
    # Prepare the sample points
    inlet_surface_points, inlet_surface_labels = get_inlet_surface_points(obj,200)
    other_surface_points, other_surface_labels = get_other_surface_points(obj,800)   
    volume_points, volume_labels = get_volume_points(obj,1000)   

    # Combine points and labels
    all_points = torch.cat([inlet_surface_points, other_surface_points, volume_points], dim = 0)
    all_labels = torch.cat([inlet_surface_labels, other_surface_labels, volume_labels], dim = 0)

    # Shuffle points and labels together
    permutation = torch.randperm(all_points.size(0))
    shuffled_points = all_points[permutation]
    shuffled_labels = all_labels[permutation]

    # Split into training and validation sets
    split_ratio = 0.7
    split_index = int(split_ratio * shuffled_points.size(0))

    train_points = shuffled_points[:split_index]
    train_labels = shuffled_labels[:split_index]

    validation_points = shuffled_points[split_index:]
    validation_labels = shuffled_labels[split_index:]

    # Ensure training points allow gradient computation
    train_points.requires_grad_(True)

    # Get Hermite Spline coefficients from the Unet
    unet_input = prepare_mesh_for_unet(binary_mask)
    spline_coeff = unet_model(unet_input)[0]

    # Calculating various field terms using coefficients
    vx,vy,vz,p,loss_divergence,loss_momentum_x,loss_momentum_y,loss_momentum_z,loss_inlet_boundary,loss_other_boundary = get_fields_and_losses(spline_coeff, train_points, train_labels)
    loss_total = loss_divergence + loss_momentum_x + loss_momentum_y + loss_momentum_z + loss_inlet_boundary + loss_other_boundary

    wandb.log({'Divergence Loss': np.log(loss_divergence.item()), 'X Momentum Loss': np.log(loss_momentum_x.item()), 'Y Momentum Loss': np.log(loss_momentum_y.item()), 'Z Momentum Loss': np.log(loss_momentum_z.item()), 'Inlet Boundary Loss': np.log(loss_inlet_boundary.item()), 'Other Boundary Loss': np.log(loss_other_boundary.item()), 'Total Loss': np.log(loss_total.item())})

    training_loss_track.append(loss_total.item())
    print(f'Training Loss  : {loss_total.item()}')
    if (epoch+1)%100==0:
        print("vx: ", vx[:20])
        print("vy: ", vy[:20])
        print("vz: ", vz[:20])
        print("p: ", p[:20])

    optimizer.zero_grad()
    loss_total.backward()

    # Validation
    # Switch model to evaluation mode
    unet_model.eval()
    with torch.no_grad():
        unet_model.eval()
        validation_vx,validation_vy,validation_vz,validation_p,validation_loss_divergence,validation_loss_momentum_x,validation_loss_momentum_y,validation_loss_momentum_z,validation_loss_inlet_boundary,validation_loss_other_boundary = get_fields_and_losses(spline_coeff, validation_points, validation_labels)
        validation_loss_total = validation_loss_divergence + validation_loss_momentum_x + validation_loss_momentum_y + validation_loss_momentum_z + validation_loss_inlet_boundary + validation_loss_other_boundary
        
        wandb.log({'Validation Divergence Loss': np.log(validation_loss_divergence.item()), 'Validation X Momentum Loss': np.log(validation_loss_momentum_x.item()), 'Validation Y Momentum Loss': np.log(validation_loss_momentum_y.item()), 'Validation Z Momentum Loss': np.log(validation_loss_momentum_z.item()), 'Validation Inlet Boundary Loss': np.log(validation_loss_inlet_boundary.item()), 'Validation Other Boundary Loss': np.log(validation_loss_other_boundary.item()), 'Validation Total Loss': np.log(validation_loss_total.item())})

        validation_loss_track.append(validation_loss_total.item())
        print(f'Validation Loss: {validation_loss_total.item()}')

    unet_model.train()
    optimizer.step()


stop_time = time.time()
print(f'Time taken for training is: {stop_time - start_time}')