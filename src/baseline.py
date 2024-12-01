import numpy as np
import torch
import torch.nn as nn
import random
from torch.optim import Adam, LBFGS
import trimesh
import time
import wandb

obj = trimesh.load("./Baseline_ML4Science.stl")

# lr = 1e-3
epochs = 1000
seed = 42
hidden_dim = 128
num_layer = 4
inlet_velocity = 0.5

wandb.init(
    # set the wandb project where this run will be logged
    project="PINNs-baseline",

    # track hyperparameters and run metadata
    config={
        # "learning_rate": lr,
        "optimizer": "LBFGS",
        "architecture": "FFNN",
        "epochs": epochs,
        "seed": seed,
        "hidden_dim": hidden_dim,
        "num_layers": num_layer,
        "inlet_velocity":inlet_velocity
    }
)

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = 'cuda'

class PINNs(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        super(PINNs, self).__init__()

        layers = []
        for i in range(num_layer-1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
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

    # Compute all gradients of vx with respect to points in one pass
    grad_vx = torch.autograd.grad(
        outputs=vx,
        inputs=points,
        grad_outputs=torch.ones_like(vx),
        retain_graph=True,
        create_graph=True
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
        create_graph=True
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
        create_graph=True
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
        create_graph=True
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
        create_graph=True
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
        create_graph=True
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
        create_graph=True
    )[0]
    
    # Extract first derivatives
    p_x = grad_p[:, 0:1]  # Gradient w.r.t x
    p_y = grad_p[:, 1:2]  # Gradient w.r.t y
    p_z = grad_p[:, 2:3]  # Gradient w.r.t z
    
    return vx, vy, vz, p, vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz, vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz, vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz, p_x, p_y, p_z
    
def get_loss(vx, vy, vz, p, vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz, vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz, vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz, p_x, p_y, p_z, labels):
    loss_divergence = torch.mean((vx_x + vy_y + vz_z)**2)
    loss_momentum_x = torch.mean((vx*vx_x + vy*vx_y + vz*vx_z + (1/rho)*p_x - (mu/rho)*(vx_xx + vx_yy + vx_zz))**2)
    loss_momentum_y = torch.mean((vx*vy_x + vy*vy_y + vz*vy_z + (1/rho)*p_y - (mu/rho)*(vy_xx + vy_yy + vy_zz))**2)
    loss_momentum_z = torch.mean((vx*vz_x + vy*vz_y + vz*vz_z + (1/rho)*p_z - (mu/rho)*(vz_xx + vz_yy + vz_zz))**2)
    inlet_boundary_loss = torch.mean((vx[labels == 1] - inlet_velocity)**2) + torch.mean((vy[labels == 1])**2) + torch.mean((vz[labels == 1])**2)
    other_boundary_loss = torch.mean((vx[labels == 2])**2) + torch.mean((vy[labels == 2])**2) + torch.mean((vz[labels == 2])**2)
    loss = loss_divergence + 10.0*loss_momentum_x + 10.0*loss_momentum_y + 10.0*loss_momentum_z + 100.0*inlet_boundary_loss + 1000.0*other_boundary_loss
    print(f'Loss: {loss_divergence.item()}, {loss_momentum_x.item()}, {loss_momentum_y.item()}, {loss_momentum_z.item()}, {inlet_boundary_loss.item()}, {other_boundary_loss.item()}')
    wandb.log({'Divergence Loss': np.log(loss_divergence.item()), 'X Momentum Loss': np.log(loss_momentum_x.item()), 'Y Momentum Loss': np.log(loss_momentum_y.item()), 'Z Momentum Loss': np.log(loss_momentum_z.item()), 'Inlet Boundary Loss': np.log(inlet_boundary_loss.item()), 'Other Boundary Loss': np.log(other_boundary_loss.item())})
    return loss
        
model = PINNs(in_dim=3, hidden_dim=hidden_dim, out_dim=4, num_layer=num_layer).to(device)
model.apply(init_weights)
model = model.double()
# optim = Adam(model.parameters(), lr = lr)
optim = LBFGS(model.parameters(), line_search_fn = 'strong_wolfe')
rho = 1.010427
mu = 2.02e-5
loss_track = []
start_time = time.time()

def get_inlet_surface_points(num_points):
    threshold = 1e-5
    faces_x_zero = [
        i for i, face in enumerate(obj.faces)
        if np.all(np.abs(obj.vertices[face, 0]) < threshold)  # Check if all vertices' x-coordinates are 0
    ]
    subset_mesh = obj.submesh([faces_x_zero], only_watertight=False)[0]
    points, _ = trimesh.sample.sample_surface(subset_mesh, count=num_points)
    return torch.tensor(points / 1000.0, dtype = torch.float64, device = device)

def get_other_surface_points(num_points):
    threshold = 1e-5
    points, _ = trimesh.sample.sample_surface(obj, count=num_points)
    filtered_points = points[np.abs(points[:, 0]) > threshold]
    return torch.tensor(filtered_points / 1000.0, dtype = torch.float64, device = device)

# surface_points = torch.tensor(trimesh.sample.sample_surface(obj, 500)[0] / 1000.0, device=device, dtype=torch.float64)
# volume_points = torch.tensor(trimesh.sample.volume_mesh(obj, 500) / 1000.0, device=device, dtype=torch.float64)
# surface_labels = torch.ones(surface_points.size(0), dtype=torch.int64)
# volume_labels = torch.zeros(volume_points.size(0), dtype=torch.int64)
# valid_points = torch.cat([surface_points, volume_points], dim = 0)
# valid_labels = torch.cat([surface_labels, volume_labels], dim = 0)

for i in range(epochs):
    def closure():
        # surface_points = torch.tensor(trimesh.sample.sample_surface(obj, 500)[0] / 1000.0, device=device, dtype=torch.float64)
        inlet_surface_points = get_inlet_surface_points(100)
        other_surface_points = get_other_surface_points(400)
        volume_points = torch.tensor(trimesh.sample.volume_mesh(obj, 500) / 1000.0, device=device, dtype=torch.float64)
        inlet_surface_labels = torch.ones(inlet_surface_points.size(0), dtype=torch.int64)
        other_surface_labels = 2*torch.ones(other_surface_points.size(0), dtype=torch.int64)
        volume_labels = torch.zeros(volume_points.size(0), dtype=torch.int64)
        # Combine points and labels
        all_points = torch.cat([inlet_surface_points, other_surface_points, volume_points], dim = 0)
        all_labels = torch.cat([inlet_surface_labels, other_surface_labels, volume_labels], dim = 0)
        # Shuffle points and labels together
        permutation = torch.randperm(all_points.size(0))
        train_points = all_points[permutation]
        train_points.requires_grad_(True)
        train_labels = all_labels[permutation]
        train_fields = model(train_points)
        vx, vy, vz, p, vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz, vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz, vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz, p_x, p_y, p_z = get_values_and_derivatives(train_fields, train_points)
        train_loss = get_loss(vx, vy, vz, p, vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz, vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz, vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz, p_x, p_y, p_z, train_labels)
        loss_track.append(train_loss.item())
        # print(f'Train Loss: {train_loss.item()}')
        optim.zero_grad()
        train_loss.backward()
        return train_loss
    optim.step(closure)

stop_time = time.time()
print(f'Time taken for training is: {stop_time - start_time}')