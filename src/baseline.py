import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import Adam
from tqdm import tqdm
import trimesh

obj = trimesh.load("../Baseline_ML4Science.stl")

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = 'cpu'

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = 'cpu'

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
        
model = PINNs(in_dim=3, hidden_dim=128, out_dim=4, num_layer=2).to(device)
model.apply(init_weights)
model = model.double()
optim = Adam(model.parameters(), lr = 1e-3)
print(model)

loss_track = []

for i in tqdm(range(1000)):
    def closure():
        surface_points = torch.tensor(trimesh.sample.sample_surface(obj, 100)[0])
        volume_points = torch.tensor(trimesh.sample.volume_mesh(obj, 100))
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
        
        fields = model(train_points)
        vx = fields[:, 0:1]
        vy = fields[:, 1:2]
        vz = fields[:, 2:3]
        p = fields[:, 3:4]

        # Compute all gradients of vx with respect to train_points in one pass
        grad_vx = torch.autograd.grad(
            outputs=vx,
            inputs=train_points,
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
            inputs=train_points,
            grad_outputs=torch.ones_like(grad_vx),
            retain_graph=True,
            create_graph=True
        )[0]
        
        # Extract second derivatives
        vx_xx = second_grad_vx[:, 0:1]  # Second derivative w.r.t x
        vx_yy = second_grad_vx[:, 1:2]  # Second derivative w.r.t y
        vx_zz = second_grad_vx[:, 2:3]  # Second derivative w.r.t z

        # Compute all gradients of vy with respect to train_points in one pass
        grad_vy = torch.autograd.grad(
            outputs=vy,
            inputs=train_points,
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
            inputs=train_points,
            grad_outputs=torch.ones_like(grad_vy),
            retain_graph=True,
            create_graph=True
        )[0]
        
        # Extract second derivatives
        vy_xx = second_grad_vy[:, 0:1]  # Second derivative w.r.t x
        vy_yy = second_grad_vy[:, 1:2]  # Second derivative w.r.t y
        vy_zz = second_grad_vy[:, 2:3]  # Second derivative w.r.t z

        # Compute all gradients of vz with respect to train_points in one pass
        grad_vz = torch.autograd.grad(
            outputs=vz,
            inputs=train_points,
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
            inputs=train_points,
            grad_outputs=torch.ones_like(grad_vz),
            retain_graph=True,
            create_graph=True
        )[0]
        
        # Extract second derivatives
        vz_xx = second_grad_vz[:, 0:1]  # Second derivative w.r.t x
        vz_yy = second_grad_vz[:, 1:2]  # Second derivative w.r.t y
        vz_zz = second_grad_vz[:, 2:3]  # Second derivative w.r.t z

        # Compute all gradients of p with respect to train_points in one pass
        grad_p = torch.autograd.grad(
            outputs=p,
            inputs=train_points,
            grad_outputs=torch.ones_like(p),
            retain_graph=True,
            create_graph=True
        )[0]
        
        # Extract first derivatives
        p_x = grad_p[:, 0:1]  # Gradient w.r.t x
        p_y = grad_p[:, 1:2]  # Gradient w.r.t y
        p_z = grad_p[:, 2:3]  # Gradient w.r.t z


        loss_divergence = torch.mean((vx_x + vy_y + vz_z)**2)
        loss_momentum_x = torch.mean((vx*vx_x + vy*vx_y + vz*vx_z + (1/rho)*p_x - (mu/rho)*(vx_xx + vx_yy + vx_zz))**2)
        loss_momentum_y = torch.mean((vx*vy_x + vy*vy_y + vz*vy_z + (1/rho)*p_y - (mu/rho)*(vy_xx + vy_yy + vy_zz))**2)
        loss_momentum_z = torch.mean((vx*vz_x + vy*vz_y + vz*vz_z + (1/rho)*p_z - (mu/rho)*(vz_xx + vz_yy + vz_zz))**2)
        boundary_loss = torch.mean((vx[train_labels == 1])**2) + torch.mean((vy[train_labels == 1])**2) + torch.mean((vz[train_labels == 1])**2)
        loss = loss_divergence + loss_momentum_x + loss_momentum_y + loss_momentum_z + 10*boundary_loss
        loss_track.append(loss.item())
        print(f'Loss: {loss.item()}')

        optim.zero_grad()
        loss.backward()
        return loss

    optim.step(closure)
