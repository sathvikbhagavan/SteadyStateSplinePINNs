import trimesh
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load and process the mesh
model = trimesh.load("/Users/macbookpro16/mlproject2/SteadyStateSplinePINNs/src/Baseline_ML4Science.stl")

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
        index_x = np.searchsorted(x, point[0])
        index_y = np.searchsorted(y, point[1])
        index_z = np.searchsorted(z, point[2])
        if (
            0 <= index_x < grid_resolution[0]
            and 0 <= index_y < grid_resolution[1]
            and 0 <= index_z < grid_resolution[2]
        ):
            binary_volume[index_x, index_y, index_z] = 1
    return binary_volume

# Process the mesh to get binary mask
grid_resolution = (20, 20, 20)  # Match UNet input size
binary_mask = get_binary_mask(model, grid_resolution)

# Convert binary mask to PyTorch tensor and reshape for UNet
def prepare_mesh_for_unet(binary_mask):
    # Convert to float32 tensor
    tensor_mask = torch.from_numpy(binary_mask).float()
    
    # Add batch and channel dimensions
    # From (20, 20, 20) to (1, 1, 20, 20, 20)
    tensor_mask = tensor_mask.unsqueeze(0).unsqueeze(0)
    
    return tensor_mask

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, coefficients_per_channel=8):
        super(UNet3D, self).__init__()

        # Encoder (Contracting Path)
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)

        # Bottleneck
        self.bottleneck = self.conv_block(64, 128)

        # Decoder (Expansive Path) with skip connections
        self.upconv2 = self.upconv_block(128, 64)
        self.dec2 = self.conv_block(128, 64)  # 128 because of concatenation

        self.upconv1 = self.upconv_block(64, 32)
        self.dec1 = self.conv_block(64, 32)  # 64 because of concatenation

        # Final output layer
        self.final_conv = nn.Conv3d(32, out_channels * coefficients_per_channel, kernel_size=1)
        self.out_channels = out_channels
        self.coefficients_per_channel = coefficients_per_channel

    def conv_block(self, in_channels, out_channels):
        """Helper function to define 3D convolutional blocks."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        """Helper function to define 3D upconvolution blocks."""
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # Level 1
        enc2 = self.enc2(F.max_pool3d(enc1, kernel_size=2, stride=2))  # Level 2

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool3d(enc2, kernel_size=2, stride=2))  # Bottleneck

        # Decoder
        up2 = self.upconv2(bottleneck)  # Upsample bottleneck
        up2 = torch.cat([up2, enc2], dim=1)  # Skip connection with encoder level 2
        up2 = self.dec2(up2)

        up1 = self.upconv1(up2)  # Upsample decoder level 2
        up1 = torch.cat([up1, enc1], dim=1)  # Skip connection with encoder level 1
        up1 = self.dec1(up1)

        # Final output
        output = self.final_conv(up1)  # [batch, out_channels * coefficients_per_channel, depth, height, width]

        # Reshape output to [batch, out_channels, coefficients_per_channel, depth, height, width]
        batch_size = output.shape[0]
        output = output.view(
            batch_size,
            self.out_channels,
            self.coefficients_per_channel,
            *output.shape[2:]  # Spatial dimensions
        )
        return output

# Main processing pipeline
def process_mesh(model_path, grid_resolution=(20, 20, 20)):
    # Load model
    model_mesh = trimesh.load(model_path)
    
    # Get binary mask
    binary_mask = get_binary_mask(model_mesh, grid_resolution)
    
    # Prepare for UNet
    unet_input = prepare_mesh_for_unet(binary_mask)
    
    # Initialize UNet
    unet_model = UNet3D(in_channels=1, out_channels=4)
    
    # Process through UNet
    with torch.no_grad():  # If just inferencing
        coefficients = unet_model(unet_input)
    
    return coefficients

# Main
if __name__ == "__main__":
    # Process the mesh and get coefficients
    mesh_path = "/Users/macbookpro16/mlproject2/SteadyStateSplinePINNs/src/Baseline_ML4Science.stl"
    coefficients = process_mesh(mesh_path)
    
    # Print shapes to verify
    print(f"Coefficients shape: {coefficients.shape}")  # Should be [1, 4, 8, 20, 20, 20]
    
    # If you want to access individual coefficient volumes
    coeff1 = coefficients[0, 0].numpy()  # First coefficient volume
    coeff2 = coefficients[0, 1].numpy()  # Second coefficient volume
    coeff3 = coefficients[0, 2].numpy()  # Third coefficient volume
    coeff4 = coefficients[0, 3].numpy()  # Fourth coefficient volume

    #print(coeff1)
    #print(coeff2)
    #print(coeff3)
    #print(coeff4)