import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


# Convert binary mask to PyTorch tensor and reshape for UNet
def prepare_mesh_for_unet(binary_mask):
    tensor_mask = torch.from_numpy(binary_mask).double()

    # Add batch and channel dimensions
    # From (20, 20, 20) to (1, 1, 20, 20, 20)
    tensor_mask = tensor_mask.unsqueeze(0).unsqueeze(0)

    return tensor_mask


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=5, coefficients_per_channel=8):
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
        self.final_conv = nn.Conv3d(
            32, out_channels * coefficients_per_channel, kernel_size=1
        )
        self.out_channels = out_channels
        self.coefficients_per_channel = coefficients_per_channel

    def conv_block(self, in_channels, out_channels):
        """Helper function to define 3D convolutional blocks."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        """Helper function to define 3D upconvolution blocks."""
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # Level 1
        enc2 = self.enc2(F.max_pool3d(enc1, kernel_size=2, stride=2))  # Level 2

        # Bottleneck
        bottleneck = self.bottleneck(
            F.max_pool3d(enc2, kernel_size=2, stride=2)
        )  # Bottleneck

        # Decoder
        up2 = self.upconv2(bottleneck)  # Upsample bottleneck
        up2 = torch.cat([up2, enc2], dim=1)  # Skip connection with encoder level 2
        up2 = self.dec2(up2)

        up1 = self.upconv1(up2)  # Upsample decoder level 2
        up1 = torch.cat([up1, enc1], dim=1)  # Skip connection with encoder level 1
        up1 = self.dec1(up1)

        # Final output
        output = self.final_conv(
            up1
        )  # [batch, out_channels * coefficients_per_channel, depth, height, width]

        # Reshape output to [batch, out_channels, coefficients_per_channel, depth, height, width]
        batch_size = output.shape[0]
        output = output.view(
            batch_size,
            self.out_channels,
            self.coefficients_per_channel,
            *output.shape[2:],  # Spatial dimensions
        )
        return output


def initialize_weights(m):
    if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
