import torch
import torch.nn as nn
from einops import rearrange

class Transformer3D(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, mlp_dim, channels, output_dim):
        super(Transformer3D, self).__init__()
        self.embed_dim = embed_dim
        self.channels = channels
        self.output_dim = output_dim
        
        # Embedding: Flatten patches and map to embed_dim
        self.embedding = nn.Linear(1, embed_dim)  # Input_dim is 1 for binary matrix
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, embed_dim))
        
        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, activation="gelu"),
            num_layers=depth
        )
        
        # Output projection
        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim, channels * output_dim)
        )
        
    def forward(self, x):
        # x shape: (B, L, W, H)
        B, L, W, H = x.shape
        
        # Compute number of patches dynamically
        num_patches = L * W * H
        
        # Flatten spatial dimensions into patches
        x = rearrange(x, 'b l w h -> b (l w h) 1')
        
        # Embed patches
        x = self.embedding(x)  # (B, num_patches, embed_dim)
        
        # Expand positional encodings to match num_patches
        pos_embedding = self.pos_embedding.expand(B, num_patches, -1)
        x += pos_embedding
        
        # Pass through transformer
        x = self.transformer(x)  # (B, num_patches, embed_dim)
        
        # Project to the output dimensions
        x = self.fc_out(x)  # (B, num_patches, channels * output_dim)
        
        # Reshape to (B, Channels, Outputs, L, W, H)
        x = rearrange(x, 'b (l w h) (c o) -> b c o l w h', l=L, w=W, h=H, c=self.channels, o=self.output_dim)
        return x


# Instantiate the model
def create_model(embed_dim=128, num_heads=4, depth=1, mlp_dim=256, channels=5, output_dim=8):
    model = Transformer3D(
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        mlp_dim=mlp_dim,
        channels=channels,
        output_dim=output_dim
    )
    return model
