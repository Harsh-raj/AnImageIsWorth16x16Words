from torch import nn
import torch
import einops
from hyperparameters import patch_size, n_channels, device, latent_size, batch_size

class InputEmbedding(nn.Module):
  def __init__(self, patch_size = patch_size, n_channels=n_channels, device=device, latent_size=latent_size, batch_size=batch_size):
    super(InputEmbedding, self).__init__()
    self.latent_size = latent_size
    self.patch_size = patch_size
    self.n_channels = n_channels
    self.device = device
    self.batch_size = batch_size
    self.input_size = self.patch_size*self.patch_size*self.n_channels
    
    # Linear projection
    self.linearProjection = nn.Linear(self.input_size, self.latent_size)
    
    # Class token
    self.class_token = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)
    
    # Positional embedding
    self.pos_embedding = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)
    
    
  def forward(self, input_data):
    input_data = input_data.to(self.device)
    
    # Rearrange image into patches
    patches = einops.rearrange(
      input_data, 'b c (h h1) (w w1) -> b (h w) (h1 w1 c)', h1=self.patch_size, w1=self.patch_size
    )
    
    linear_projection = self.linearProjection(patches).to(self.device)
    b, n, _ = linear_projection.shape
    
    # Prepend the [class] token to the original linear projection
    linear_projection = torch.cat((self.class_token, linear_projection), dim=1)
    pos_embed = einops.repeat(self.pos_embedding, 'b 1 d -> b m d', m=n+1)
    
    # Add positional embedding to linear projection
    linear_projection += pos_embed
    return linear_projection