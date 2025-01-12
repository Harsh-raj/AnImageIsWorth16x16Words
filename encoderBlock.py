from torch import nn
import torch
from vit import Vit

class EncoderBlock(nn.Module):
  def __init__(self, latent_size=Vit.latent_size, num_heads=Vit.num_heads, device=Vit.device, dropout=Vit.dropout):
    super(EncoderBlock, self).__init__()
    
    self.latent_size = latent_size
    self.num_heads = num_heads
    self.device = device
    self.dropout = dropout
    
    # Normalization layer
    self.norm = nn.LayerNorm(self.latent_size)
    
    self.multihead = nn.MultiheadAttention(
      self.latent_size, self.num_heads, dropout=self.dropout
    )
    
    self.enc_MLP = nn.Sequential(
      nn.Linear(self.latent_size, self.latent_size*4),
      nn.GELU(),
      nn.Dropout(self.dropout),
      nn.Linear(self.latent_size*4, self.latent_size),
      nn.Dropout(self.dropout)
    )
    
  def forward(self, embedded_patches):
    firstnorm_out = self.norm(embedded_patches)
    attention_out = self.multihead(firstnorm_out, firstnorm_out, firstnorm_out)[0]
    
    # first residual connection
    first_added = attention_out + embedded_patches
    
    secondnorm_out = self.norm(first_added)
    ff_out = self.enc_MLP(secondnorm_out)
    
    return ff_out + first_added