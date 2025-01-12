import einops
from tqdm import tqdm

from torchsummary import summary

import torch
from torch import nn
import torchvision
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop

from inputEmbedding import InputEmbedding
from encoderBlock import EncoderBlock
  
class Vit(nn.Module):

  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  print(device)

  patch_size = 16
  latent_size = 768
  n_channels = 3
  num_heads = 12
  num_encoders = 12
  dropout = 0.1
  num_classes = 10
  size = 224

  epochs = 10
  base_lr = 10e-3
  weight_decay = 0.03
  batch_size = 4
  
  def __init__(self, num_encoders=num_encoders, latent_size=latent_size, device=device, num_classes=num_classes, dropout=dropout):
    super(Vit, self).__init__()
    self.num_encoder = num_encoders
    self.latent_size = latent_size
    self.device = device
    self.num_classes = num_classes
    self.dropout = dropout
    
    self.embedding = InputEmbedding()
    
    #Create the stack of encoders
    self.encStack = nn.ModuleList([EncoderBlock() for i in range(self.num_encoder)])
    
    self.MLP_head = nn.Sequential(
      nn.LayerNorm(self.latent_size),
      nn.Linear(self.latent_size, self.latent_size),
      nn.Linear(self.latent_size, self.num_classes)
    )
    
  def forward(self, test_input):
    enc_output = self.embedding(test_input)
    
    for enc_layer in self.encStack:
      enc_output = enc_layer(enc_output)
      
    cls_token_embed = enc_output[:, 0]
    
    return self.MLP_head(cls_token_embed)