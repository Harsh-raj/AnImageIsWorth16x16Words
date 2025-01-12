from vit import Vit
from prepareDataset import PrepareDataset
from tqdm import tqdm
from torch import nn
import torch.optim as optim
from torchsummary import summary

class Train:
  model = Vit(Vit.num_encoders, Vit.latent_size, Vit.device, Vit.num_classes).to(Vit.device)
  
  # Betas used for Adam in paper are 0.9 and 0.999
  optimizer = optim.Adam(model.parameters(), lr=Vit.base_lr, weight_decay=Vit.weight_decay)
  criterion = nn.CrossEntropyLoss()
  scheduler = optim.lr_scheduler.LinearLR(optimizer)
  
  @staticmethod
  def train():
    Vit.model.train().to(Vit.device)
    
    for epoch in tqdm(range(Vit.epochs), total=Vit.epochs):
      running_loss = 0.0
      for batch_idx, (inputs, targets) in enumerate(tqdm(PrepareDataset.trainloader)):
        inputs, targets = inputs.to(Vit.device), targets.to(Vit.device)
        
        optimizer.zero_grad()
        
        outputs = Vit.model(inputs)
        
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 200 == 0:
          print('Batch {} epoch {} has loss = {}'.format(batch_idx, epoch, running_loss/200))
          running_loss = 0
      
    scheduler.step()
    summary(Vit.model)
      
if __name__ == "__main__":
  Train.train()