from hyperparameters import num_classes, num_encoders, latent_size, device, base_lr, weight_decay, epochs
from prepareDataset import PrepareDataset
from tqdm import tqdm
from torch import nn
import torch.optim as optim
from torchsummary import summary
from vit import Vit

class Train:
  model = Vit(num_encoders, latent_size, device, num_classes).to(device)
  
  # Betas used for Adam in paper are 0.9 and 0.999
  optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
  criterion = nn.CrossEntropyLoss()
  scheduler = optim.lr_scheduler.LinearLR(optimizer)
  
  @staticmethod
  def train():
    Train.model.train().to(device)
    
    for epoch in tqdm(range(epochs), total=epochs):
      running_loss = 0.0
      for batch_idx, (inputs, targets) in enumerate(tqdm(PrepareDataset.trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 200 == 0:
          print('Batch {} epoch {} has loss = {}'.format(batch_idx, epoch, running_loss/200))
          running_loss = 0
      
    scheduler.step()
    summary(model)
      
if __name__ == "__main__":
  Train.train()