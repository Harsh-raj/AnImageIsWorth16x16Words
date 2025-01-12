from torchvision.transform import Compose, RandomCrop, Resize, RandomHorizontalFlip, ToTensor, Normalize

import torchvision
import torch

from vit import Vit
class PrepareDataset:
  transform_training_data = Compose([RandomCrop(32, padding=4), Resize((224)), RandomHorizontalFlip(), ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

  train_data = torchvision.datasets.CIFAR10(root='/home/achalhoub/Desktop/ViT_CIFAR10_data', train=True, download=False, transform=transform_training_data)

  trainloader = torch.utils.data.DataLoader(train_data, batch_size=Vit.batch_size,shuffle=True, num_workers=2)

  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')