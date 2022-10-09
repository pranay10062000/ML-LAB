import torch
import torchvision.transforms as transforms
from itertools import cycle
from torch.utils.data import DataLoader
import glob
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
img_size = 48 # before 28
train_batch_size = 100
epochs = 40

transforms_ = [
    transforms.ToTensor(),
    transforms.Resize((img_size, img_size)),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    transforms.Grayscale(num_output_channels=1),
] # Transform for Train Data

transforms_labels = [
    transforms.ToTensor(),
    transforms.Resize((img_size, img_size)),
    transforms.Normalize((0.5), (0.5)),
] # Transform for Train Labels

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train", train_size = 9000):
        self.transform = transforms.Compose(transforms_)
        self.mode = mode
        self.files = sorted(glob.glob("%s/*.jpg" % root))
        
        if mode == "train":
            self.files = self.files[:train_size]
        else:
            self.files = self.files[train_size:] 
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        return img


train_dataloader = DataLoader(
    ImageDataset("D:\\FYP\\context_encoder\\img_align_celeba_10k", transforms_=transforms_), 
    batch_size = train_batch_size,
    shuffle=False,
) # TrainSet DataLoader

test_dataloader = DataLoader(
    ImageDataset("D:\\FYP\\context_encoder\\img_align_celeba_10k", transforms_=transforms_, mode="val"),
    batch_size=1,
    shuffle=False,
) # TestSet DataLoader

train_label_dataloader = DataLoader(
    ImageDataset("D:\\FYP\\New folder\\images", transforms_=transforms_labels),
    batch_size = train_batch_size,
    shuffle=False,
) # EdgeDetected Images of TrainSet DataLoader

print(device)