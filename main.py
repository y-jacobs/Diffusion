# diffusion model u-net fashion-mnist

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('TkAgg')  # Use a suitable backend here (e.g., 'TkAgg', 'Qt5Agg', etc.)
import matplotlib.pyplot as plt
from model import UNet
from utils import visualize_results


# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)




# load fashion mnist dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


model = UNet(in_channels=1, out_channels=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5
# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
# Save the model
torch.save(model.state_dict(), 'unet_fashion_mnist.pth')
# Load the model
model = UNet(in_channels=1, out_channels=1)
model.load_state_dict(torch.load('unet_fashion_mnist.pth'))
model.eval()
# Test the model
test_loss = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, data).item()
test_loss /= len(test_loader.dataset)
print(f'Test Loss: {test_loss}')
# Visualize some results

visualize_results(model, test_loader)
