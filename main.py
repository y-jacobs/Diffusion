import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# Visualization tools
# import graphviz
# from torchview import draw_graph
import torchvision
import torchvision.transforms as transforms
import matplotlib

import utils

matplotlib.use('TkAgg')  # or 'QtAgg' if you installed PyQt5
import matplotlib.pyplot as plt
from model import UNet
from utils import add_noise, load_transformed_fashionMNIST


def plot_sample(images):
    plt.figure(figsize=(16, 1))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        img = images[i].detach().cpu().squeeze().numpy()
        plt.imshow(img, cmap='gray')
        plt.axis("off")
    plt.show()


# def test_model(model):
#     model.eval()
#     plt.figure(figsize=(8, 8))
#     plt.axis("off")
#
#     x = torch.randn(5, IMG_CH, IMG_SIZE, IMG_SIZE).to(device)
#
#     with torch.no_grad():
#         for t_val in reversed(range(T)):
#             t_batch = torch.full((x.size(0),), t_val, device=device, dtype=torch.long)  # [B]
#             noise_pred = model(x, t_batch.unsqueeze(-1).float())
#             x = reverse_q(x, t_batch, noise_pred, B, pred_noise_coeff, sqrt_a_inv)
#
#     for i in range(5):
#         plt.subplot(1, 5, i + 1)
#         img = x[i].detach().cpu().squeeze().numpy()
#         plt.imshow(img, cmap='gray')
#         plt.axis("off")
#
#     plt.savefig("test_model.png", bbox_inches='tight', pad_inches=0.1)
#     plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()


train_set = torchvision.datasets.FashionMNIST(
    "./data/", download=True, transform=transforms.Compose([transforms.ToTensor()])
)
NUM_CLASSES = 10


IMG_SIZE = 16 # Due to stride and pooling, must be divisible by 2 multiple times
IMG_CH = 1 # Black and white image, no color channels
BATCH_SIZE = 128


nrows = 10
ncols = 15

T = nrows * ncols
start = 0.0001
end = 0.02
B = torch.linspace(start, end, T).to(device)

a = 1. - B
a_bar = torch.cumprod(a, dim=0)
sqrt_a_bar = torch.sqrt(a_bar)  # Mean Coefficient
sqrt_one_minus_a_bar = torch.sqrt(1 - a_bar) # St. Dev. Coefficient

ddpm = utils.DDPM(B, device)


data = load_transformed_fashionMNIST()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


model = UNet(IMG_SIZE, IMG_CH, T)
print("Num params: ", sum(p.numel() for p in model.parameters()))
model.to(device)

optimizer = Adam(model.parameters(), lr=0.0001)
epochs = 5

model.train()
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        t = torch.randint(0, T, (BATCH_SIZE,), device=device).float()
        x = batch[0].to(device)
        loss = ddpm.get_loss(model, x, t)
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0 and step % 100 == 0:
            print(f"Epoch {epoch} | Step {step:03d} Loss: {loss.item()} ")
            # plot_sample(images_pred)


model.eval()
plt.figure(figsize=(8,8))
ncols = 3 # Should evenly divide T
fig, axs = plt.subplots(10, ncols, figsize=(ncols * 3, 10 * 3))
for _ in range(10):
    ddpm.sample_images(model, IMG_CH, IMG_SIZE, ncols)
plt.tight_layout()
plt.show()