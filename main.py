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
matplotlib.use('TkAgg')  # or 'QtAgg' if you installed PyQt5
import matplotlib.pyplot as plt
from model import UNet
from utils import add_noise, load_transformed_fashionMNIST, q, reverse_q


def plot_sample(images):
    plt.figure(figsize=(16, 1))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        img = images[i].detach().cpu().squeeze().numpy()
        plt.imshow(img, cmap='gray')
        plt.axis("off")
    plt.show()

def test_model(model):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    # Noise to generate images from
    x_t = torch.randn((1, IMG_CH, IMG_SIZE, IMG_SIZE), device=device)

    # Go from T to 0 removing and adding noise until t = 0
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device)
        e_t = model(x_t, t)  # Predicted noise
        x_t = reverse_q(x_t, t, e_t, B, pred_noise_coeff, sqrt_a_inv)
        plt.subplot(1, T+1, i+1)
        plt.axis('off')


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




data = load_transformed_fashionMNIST()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


model = UNet(IMG_CH, IMG_SIZE)
print("Num params: ", sum(p.numel() for p in model.parameters()))

optimizer = Adam(model.parameters(), lr=0.0001)
epochs = 3

model.train()
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        t = torch.randint(0, T, (BATCH_SIZE,), device=device)
        images = batch[0].to(device)

        images_noisy, noise = q(images, t, sqrt_a_bar, sqrt_one_minus_a_bar)
        noise_pred = model(images_noisy)
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0 and step % 100 == 0:
            print(f"Epoch {epoch} | Step {step:03d} Loss: {loss.item()} ")
            # plot_sample(images_pred)


sqrt_a_inv = torch.sqrt(1 / a)
pred_noise_coeff = (1 - a) / torch.sqrt(1 - a_bar)

test_model(model)