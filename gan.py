import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DATA_DIR = "/home/groups/comp3710/OASIS/"
SAVE_DIR = "./gan_results"
os.makedirs(SAVE_DIR, exist_ok=True)

image_size = 32   # smaller images = faster training
batch_size = 32
z_dim = 64
epochs = 3
lr = 0.0002
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATA ----------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---------------- MODELS ----------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 128, 4, 1, 0),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, x): return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 0), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

G, D = Generator().to(device), Discriminator().to(device)
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))

fixed_noise = torch.randn(16, z_dim, 1, 1, device=device)
G_losses, D_losses = [], []

# ---------------- TRAINING ----------------
for epoch in range(epochs):
    for real, _ in dataloader:
        real = real.to(device)
        bs = real.size(0)

        # Train D
        noise = torch.randn(bs, z_dim, 1, 1, device=device)
        fake = G(noise)
        D_loss = (criterion(D(real).view(-1), torch.ones(bs, device=device)) +
                  criterion(D(fake.detach()).view(-1), torch.zeros(bs, device=device)))
        opt_D.zero_grad(); D_loss.backward(); opt_D.step()

        # Train G
        G_loss = criterion(D(fake).view(-1), torch.ones(bs, device=device))
        opt_G.zero_grad(); G_loss.backward(); opt_G.step()

    G_losses.append(G_loss.item()); D_losses.append(D_loss.item())

    with torch.no_grad():
        fake = G(fixed_noise).cpu()
        pl

