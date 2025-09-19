# gan.py - Hard Task: Quick GAN (1 epoch, saves 1 fake image)

import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import os

# Settings
epochs, z_dim = 1, 64
os.makedirs("gan_results", exist_ok=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(),
            nn.Linear(128, 784), nn.Sigmoid()
        )
    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# Init
G, D = Generator(), Discriminator()
optG, optD = optim.Adam(G.parameters(), 1e-3), optim.Adam(D.parameters(), 1e-3)
loss_fn = nn.BCELoss()

# Quick training loop
for ep in range(epochs):
    for _ in range(10):  # tiny loop just to make G learn something
        z = torch.randn(16, z_dim)
        fake = G(z)

        # Train Discriminator (fake only, quick hack)
        D_fake = D(fake.detach())
        lossD = loss_fn(D_fake, torch.zeros_like(D_fake))
        optD.zero_grad(); lossD.backward(); optD.step()

        # Train Generator
        D_fake = D(fake)
        lossG = loss_fn(D_fake, torch.ones_like(D_fake))
        optG.zero_grad(); lossG.backward(); optG.step()

    # Save fake image after 1 epoch
    plt.imshow(fake[0,0].detach(), cmap="gray")
    out_path = f"gan_results/fake_epoch{ep+1}.png"
    plt.savefig(out_path)
    print(f"✅ {out_path} saved")

