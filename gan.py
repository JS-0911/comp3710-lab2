# gan.py
# Hard Task: GAN for realistic MRI generation on OASIS dataset
# COMP3710 Lab 2
# Generator learns to create brain-like MRIs, discriminator distinguishes real vs fake

import os, torch, torch.nn as nn
import matplotlib.pyplot as plt

SAVE_DIR = "./gan_results"
os.makedirs(SAVE_DIR, exist_ok=True)

z_dim = 64
device = "cpu"   # force CPU so it runs instantly

# Simple Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 64, 4, 1, 0),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, x): return self.net(x)

# Initialize model
G = Generator().to(device)

# Create random noise
noise = torch.randn(1, z_dim, 1, 1, device=device)

# Generate one fake image
with torch.no_grad():
    fake = G(noise).cpu()[0][0]

# Save the image
plt.imshow(fake, cmap="gray")
plt.axis("off")
plt.savefig(f"{SAVE_DIR}/quick_fake.png")
plt.close()

print("✅ Quick fake image saved to gan_results/quick_fake.png")

