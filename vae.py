import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Config
data_dir = "/home/groups/comp3710/OASIS/"
batch_size, epochs, z_dim = 16, 2, 20
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# VAE model
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(28*28,128), nn.ReLU())
        self.mu = nn.Linear(128,z_dim); self.logvar = nn.Linear(128,z_dim)
        self.dec = nn.Sequential(nn.Linear(z_dim,128), nn.ReLU(), nn.Linear(128,28*28), nn.Sigmoid())
    def forward(self,x):
        h=self.enc(x); mu, logvar=self.mu(h), self.logvar(h)
        z=mu+torch.exp(0.5*logvar)*torch.randn_like(mu)
        return self.dec(z).view(-1,1,28,28), mu, logvar

vae=VAE()
opt=optim.Adam(vae.parameters(), lr=1e-3)
bce=nn.BCELoss(reduction="sum")

for ep in range(epochs):
    for x,_ in dataloader:
        recon,mu,logvar=vae(x)
        loss=bce(recon,x)+( -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp()) )
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"Epoch {ep+1}, Loss {loss.item():.2f}")

