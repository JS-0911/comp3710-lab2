# unet.py - Medium Task: UNet (1 epoch, saves output mask)

import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

data_dir = "/home/groups/comp3710/OASIS/"
batch_size, epochs = 2, 1
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64,64)),
    transforms.ToTensor()
])
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1=nn.Sequential(nn.Conv2d(1,16,3,1,1),nn.ReLU(),nn.MaxPool2d(2))
        self.down2=nn.Sequential(nn.Conv2d(16,32,3,1,1),nn.ReLU(),nn.MaxPool2d(2))
        self.up1=nn.ConvTranspose2d(32,16,2,2)
        self.out=nn.Conv2d(16,1,1)
    def forward(self,x):
        d1=self.down1(x); d2=self.down2(d1); u=self.up1(d2)
        return torch.sigmoid(self.out(u))

model=UNet()
opt=optim.Adam(model.parameters(), lr=1e-3)
loss_fn=nn.BCELoss()

for ep in range(epochs):
    x,_ = next(iter(dataloader))
    yhat=model(x)
    y=torch.ones_like(yhat)
    loss=loss_fn(yhat,y)
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"Epoch {ep+1}, Loss {loss.item():.3f}")

x,_=next(iter(dataloader))
mask=model(x).detach()[0,0]
plt.subplot(1,2,1); plt.imshow(x[0,0],cmap="gray"); plt.title("MRI")
plt.subplot(1,2,2); plt.imshow(mask,cmap="gray"); plt.title("Predicted Mask")
plt.savefig("segmentation.png"); print("✅ segmentation.png saved")

