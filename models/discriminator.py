import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Use "disc" as the sequential name to match saved model
        self.disc = nn.Sequential(
            # Layer 0: Conv2d 3->64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: Conv2d 64->128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 5: Conv2d 128->256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 8: Conv2d 256->512
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 11: Conv2d 512->1
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # Layer 12: Sigmoid
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.disc(x)
        return x.view(-1, 1).squeeze(1)