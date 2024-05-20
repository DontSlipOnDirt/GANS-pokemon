import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # 128x128x3 -> 64x64x64
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64x64 -> 32x32x128
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32x128 -> 16x16x256
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16x256 -> 8x8x512
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(512, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8x512 -> 4x4x1024
            nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(1024, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        
        self.init_size = 8  # Initial size before upsampling
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 1024 * self.init_size * self.init_size)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(1024, momentum=0.1, eps=0.8),

            # 8x8x1024 -> 16x16x512
            nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=0.1, eps=0.8),
            nn.ReLU(inplace=True),

            # 16x16x512 -> 32x32x256
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.1, eps=0.8),
            nn.ReLU(inplace=True),

            # 32x32x256 -> 64x64x128
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.1, eps=0.8),
            nn.ReLU(inplace=True),

            # 64x64x128 -> 128x128x64
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1, eps=0.8),
            nn.ReLU(inplace=True),

            # 128x128x64 -> 128x128x3
            nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 1024, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
