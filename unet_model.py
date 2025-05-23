import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        # Simplified U-Net for demonstration
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.pool = nn.MaxPool2d(2, 2)
        self.enc2 = conv_block(64, 128)
        self.dec1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.pool(e1)
        e2 = self.enc2(e2)
        d1 = self.dec1(e2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.conv1(d1)
        out = self.final(d1)
        return out