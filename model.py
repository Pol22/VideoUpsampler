import torch.nn as nn


class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU(out_channels)
        )

    def forward(self, x):
        return x + self.conv1(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', 
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(self.up(x))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UNet, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)
        self.inc = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv1 = ResConv(32, 32)
        self.down1 = Down(32, 64)
        self.conv2 = ResConv(64, 64)
        self.down2 = Down(64, 128)
        self.conv3 = ResConv(128, 128)
        self.down3 = Down(128, 256)

        self.mid_conv = ResConv(256, 256)

        self.up1 = Up(256, 128, bilinear)
        self.up_conv1 = ResConv(128, 128)
        self.up2 = Up(128, 64, bilinear)
        self.up_conv2 = ResConv(64, 64)
        self.up3 = Up(64, 32, bilinear)
        self.up_conv3 = ResConv(32, 32)
        self.outc = OutConv(32, out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.inc(x)
        x1 = self.conv1(x)
        x = self.down1(x1)
        x2 = self.conv2(x)
        x = self.down2(x2)
        x3 = self.conv3(x)
        x = self.down3(x3)
        x = self.mid_conv(x)
        x = self.up1(x)
        x = self.up_conv1(x + x3)
        x = self.up2(x)
        x = self.up_conv2(x + x2)
        x = self.up3(x)
        x = self.up_conv3(x + x1)
        return self.outc(x)
