import torch 
import torch.nn as nn
import torch.nn.functional as F

class single_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class up(nn.Module):
    def __init__(self, in_channels):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size = 2, stride = 2)
    
    # 残差连接
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = x1 + x2
        return x

class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        
        self.encoder1 = nn.Sequential(
            single_conv(3, 64),
            single_conv(64, 64)
        )
        self.down1 = nn.AvgPool2d(2)
        self.encoder2 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128)
        )
        self.down2 = nn.AvgPool2d(2)
        self.bottleneck = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
        )
        self.up1 = up(256)
        self.decoder1 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )
        self.up2 = up(128)
        self.decoder2 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )
        self.outc = nn.Conv2d(64, 3, 1)


    def forward(self, x):
        inx = self.encoder1(x)

        down1 = self.down1(inx)
        conv1 = self.encoder2(down1)

        down2 = self.down2(conv1)
        conv2 = self.bottleneck(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.decoder1(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.decoder2(up2)

        out = self.outc(conv4)
        return out