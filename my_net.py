import torch 
import torch.nn as nn

# 总参数量75456
class mynet(nn.Module):
    def __init__(self, channels, num_layers = 10):
        super(mynet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 32
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size,padding=padding))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)