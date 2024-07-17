import torch 
import torch.nn as nn
import utils

# 总参数量75456
class mynet(nn.Module):
    def __init__(self, num_layers = 10):
        super(mynet, self).__init__()
        
        kernel_size = 3
        padding = 1
        downsampled_channels = 15
        output_features = 12
        features = 32

        layers = []
        layers.append(nn.Conv2d(in_channels=downsampled_channels, out_channels=features, kernel_size=kernel_size, padding=padding))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size,padding=padding))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=output_features, kernel_size=kernel_size, padding=padding))
        self.net = nn.Sequential(*layers)

    def forward(self, x, noise_sigma):
        noise_map = noise_sigma.view(x.shape[0], 1, 1, 1).repeat(1, x.shape[1], x.shape[2] // 2, x.shape[3] // 2)
        
        x_up = utils.downsample(x.data) # 4 * C * H/2 * W/2
        x_cat = torch.cat((noise_map.data, x_up), 1) # 4 * (C + 1) * H/2 * W/2

        h_dncnn = self.net(x_cat)
        y_pred = utils.upsample(h_dncnn)

        return y_pred