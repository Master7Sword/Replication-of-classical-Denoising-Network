import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, bias = True):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

# 下采样，(H,W) -> (H/2, W/2)
def conv_down(in_channels, out_channels, bias = False):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride = 2, padding = 1, bias = bias)
    return layer

class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, slope):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(slope, inplace=True)
        )
        if downsample:
            self.downsample = conv_down(out_channels, out_channels, bias=False)

    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_channels, out_channels, downsample=False, slope=slope)

    def forward(self, x, bridge):
        up = self.up(x)
        if up.dim() == 3:
            up = up.unsqueeze(0)
        if bridge.dim() == 3:
            bridge = bridge.unsqueeze(0)
        out = torch.cat([up, bridge], 1) # concatenate in channels
        # print(f"[out] {out.shape}")
        out = self.conv_block(out)

        return out

class DiscriminatorLinear(nn.Module):
    def __init__(self, in_channels, filters = 64, slope = 0.2):
        super(DiscriminatorLinear, self).__init__()
        self.filters = filters

        # input (N, C, 128, 128)
        self.main = nn.Sequential(
            conv_down(in_channels, filters, bias=False),
            nn.LeakyReLU(slope, inplace=True),
            conv_down(filters, filters*2, bias=False),
            nn.LeakyReLU(slope, inplace=True),
            conv_down(filters*2, filters*4, bias=False),
            nn.LeakyReLU(slope, inplace=True),
            conv_down(filters*4, filters*8, bias=False),
            nn.LeakyReLU(slope, inplace=True),
            conv_down(filters*8, filters*16,bias=False),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(filters*16,filters*32, kernel_size = 4,stride = 1, padding = 0, bias=False),
            nn.LeakyReLU(slope, inplace=True),
        )
        self.output = nn.Linear(filters*32, 1)

        self._initialize()
    
    def forward(self, x):
        feature = self.main(x)
        feature = feature.view(-1,self.filters*32)
        out = self.output(feature)
        return out.view(-1)
    
    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.normal_(m.weight.data, 0., 0.02)
                if not m.bias is None:
                    init.constant_(m.bias, 0)
    
class UNetD(nn.Module):
    """
    Denoiser Network
    """
    def __init__(self, in_channels, filters=32, depth=5, slope=0.2):
        super(UNetD, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        prev_channels = self.get_input_chn(in_channels)
        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*filters, downsample, slope))
            prev_channels = (2**i)*filters

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2**i)*filters, slope))
            prev_channels = (2**i)*filters
        
        self.last = conv3x3(prev_channels, in_channels, bias=True)

    def forward(self, x1):
        blocks = []
        for i, down in enumerate(self.down_path):
            if(i+1) < self.depth:
                x1, x1_up = down(x1)
                blocks.append(x1_up)
            else:
                x1 = down(x1)
        
        for i, up in enumerate(self.up_path):
            x1 = up(x1, blocks[-i-1])
        
        out = self.last(x1)
        return out
    
    def get_input_chn(self, in_channels):
        return in_channels
    
    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

class UNetG(UNetD):
    def __init__(self, in_channels, filters=32, depth=5, slope=0.20):
        super(UNetG, self).__init__(in_channels, filters, depth, slope)

    def get_input_chn(self, in_channels):
        return in_channels+1
    
def sample_generator(netG, x):
    z = torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]], device=x.device)
    x1 = torch.cat([x, z], dim=1)
    out = netG(x1)

    return out+x
