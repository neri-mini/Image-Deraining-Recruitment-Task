import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        y = self.body(x)
        return F.relu(x + y)      

class BaselineNet(nn.Module):
    """8 个残差块的 ResNet"""
    def __init__(self, in_channels=3, out_channels=3, channels=64, num_blocks=8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels,channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),    
        )

        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
             
        self.tail = nn.Conv2d(channels,out_channels,kernel_size=3,stride=1,padding=1)  
        # tail 零初始化
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        step1 = self.head(x)
        step2 = self.body(step1)
        rain = self.tail(step2)
        out = x - rain
        return out           