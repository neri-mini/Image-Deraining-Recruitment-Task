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

class DerainNet(nn.Module):
    def __init__(self,raw_channels=3,inter_channels=64,num_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(raw_channels,inter_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.pool_half = nn.MaxPool2d(kernel_size=2,stride=2)
        self.pool_quarter = nn.MaxPool2d(kernel_size=4,stride=4)
        self.back_double = nn.ConvTranspose2d(inter_channels,inter_channels,kernel_size=2,stride=2)
        self.back_four = nn.ConvTranspose2d(inter_channels,inter_channels,kernel_size=4,stride=4)
        self.tail = nn.Sequential(
            nn.Conv2d(inter_channels * 3,inter_channels * 3,kernel_size=3,stride=1,padding=1,bias=False),
            nn.InstanceNorm2d(inter_channels * 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels * 3, raw_channels,kernel_size=3,stride=1,padding=1),
        )
        
        nn.init.zeros_(self.tail[-1].weight)
        nn.init.zeros_(self.tail[-1].bias)
        # 三个分支用不同的残差参数
        self.rest_full = nn.Sequential(*[ResidualBlock(channels=64) for _ in range(num_blocks)])
        self.rest_half = nn.Sequential(*[ResidualBlock(channels=64) for _ in range(num_blocks)])
        self.rest_quarter = nn.Sequential(*[ResidualBlock(channels=64) for _ in range(num_blocks)])
    def forward(self,x):
        x_0 = self.head(x)

        h_1 = x_0
        x_1 = self.rest_full(h_1) + h_1

        h_2 = self.pool_half(x_0)
        x_2 = self.back_double(self.rest_half(h_2) + h_2 )

        h_3 = self.pool_quarter(x_0)
        x_3 = self.back_four(self.rest_quarter(h_3) + h_3 )

        inter = torch.cat([x_1,x_2,x_3], dim= 1)
        rain = self.tail(inter)
        return x - rain
