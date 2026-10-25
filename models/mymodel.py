import torch
import torch.nn as nn
import torch.nn.functional as F

# 依旧残差块起手
class ResidualBlock(nn.Module):
        def __init__(self, channels=64):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,bias=False),
                nn.InstanceNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,bias=False),
                nn.InstanceNorm2d(channels),
            )

        def forward(self,x):
            y = self.body(x)
            return F.relu(x + y)


# 丐版的4方向累积注意力机制
class SPANet(nn.Module):
        def __init__(self,channels=64):
            super().__init__()
            self.conv1 = nn.Conv2d(channels * 4,channels,kernel_size=3,stride=1,padding=1)
            self.conv2 = nn.Conv2d(channels,1,kernel_size=3,stride=1,padding=1)
        def forward(self,x):
            right = torch.cumsum(x,dim=3)
            left = torch.cumsum(x.flip(3),dim=3).flip(3)
            down = torch.cumsum(x,dim=2)
            up = torch.cumsum(x.flip(2),dim=2).flip(2)    
            full = self.conv1(torch.cat([right,left,down,up],dim=1))
            a = torch.sigmoid(self.conv2(full))
            return x * (a + 1)       

class Mymodel(nn.Module):
        def __init__(self, raw_channels=3, inter_channels=64, num_blocks=8):
            super().__init__()
            # 初处理，把channels从3变成64
            self.head = nn.Conv2d(raw_channels,inter_channels, kernel_size=3, stride=1, padding=1)
            self.rest_full    = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
            self.rest_half    = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
            self.rest_quarter = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
            # 池化，造分支
            self.pool_half = nn.MaxPool2d(kernel_size=2,stride=2)
            self.pool_quarter = nn.MaxPool2d(kernel_size=4,stride=4)
            # 转置卷积，变回原图
            self.back_double = nn.ConvTranspose2d(inter_channels,inter_channels,kernel_size=2,stride=2)
            self.back_four = nn.ConvTranspose2d(inter_channels,inter_channels,kernel_size=4,stride=4)
            # 尾处理，把channels从64变成3
            self.tail = nn.Sequential(
                nn.Conv2d(inter_channels * 3,inter_channels,kernel_size=3,stride=1,padding=1,bias=False),
                nn.InstanceNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, raw_channels,kernel_size=3,stride=1,padding=1),
            )
            nn.init.zeros_(self.tail[-1].weight)
            nn.init.zeros_(self.tail[-1].bias)
            self.spa1 = SPANet()
            self.spa2 = SPANet()
            self.spa3 = SPANet()
        def forward(self,x):
            x_0 = self.head(x)

            h_1 = x_0
            b_1 = self.rest_full(h_1) + h_1
            x_1 = self.spa1(b_1)

            h_2 = self.pool_half(x_0)
            b_2 = self.back_double(self.rest_half(h_2) + h_2 )
            x_2 = self.spa2(b_2)

            h_3 = self.pool_quarter(x_0)
            b_3 = self.back_four(self.rest_quarter(h_3) + h_3 )
            x_3 = self.spa3(b_3)

            inter = torch.cat([x_1,x_2,x_3], dim= 1)
            rain = self.tail(inter)
            return x - rain

                