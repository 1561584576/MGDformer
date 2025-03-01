import torch
from torch import nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.maxpool_conv = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool_conv = nn.AvgPool1d(kernel_size=4, stride=4)
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.press_conv=nn.Conv1d(4, 1, kernel_size=17,padding=8)

    def random_sample(self, x):
        B,N,L=x.size()

        x=x.view(B,N,L//4,4)
        # 在最后一个维度随机选择索引
        random_indices = torch.randint(0, 4, (x.size(0), x.size(1), x.size(2), 1)).to('cuda')
        # 使用随机索引提取值
        sampled_values = x.gather(-1, random_indices).squeeze(-1)  # 在最后一个维度提取值

        return sampled_values

    def segment_sample(self, x):
        B,N,L=x.size()
        x=x.view(B,N,4,L//4)
        x=x.view(B*N,4,L//4)
        x=self.press_conv(x).view(B,N,L//4)
        return x


    def forward(self, x,slection):

        if slection=='r':
            x = self.random_sample(x)
        elif slection=='s':
            x = self.segment_sample(x)
        else:
            x = self.maxpool_conv(x)
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, linear=True):
        super().__init__()
        if linear:
            # self.up = nn.Upsample(scale_factor=4, mode='linear', align_corners=True)
            # self.up = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=4, stride=4)
            self.up = nn.Linear(128,512)
            self.up1 = nn.Linear(128,512)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # print(x1.shape)
        x1 = self.up(x1)
        # print(x1.shape)
        x2 = self.up1(x2)
        # input is CHW
        diffY = x2.size(2) - x1.size(2)
        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, linear=True):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.inc1 = DoubleConv(128, 64)
        self.down1 = Down(64, 64)
        self.down2 = Down(64, 64)
        self.down3 = Down(64, 64)
        # self.down4 = Down(512, 512)
        # self.up1 = Up(1024, 256, linear)
        self.up2 = Up(128, 64, linear)
        self.up3 = Up(128, 64, linear)
        self.up4 = Up(128, 64, linear)
        self.outc = OutConv(64, out_channels)
        self.linear = nn.Linear(512,128)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1,'r')
        x3 = self.down2(x1,'s')
        x4 = self.down3(x1,'m')
        # print(x4.shape)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x2, x3)

        x1=self.linear(x1)

        y = self.up3(x1, x4)

        # x = self.up4(x, y)
        x=torch.cat([x, y], dim=1)
        x = self.inc1(x)
        result = self.outc(x)
        return result

