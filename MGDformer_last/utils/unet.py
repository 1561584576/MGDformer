# # 康汝兵 哈哈哈
##版本1
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
        self.maxpool_conv = nn.MaxPool1d(kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool_conv(x)
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, linear=True):
        super().__init__()
        if linear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
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
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, linear)
        self.up2 = Up(512, 128, linear)
        self.up3 = Up(256, 64, linear)
        self.up4 = Up(128, 64, linear)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        result = self.outc(x)
        return result

# import torch
# from torch import nn
# import torch.nn.functional as F
# #版本2
# class DoubleConv(nn.Module):
#     """Independent convolution for each channel."""
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         # Create a separate conv block for each input channel
#         self.convs = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(1, 1, kernel_size=7, padding=3),
#                 nn.BatchNorm1d(1),
#                 nn.ReLU(inplace=True),
#                 # nn.Conv1d(1, 1, kernel_size=7, padding=3),
#                 # nn.BatchNorm1d(1),
#                 # nn.ReLU(inplace=True)
#             ) for _ in range(in_channels)
#         ])
#
#     def forward(self, x):
#         # Split along channel dimension (dim=1) to process each channel independently
#         # print("==================")
#         # print(x.shape)
#         channels = [conv(x[:, i:i+1, :]) for i, conv in enumerate(self.convs)]
#         # print(channels)
#         # print("--------------")
#         # Stack the processed channels back together along the channel dimension
#         return torch.cat(channels, dim=1)
#
# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.double_conv = DoubleConv(in_channels, out_channels)
#
#     def forward(self, x):
#         # print(x.shape)
#         x = self.maxpool_conv(x)
#         # print(x.shape)
#         # print("====")
#         return self.double_conv(x)
#
# class Up(nn.Module):
#     """Upscaling then double conv"""
#     def __init__(self, in_channels, out_channels, linear=True):
#         super().__init__()
#         if linear:
#             self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
#         self.conv = DoubleConv(in_channels, out_channels)
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size(2) - x1.size(2)
#         x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)
#
# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         return self.conv(x)
#
# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels, linear=True):
#         super(UNet, self).__init__()
#         self.inc = DoubleConv(in_channels, 64)
#         self.down1 = Down(in_channels, 128)
#         self.down2 = Down(in_channels, 256)
#         self.down3 = Down(in_channels, 512)
#         self.down4 = Down(in_channels, 512)
#         self.up1 = Up(in_channels, 256, linear)
#         self.up2 = Up(in_channels, 128, linear)
#         self.up3 = Up(in_channels, 64, linear)
#         self.up4 = Up(in_channels, 64, linear)
#         self.outc = OutConv(in_channels, out_channels)
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         result = self.outc(x)
#         return result
