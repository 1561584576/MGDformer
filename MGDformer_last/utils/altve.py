from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F

class FullAttention(nn.Module):
    '''
    The Attention operation
    '''

    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()

class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout = 0.2):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout = dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape       #L：段数量
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = out.view(B, L, -1)

        return self.out_projection(out)

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
    def __init__(self, configs,num, in_channels, out_channels):
        super().__init__()
        # self.maxpool_conv = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool_conv = nn.AvgPool1d(kernel_size=4, stride=4)
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample=nn.Sequential(nn.Linear(configs.d_model//(4**num),256),nn.ReLU(inplace=True)
                                       ,nn.Linear(256,128),nn.ReLU(inplace=True)
                                       ,nn.Linear(128,configs.d_model//(4**num)),
                                       nn.Linear(configs.d_model//(4**num),configs.d_model//(4**(num+1))))

        self.attention = AttentionLayer(configs.d_model//(4**num), 4, dropout=0.2)

    def forward(self, x):#B N L
        # x = x.view(x.size(0),x.size(1),x.size(2)//4,4)
        # print(x.shape)
        y=x.view(x.size(0)*x.size(1),4,x.size(2)//4)
        # print(y.shape)
        z=self.down_sample(y)
        # print(z.shape)
        z=z.view(x.size(0),x.size(1),x.size(2)//4)
        # print(z.shape)
        attention_out = self.attention(z, z, z)
        # print(attention_out.shape)
        return attention_out

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, linear=True):
        super().__init__()

    def forward(self, x1, x2):

        x=torch.cat([x1,x2], dim=-1)


        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, configs, in_channels, out_channels, linear=True):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(configs,1, 64, 128)
        self.down2 = Down(configs, 2,128, 256)
        self.down3 = Down(configs, 3,256, 256)
        # self.down4 = Down(512, 512)
        # self.up1 = Up(1024, 256, linear)
        self.up2 = Up(512, 128, linear)
        self.up3 = Up(256, 64, linear)
        self.up4 = Up(128, 64, linear)
        self.outc = nn.Sequential(nn.Linear(680,1024),nn.LayerNorm(1024),nn.ReLU(),nn.Linear(1024,512)
                                  ,nn.LayerNorm(512),nn.ReLU(),nn.Linear(512,256),nn.LayerNorm(256)
                                  ,nn.ReLU(),nn.Linear(256,128),nn.LayerNorm(128),
                                  nn.ReLU(),nn.Linear(128,configs.pred_len))

        self.outc1 = nn.Sequential(nn.Linear(680, 1024), nn.BatchNorm1d(in_channels), nn.ReLU(), nn.Linear(1024, 512)
                                  , nn.BatchNorm1d(in_channels), nn.ReLU(), nn.Linear(512, 256), nn.BatchNorm1d(in_channels)
                                  , nn.ReLU(), nn.Linear(256, 128), nn.BatchNorm1d(in_channels),
                                  nn.ReLU(), nn.Linear(128, configs.pred_len))
        self.attention = AttentionLayer(configs.d_model, 4, dropout=0.2)

    def forward(self, x):
        # attention_out = self.attention(x, x, x)
        x1 = x
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        result = self.outc1(x)
        return result

