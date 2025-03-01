from math import sqrt

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
import torch.nn.functional as F
from layers.Embed import DataEmbedding_inverted,Dimission_Embedding



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
        self.down_sample=nn.Sequential(nn.Linear(configs.d_model//(4**num),256),nn.GELU()
                                       ,nn.Linear(256,128),nn.GELU()
                                       ,nn.Linear(128,configs.d_model//(4**num)),
                                       nn.Linear(configs.d_model//(4**num),configs.d_model//(4**(num+1))))
        self.down_sample1 = nn.Sequential(nn.Linear(4, 8),nn.GELU()
                                         , nn.Linear(8, 4),nn.GELU()
                                         , nn.Linear(4, 1))
        self.down_sample2 = nn.Sequential(nn.Linear(2*configs.d_model//(4**num), 128),nn.GELU(),
                                          nn.Linear(128,64),nn.GELU(),
                                          nn.Linear(64,configs.d_model//(4**num)))

        self.attention0 = AttentionLayer(configs.d_model//(4**num), 4, dropout=0.2)
        self.attention1 = AttentionLayer(configs.d_model//(4**num), 4, dropout=0.2)
        self.attention2 = AttentionLayer(configs.d_model//(4**num), 4, dropout=0.2)
        self.para=nn.Parameter(torch.rand(1))

    def forward(self, x):#B N L

        m=x.view(x.size(0)*x.size(1),x.size(2)//4,4)

        n=self.down_sample1(m).squeeze(-1)

        m=n=n.view(x.size(0),x.size(1),x.size(2)//4)

        # m=n=self.maxpool_conv(x)
        n = self.attention1(n, n, n)
        return n,m

class Down_Time(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, configs,num, in_channels, out_channels):
        super().__init__()
        # self.maxpool_conv = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool_conv = nn.AvgPool1d(kernel_size=4, stride=4)
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample=nn.Sequential(nn.Linear(configs.d_model//(4**num),256),nn.GELU()
                                       ,nn.Linear(256,128),nn.GELU()
                                       ,nn.Linear(128,configs.d_model//(4**num)),
                                       nn.Linear(configs.d_model//(4**num),configs.d_model//(4**(num+1))))
        self.down_sample1 = nn.Sequential(nn.Linear(4, 8),nn.GELU()
                                         , nn.Linear(8, 4),nn.GELU()
                                         , nn.Linear(4, 1))
        self.down_sample2 = nn.Sequential(nn.Linear(2*configs.d_model//(4**num), 128),nn.GELU(),
                                          nn.Linear(128,64),nn.GELU(),
                                          nn.Linear(64,configs.d_model//(4**num)))

        self.attention0 = AttentionLayer(configs.d_model//(4**num), 4, dropout=0.2)
        self.attention1 = AttentionLayer(configs.d_model//(4**num), 4, dropout=0.2)
        self.attention2 = AttentionLayer(configs.d_model//(4**num), 4, dropout=0.2)
        self.para=nn.Parameter(torch.rand(1))

    def forward(self, x):#B N L

        m=x.view(x.size(0)*x.size(1),4,x.size(2)//4)
        # print(m.shape)
        m=self.attention0(m,m,m).view(x.size(0),x.size(1),x.size(2))

        l=x.view(x.size(0)*x.size(1),x.size(2)//4,4)
        n=self.down_sample1(l).squeeze(-1)

        n=n.view(x.size(0),x.size(1),x.size(2)//4)

        return m,n

class Up_Time(nn.Module):
    def __init__(self,configs,num):
        super().__init__()
        self.num=num
        if num==0:
            # self.linear = nn.Linear(configs.d_model // 4 + configs.d_model ,configs.d_model)
            self.linear = nn.Sequential(
                nn.Linear(configs.d_model+configs.d_model, configs.d_model)
                # nn.Linear(configs.d_model // 4, configs.d_model)
                ,nn.BatchNorm1d(configs.enc_in)
                ,nn.GELU(), nn.Linear(configs.d_model, configs.d_model))

        else:
            # self.linear=nn.Linear(configs.d_model//(4**(num+1))+configs.d_model//4,configs.d_model//4)
            self.linear=nn.Sequential(nn.Linear(configs.d_model//(4**(num-1))+configs.d_model//4**num,configs.d_model)
                                      ,nn.BatchNorm1d(configs.enc_in)
                                      ,nn.GELU(),nn.Linear(configs.d_model,configs.d_model//4**(num-1)))
        self.attention0 = AttentionLayer(configs.d_model // (4 ** num), 4, dropout=0.2)

    def forward(self, x1, x2):


        x=torch.cat([x1,x2], dim=-1)
        # print(x.shape)
        x=self.linear(x)
        return x

class Up1(nn.Module):
    def __init__(self,configs,num):
        super().__init__()
        self.num=num
        if num==0:
            # self.linear = nn.Linear(configs.d_model // 4 + configs.d_model ,configs.d_model)
            self.linear = nn.Sequential(
                nn.Linear(configs.d_model // 4+configs.d_model, configs.d_model)
                # nn.Linear(configs.d_model // 4, configs.d_model)
                ,nn.BatchNorm1d(configs.enc_in)
                ,nn.GELU(), nn.Linear(configs.d_model, configs.d_model))

        else:
            # self.linear=nn.Linear(configs.d_model//(4**(num+1))+configs.d_model//4,configs.d_model//4)
            self.linear=nn.Sequential(nn.Linear(configs.d_model//(4**(num+1))+configs.d_model//4**num,configs.d_model//4)
                                      ,nn.BatchNorm1d(configs.enc_in)
                                      ,nn.GELU(),nn.Linear(configs.d_model//4,configs.d_model//4**num))
        self.attention0 = AttentionLayer(configs.d_model // (4 ** num), 4, dropout=0.2)

    def forward(self, x1, x2):

        # if self.num==0:
        #     x=x1+x2
        #     x=self.linear(x)
        #     return x

        x=torch.cat([x1,x2], dim=-1)
        # print(x.shape)
        x=self.linear(x)
        return x
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

        self.down_1 = Down_Time(configs, 1, 64, 128)
        self.down_2 = Down_Time(configs, 2, 128, 256)
        self.down_3 = Down_Time(configs, 3, 256, 256)

        # self.down4 = Down(512, 512)
        # self.up1 = Up(1024, 256, linear)
        self.up1 = Up1(configs,0)
        self.up2 = Up1(configs,1)
        self.up3 = Up1(configs,2)

        self.up_1 = Up_Time(configs, 0)
        self.up_2 = Up_Time(configs, 1)
        self.up_3 = Up_Time(configs, 2)

        self.outc2 = nn.Sequential(nn.Linear(512, 1024),nn.LayerNorm(1024),nn.GELU(), nn.Linear(1024, 512)
                                  ,nn.LayerNorm(512), nn.GELU(), nn.Linear(512, 256),nn.LayerNorm(256)
                                  ,nn.GELU(), nn.Linear(256, 128),nn.LayerNorm(128)
                                  ,nn.GELU(), nn.Linear(128, configs.pred_len))
        self.outc3 = nn.Linear(512,configs.pred_len)
        self.outc4 = nn.Linear(512,configs.seq_len)

        self.attention = AttentionLayer(configs.d_model, 4, dropout=0.2)
        self.attention_o = AttentionLayer(configs.pred_len, 4, dropout=0.2)
        self.catagory= np.floor(sqrt(configs.enc_in)).astype(int) if np.floor(sqrt(configs.enc_in)).astype(int)<10 else 10
        # self.catagory= 2
        self.router = nn.Parameter(torch.randn(1, self.catagory, configs.d_model))
        self.out_router = nn.Parameter(torch.randn(1, configs.enc_in, configs.d_model))
        self.result = nn.Parameter(torch.randn(1, configs.enc_in, configs.pred_len))
        self.linear = nn.ModuleList([nn.Linear(configs.d_model, configs.pred_len) for _ in range(self.catagory)])

        self.avgpool_conv = nn.AvgPool1d(kernel_size=2, stride=2)

        self.predicts = []
        for _ in range(self.catagory):
            self.predicts.append(nn.Linear(configs.d_model, configs.pred_len).to("cuda:0"))


        self.avgpool_conv = nn.AvgPool1d(kernel_size=4, stride=4)
        self.param=nn.Parameter(torch.randn(1))
        self.lineara_predict = nn.Linear(configs.batch_size*configs.seq_len, configs.seq_len)
        self.attention0 = AttentionLayer(configs.d_model // 4 , 4, dropout=0.2)

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scale = 0.02

        self.embed_size = self.seq_len
        self.hidden_size = configs.d_model

        self.w = nn.Parameter(self.scale * torch.randn(1, self.embed_size))
        self.w1 = nn.Parameter(torch.randn(self.embed_size//2 + 1))
        self.w2 = nn.Parameter(self.scale * torch.randn(self.embed_size//2 + 1))
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.conv1 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=configs.d_ff, out_channels=configs.d_model, kernel_size=1)
        self.activation = F.gelu
        self.norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(0.1)

    def circular_convolution(self, x, w):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(w, dim=1, norm='ortho')
        y = x * w
        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out

    def circular_convolution_1(self, x, w):
        x = torch.fft.rfft(x, dim=2, norm='ortho')

        y = x * w


        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out

    def circular_convolution_2(self, x, w):
        x = torch.fft.rfft(x, dim=2, norm='ortho')

        w = self.sigmoid(w)

        y = x * w.round()

        # print(x[-1, -1, :])
        # print(w.round())
        # print(y[-1, -1, :])

        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out

    def forward(self, x,y):

        x1 = x
        x2,x22 = self.down1(x1)
        x3,x33 = self.down2(x22)
        x4, _ = self.down3(x33)

        m=self.up3(x3,x4)
        m=m+x3
        # print(m.shape)
        n=self.up2(x2,m)
        n=n+x2
        k=self.up1(x1,n)
        z=k=k+x

        k = self.dropout(self.activation(self.conv1(k.transpose(-1, 1))))
        k = self.dropout(self.conv2(k).transpose(-1, 1))

        result = self.outc3(self.norm(k+z))



        return result


