from math import sqrt

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
from einops import rearrange, repeat

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
        # self.inner_attention = VarCorAttention(scale=None, attention_dropout = dropout)
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


class AbsMaxPool1dWithValue(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(AbsMaxPool1dWithValue, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # 计算输入的绝对值
        abs_x = torch.abs(x)

        # 使用max_pool1d找到最大绝对值的位置
        # 返回的索引是max_pool1d在绝对值上的位置
        abs_max, abs_max_idx = F.max_pool1d(abs_x, self.kernel_size, self.stride, self.padding, return_indices=True)

        # 使用索引找到原始输入值
        output = torch.gather(x, 2, abs_max_idx)  # abs_max_idx 是在绝对值中的最大位置，gather用它来索引原始值

        return output

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, configs,num, flag):
        super().__init__()

        self.down_sample = nn.Sequential(nn.Linear(4, 8),nn.GELU()
                                         , nn.Linear(8, 4),nn.GELU()
                                         , nn.Linear(4, 1))
        self.down_sample_1=nn.AvgPool1d(4)
        self.down_sample_2=nn.MaxPool1d(4)
        self.down_sample_3=AbsMaxPool1dWithValue(4)

        self.attention = AttentionLayer(configs.d_model//(4**num), 4, dropout=0.2)

        self.flag=flag


    def forward(self, x):#B N L

        m=x.view(x.size(0)*x.size(1),x.size(2)//4,4)

        if self.flag==0:
            n=self.down_sample_1(m).squeeze(-1)

        else:
            n=self.down_sample_1(m).squeeze(-1)

        m=n=n.view(x.size(0),x.size(1),x.size(2)//4)

        z = self.attention(n, n, n)


        return z,m

class Up1(nn.Module):
    def __init__(self,configs,num):
        super().__init__()
        self.num=num
        if num==0:

            self.linear = nn.Sequential(
                nn.Linear(configs.d_model // 4+configs.d_model, configs.d_model)
                ,nn.BatchNorm1d(configs.enc_in)
                ,nn.GELU(), nn.Linear(configs.d_model, configs.d_model))

        else:
            self.linear=nn.Sequential(nn.Linear(configs.d_model//(4**(num+1))+configs.d_model//4**num,configs.d_model//4)
                                      ,nn.BatchNorm1d(configs.enc_in)
                                      ,nn.GELU(),nn.Linear(configs.d_model//4,configs.d_model//4**num))

    def forward(self, x1, x2):


        x=torch.cat([x1,x2], dim=-1)
        # print(x.shape)
        x=self.linear(x)
        return x



class UNet(nn.Module):
    def __init__(self, configs):
        super(UNet, self).__init__()
        self.down1 = Down(configs,1,0 )
        self.down2 = Down(configs, 2,1)
        self.down3 = Down(configs, 3,1)

        self.up1 = Up1(configs,0)
        self.up2 = Up1(configs,1)
        self.up3 = Up1(configs,2)

        self.outc3 = nn.Linear(512,configs.pred_len)
        self.attention = AttentionLayer(configs.d_model, 4, dropout=0.2)


    def forward(self, x):

        x1 = x
        x2,x22 = self.down1(x1)
        x3,x33 = self.down2(x22)
        x4, _ = self.down3(x33)

        m=self.up3(x3,x4)
        m=m+x33
        n=self.up2(x2,m)
        n=n+x22
        k=self.up1(x1,n)

        return k



