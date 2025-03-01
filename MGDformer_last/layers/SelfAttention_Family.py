import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask

from einops import rearrange

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class VarCorAttention(nn.Module):
    def __init__(self, args, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,dmodel=512) -> None:
        super(VarCorAttention, self).__init__()

        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.enc_in=args.enc_in
        self.para=nn.Parameter(torch.rand((dmodel//2+1)//10))
        self.para2=nn.Parameter(torch.rand((self.enc_in,self.enc_in)))

    def origin_compute_cross_cor(self, queries, keys):
        q_fft = torch.fft.rfft(queries, dim=-1)
        k_fft = torch.fft.rfft(keys, dim=-1)

        res = q_fft*k_fft
        corr = torch.fft.irfft(res, dim=-1)
        corr = corr.mean(dim=-1)
        return corr

    def compute_cross_cor(self, queries, keys):

        q_fft = torch.fft.rfft(queries, dim=-1)
        k_fft = torch.fft.rfft(keys, dim=-1)
        q_fft = q_fft.unsqueeze(1)  # [D,1,T/2+1]
        k_fft = torch.conj(k_fft.unsqueeze(0))  # [1,D,T/2+1]
        res = q_fft*k_fft  # [D,D,T/2+1]
        corr = torch.fft.irfft(res, dim=-1)
        corr = corr.mean(dim=-1)

        return corr

    def optimized_compute_cross_cor(self, queries, keys):
        # Perform batched FFT
        q_fft = torch.fft.rfft(queries, dim=-1)
        k_fft = torch.fft.rfft(keys, dim=-1)

        # Expand dimensions for broadcasting
        q_fft = q_fft.unsqueeze(2)  # [B, D, 1, T/2+1]
        k_fft = torch.conj(k_fft.unsqueeze(1))  # [B, 1, D, T/2+1]

        # Element-wise multiplication and batched inverse FFT
        res = q_fft * k_fft  # [B, D, D, T/2+1]
        corr = torch.fft.irfft(res, dim=-1)

        # Mean across the time dimension
        corr = corr.mean(dim=-1)

        return corr

    def roll_optimized_compute_cross_cor(self, queries, keys):
        # Perform batched FFT
        q_fft = torch.fft.rfft(queries, dim=-1)
        k_fft = torch.fft.rfft(keys, dim=-1)
        # Expand dimensions for broadcasting
        q_fft = q_fft.unsqueeze(2)  # [B, D, 1, T/2+1]
        res=torch.zeros(k_fft.shape[0],k_fft.shape[1],k_fft.shape[1],k_fft.shape[2]).to(queries.device)


        for i in range(k_fft.shape[-1]//10):
            k_fft=torch.roll(k_fft, shifts=10, dims=-1)

            k_fft_1 = torch.conj(k_fft.unsqueeze(1))  # [B, 1, D, T/2+1]

            res = q_fft * k_fft_1*self.para[i]+res  # [B, D, D, T/2+1]

        corr = torch.fft.irfft(res, dim=-1)

        # Mean across the time dimension
        corr = corr.mean(dim=-1)

        return corr

    def roll_optimized_compute_cross_cor_test(self, queries, keys):
        # Perform batched FFT
        q_fft = torch.fft.rfft(queries, dim=-1)
        k_fft = torch.fft.rfft(keys, dim=-1)
        # Expand dimensions for broadcasting
        q_fft = q_fft.unsqueeze(2)  # [B, D, 1, T/2+1]
        k_fft = k_fft.unsqueeze(2)

        # 存储每次roll操作后的结果
        rolled_output1 = [k_fft]  # 初始包含原始k_fft
        rolled_output2 = [q_fft]  # q_fft

        # 执行10次roll操作并拼接结果
        for i in range(k_fft.shape[-1]//10-1):
            rolled_k_fft = torch.roll(k_fft, shifts=10, dims=-1)
            rolled_output1.append(rolled_k_fft)
            rolled_output2.append(q_fft)
        # 在第三个维度上拼接所有结果
        k_fft_output = torch.cat(rolled_output1, dim=2).unsqueeze(1) # [B, 1, D,  ,T/2+1]
        # print(k_fft_output.shape)
        # print(self.para.shape)
        k_fft_output = self.para.view(1,1,1,k_fft.shape[-1]//10,1)*k_fft_output
        q_fft_output = torch.cat(rolled_output2, dim=2).unsqueeze(2) # [B, D, 1,  ,T/2+1]
        # print(q_fft_output.shape)
        corr=k_fft_output*q_fft_output
        corr=torch.sum(corr, dim=3).squeeze(3)
        corr = torch.fft.irfft(corr, dim=-1)

        corr = corr.mean(dim=-1)

        return corr

    def roll_optimized_compute_cross_cor_test_1(self, queries, keys):
        # Perform batched FFT
        q_fft = torch.fft.rfft(queries, dim=-1)
        k_fft = torch.fft.rfft(keys, dim=-1)
        # Expand dimensions for broadcasting
        q_fft = q_fft.unsqueeze(2)  # [B, D, 1, T/2+1]
        k_fft = k_fft.unsqueeze(2)

        # 存储每次roll操作后的结果
        rolled_output1 = [k_fft]  # 初始包含原始k_fft
        rolled_output2 = [q_fft]  # q_fft

        # 执行10次roll操作并拼接结果
        shifts=10
        for i in range(k_fft.shape[-1]//10-1):

            new_tensor = torch.zeros_like(k_fft)
            # 将有效的部分赋值到新的张量中
            if shifts < k_fft.size(-1):
                new_tensor[:, :-shifts] = k_fft[:, shifts:]
            else:
                # 如果位移大于或等于张量长度，结果将全为0
                new_tensor = new_tensor

            # rolled_k_fft = torch.roll(k_fft, shifts=10, dims=-1)
            rolled_output1.append(new_tensor)
            rolled_output2.append(q_fft)
        # 在第三个维度上拼接所有结果
        k_fft_output = torch.cat(rolled_output1, dim=2).unsqueeze(1) # [B, 1, D,  ,T/2+1]
        # print(k_fft_output.shape)
        # print(self.para.shape)
        k_fft_output = self.para.view(1,1,1,k_fft.shape[-1]//10,1)*k_fft_output
        q_fft_output = torch.cat(rolled_output2, dim=2).unsqueeze(2) # [B, D, 1,  ,T/2+1]
        # print(q_fft_output.shape)
        corr=k_fft_output*q_fft_output
        corr=torch.sum(corr, dim=3).squeeze(3)
        corr = torch.fft.irfft(corr, dim=-1)

        corr = corr.mean(dim=-1)

        return corr

    def roll_optimized_compute_cross_cor_test_2(self, queries, keys):
        # Perform batched FFT
        q_fft = torch.fft.rfft(queries, dim=-1)
        k_fft = torch.fft.rfft(keys, dim=-1)
        # Expand dimensions for broadcasting
        q_fft = q_fft.unsqueeze(2)  # [B, D, 1, T/2+1]
        k_fft = k_fft.unsqueeze(2)
        # print(k_fft.shape)


        # 存储每次roll操作后的结果
        rolled_output1 = [k_fft]  # 初始包含原始k_fft
        rolled_output2 = [q_fft]  # q_fft

        # 执行10次roll操作并拼接结果
        shifts = 10
        for i in range(k_fft.shape[-1] // 10 - 1):

            new_tensor = torch.zeros_like(k_fft)
            # 将有效的部分赋值到新的张量中
            if shifts < k_fft.size(-1):
                new_tensor[:,:, :-shifts] = k_fft[:,:, shifts:]
                k_fft=new_tensor
            else:
                # 如果位移大于或等于张量长度，结果将全为0
                new_tensor = new_tensor

            # rolled_k_fft = torch.roll(k_fft, shifts=10, dims=-1)
            rolled_output1.append(new_tensor)
            rolled_output2.append(q_fft)

        # 在第三个维度上拼接所有结果
        k_fft_output = torch.cat(rolled_output1, dim=2).unsqueeze(1)  # [B, 1, D,  ,T/2+1]

        k_fft_output = self.para.view(1, 1, 1, k_fft.shape[-1] // 10, 1) * k_fft_output
        q_fft_output = torch.cat(rolled_output2, dim=2).unsqueeze(2)  # [B, D, 1,  ,T/2+1]

        corr = k_fft_output * q_fft_output
        corr = torch.sum(corr, dim=3).squeeze(3)
        corr=corr*self.para2.view(1,q_fft.shape[1],q_fft.shape[1],1)
        corr = torch.fft.irfft(corr, dim=-1)

        corr = corr.mean(dim=-1)

        return corr

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):

        B, D, T = queries.shape
        _, S, _ = values.shape
        corr = torch.zeros(B, D, D).to(queries.device)
        scale = self.scale or 1./sqrt(T)

        # corr=corr+self.roll_optimized_compute_cross_cor(queries,keys)

        corr=corr+self.roll_optimized_compute_cross_cor_test_2(queries,keys)

        if self.mask_flag:
            if attn_mask is None:
                # attn_mask = TriangularCausalMask(B, T, device=queries.device)
                print("==================有问题===================")
        corr = torch.softmax(corr*scale, dim=-1)

        V = torch.einsum("bsd,bde->bse", corr, values)
        if self.output_attention:
            return (V.contiguous(), corr)
        else:
            return (V.contiguous(), None)

class VarCorAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(VarCorAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

        self.d_model = d_keys*n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, D, L = queries.shape
        _, _, S = keys.shape

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, D, -1)

        return self.out_projection(out), attn

