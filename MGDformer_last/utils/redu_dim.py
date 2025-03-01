from math import sqrt
from einops import rearrange, repeat
import torch
from torch import nn


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
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout = 0.1):
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

class redu_dim(nn.Module):
    def __init__(self,catagory,configs):
        super().__init__()
        self.attention = AttentionLayer(configs.d_model, 4, dropout=0.1)
        self.catagory = catagory
        self.router=nn.Parameter(torch.randn(1,self.catagory,configs.d_model))
        self.covs=nn.ModuleList([nn.Conv1d(2,1,25,padding=12) for _ in range(self.catagory)])
        self.conv=nn.Conv1d(2,1,25,padding=12)

        self.pred_len = configs.pred_len
        self.catagory = catagory
        self.predicts = nn.ModuleList()
        for _ in range(self.catagory):
            self.predicts.append(nn.Linear(configs.d_model, configs.pred_len))
        self.classifier = nn.Sequential(nn.Linear(configs.d_model, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                        nn.Dropout(0.1), nn.Linear(128, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                        nn.Dropout(0.1), nn.Linear(32, 1), nn.Sigmoid())

    def normalize_similarity0(self, c, h):
        # c 和 h 的形状是 (n, d)
        # n 是特征向量的数量，d 是每个向量的维度

        # 计算内积
        dot_product = torch.matmul(c, h.t())  # 计算 n x m 的内积矩阵

        # 计算范数
        norm_c = torch.norm(c, dim=1, keepdim=True)  # (n, 1)
        norm_h = torch.norm(h, dim=1, keepdim=True)  # (m, 1)

        # 计算余弦相似度
        pi = dot_product / (norm_c * norm_h.t())  # 结果是 n x m 的矩阵

        # 映射到 [0, 1] 之间
        pi_normalized = (1 + pi) / 2

        # 使用 softmax 将每一行归一化，使其和为 1
        softmax_result = torch.softmax(pi_normalized, dim=1)

        return softmax_result

    def normalize_similarity(self, c, h):
        # 假设 c 和 h 的形状是 (B, N, D)
        # B 是批次大小，N 是特征数量，D 是每个特征的维度

        # 计算内积
        # c 的形状为 (B, N, D)，h 的形状为 (B, M, D)（假设 h 的第二维是 M）
        # 计算内积，得到 (B, N, M)
        dot_product = torch.bmm(c, h.transpose(1, 2))  # c 与 h 的转置进行批量矩阵乘法

        # 计算范数
        norm_c = torch.norm(c, dim=2, keepdim=True)  # (B, N, 1)
        norm_h = torch.norm(h, dim=2, keepdim=True)  # (B, M, 1)

        # 计算余弦相似度
        # 将 norm_h 从 (B, M, 1) 变形为 (B, 1, M)
        pi = dot_product / (norm_c * norm_h.transpose(1, 2))  # 结果是 (B, N, M)

        # 映射到 [0, 1] 之间
        pi_normalized = (1 + pi) / 2

        # 使用 softmax 将每一行归一化，使其和为 1
        # 对于每个样本的每一行进行 softmax，dim=2 表示在特征维度上归一化
        # softmax_result = torch.softmax(pi_normalized, dim=2)#加上时间太久

        return pi_normalized

    # def forward(self, x):
    #     batch = x.shape[0]
    #     B,N,L=x.shape
    #     y=torch.zeros((B,N,L)).to(x.device)
    #     batch_router = repeat(self.router, 'b dim d_model -> (repeat b) dim d_model', repeat=batch)
    #     # print(batch_router.shape)
    #     attention_out = self.attention(batch_router,x,x) #batch catory dmodel
    #     score=self.normalize_similarity(attention_out,x)
    #     # 找到每行的最大值和对应的索引
    #     max_values, max_indices = torch.max(score, dim=1) #batch dim
    #     # print(max_indices.shape)
    #     # print(max_indices)
    #     for i in range(B):
    #         for j in range(N):
    #             # print(max_indices[i,j])
    #             y[i:i+1,j:j+1,:]=self.covs[max_indices[i,j]](torch.stack([x[i,j,:],attention_out[i,max_indices[i,j],:]],dim=0))
    #
    #
    #     return y

    def forward(self, x):
        batch = x.shape[0]
        B, N, L = x.shape

        batch_router = repeat(self.router, 'b dim d_model -> (repeat b) dim d_model', repeat=batch)

        attention_out = self.attention(batch_router, x, x)  # batch catory dmodel
        score = self.normalize_similarity(attention_out, x)
        # 找到每行的最大值和对应的索引
        max_values, max_indices = torch.max(score, dim=1)  # B N

        # 使用 arange 获取当前批次的索引
        batch_indices = torch.arange(batch, device=x.device).unsqueeze(1)  # (batch, 1)
        z=attention_out[batch_indices, max_indices, :]
        result=torch.stack([x,z],dim=-2) #B N 2 512
        # result=result.view(B*N,2,512)
        # max_indices=max_indices.view(B*N)
        #
        # last_result=torch.zeros(B*N,512).to(x.device)
        #
        # for i in range(len(max_indices)):
        #     last_result[i,:]=self.covs[max_indices[i]](result[i,:,:])
        #
        # # last_result=self.covs[max_indices[:]](result)
        #
        # result=last_result.view(B,N,512)

        last_result = torch.stack(
            [self.predicts[max_indices.view(-1)[i]](self.covs[max_indices.view(-1)[i]](result.view(B * N, 2, 512)[i, :, :])) for i in range(B * N)]
        ).view(B, N, self.pred_len)



        return last_result

    # def forward(self, x):
    #     batch, N, L = x.shape
    #     # 创建一个输出张量
    #     y = torch.zeros((batch, N, L), device=x.device)
    #
    #     # 扩展 router 以匹配批次大小
    #     batch_router = repeat(self.router, 'b dim d_model -> (repeat b) dim d_model', repeat=batch)
    #
    #     # 计算注意力输出
    #     attention_out = self.attention(batch_router, x, x)  # batch catory dmodel
    #
    #     # 计算相似度
    #     score = self.normalize_similarity(attention_out, x)
    #
    #     # 找到每行的最大值和对应的索引
    #     max_values, max_indices = torch.max(score, dim=1)  # batch dim
    #
    #     # 使用 `gather` 进行批量处理
    #     # 使用 `max_indices` 选择对应的 attention_out
    #     # max_indices 应该是形状为 (batch, N) 的整数张量
    #     selected_attention_out = attention_out[torch.arange(batch).unsqueeze(1), max_indices]  # (B, N, D)
    #
    #     # 将选定的 `attention_out` 和原始 `x` 一起堆叠并传入 `covs`
    #     combined_input = torch.stack([x, selected_attention_out], dim=2)  # (B, N, 2, L)
    #
    #     # 将 max_indices 转换为整型
    #     max_indices = max_indices.long()  # 确保 max_indices 是长整型张量
    #
    #     # 使用 `covs` 进行处理，确保 covs 是一个可调用的对象
    #     y = torch.stack([self.covs[i](combined_input) for i in max_indices.view(-1)])  # 批量处理
    #
    #     return y


