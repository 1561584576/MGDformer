# 康汝兵 哈哈哈
import torch
from torch import nn



class class_predictor(nn.Module):
    def __init__(self,catagory,configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.catagory =catagory
        self.predicts=nn.ModuleList()
        for _ in range(self.catagory):
            self.predicts.append(nn.Linear(configs.d_model, configs.pred_len))
        self.classifier = nn.Sequential(nn.Linear(configs.d_model, 128),nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.1),nn.Linear(128, 32),nn.BatchNorm1d(32),nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(32, 1),nn.Sigmoid())

    def forward(self, x):

        B, N, L = x.size()
        x_reshaped = x.view(B * N, L)

        # y = torch.mean(self.classifier(x_reshaped).view(B , N, 1).squeeze(-1),dim=0)

        z = self.classifier(x_reshaped).view(B , N, 1).squeeze(-1)
        output = torch.zeros(x.size(0), x.size(1), self.pred_len)
        for i in range(x.shape[1]):
            output[:, i, :] = self.predicts[z[:,i]](x[:, i, :])

        # # 使用 torch.unique 统计各个元素的数量
        # unique_elements, counts = torch.unique(y, return_counts=True)
        #
        # # 打印结果
        # for element, count in zip(unique_elements, counts):
        #     print(f"Element: {element.item()}, Count: {count.item()}")

        # print(self.classifier(x_reshaped).view(B , N, 1).shape)
        # output = torch.zeros(x.size(0), x.size(1), self.pred_len)
        # for i in range(x.shape[1]):
        #     output[:,i,:] = self.predicts[int(y[i]*self.catagory)](x[:,i,:])

        return output
    # def forward(self,x):
    #     y=self.classifier(x)
    #     output=torch.zeros(x.size(0),x.size(1),self.pred_len)
    #     for i in range(x.shape[0]):
    #         for j in range(x.shape[1]):
    #             output[i][j]=self.predicts[int(torch.floor(y[i,j,0]*self.catagory))](x[i][j])
    #
    #     return output