from __future__ import division
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 源码是type == 'RNN'
class GCRN(nn.Module):
    def __init__(self, dims, gdep, alpha, beta, gamma):
        super(GCRN, self).__init__()

        self.mlp = nn.Linear((gdep + 1) * dims[0], dims[1]).to(device)

        self.gdep = gdep
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


    def forward(self, x, supports):

        h = x.contiguous()
        out = [h]
        for _ in range(self.gdep):
                h = self.alpha * x.to(device) + \
                    self.beta * torch.einsum('bnc,nw->bwc', (h.to(device), supports[0].to(device))) + \
                    self.gamma * torch.einsum('bnc,nw->bwc', (h.to(device), supports[1].to(device)))
                out.append(h.to(device))

        ho = torch.cat([i.to(device) for i in out], dim=-1)
        ho = self.mlp(ho)

        return ho

class feature_agg(nn.Module):
    def __init__(self, input_size, output_size):
        super(feature_agg, self).__init__()
        self.CAM = nn.Sequential(nn.Linear(input_size, output_size).to(device), nn.GELU().to(device))

    def forward(self, x1, x2=None):
        if x2 == None:
            score = self.CAM(x1.to(device))  # (B,N,1)
        else:
            score = self.CAM(torch.cat((x1.to(device), x2.to(device)), dim=-1))  # (B,N,1)
        return score

class IDWTL(nn.Module):
    def __init__(self, num_nodes, hidden_size):
        super(IDWTL, self).__init__()

        # 定义可学习参数，即一个形状为(207, 64)的矩阵
        self.weight = nn.Parameter(torch.Tensor(num_nodes, hidden_size))
        self.gelu = nn.GELU()

        # 初始化权重
        nn.init.uniform_(self.weight, a=15, b=30)

    def forward(self, x):
        # x的shape为(B, N, Hidden_size)
        # 对输入张量的每个切片与可学习矩阵进行Hadamard乘积
        x = x * self.weight.unsqueeze(0)
        x = self.gelu(x)

        return x
