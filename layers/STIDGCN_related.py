import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out

'''源码是对数据集整个序列打乱的，这里是按顺序的'''
class TemporalEmbedding(nn.Module):
    def __init__(self, time, features, points_per_hour, num_nodes):
        super(TemporalEmbedding, self).__init__()

        self.num_nodes = num_nodes
        self.points_per_hour = points_per_hour
        self.time = time
        self.time_day = nn.Parameter(torch.empty(time, features), requires_grad=True)

        self.time_week = nn.Parameter(torch.empty(7, features), requires_grad=True)

        self._reset_parameter()

    def _reset_parameter(self):
        nn.init.xavier_uniform_(self.time_day)
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, seq_time):
        hour = (seq_time[:, -2:-1, ...] + 0.5) * 23  # 得到第几个小时
        min = (seq_time[:, -1:, ...] + 0.5) * 59  # 得到第几分钟
        hour_index = ((hour * 60 + min) / (60 / self.points_per_hour)).squeeze().type(torch.LongTensor)  # 得到第几个时间步

        day = ((seq_time[:, 2:3, ...] + 0.5) * (6 - 0)).squeeze().type(torch.LongTensor)  # 一周的第几天

        # 用广播机制构建emb
        time_day = self.time_day[hour_index.unsqueeze(-1).repeat(1, 1, self.num_nodes)]
        time_week = self.time_week[day.unsqueeze(-1).repeat(1, 1, self.num_nodes)]

        tem_emb = time_day + time_week  # (B,L,N,C)

        tem_emb = tem_emb.permute(0, 3, 2, 1)  # (B,C,N,L)

        return tem_emb

'论文是说，构建邻接矩阵时会进行时间维度上的消除，论文是说用先加和再用FC进行消除(有点奇怪)，源码是用einsum函数直接进行计算(相当于直接加和加和的方法),这里是和源码一样的'
class Graph_Generator(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        adj_dyn_1 = torch.softmax(F.relu(torch.einsum("bcnl, cm->bnm", x, self.memory).contiguous()/ math.sqrt(x.shape[1])),-1,)
        adj_dyn_2 = torch.softmax(F.relu(torch.einsum("bcn, bcm->bnm", x.sum(-1), x.sum(-1)).contiguous()/ math.sqrt(x.shape[1])),-1,)

        adj_f = torch.cat([(adj_dyn_1).unsqueeze(-1)] + [(adj_dyn_2).unsqueeze(-1)], dim=-1)
        adj_f = torch.softmax(self.fc(adj_f).squeeze(), -1)

        topk_values, topk_indices = torch.topk(adj_f, k=int(adj_f.shape[1] * 0.8), dim=-1)
        mask = torch.zeros_like(adj_f)
        mask.scatter_(-1, topk_indices, 1)
        adj_f = adj_f * mask  # 确保稀疏性

        return adj_f

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class DGCN(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1, emb=None):
        super().__init__()
        self.mlp = linear(diffusion_step * channels, channels)
        self.emb = emb
        self.diffusion_step = diffusion_step

        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(channels, channels, (1, 1))
        self.generator = Graph_Generator(channels, num_nodes, diffusion_step, dropout)

    def forward(self, x):
        skip = x
        x = self.conv(x)
        adj_dyn = self.generator(x)

        out = []
        for i in range(0, self.diffusion_step):
            if adj_dyn.dim() == 3:
                x = torch.einsum("bcnl,bnm->bcml", x, adj_dyn).contiguous()
                out.append(x)
            elif adj_dyn.dim() == 2:
                x = torch.einsum("bcnl,nm->bcml", x, adj_dyn).contiguous()
                out.append(x)
        x = torch.cat(out, dim=1)
        x = self.conv(x)
        x = self.dropout(x)

        x = x * self.emb + skip
        return x

class Splitting(nn.Module):  # 按奇偶时间步进行分离
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, :, :, ::2]

    def odd(self, x):
        return x[:, :, :, 1::2]

    def forward(self, x):
        return (self.even(x), self.odd(x))


class IDGCN(nn.Module):
    def __init__(
            self,
            channels=64,
            diffusion_step=1,
            splitting=True,
            num_nodes=170,
            dropout=0.2, emb=None
    ):
        super(IDGCN, self).__init__()

        self.dropout = dropout
        self.num_nodes = num_nodes
        self.splitting = splitting
        self.split = Splitting()

        pad_l = 3
        pad_r = 3
        k1 = 5
        k2 = 3

        TConv = [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels, kernel_size=(1, k2)),
            nn.Tanh(),
        ]

        self.TSConv = nn.ModuleList([nn.Sequential(*TConv) for i in range(4)])
        self.dgcn = DGCN(channels, num_nodes, diffusion_step, dropout, emb)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        x1 = self.TSConv[0](x_even)
        x1 = self.dgcn(x1)
        d = x_odd.mul(torch.tanh(x1))

        x2 = self.TSConv[1](x_odd)
        x2 = self.dgcn(x2)
        c = x_even.mul(torch.tanh(x2))

        x3 = self.TSConv[2](c)
        x3 = self.dgcn(x3)
        x_odd_update = d + x3

        x4 = self.TSConv[3](d)
        x4 = self.dgcn(x4)
        x_even_update = c + x4

        return [x_even_update, x_odd_update]

'''IDGCN树'''
class IDGCN_Tree(nn.Module):
    def __init__(
            self, channels=64, diffusion_step=1, num_nodes=170, dropout=0.1
    ):
        super().__init__()

        self.memory = nn.ParameterList([nn.Parameter(torch.randn(channels, num_nodes, 6)),
                                        nn.Parameter(torch.randn(channels, num_nodes, 3)),
                                        nn.Parameter(torch.randn(channels, num_nodes, 3))])


        self.IDGCN = nn.ModuleList([IDGCN(
            splitting=True,
            channels=channels,
            diffusion_step=diffusion_step,
            num_nodes=num_nodes,
            dropout=dropout, emb=self.memory[i]
        ) for i in range(3)])

    def concat(self, even, odd):
        even = even.permute(3, 1, 2, 0)
        odd = odd.permute(3, 1, 2, 0)
        len = even.shape[0]
        _ = []
        for i in range(len):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        return torch.cat(_, 0).permute(3, 1, 2, 0)

    def forward(self, x):
        x_even_update1, x_odd_update1 = self.IDGCN[0](x)
        x_even_update2, x_odd_update2 = self.IDGCN[1](x_even_update1)
        x_even_update3, x_odd_update3 = self.IDGCN[2](x_odd_update1)
        concat1 = self.concat(x_even_update2, x_odd_update2)
        concat2 = self.concat(x_even_update3, x_odd_update3)
        concat0 = self.concat(concat1, concat2)
        output = concat0 + x
        return output
