import torch
import torch.nn.functional as F
import torch.nn as nn

'''该模型类似与DLinear思想：使用最简单的模型，conv1d。'''
class SANN(nn.Module):
    def __init__(self, n_inp, n_out, t_inp, t_out, n_points, past_t, hidden_dim, dropout):
        '''
        Args:
            n_inp:输入的特征维度
            n_out: 输出的特征维度
            t_inp: 输入的时间长度
            t_out: 预测的时间长度
            n_points: 节点数
            past_t: 延时(卷积时候使用的，时间维度的Kernel_size)
            hidden_dim: 隐藏层维度
            dropout: Dropout（丢弃率）的比例
        '''
        super(SANN, self).__init__()
        # Variables
        self.n_inp = n_inp
        self.n_out = n_out
        self.t_inp = t_inp
        self.t_out = t_out
        self.n_points = n_points
        self.past_t = past_t
        self.hidden_dim = hidden_dim
        # Convolutional layer
        self.conv_block = AgnosticConvBlock(n_inp, n_points, past_t, hidden_dim, num_conv=1)
        self.convT = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, n_points))
        # Regressor layer
        self.regressor = ConvRegBlock(t_inp, t_out,n_out ,n_points, hidden_dim)
        # Dropout
        self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x,adj,**kargs):
        '''x and output:(N,C,S,T) 其中N表示的是batchsize，C表示的是特征维度，S表示的是节点数，T表示的是时间长度'''
        x=x.permute(0,1,3,2)
        N, C, T, S = x.size()  # 其中N为batchsize，T为输入的时间步，S为节点数
        # Padding
        xp = F.pad(x, pad=(0, 0, self.past_t - 1, 0))
        # NxCxTxS ---> NxHxTx1
        # TODO 这里会利用空间信息，但是这个空间信息不是通过邻接矩阵来定义的，因为其认为邻接空间信息不是最好的，这里使用自己学习权重的方式来学习邻接信息。
        out = self.conv_block(xp)  # 将节点数降为1，顺便升特征维度
        out = out.view(N, self.hidden_dim, T, 1)
        # NxHxTx1 ---> NxHxTxS
        out = self.convT(out)  # 转置卷积 将节点数从1变为S (特征维度不变)
        # 2D dropout
        out = self.drop(out)# 源代码Dropout率为0，即全部保留
        # NxHxTxS ---> NxC'xT'xS
        out = self.regressor(out.view(N, -1, S))  # 将隐藏层和时间步融合为1，使用conv1d映射到-->预测长度*预测的特征维度
        return out.reshape(N, self.n_out, self.n_points,self.t_out)

'''这一个block作用：(1)聚合邻接信息;(2)将全部节点的信息进行聚合为一个点;(3)将特征维度进行升维'''
class AgnosticConvBlock(nn.Module):
    def __init__(self, n_inp, n_points, past_t, hidden_dim, num_conv):
        super(AgnosticConvBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=n_inp, out_channels=hidden_dim, kernel_size=(past_t, n_points), bias=True))
        layers.append(nn.BatchNorm2d(num_features=hidden_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU())
        self.op = nn.Sequential(*layers)

    def forward(self, x):
        return self.op(x)


class ConvRegBlock(nn.Module):
    def __init__(self, t_inp, t_out,n_out, n_points, hidden_dim):
        super(ConvRegBlock, self).__init__()
        layers = []
        layers.append(nn.Conv1d(in_channels=hidden_dim * t_inp, out_channels=t_out*n_out, kernel_size=1, bias=True))
        layers.append(nn.BatchNorm1d(num_features=t_out*n_out, affine=True, track_running_stats=True))
        self.op = nn.Sequential(*layers)

    def forward(self, x):
        return self.op(x)



