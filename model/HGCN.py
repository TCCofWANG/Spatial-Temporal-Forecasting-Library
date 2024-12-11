from layers.HGCN_related import *
from torch_utils.graph_process import *
import torch
import torch.nn as nn
import torch.nn.functional as F

'''该版本的是HGCN是没有区域卷积部分的，因为公共数据集上没有对应的数据（原paper是在私有的数据集上跑）。（因此该模型的核心的创新点没有了）'''
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
class H_GCN_wh(nn.Module):
    def __init__(self, num_nodes, seq_len=12,num_features=3,pred_len=12,supports=None,dropout=0.3,residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, K=3, Kt=3):
        super(H_GCN_wh, self).__init__()
        # 更改变量名称
        length=seq_len
        in_dim=num_features
        out_dim=pred_len

        self.dropout = dropout
        self.num_nodes = num_nodes
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        supports=transition_matrix(supports)
        self.supports = [supports]

        self.supports_len = 0

        if supports is not None:
            self.supports_len += len(self.supports)

        if supports is None:
            self.supports = []
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.h = nn.Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)

        self.supports_len += 1

        Kt1 = 2
        self.block1 = GCNPool(dilation_channels, dilation_channels, num_nodes, length - 6, 3, dropout, num_nodes,
                              support_len=self.supports_len)
        self.block2 = GCNPool(dilation_channels, dilation_channels, num_nodes, length - 9, 2, dropout, num_nodes,
                              support_len=self.supports_len)

        self.skip_conv1 = nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)
        self.skip_conv2 = nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 3),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.bn = nn.BatchNorm2d(in_dim, affine=False)

    def forward(self, input,adj,**kargs):
        x = self.bn(input) # 在特征维度进行归一化
        shape = x.shape

        if self.supports is not None:
            # nodes
            # A=A+self.supports[0]
            A = F.relu(torch.mm(self.nodevec1, self.nodevec2)) # 可训练邻接矩阵
            d = 1 / (torch.sum(A, -1)) # 归一化
            D = torch.diag_embed(d) # 对角化
            A = torch.matmul(D, A) # 得到归一化的拉普拉斯矩阵
            if not isinstance(self.supports,list):
                self.supports=[self.supports]
            new_supports = self.supports + [A.to(device)]

        skip = 0
        x = self.start_conv(x) # 升特征维度

        # S-T block1
        x = self.block1(x, new_supports)

        s1 = self.skip_conv1(x) # 升维为了后续残差连接
        skip = s1 + skip

        # S-T block2
        x = self.block2(x, new_supports)

        s2 = self.skip_conv2(x)
        skip = skip[:, :, :, -s2.size(3):]
        skip = s2 + skip

        # Forcing Block 类似与FeedForward
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))# F_sum1 将时间维度压成1
        x = self.end_conv_2(x) # 将特征维度降维成需要预测的时间长度
        x=x.transpose(1,3)
        # output = [batch_size,1=dim,num_nodes,12=pred_len]
        return x
