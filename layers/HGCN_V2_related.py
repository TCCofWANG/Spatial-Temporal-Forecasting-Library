import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, BatchNorm1d

'''时间Attention'''
class TATT_1(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT_1, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)

        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn = BatchNorm1d(tem_size)

    def forward(self, seq):
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n 降维特征维度

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        # print(c2.shape)
        f2 = self.conv2(c2).squeeze()  # b,c,n 降维节点维度

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b) # 套公式
        logits = torch.matmul(self.v, logits)
        logits = logits.permute(0, 2, 1).contiguous()
        logits = self.bn(logits).permute(0, 2, 1).contiguous()
        coefs = torch.softmax(logits, -1)
        return coefs


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        A=A.transpose(-1,-2).to(x.device)
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear_time(nn.Module):
    def __init__(self,c_in,c_out,Kt):
        super(linear_time,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

'''这里就是使用切比雪夫多项式实现的GCN'''
# class multi_gcn_time(nn.Module):
#     def __init__(self,c_in,c_out,Kt,dropout,support_len=3,order=2):
#         super(multi_gcn_time,self).__init__()
#         self.nconv = nconv()
#         c_in = (order*support_len+1)*c_in
#         self.mlp = linear_time(c_in,c_out,Kt)
#         self.dropout = dropout
#         self.order = order
#
#     def forward(self,x,support):
#         out = [x] # bcnl
#         for a in support:
#             x1 = self.nconv(x,a)
#             out.append(x1)
#             for k in range(2, self.order + 1):
#                 x2 = self.nconv(x1,a)
#                 out.append(x2)
#                 x1 = x2
#
#         h = torch.cat(out,dim=1)
#         h = self.mlp(h)
#         h = F.dropout(h, self.dropout, training=self.training)
#         return h

class GCN(nn.Module):
    def __init__(self,c_in,c_out,dropout,node_nums,support_len=2,order=3,embed_dim=10):
        super(GCN,self).__init__()
        self.nconv = nconv()
        c_in_ = support_len*c_in
        self.mlp = linear_time(c_in_,c_out,1)
        self.dropout = dropout
        self.order = order

        self.node_embedding=nn.Linear(node_nums,embed_dim)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, order, c_in, c_in))  # 不同的阶次的切比雪夫多项式对应的w都不同
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, c_in))

    def forward(self,x,support):
        out=[]
        node_embeddings=self.node_embedding(support[0].cuda())# 对固定的adj进行Embedding
        for a in support:
            node_num = a.shape[0]
            support_set = [torch.eye(node_num).to(a.device), a]  # 切比雪夫多项式的前两项的结果
            #  切比雪夫多项式近似图卷积 以下的实现方式和常见的实现方式有些不同，但是是等价的
            for k in range(2, self.order):
                support_set.append(torch.matmul(2 * a, support_set[-1]) - support_set[-2])  # 依据切比雪夫多项式的公式
            supports = torch.stack(support_set, dim=0)#(3,N,N)
            weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
            bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
            x_g = torch.einsum("knm,bcml->bkncl", supports.to(x.device), x)  # B, cheb_k, N, dim_in,L
            b,k,n,c,l=x_g.shape
            x_g=x_g.reshape(-1,k,n,c)#(b*l,k,n,c)相当于把b和l融起来
            x_g = x_g.permute(0, 2, 1, 3)  # B*l, N, cheb_k, dim_in
            x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b*l, N, C 对于不同的节点有不同的W特征变换矩阵
            out.append(x_gconv.reshape(b,-1,n,l)) #(B,C,N,L)
        h = torch.cat(out, dim=1) #在特征维度上进行拼接
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GCNPool(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size,
                 Kt, dropout, pool_nodes, support_len=2, order=3):
        super(GCNPool, self).__init__()
        self.conv1 = Conv2d(c_in, c_in, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        # 以下都是时间维度处理部分
        self.time_conv = Conv2d(c_in, 2 * c_out,kernel_size=(1,1),stride=(1,1),bias=True)
        self.linear_one=nn.Linear(tem_size,tem_size)
        self.linear_two = nn.Linear(tem_size, tem_size)

        self.multigcn = GCN(c_out, 2 * c_out, dropout,num_nodes, support_len, order,embed_dim=64) #FIXME64是d_mdoel

        self.num_nodes = num_nodes
        self.tem_size = tem_size
        self.TAT = TATT_1(c_out, num_nodes, tem_size)
        self.c_out = c_out
        # self.bn=LayerNorm([c_out,num_nodes,tem_size])
        self.bn = BatchNorm2d(c_out)

    def time_linear(self,x):
        x=self.time_conv(x)
        x_one=self.linear_one(x)
        x_two=self.linear_two(x)
        x=x_two+x_one
        return x

    def forward(self, x, support):
        residual = self.conv1(x)
        # 时间处理部分的门控机制
        x = self.time_linear(x)
        x1, x2 = torch.split(x, [self.c_out, self.c_out], 1)
        x = torch.tanh(x1) * torch.sigmoid(x2)
        # 空间卷积的门控机制
        x = self.multigcn(x, support)# -->现在已经改成节点不共享GCN参数
        x1, x2 = torch.split(x, [self.c_out, self.c_out], 1)
        x = torch.tanh(x1) * (torch.sigmoid(x2))
        # x=F.dropout(x,0.3,self.training)

        T_coef = self.TAT(x)
        T_coef = T_coef.transpose(-1, -2)
        x = torch.einsum('bcnl,blq->bcnq', x, T_coef)
        out = self.bn(x + residual[:, :, :, -x.size(3):]) # 残差连接
        return out
