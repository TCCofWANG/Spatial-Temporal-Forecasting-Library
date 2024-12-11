import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import math
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, BatchNorm1d

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden

def do_patching(z,stride,patch_len):
    # Patching
    z = nn.functional.pad(z, (0,stride))
    if z.shape[-1]<patch_len:
        z= nn.functional.pad(z, (0,patch_len-z.shape[-1]))
    # real doing patching
    z = z.unfold(dimension=-1, size=patch_len, step=stride)  # z: [bs x nvars x patch_num x patch_len]
    return z


# TODO 新增的节点间不共享的时间Attention
class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout,stride,patch_len,**kwargs):
        super(TTransformer, self).__init__()
        # Temporal embedding One hot
        self.time_num = time_num
        self.temporal_embedding = nn.Embedding(int(time_num), embed_size*patch_len)  # temporal embedding选用nn.Embedding

        self.attention = TSelfattention(embed_size*patch_len, heads)
        self.norm1 = nn.LayerNorm(embed_size*patch_len)
        self.norm2 = nn.LayerNorm(embed_size*patch_len)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size*patch_len, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size*patch_len)
        )
        self.stride  = stride
        self.patch_len = patch_len
        out_len = kwargs.get('out_len')
        if out_len<patch_len:
            out_len_new=patch_len-stride # 因为下面计算patch_num的时候会默认补stride
        else:
            out_len_new=out_len
        patch_num = int(( out_len_new- patch_len) /stride + 1) + 1  # 根据公式直接计算
        self.re_patch = nn.Linear(patch_num*patch_len,out_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # q, k, v：[B, N, T, C]
        x = do_patching(x ,patch_len = self.patch_len,stride=self.stride)
        query = x.reshape(x.size(0), x.size(2), x.size(3), x.size(1) * x.size(-1))
        B, N, T, C = query.shape
        # D_T = self.one_hot(t, N, T)  # temporal embedding选用one-hot方式
        D_T = self.temporal_embedding(torch.arange(0, T).to(query.device))  # temporal embedding选用nn.Embedding
        D_T = D_T.expand(B, N, T, C)

        # TTransformer
        x = D_T + query
        attention = self.attention(x, x, x)
        M_t = self.dropout(self.norm1(attention + x))
        feedforward = self.feed_forward(M_t)
        U_t = self.dropout(self.norm2(M_t + feedforward))

        out = U_t + x + M_t # 残差连接
        out =out.reshape(out.size(0), out.size(1), -1,out.size(2)* self.patch_len)
        out=self.re_patch(out)
        out=out.transpose(1,2)
        return out

class TSelfattention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TSelfattention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = self.embed_size // heads
        self.queries = nn.Linear(self.per_dim, self.per_dim)
        self.keys = nn.Linear(self.per_dim, self.per_dim)
        self.values = nn.Linear(self.per_dim, self.per_dim)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query):
        # q, k, v:[B, N, T, C]
        B, N, T, C = query.shape

        # q, k, v:[B, N,T,heads, per_dim]
        keys = key.reshape(B, N, T, self.heads, self.per_dim)
        queries = query.reshape(B, N, T, self.heads, self.per_dim)
        values = value.reshape(B, N, T, self.heads, self.per_dim)

        keys = self.keys(keys)
        values = self.values(values)
        queries = self.queries(queries)

        # compute temperal self-attention
        attnscore = torch.einsum("bnqhd, bnkhd->bnqkh", (queries, keys))  # [B,N, T, T, heads]
        attention = torch.softmax(attnscore / (self.embed_size ** (1/2)), dim=2)

        out = torch.einsum("bnqkh, bnkhd->bnqhd", (attention, values)) # [B, N, T, heads, per_dim]
        out = out.reshape(B,N, T, self.embed_size)
        out = self.fc(out)

        return out


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




class GCNPool(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size,
                 Kt, dropout, pool_nodes, support_len=3, order=2,**kwargs):
        super(GCNPool, self).__init__()
        args = kwargs.get('args')
        self.norm_T=kwargs.get('t_norm')
        self.norm_S = kwargs.get('s_norm')

        self.time_conv = Conv2d(3*c_in, 2 * c_out, kernel_size=(1, Kt), padding=(0, 0),
                                stride=(1, 1), bias=True, dilation=2)

        self.multigcn = multi_gcn_time(c_out,2 * c_out,dropout=dropout,support_len=support_len,Kt=Kt,nhid=args.d_model,num_gcn=args.num_gcn)
        self.num_nodes = num_nodes
        self.tem_size = tem_size
        time_num=24*60/args.points_per_hour

        out_len=tem_size  # 根据膨胀卷积的公式计算
        self.TAT = TTransformer(c_out,heads=args.heads,time_num=time_num,dropout=dropout,stride=args.stride,patch_len=args.patch_len,out_len=out_len)
        self.c_out = c_out

        self.bn = BatchNorm2d(c_out)

        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)

    def forward(self, x, support):
        residual = self.conv1(x)
        x_list = [residual]
        x_list.append(self.norm_T(residual))
        x_list.append(self.norm_S(residual))
        x = torch.concat(x_list, dim=1)

        # 时间卷积的门控机制
        x = self.time_conv(x) # 时间维度卷积
        x1, x2 = torch.split(x, [self.c_out, self.c_out], 1)
        x = torch.tanh(x1) * torch.sigmoid(x2)

        # 空间卷积的门控机制
        x = self.multigcn(x, support)
        x1,x2 = torch.split(x, [self.c_out, self.c_out], 1)
        x = torch.tanh(x1) * torch.sigmoid(x2)

        x = self.TAT(x)
        out = self.bn(x + residual[:, :, :, -x.size(3):]) # 残差连接
        return out



class GraphConvolution_(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True,num=10):
        super(GraphConvolution_, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_q = nn.Parameter(torch.FloatTensor(num,in_features, out_features))
        self.weight_v = nn.Parameter(torch.FloatTensor(num,in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1,out_features,1,1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_v.size(1))
        self.weight_v.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_q.size(1))
        self.weight_q.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support_q = torch.sigmoid(torch.einsum('bcnl,kcd->kbdnl',x,self.weight_q))
        support_v=torch.einsum('bcnl,kcd->kbdnl',x,self.weight_v)
        support=torch.sum(torch.multiply(support_q,support_v),dim=0)
        output=torch.einsum('nk,bcnl->bckl',adj.cuda(),support.cuda())
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class multi_gcn_time(nn.Module):
    def __init__(self, dim_in,dim_out,Kt=2,dropout=0.0,support_len=2,nhid=64,num_gcn =10):
        super(multi_gcn_time, self).__init__()
        self.gc1=nn.ModuleList()
        self.gc2=nn.ModuleList()
        for i in range(support_len):
            self.gc1.append(GraphConvolution_(dim_in, nhid,num=num_gcn))
            self.gc2.append(GraphConvolution_(nhid, dim_out,num=num_gcn))
        self.dropout = dropout
        self.mlp = linear_time(support_len*dim_out,dim_out,Kt)

    def forward(self, x, adj_list):
        out=[]
        x_old=x.clone()
        for i in range(len(adj_list)):
            adj=adj_list[i]
            x = F.relu(self.gc1[i](x_old, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2[i](x, adj)
            out.append(x)
        out=torch.concat(out,dim=1)
        out=self.mlp(out)
        return out


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        # affine 默认False
        # 利用仿射函数作为学习参数，不用仿射的话就是单纯的实例标准化
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        # 对称结构
        B, C, N, L = x.shape
        x = x.reshape(B * N, L,C)
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        x = x.reshape(B, C, N, L)
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
        # self.affine_std = nn.Linear(self.num_features,self.num_features)
        # self.affine_mean = nn.Linear(self.num_features,self.num_features)

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    # 对统计量也进行映射
    # def _statistics_affine(self):
    #     self.mean=self.affine_mean(self.mean)
    #     self.stdev=self.affine_std(self.stdev)

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        # self._statistics_affine()
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x



