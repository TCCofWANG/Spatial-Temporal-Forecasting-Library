import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math


# 单位矩阵生成one-hot编码，线性层降维
class One_hot_encoder(nn.Module):
    def __init__(self, embed_size, time_num=288):
        super(One_hot_encoder, self).__init__()

        self.time_num = time_num
        self.I = nn.Parameter(torch.eye(time_num, time_num, requires_grad=True))
        self.onehot_Linear = nn.Linear(time_num, embed_size)  # 线性层改变one hot编码维度

    def forward(self, i, N=25, T=12):

        if i % self.time_num + T > self.time_num:
            o1 = self.I[i % self.time_num:, :]
            o2 = self.I[0: (i + T) % self.time_num, :]
            onehot = torch.cat((o1, o2), 0)
        else:
            onehot = self.I[i % self.time_num: i % self.time_num + T, :]

        # onehot = onehot.repeat(N, 1, 1)
        onehot = onehot.expand(N, T, self.time_num)
        onehot = self.onehot_Linear(onehot)
        return onehot


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)




class Transformer(nn.Module):
    def __init__(self, embed_size, heads, adj, time_num, dropout, forward_expansion):
        super(Transformer, self).__init__()
        self.sttnblock = STTNSNetBlock(embed_size, heads, adj, time_num, dropout, forward_expansion)

    def forward(self, query, key, value, t, num_layers):
        q, k, v = query, key, value
        for i in range(num_layers): # 堆叠层数
            out = self.sttnblock(q, k, v, t)
            q, k, v = out, out, out
        return out


# model input:[B, N, T, C]
# model output[B, N, T, C]
class STTNSNetBlock(nn.Module):
    def __init__(self, embed_size, heads, adj, time_num, dropout, forward_expansion):
        super(STTNSNetBlock, self).__init__()
        self.SpatialTansformer = STransformer(embed_size, heads, adj, dropout, forward_expansion)
        self.TemporalTransformer = TTransformer(embed_size, heads, time_num, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, t):
        out1 = self.norm1(self.SpatialTansformer(query, key, value) + query) # 空间Attention
        out2 = self.dropout(self.norm2(self.TemporalTransformer(out1, out1, out1, t) + out1)) # 时间Attention

        return out2

'''STTNS的效果比GMAN的效果好原因：STTNS的Attention有Feedforward和残差连接'''
# model input:[B, N, T, C]
# model output:[B, N, T, C]
class STransformer(nn.Module):
    def __init__(self, embed_size, heads, adj, dropout, forward_expansion):
        super(STransformer, self).__init__()
        self.adj = adj
        self.D_S = nn.Parameter(adj)
        self.embed_linear = nn.Linear(adj.shape[0], embed_size)
        self.attention = SSelfattention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        # 调用GCN
        self.gcn = GCN(embed_size, embed_size * 2, embed_size, dropout)
        self.norm_adj = nn.InstanceNorm2d(1)  # 对邻接矩阵归一化，理解：归一化拉普拉斯矩阵的近似表示

        self.dropout = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value):
        # Spatial Embedding 部分
        B, N, T, C = query.shape
        D_S = self.embed_linear((self.D_S))
        D_S = D_S.expand(B,T, N, C)
        D_S = D_S.permute(0, 2, 1, 3) #(B,N,L,D)

        # GCN 部分
        X_G = torch.Tensor(0,N,T,C).to(query.device)
        self.adj = self.adj.unsqueeze(0).unsqueeze(0)
        self.adj = self.norm_adj(self.adj)
        self.adj = self.adj.squeeze(0).squeeze(0)
        for b in range(query.shape[0]):
            tmp_list=[]
            for t in range(query.shape[2]):
                o = self.gcn(query[b,:, t, :], self.adj.to(query.device))
                o = o.unsqueeze(1).unsqueeze(0)   # shape [1,N, 1, C]
                tmp_list.append(o)
            tmp=torch.cat(tmp_list,dim=2)
            X_G = torch.cat((X_G, tmp), dim=0) # 固定的图卷积
        # spatial transformer -->动态的图卷积
        query = query + D_S # 位置编码
        value = value + D_S # 位置编码
        key = key + D_S
        attn = self.attention(value, key, query)  # [B,N, T, C]
        M_s = self.dropout(self.norm1(attn + query)) # 残差连接
        feedforward = self.feed_forward(M_s) # Feedforward
        U_s = self.dropout(self.norm2(feedforward + M_s)) # 残差连接

        # 融合-->将动态的图卷积和固定的图卷积进行融合
        g = torch.sigmoid(self.fs(U_s) + self.fg(X_G)) #FIXME 但是是否可以在adj上加上一个可学习的参数来完成动态的
        out = g * U_s + (1 - g) * X_G # 门控机制

        return out


# model input:[B,N,T,C]
# model output:[B,N,T,C]
class SSelfattention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SSelfattention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = embed_size // heads
        self.values = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.queries = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.keys = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query):
        B,N, T, C = query.shape
        query = values.reshape(B,N, T, self.heads, self.per_dim)
        keys = keys.reshape(B,N, T, self.heads, self.per_dim)
        values = values.reshape(B,N, T, self.heads, self.per_dim)

        # q, k, v:[B, N, T, heads, per_dim]
        queries = self.queries(query)
        keys = self.keys(keys)
        values = self.values(values)

        # spatial self-attention
        attn = torch.einsum("bqthd, bkthd->bqkth", (queries, keys))  # [B, N, N, T, heads]
        attention = torch.softmax(attn / (self.embed_size ** (1 / 2)), dim=1)

        out = torch.einsum("bqkth,bkthd->bqthd", (attention, values))  # [B,N, T, heads, per_dim]
        out = out.reshape(B, N, T, self.heads * self.per_dim)  # [B, N, T, C]

        out = self.fc(out)

        return out


# input[N, T, C]
class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion):
        super(TTransformer, self).__init__()
        # Temporal embedding One hot
        self.time_num = time_num
        self.one_hot = One_hot_encoder(embed_size, time_num)          # temporal embedding选用one-hot方式 或者
        self.temporal_embedding = nn.Embedding(time_num, embed_size)  # temporal embedding选用nn.Embedding

        self.attention = TSelfattention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, t):
        # q, k, v：[B, N, T, C]
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


