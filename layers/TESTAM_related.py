import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy as cp

device=torch.device('cuda' if torch.cuda.is_available() else'cpu')

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class DCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, supports_len=3, order=2):
        super(DCN,self).__init__()
        c_in = (order*supports_len+1)*c_in
        self.mlp = nn.Linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = torch.einsum('bnlc,nw->bwlc', (x, a))
            out.append(x1.contiguous())
            for k in range(2, self.order + 1):
                x2 = torch.einsum('bnlc,nw->bwlc', (x1, a))
                out.append(x2.contiguous())
                x1 = x2.contiguous()

        h = torch.cat(out, dim=-1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

# 多头注意力
class Attention(nn.Module):
    """
    Assume input has shape B, N, T, C or B, T, N, C
    Note: Attention map will be B, N, T, T or B, T, N, N
        - Could be utilized for both spatial and temporal modeling
        - Able to get additional kv-input (for Time-Enhanced Attention)
    """
    def __init__(self, in_dim, hidden_size, dropout, num_heads = 4):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Parameter(torch.randn((in_dim, self.hidden_size)))
        self.key = nn.Parameter(torch.randn((in_dim, self.hidden_size)))
        self.value = nn.Parameter(torch.randn((in_dim, self.hidden_size)))
        self.num_heads = num_heads
        self.proj = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.dropout = nn.Dropout(p=dropout)
        assert hidden_size % num_heads == 0

    def forward(self, x, kv = None):  # B,X,Y,C
        if kv is None:
            kv = x
        query = torch.einsum('bxyz,zd->bxyd', (x, self.query))  # B,X,Y,D
        key = torch.einsum('bxyz,zd->bxyd', (kv, self.key))  # B,X,Y,D
        value = torch.einsum('bxyz,zd->bxyd', (kv, self.value))  # B,X,Y,D

        query = torch.cat(torch.chunk(query, self.num_heads, dim=-1), dim=0)  # B*heads,X,Y,D/heads
        key = torch.cat(torch.chunk(key, self.num_heads, dim=-1), dim=0)  # B*heads,X,Y,D/heads
        value = torch.cat(torch.chunk(value, self.num_heads, dim=-1), dim=0)  # B*heads,X,Y,D/heads

        energy = torch.einsum('bxyz,bxzd->bxyd', (query, key.transpose(-1, -2)))  # B*heads,X,Y,Y
        energy = energy / ((self.hidden_size/self.num_heads) ** 0.5)
        score = torch.softmax(energy, dim=-1)
        head_out = torch.einsum('bxyz,bxzd->bxyd', (score, value))  # B*heads,X,Y,D/heads
        out = torch.cat(torch.chunk(head_out, self.num_heads, dim=0), dim=-1)  # B,X,Y,D

        return self.dropout(torch.matmul(out, self.proj))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, in_dim, hidden_size, dropout, activation=nn.GELU()):
        super(PositionwiseFeedForward, self).__init__()
        self.act = activation
        self.l1 = nn.Parameter(torch.randn(in_dim, hidden_size))
        self.l2 = nn.Parameter(torch.randn(hidden_size, in_dim))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(torch.matmul(self.act(torch.matmul(x, self.l1)), self.l2))


class TemporalInformationEmbedding(nn.Module):
    """
    We assume that input shape is B, T
        - Only contains temporal information with index
    Arguments:
        - vocab_size: total number of temporal features (e.g., 7 days)
        - freq_act: periodic activation function
        - n_freq: number of hidden elements for frequency components
            - if 0 or H, it only uses linear or frequency component, respectively
    """
    def __init__(self, hidden_size, vocab_size, freq_act = torch.sin, n_freq=1):
        super(TemporalInformationEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size).to(device)
        self.linear = nn.Parameter(torch.randn(hidden_size, hidden_size)).to(device)
        self.freq_act = freq_act
        self.n_freq = n_freq

    def forward(self, seq_len):
        emb = self.embedding(seq_len.to(device))
        weight = torch.einsum('blc,cd->bld', (emb, self.linear.to(emb.device)))

        # 这里是建模周期性的，n_freq表示论文中的条件分界点，但是源码与论文的条件是反过来的，这里依照论文的条件
        if self.n_freq == 0:
            return weight
        if self.n_freq == emb.size(-1):
            return self.freq_act(weight)
        x_linear = weight[..., :self.n_freq:]
        x_act = self.freq_act(weight[..., self.n_freq:])

        return torch.cat([x_linear, x_act], dim=-1)

