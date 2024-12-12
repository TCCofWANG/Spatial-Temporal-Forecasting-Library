import torch
from torch import nn
import torch.nn.functional as F
import math

class GATConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, dropout=0.1, alpha=1e-2, concat=True):
        super(GATConv, self).__init__()
        self.concat = concat
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.w = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.a = nn.Parameter(torch.FloatTensor(2 * out_channels, num_nodes))

        self.leaky_relu = nn.LeakyReLU(alpha, inplace=True)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, x, adj):
        # x: (n_batch, channels, num_nodes, seq_len
        wh = torch.einsum('bivl,io->bovl', x, self.w) # 与可训练权重相乘，变换特征维度
        wh1 = torch.einsum('bivl,io->bovl', wh, self.a[:self.out_channels, :])
        wh2 = torch.einsum('bivl,io->bovl', wh, self.a[self.out_channels:, :])
        e = wh1 + wh2 #(B,N,N,seq_len)
        adj = adj.to(e.device)
        e = self.leaky_relu(e)
        attention = torch.where(adj[None, ..., None] > 1e-6, e, -1e10)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.einsum('binl,bovl->bonl', attention, wh)

        if self.concat:
            return F.elu(h_prime, inplace=True)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, dropout=0.1, support_len=2):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList([GATConv(in_channels, out_channels, num_nodes, dropout=dropout, concat=True) for _ in range(support_len)])

    def forward(self, x, supports):
        out = []
        for att, support in zip(self.attentions, supports):
            support=support
            out.append(att(x, support))
        x = torch.cat(out, dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(x, inplace=True)
        return x

# 多头GAT
class GAT_MH(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, n_hidden, dropout=0.1, support_len=2, heads=1, type='concat'):
        """Dense version of GAT."""
        super(GAT_MH, self).__init__()
        self.dropout = dropout
        self.type = type
        if type == 'concat':
            assert n_hidden % heads == 0
            one_hidden = int(n_hidden/heads)
        elif type == 'mean':
            one_hidden = n_hidden
        else:
            raise NotImplementedError

        self.attentions = nn.ModuleList([GAT(in_channels, one_hidden, num_nodes, dropout=dropout, support_len=support_len) for _ in range(heads)])
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GAT(n_hidden, out_channels, num_nodes, dropout=dropout, support_len=support_len)

    def forward(self, x, supports):
        x = F.dropout(x, self.dropout, training=self.training)
        if self.type == 'concat':
            x = torch.concat([att(x, supports) for att in self.attentions], dim=1)
        else:
            x = torch.mean(torch.concat([torch.unsqueeze(att(x, supports), dim=1) for att in self.attentions], dim=1))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, supports))
        return x

# 单层LSTM
class G_LSTM_layer(nn.Module):
    def __init__(self, seq_len, num_nodes, in_channels, out_channels, n_hidden, dropout, support_len, head=1, type='concat'):
        super(G_LSTM_layer, self).__init__()
        self.seq_len = seq_len
        self.i_bias = nn.Parameter(torch.empty(in_channels, num_nodes, seq_len))
        self.f_bias = nn.Parameter(torch.empty(in_channels, num_nodes, seq_len))
        self.o_bias = nn.Parameter(torch.empty(in_channels, num_nodes, seq_len))
        self.c_bias = nn.Parameter(torch.empty(in_channels, num_nodes, seq_len))

        self.G = nn.ModuleList([nn.ModuleList([GAT_MH(in_channels, out_channels, num_nodes, n_hidden, dropout, support_len, head, type) for i in range(8)]) for j in range(seq_len)])
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.i_bias.size(1))
        self.i_bias.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.f_bias.size(1))
        self.f_bias.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.o_bias.size(1))
        self.o_bias.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.c_bias.size(1))
        self.c_bias.data.uniform_(-stdv, stdv)

        self.bias = [self.i_bias, self.f_bias, self.o_bias, self.c_bias]

    def forward(self, input, cell, supports):  # cell:B,C,N,1
        output = []
        for i in range(self.seq_len):
            x = input
            # B,C,N->B,C,N,1->F(x)->B,C,N,1->B,C,N,1+C,N->B,C,N,1+1,C,N,1->B,C,N,1
            x_list = [self.G[i][j](input[:,:,:,i].unsqueeze(-1), supports)+self.bias[j][:,:,i].unsqueeze(0).unsqueeze(-1) for j in range(4)]
            # B,C,N,1->F(x)->B,C,N,1
            c_list = [self.G[i][j](cell, supports) for j in range(4, 8)]
            door_list = [torch.sigmoid(unpack[0]+unpack[1]) if index<3 else torch.tanh(unpack[0]+unpack[1]) for index, unpack in enumerate(zip(x_list, c_list))]

            cell = door_list[1]*cell + door_list[0]*door_list[-1]
            H = door_list[2] * torch.tanh(cell)
            output.append(H)
        output = torch.concat(output, dim=-1)  # B, C, N, L
        return output, cell  # output:B,C,N,L    cell:B,C,N,1














