import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils.graph_process import *

class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, feature:int,output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias

        self.laplacian=nn.Parameter(torch.FloatTensor(adj),requires_grad=False)

        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + feature, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

        # 这里的输入信息x会和隐藏层输出一起做图卷积(一层)
    def forward(self, inputs, hidden_state):
        batch_size,feature_dim, num_nodes = inputs.shape
        # [x, h] (batch_size,num_gru_units + C,num_nodes)
        concatenation = torch.cat((inputs, hidden_state), dim=1) # 表示输入的特征数据与隐藏层输出进行拼接
        # [x, h]A (batch_size,(num_gru_units + C),num_nodes)
        a_times_concat = torch.matmul(concatenation,self.laplacian)# 图卷积，会把隐藏层输出也进行一下图卷积
        # [x, h]A (batch_size, num_nodes, num_gru_units + C)
        a_times_concat = a_times_concat.transpose(-1,-2)
        # [x, h]AW + b (batch_size , num_nodes, output_dim)
        outputs = torch.matmul(a_times_concat,self.weights) + self.biases # 变换特征维度 相当于把当前时间步的输入信息与上一个时间步的隐藏层输出结合
        outputs=outputs.transpose(-1,-2)
        return outputs #(B,C,N)

def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum('ncvl,vw->ncwl', (x, A.to(x.device))).contiguous()


class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout,support, support_len=3, order=2):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.final_conv = nn.Conv2d(c_in, c_out, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order
        self.support=[support]

    def forward(self, x, hidden_state):
        x = torch.cat((x, hidden_state), dim=1).unsqueeze(-1)
        out = [x]#ncvl 一阶的结果
        for a in self.support:
            x1 = nconv(x, a) # 二阶的结果
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a) # 将二阶的结果在与拉普拉斯矩阵进行相乘
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.final_conv(h)
        h = F.dropout(h, self.dropout, training=self.training) #与另一份代码相比多出来的 测试过加上效果会差很多
        return h.squeeze(-1)



class TGCNCell(nn.Module):
    def __init__(self, adj, features:int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", adj.float())
        # self.graph_conv1 = TGCNGraphConvolution(
        #     adj=self.adj, num_gru_units=self._hidden_dim ,feature=features,output_dim=self._hidden_dim * 2, bias=1.0
        # ) # self._hidden_dim * 2 是为了得到r和u
        # self.graph_conv2 = TGCNGraphConvolution(
        #     adj=self.adj, num_gru_units=self._hidden_dim ,feature=features,output_dim=self._hidden_dim,
        # )
        self.graph_conv1 = GraphConvNet(features+self._hidden_dim,self._hidden_dim * 2,dropout=0,support=self.adj,support_len=1
        )
        self.graph_conv2 = GraphConvNet(
            features+self._hidden_dim,self._hidden_dim,dropout=0,support=self.adj,support_len=1)


    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid([x, h]AW + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_gru_units, num_nodes)
        # u (batch_size, num_gru_units, num_nodes)
        r, u = torch.chunk(concatenation, chunks=2, dim=1) # 根据GRU的流程图图
        # c = tanh([x, (r * h)]AW + b)
        # c (batch_size,num_gru_units,num_nodes )
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state)) # 根据GRU的公式
        # h := u * h + (1 - u) * c
        # h (batch_size, num_gru_units,num_nodes )
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state


class TGCN(nn.Module):
    def __init__(self, adj,features:int, hidden_dim: int,pred_len, **kwargs):
        super(TGCN, self).__init__()
        self.num_nodes = adj.shape[0]
        self._hidden_dim = hidden_dim
        adj=transition_matrix(adj)
        self.register_buffer("adj", adj.float())
        self.tgcn_cell = TGCNCell(self.adj,features,self._hidden_dim)


        self.conv=nn.Conv1d(self._hidden_dim,pred_len,kernel_size=1,stride=1)

    def forward(self, inputs,adj,**kargs):
        batch_size, feature_dim,num_nodes,seq_len = inputs.shape
        assert self.num_nodes == num_nodes
        hidden_state = (torch.zeros(batch_size,self._hidden_dim,num_nodes)
                        .type_as(inputs))#少一维是因为不用考虑时间维度
        output = None
        for i in range(seq_len): # 循环时间步 每次的输入就是特征和节点
            output, hidden_state = self.tgcn_cell(inputs[:, :, :,i], hidden_state)
        output=self.conv(output).unsqueeze(-1) # 拿一个pred_len个卷积核，每一个卷积核将所有的特征隐藏层维度压为1维度
        output=output.transpose(1,-1) #(B,C,N,L)
        return output
