import torch
import torch.nn.functional as F
import torch.nn as nn


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out)) # 不同的阶次的切比雪夫多项式对应的w都不同
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports] # 切比雪夫多项式的前两项的结果
        # default cheb_k = 3 切比雪夫多项式近似图卷积
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2]) #依据切比雪夫多项式的公式
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x) # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     # b, N, dim_out 对于不同的节点有不同的W特征变换矩阵
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim) # 这个是得到z和r
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim) # 这个是得到\hat{h}

    def forward(self, x, state, node_embeddings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, num_nodes,feature_dim,rnn_units,embed_dim,num_layers=2,cheb_order = 3):
        super(AVWDCRNN, self).__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.hidden_dim = rnn_units
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.cheb_k = cheb_order
        assert self.num_layers >= 1, 'At least one DCRNN layer in the Encoder.'

        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.feature_dim,
                                          self.hidden_dim, self.cheb_k, self.embed_dim))
        for _ in range(1, self.num_layers):
            self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.hidden_dim,
                                              self.hidden_dim, self.cheb_k, self.embed_dim))

    def forward(self, x, init_state, node_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.num_nodes and x.shape[3] == self.feature_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers): # 循环GRU的层数(串联堆叠的)
            state = init_state[i]#(B,N,hidden_dim)
            inner_states = []
            for t in range(seq_length): #循环时间步
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden # current_inputs最后一个时间步的输出==output_hidden的

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers): # 初始化GRU的隐藏层
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      # (num_layers, B, N, hidden_dim)


class AGCRN(nn.Module):
    def __init__(self, num_nodes,num_feature,seq_len,pred_len,rnn_units=64,embed_dim=10,device = "cuda"):
        super(AGCRN, self).__init__()
        self.num_nodes = num_nodes
        self.feature_dim =num_feature
        self.num_nodes = num_nodes
        self.rnn_units = rnn_units
        self.input_window = seq_len
        self.output_window = pred_len
        self.output_dim = num_feature
        self.hidden_dim = rnn_units
        self.embed_dim = embed_dim

        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True) #初始化的节点Embedding(为了建模节点间的依赖性)
        self.encoder = AVWDCRNN(self.num_nodes,self.feature_dim,self.rnn_units,self.embed_dim)
        self.end_conv = nn.Conv2d(1, self.output_window * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.device = device

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, seqs,adj,**kwargs):
        # source: B, C, N, L
        # target: B, C, N, L
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        source = seqs.permute(0, 3, 2, 1)#(B,L,N,C)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)  # B, T, N, hidden
        output =  output[:, -1:, :, :]  # B, 1, N, hidden 取最后一个时间步的输出

        # CNN based predictor
        output = self.end_conv(output) # B, T*C, N, 1  将最后一个时间步的输出的时间维度进行升维
        output = output.squeeze(-1).reshape(-1, self.output_window, self.output_dim, self.num_nodes)
        output = output.permute(0, 2, 3, 1)                  # B, C, N, L
        return output


