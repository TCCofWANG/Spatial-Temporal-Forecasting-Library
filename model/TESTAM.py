from layers.TESTAM_related import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


# 因实验，未特别设置损失函数
class TESTAM(nn.Module):
    """
    TESTAM model
    """
    def __init__(self, num_nodes, seq_len=12, dropout=0.3, in_dim=2, out_dim=12, hidden_size=32, layers=3, points_per_hour=15, prob_mul=False,
                 **kargs):
        super(TESTAM, self).__init__()
        self.dropout = dropout
        self.seq_len = seq_len
        self.prob_mul = prob_mul

        self.supports_len = 2
        self.points_per_hour = points_per_hour
        self.identity_expert = TemporalModel(hidden_size, num_nodes, in_dim=in_dim, layers=layers, dropout=dropout)
        self.adaptive_expert = STModel(hidden_size, self.supports_len, num_nodes, in_dim=in_dim, layers=layers,
                                       dropout=dropout)
        self.attention_expert = AttentionModel(hidden_size, in_dim=in_dim, layers=layers, dropout=dropout)

        self.gate_network = MemoryGate(hidden_size, num_nodes, input_dim=in_dim)

        self.output_linear = nn.Linear(seq_len,out_dim)

        for model in [self.identity_expert, self.adaptive_expert, self.attention_expert]:
            for n, p in model.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, input, adj,**kwargs):
        """
        input: B, in_dim, N, T
        o_identity shape B, N, T, 1
        """
        seqs_time=kwargs.get("seqs_time")
        targets_time=kwargs.get('targets_time')
        input = input.permute(0, 2, 3, 1)  # B,N,L,C

        # 以下过程是获取动态邻接矩阵，更好的学习静态图(应该是因为元学习的原因)
        n1 = torch.matmul(self.gate_network.We1, self.gate_network.memory)
        n2 = torch.matmul(self.gate_network.We2, self.gate_network.memory)
        g1 = torch.softmax(torch.relu(torch.mm(n1, n2.T)), dim=-1)
        g2 = torch.softmax(torch.relu(torch.mm(n2, n1.T)), dim=-1)
        new_supports = [g1, g2]

        hour = (seqs_time[:, -2:-1, ...] + 0.5) * 23  # 得到第几个小时
        min = (seqs_time[:, -1:, ...] + 0.5) * 59  # 得到第几分钟
        cur_time_index = ((hour * 60 + min) / (60 / self.points_per_hour)).squeeze().type(torch.LongTensor)  # 得到第几个时间步

        next_time_index = cur_time_index+self.seq_len
        if True in (next_time_index > 287):
            next_time_index[next_time_index > 287] = next_time_index[next_time_index > 287] - 288

        # 每个专家都能输出自己的输出和最后一层的隐状态
        o_identity, h_identity = self.identity_expert(cur_time_index, input)

        _, h_future = self.identity_expert(next_time_index)  # 伪标签的获取

        o_adaptive, h_adaptive = self.adaptive_expert(input, h_future, new_supports)

        o_attention, h_attention = self.attention_expert(input, h_future)

        B, N, T, _ = o_identity.size()
        gate_in = [h_identity, h_adaptive, h_attention]

        # 打分
        gate = torch.softmax(self.gate_network(input, gate_in), dim=-1)
        out = torch.zeros_like(o_identity).view(-1, 1)

        outs = [o_identity, o_adaptive, o_attention]

        route_prob_max, routes = torch.max(gate, dim=-1)
        route_prob_max = route_prob_max.view(-1)
        routes = routes.view(-1)

        # 选最好的专家
        for i in range(len(outs)):
            cur_out = outs[i].view(-1, 1)
            indices = torch.eq(routes, i).nonzero(as_tuple=True)[0]
            out[indices] = cur_out[indices]

        if self.prob_mul:
            out = out * (route_prob_max).unsqueeze(dim=-1)

        out = out.view(B,1, N,T)
        out = self.output_linear(out)

        return out



# 每个专家为了提高专业性，大体架构相同但是细节不同
# 从架构来说，时间信息建模专家没有用伪标签嵌入、空间建模层和时间增强注意力层
class TemporalModel(nn.Module):
    """
    Input shape
        - x: B, T
            - Need modification to use the multiple temporal information with different indexing (e.g., dow and tod)
        - speed: B, N, T, in_dim = 1
            - Need modification to use them in different dataset
    Output shape B, N, T, O
        - In the traffic forecasting, O (outdim) is normally one
    Arguments:
        - vocab_size: total number of temporal features (e.g., 7 days)
            - Notes: in the trivial traffic forecasting problem, we have total 288 = 24 * 60 / 5 (5 min interval)
    """
    def __init__(self, hidden_size, num_nodes, layers, dropout, in_dim=1, vocab_size=288, activation=nn.ReLU()):
        super(TemporalModel, self).__init__()
        self.vocab_size = vocab_size
        self.layers = layers
        self.in_dim = in_dim
        self.act = activation
        self.embedding = TemporalInformationEmbedding(hidden_size, vocab_size=vocab_size)
        self.linear1 = nn.Parameter(torch.randn(in_dim, hidden_size))  # 将输入升维
        self.linear2 = nn.Parameter(torch.randn(hidden_size * 2, hidden_size))  # 对输入和时间嵌入进行降维
        self.linear3 = nn.Parameter(torch.randn(hidden_size, 1))  # 将结果降维

        self.node_features = nn.Parameter(torch.randn(num_nodes, hidden_size))

        self.attn_layers = nn.ModuleList([Attention(in_dim=hidden_size, hidden_size=hidden_size, dropout=dropout) for i in range(layers)])
        self.ff = nn.ModuleList([PositionwiseFeedForward(in_dim=hidden_size, hidden_size=4 * hidden_size, dropout=dropout) for i in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(normalized_shape=hidden_size) for i in range(2*layers)])


    def forward(self, seq_time, x=None):  # x:(B,N,L,C)

        TIM = self.embedding(seq_time)

        # 主要目的是与输入维度匹配+元学习(因为实验是在一个数据集上与多个模型做对比实验，所以没发挥元学习的作用)
        x_nemb = torch.einsum('blc, nc -> bnlc', TIM, self.node_features)
        if x is None:
            x = torch.zeros_like(x_nemb[...,:self.in_dim])
        x_spd = torch.matmul(x.to(device), self.linear1.to(device))
        x_nemb = torch.matmul(torch.cat([x_spd, x_nemb], dim=-1), self.linear2)

        attns = []
        for i, (attn_layer, ff) in enumerate(zip(self.attn_layers, self.ff)):

            skip1 = x_nemb.contiguous()
            x_attn = self.norm[i](attn_layer(x_nemb)+skip1)

            skip2 = x_attn.contiguous()

            x_nemb = self.norm[i+3](ff(x_attn)+skip2)

            attns.append(x_nemb)

        out = torch.matmul(x_nemb, self.linear3)

        return out, attns[-1]

# 从架构来说，时空建模专家没有用输入序列时间嵌入, 空间建模层用的是GCN方法
class STModel(nn.Module):
    """

    Arguments:
        - TS: Flag that determine when spatial attention will be performed
            - True --> spatial first and then temporal attention will be performed
    """

    def __init__(self, hidden_size, supports_len, num_nodes, dropout, layers, out_dim=1, in_dim=2, TS=True,
                 activation=nn.ReLU()):
        super(STModel, self).__init__()
        self.TS = TS
        self.act = activation

        self.spatial_layers = nn.ModuleList([DCN(c_in=hidden_size, c_out=hidden_size, dropout=dropout, supports_len=supports_len, order=2) for i in range(layers)])
        self.temporal_layers = nn.ModuleList([Attention(in_dim=hidden_size, hidden_size=hidden_size, dropout=dropout) for i in range(layers)])
        self.ed_layers = nn.ModuleList([Attention(in_dim=hidden_size, hidden_size=hidden_size, dropout=dropout) for i in range(layers)])
        self.ff = nn.ModuleList([PositionwiseFeedForward(in_dim=hidden_size, hidden_size=4 * hidden_size, dropout=dropout) for i in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(normalized_shape=hidden_size) for i in range(4*layers)])

        self.start_linear = nn.Parameter(torch.randn(in_dim, hidden_size))

        if out_dim == 1:
            self.proj = nn.Linear(hidden_size, hidden_size + out_dim)
        else:
            self.proj = nn.Linear(hidden_size, out_dim)
        self.out_dim = out_dim

    def forward(self, x, prev_hidden, supports):  # B,N,L,C
        x = torch.matmul(x.to(device), self.start_linear.to(device))
        hiddens = []
        for i, (temporal_layer, spatial_layer, ed_layer, ff) in enumerate(
                zip(self.temporal_layers, self.spatial_layers, self.ed_layers, self.ff)):

            #  与论文和源码的原参相同，先时间注意力再空间建模
            if self.TS:
                skip = x.contiguous()
                x1 = self.norm[i](skip+temporal_layer(x))  # B, N, L, C
                skip = x1.contiguous()
                x_attn = self.norm[i+3](skip+spatial_layer(x1, supports))  # B, N, L, C
            else:
                skip = x.contiguous()
                x1 = self.norm[i](skip + spatial_layer(x, supports))  # B, N, L, C
                skip = x1.contiguous()
                x_attn = self.norm[i+3](skip + temporal_layer(x1))  # B, N, L, C

            if prev_hidden is not None:
                skip = x_attn.contiguous()
                x_attn = self.norm[i+3*2](ed_layer(x_attn, prev_hidden)+skip)
            skip = x_attn.contiguous()
            x = self.norm[i+3*3](skip+ff(x_attn))
            hiddens.append(x)

        out = self.proj(self.act(x))

        return out, hiddens[-1]

# 从架构来说，注意力专家没有用输入序列时间嵌入, 空间建模层用的是注意力方法
class AttentionModel(nn.Module):

    def __init__(self, hidden_size, layers, dropout, edproj=False, in_dim=2, out_dim=1, TS=True,
                 activation=nn.ReLU()):
        super(AttentionModel, self).__init__()
        self.TS = TS
        self.act = activation

        self.start_linear = nn.Parameter(torch.randn(in_dim, hidden_size))

        self.spatial_layers = nn.ModuleList([Attention(hidden_size, hidden_size, dropout=dropout) for i in range(layers)])
        self.temporal_layers = nn.ModuleList([Attention(hidden_size, hidden_size, dropout=dropout) for i in range(layers)])
        self.ed_layers = nn.ModuleList([Attention(hidden_size, hidden_size, dropout=dropout) for i in range(layers)])
        self.ff = nn.ModuleList([PositionwiseFeedForward(hidden_size, 4 * hidden_size, dropout=dropout) for i in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(normalized_shape=hidden_size) for i in range(4*layers)])

        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, x, prev_hidden):  # B,N,L,C
        x = torch.matmul(x.to(device), self.start_linear.to(device))

        for i, (s_layer, t_layer, ff) in enumerate(zip(self.spatial_layers, self.temporal_layers, self.ff)):
            if self.TS:
                skip = x.contiguous()  # B,N,L,C
                x1 = self.norm[i](t_layer(x)+skip)  # B,N,L,C
                skip = x.contiguous().transpose(1, 2)  # B,L,N,C
                x_attn = self.norm[i+3](skip+s_layer(x1.transpose(1, 2)))  # B,L,N,C
            else:
                skip = x.contiguous().transpose(1, 2)   # B,L,N,C
                x1 = self.norm[i](skip + s_layer(x.transpose(1, 2))) # B,L,N,C
                skip = x1.contiguous().transpose(1, 2)  # B,N,L,C
                x_attn = self.norm[i+3](t_layer(x1.transpose(1, 2))+skip).transpose(1, 2)  # B,N,L,C->B,L,N,C


            skip = x_attn.contiguous().transpose(1, 2)  # B,N,L,C
            x_attn = self.norm[i+3*2](skip+self.ed_layers[i](x_attn.transpose(1, 2), prev_hidden))  # B,N,L,C

            skip = x_attn.contiguous()  # B,N,L,C
            x = self.norm[i+3*3](skip+ff(x_attn))  # B,N,L,C

        return self.proj(self.act(x)), x

# 记忆门控网络
class MemoryGate(nn.Module):
    """
    Input
     - input: B, N, T, in_dim, original input
     - hidden: hidden states from each expert, shape: E-length list of (B, N, T, C) tensors, where E is the number of experts
    Output
     - similarity score (i.e., routing probability before softmax function)
    Arguments
     - mem_hid, memory_size: hidden size and total number of memroy units
     - sim: similarity function to evaluate routing probability
     - nodewise: flag to determine routing level. Traffic forecasting could have a more fine-grained routing, because it has additional dimension for the roads
        - True: enables node-wise routing probability calculation, which is coarse-grained one
    """

    def __init__(self, hidden_size, num_nodes, mem_hid=32, input_dim=3, output_dim=1, memory_size=20,
                 sim=nn.CosineSimilarity(dim=-1), nodewise=False, ind_proj=True, attention_type='attention'):
        super(MemoryGate, self).__init__()
        self.sim = sim  # 用余弦相似度
        self.nodewise = nodewise

        self.memory = nn.Parameter(torch.empty(memory_size, mem_hid))  # 元学习方法的memory_node_bank

        self.hid_query =[nn.Parameter(torch.empty(hidden_size, mem_hid),requires_grad=True) for _ in range(3)]
        self.key = [nn.Parameter(torch.empty(hidden_size, mem_hid),requires_grad=True) for _ in range(3)]
        self.value =[nn.Parameter(torch.empty(hidden_size, mem_hid),requires_grad=True) for _ in range(3)]

        self.input_query = nn.Parameter(torch.empty(input_dim, mem_hid))

        self.We1 = nn.Parameter(torch.empty(num_nodes, memory_size))
        self.We2 = nn.Parameter(torch.empty(num_nodes, memory_size))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, input, hidden):  # 计算input_query与各个专家的输出的余弦相似度
        attention = self.attention
        B, N, T, _ = input.size()
        memories = self.query_mem(input)
        scores = []
        for i, h in enumerate(hidden):
            hidden_att = attention(h, i)
            scores.append(self.sim(memories, hidden_att))

        scores = torch.stack(scores, dim=-1)
        return scores

    def attention(self, x, i):
        B, N, T, _ = x.size()
        query = torch.matmul(x, self.hid_query[i].to(x.device))
        key = torch.matmul(x, self.key[i].to(x.device))
        value = torch.matmul(x, self.value[i].to(x.device))

        # 更细粒度的计算，让专家更专业，粗粒度不符合论文思想(粗粒度在源码仅表示为对时间维度求和)
        energy = torch.matmul(query, key.transpose(-1, -2))
        score = torch.softmax(energy, dim=-1)
        out = torch.matmul(score, value)
        return out.expand_as(value)

    def query_mem(self, input):
        B, N, T, _ = input.size()
        mem = self.memory
        query = torch.matmul(input.to(device), self.input_query.to(device))
        energy = torch.matmul(query, mem.T)
        score = torch.softmax(energy, dim=-1)
        out = torch.matmul(score, mem)
        return out

