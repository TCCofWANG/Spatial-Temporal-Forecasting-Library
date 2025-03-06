import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.TESTAM_related import *
from copy import deepcopy
import numpy as np

class TESTAM(nn.Module):
    """
    TESTAM model
    """

    def __init__(self, num_features, num_nodes, seq_len=12, dropout=0.3, in_dim=2, out_dim=12, hidden_size=32, layers=3, points_per_hour=15, prob_mul=False,
                 **args):
        super(TESTAM, self).__init__()
        self.flag = True
        self.use_uncertainty = True
        self.warmup_epoch = 0
        self.quantile = 0.7
        self.dropout = dropout
        self.seq_len = seq_len
        self.prob_mul = prob_mul
        self.threshold = 0.
        self.supports_len = 2
        self.points_per_hour = points_per_hour
        self.identity_expert = TemporalModel(hidden_size, num_nodes, in_dim=in_dim, layers=layers, dropout=dropout)
        self.adaptive_expert = STModel(hidden_size, self.supports_len, num_nodes, in_dim=in_dim, layers=layers,
                                       dropout=dropout, out_dim=1)
        self.attention_expert = AttentionModel(hidden_size, in_dim=in_dim, layers=layers, dropout=dropout)

        self.gate_network = MemoryGate(hidden_size, num_nodes, input_dim=in_dim)

        self.output_linear = nn.Linear(seq_len,out_dim)
#        self.dim_linear = nn.Linear(1, num_features)

        for model in [self.identity_expert, self.adaptive_expert, self.attention_expert]:
            for n, p in model.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def masked_mae(self, preds, labels, null_val=np.nan, reduce=True):
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels > null_val)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(preds - labels)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        if reduce:
            return torch.mean(loss)
        else:
            return loss

    def get_quantile_label(self, gated_loss, gate, real):
        gated_loss = gated_loss.unsqueeze(1)
        max_quantile = gated_loss.quantile(self.quantile)
        min_quantile = gated_loss.quantile(1 - self.quantile)
        incorrect = (gated_loss > max_quantile).expand_as(gate)
        correct = ((gated_loss < min_quantile) & (real[:, :1, :, :].unsqueeze(1) > self.threshold)).expand_as(gate)
        cur_expert = gate.argmax(dim=1, keepdim=True)
        not_chosen = gate.topk(dim=1, k=2, largest=False).indices
        selected = torch.zeros_like(gate).scatter_(-1, cur_expert, 1.0)
        scaling = torch.zeros_like(gate).scatter_(-1, not_chosen, 0.5)
        selected[incorrect] = scaling[incorrect]
        l_worst_avoidance = selected.detach()
        selected = torch.zeros_like(gate).scatter(-1, cur_expert, 1.0) * correct
        l_best_choice = selected.detach()
        return l_worst_avoidance, l_best_choice

    def get_label(self, ind_loss, gate, real):
        empty_val = (real.expand_as(gate)) <= self.threshold
        max_error = ind_loss.argmax(dim=1, keepdim=True)
        cur_expert = gate.argmax(dim=1, keepdim=True)
        incorrect = max_error == cur_expert
        selected = torch.zeros_like(gate).scatter(-1, cur_expert, 1.0)
        scaling = torch.ones_like(gate) * ind_loss
        scaling = scaling.scatter(-1, max_error, 0.)
        scaling = scaling / (scaling.sum(dim=1, keepdim=True)) * (1 - selected)
        l_worst_avoidance = torch.where(incorrect, scaling, selected)
        l_worst_avoidance = torch.where(empty_val, torch.zeros_like(gate), l_worst_avoidance)
        l_worst_avoidance = l_worst_avoidance.detach()
        min_error = ind_loss.argmin(dim=1, keepdim=True)
        correct = min_error == cur_expert
        scaling = torch.zeros_like(gate)
        scaling = scaling.scatter(-1, min_error, 1.)
        l_best_choice = torch.where(correct, selected, scaling)
        l_best_choice = torch.where(empty_val, torch.zeros_like(gate), l_best_choice)
        l_best_choice = l_best_choice.detach()
        return l_worst_avoidance, l_best_choice

    def loss(self, predict, gate, res, real, epoch):
        res = res.permute(0, 4, 3, 1, 2)
        res = self.output_linear(res)
        gate = gate.unsqueeze(-1).permute(0, 3, 4, 1, 2)
        gate = self.output_linear(gate)
        ind_loss = self.masked_mae(res, real[:, :1, :, :].unsqueeze(1), self.threshold, reduce=None)
        if self.flag:
            gated_loss = self.masked_mae(predict, real[:, :1, :, :], reduce=None)
            l_worst_avoidance, l_best_choice = self.get_quantile_label(gated_loss, gate, real)
        else:
            l_worst_avoidance, l_best_choice = self.get_label(ind_loss, gate, real)

        if self.use_uncertainty:
            uncertainty = self.get_uncertainty(real, threshold=self.threshold)[:, :1, :, :].unsqueeze(1)
        else:
            uncertainty = torch.ones_like(gate)

        worst_avoidance = -.5 * l_worst_avoidance * torch.log(gate) * (2 - uncertainty)
        best_choice = -.5 * l_best_choice * torch.log(gate) * uncertainty

        if epoch <= self.warmup_epoch:
            loss = 0
        else:
            loss = worst_avoidance.mean() + best_choice.mean()

        return loss

    def get_uncertainty(self, x, mode='psd', threshold=0.0):

        def _acorr(x, dim=-1):
            size = x.size(dim)
            x_fft = torch.fft.fft(x, dim=dim)
            acorr = torch.fft.ifft(x_fft * x_fft.conj(), dim=dim).real
            return acorr / (size ** 2)

        def nanstd(x, dim, keepdim=False):
            return torch.sqrt(
                torch.nanmean(
                    torch.pow(torch.abs(x - torch.nanmean(x, dim=dim, keepdim=True)), 2),
                    dim=dim, keepdim=keepdim
                )
            )

        with torch.no_grad():
            if mode == 'acorr':
                std = x.std(dim=-1, keepdim=True)
                corr = _acorr(x, dim=-1)
                x_noise = x + std * torch.randn((1, 1, 1, x.size(-1)), device=x.device) / 2
                corr_w_noise = _acorr(x_noise, dim=-1)
                corr_changed = torch.abs(corr - corr_w_noise)
                uncertainty = torch.ones_like(corr_changed) * (corr_changed > corr_changed.quantile(1 - self.quantile))
            elif mode == 'psd':
                from copy import deepcopy as cp
                vals = cp(x)
                vals[vals <= threshold] = torch.nan
                diff = vals[:, :, :, 1:] - vals[:, :, :, :-1]
                corr_changed = torch.nanmean(torch.abs(diff), dim=-1, keepdim=True) / (
                            nanstd(diff, dim=-1, keepdim=True) + 1e-6)
                corr_changed[corr_changed != corr_changed] = 0.
                uncertainty = torch.ones_like(corr_changed) * (corr_changed < corr_changed.quantile(self.quantile))
            else:
                raise NotImplementedError
            return uncertainty

    def forward(self, input, adj,**kwargs):
        """
        input: B, in_dim, N, T
        o_identity shape B, N, T, 1
        """
        seqs_time=kwargs.get("seqs_time")
        targets_time=kwargs.get('targets_time')
        targets = kwargs.get('targets')
        epoch = kwargs.get('epoch')

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
        ind_out = torch.stack([o_identity, o_adaptive, o_attention], dim=-1)

        route_prob_max, routes = torch.max(gate, dim=-1)
        route_prob_max = route_prob_max.view(-1)
        routes = routes.view(-1)

        # 选最好的
        for i in range(len(outs)):
            cur_out = outs[i].view(-1, 1)
            indices = torch.eq(routes, i).nonzero(as_tuple=True)[0]
            out[indices] = cur_out[indices]

        if self.prob_mul:
            out = out * (route_prob_max).unsqueeze(dim=-1)

        out = out.view(B,1, N,T)
        out = self.output_linear(out)

        if self.training:
            loss = self.loss(out, gate, ind_out, targets, epoch)
            return out, loss
        else:
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
            self.proj = nn.Linear(hidden_size, out_dim)
        else:
            self.proj = nn.Linear(hidden_size, hidden_size + out_dim)
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

