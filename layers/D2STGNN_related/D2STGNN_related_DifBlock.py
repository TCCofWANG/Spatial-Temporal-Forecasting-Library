import torch.nn as nn
import torch

class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256, use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        """Diffusion block

        Args:
            hidden_dim (int): hidden dimension.
            forecast_hidden_dim (int, optional): forecast branch hidden dimension. Defaults to 256.
            use_pre (bool, optional): if use predefined graph. Defaults to None.
            dy_graph (bool, optional): if use dynamic graph. Defaults to None.
            sta_graph (bool, optional): if use static graph (the adaptive graph). Defaults to None.
        """

        super().__init__()
        self.pre_defined_graph = model_args['adjs']

        # diffusion model
        self.localized_st_conv = STLocalizedConv(hidden_dim, pre_defined_graph=self.pre_defined_graph, use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph, **model_args)

        # forecast
        self.forecast_branch = Forecast(hidden_dim, forecast_hidden_dim=forecast_hidden_dim, **model_args)
        # backcast
        self.backcast_branch = nn.Linear(hidden_dim, hidden_dim)
        # esidual decomposition
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        """Diffusion block, containing the diffusion model, forecast branch, backcast branch, and the residual decomposition link.

        Args:
            history_data (torch.Tensor): history data with shape [batch_size, seq_len, num_nodes, hidden_dim]
            gated_history_data (torch.Tensor): gated history data with shape [batch_size, seq_len, num_nodes, hidden_dim]
            dynamic_graph (list): dynamic graphs.
            static_graph (list): static graphs (the adaptive graph).

        Returns:
            torch.Tensor: the output after the decoupling mechanism (backcast branch and the residual link), which should be fed to the inherent model.
                          Shape: [batch_size, seq_len', num_nodes, hidden_dim]. Kindly note that after the st conv, the sequence will be shorter.
            torch.Tensor: the output of the forecast branch, which will be used to make final prediction.
                          Shape: [batch_size, seq_len'', num_nodes, forecast_hidden_dim]. seq_len'' = future_len / gap.
                          In order to reduce the error accumulation in the AR forecasting strategy, we let each hidden state generate the prediction of gap points, instead of a single point.
        """

        # diffusion model
        hidden_states_dif = self.localized_st_conv(gated_history_data, dynamic_graph, static_graph)
        # forecast branch: use the localized st conv to predict future hidden states.
        forecast_hidden = self.forecast_branch(gated_history_data, hidden_states_dif, self.localized_st_conv, dynamic_graph, static_graph)
        # backcast branch: use FC layer to do backcast
        backcast_seq = self.backcast_branch(hidden_states_dif)
        # residual decomposition: remove the learned knowledge from input data
        backcast_seq = backcast_seq
        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(history_data, backcast_seq)

        return backcast_seq_res, forecast_hidden


class Forecast(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_len']
        self.forecast_fc    = nn.Linear(hidden_dim, forecast_hidden_dim)
        self.model_args     = model_args

    def forward(self, gated_history_data, hidden_states_dif, localized_st_conv, dynamic_graph, static_graph):
        predict = []
        history = gated_history_data
        predict.append(hidden_states_dif[:, -1, :, :].unsqueeze(1)) # 取最后一个时间步来预测未来的
        for _ in range(int(self.output_seq_len / self.model_args['gap'])-1):
            _1 = predict[-self.k_t:]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2  = history[:, -sub:, :, :]
                _1  = torch.cat([_2] + _1, dim=1)
            else:
                _1  = torch.cat(_1, dim=1)
            predict.append(localized_st_conv(_1, dynamic_graph, static_graph))
        predict = torch.cat(predict, dim=1)
        predict = self.forecast_fc(predict)
        return predict

class STLocalizedConv(nn.Module):
    def __init__(self, hidden_dim, pre_defined_graph=None, use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        # gated temporal conv
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = hidden_dim

        # graph conv
        self.pre_defined_graph = pre_defined_graph
        self.use_predefined_graph = use_pre
        self.use_dynamic_hidden_graph = dy_graph
        self.use_static__hidden_graph = sta_graph

        self.support_len = len(self.pre_defined_graph) + \
            int(dy_graph) + int(sta_graph)
        self.num_matric = (int(use_pre) * len(self.pre_defined_graph) + len(
            self.pre_defined_graph) * int(dy_graph) + int(sta_graph)) * self.k_s + 1
        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self.get_graph(self.pre_defined_graph)

        self.fc_list_updt = nn.Linear(
            self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        self.gcn_updt = nn.Linear(
            self.hidden_dim*self.num_matric, self.hidden_dim)

        # others
        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.activation = nn.ReLU()

    def gconv(self, support, X_k, X_0):
        out = [X_0]
        for graph in support:
            if len(graph.shape) == 2:  # staitic or predefined graph
                pass
            else:
                graph = graph.unsqueeze(1)
            H_k = torch.matmul(graph, X_k)
            out.append(H_k)
        out = torch.cat(out, dim=-1)
        out = self.gcn_updt(out)
        out = self.dropout(out)
        return out

    def get_graph(self, support):
        # Only used in static including static hidden graph and predefined graph, but not used for dynamic graph.
        graph_ordered = []
        mask = 1 - torch.eye(support[0].shape[0]).to(support[0].device) # 对角阵为0，其余都为1
        for graph in support:
            k_1_order = graph                           # 1 order
            graph_ordered.append(k_1_order * mask)
            # e.g., order = 3, k=[2, 3]; order = 2, k=[2]
            for k in range(2, self.k_s+1):
                k_1_order = torch.matmul(graph, k_1_order)
                graph_ordered.append(k_1_order * mask)
        # get st localed graph 得到kt个P^{local}
        st_local_graph = []
        for graph in graph_ordered:
            graph = graph.unsqueeze(-2).expand(-1, self.k_t, -1)
            graph = graph.reshape(
                graph.shape[0], graph.shape[1] * graph.shape[2])
            # [num_nodes, kernel_size x num_nodes]
            st_local_graph.append(graph)
        # [order, num_nodes, kernel_size x num_nodes]
        return st_local_graph

    def forward(self, X, dynamic_graph, static_graph):
        # X: [bs, seq, nodes, feat]
        # [bs, seq, num_nodes, ks, num_feat]
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3) # 这个是按照self.k_t为窗口长度，每一个次滑动1，来展开维度1，主要为了后面有根据窗口进行操作的
        # seq_len is changing
        batch_size, seq_len, num_nodes, kernel_size, num_feat = X.shape

        # support
        support = []
        # predefined graph
        if self.use_predefined_graph:
            support = support + self.pre_defined_graph
        # dynamic graph
        if self.use_dynamic_hidden_graph:
            # k_order is caled in dynamic_graph_constructor component
            support = support + dynamic_graph
        # predefined graphs and static hidden graphs
        if self.use_static__hidden_graph:
            support = support + self.get_graph(static_graph)

        # parallelize
        X = X.reshape(batch_size, seq_len, num_nodes, kernel_size * num_feat)
        # batch_size, seq_len, num_nodes, kernel_size * hidden_dim
        out = self.fc_list_updt(X)
        out = self.activation(out)
        out = out.view(batch_size, seq_len, num_nodes, kernel_size, num_feat)
        X_0 = torch.mean(out, dim=-2) # 根据原始论文的代码这里应该是sum，但影响不大
        # batch_size, seq_len, kernel_size x num_nodes, hidden_dim
        X_k = out.transpose(-3, -2).reshape(batch_size,
                                            seq_len, kernel_size*num_nodes, num_feat)
        # Nx3N 3NxD -> NxD: batch_size, seq_len, num_nodes, hidden_dim
        hidden = self.gconv(support, X_k, X_0) # 得到所有时间步的H_t^{dif}
        return hidden

class ResidualDecomp(nn.Module):
    """Residual decomposition."""

    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ReLU()

    def forward(self, x, y):
        u = x - self.ac(y)
        u = self.ln(u)
        return u





