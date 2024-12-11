import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.D2STGNN_related.D2STGNN_related_DifBlock import DifBlock
from layers.D2STGNN_related.D2STGNN_related_InhBlock import InhBlock
from layers.D2STGNN_related.D2STGNN_related_DynamicGraphConstructor import DynamicGraphConstructor


class D2STGNN(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self._in_feat       = model_args['num_features']
        self._hidden_dim    = model_args['num_hidden']
        self._node_dim      = model_args['node_hidden']
        self._forecast_dim  = 256
        self._output_hidden = 512
        self._output_dim    = model_args['seq_len']

        self._num_nodes     = model_args['num_nodes']
        self._k_s           = model_args['k_s']
        self._k_t           = model_args['k_t']
        self._num_layers    = 2 # 原始设置为5，但是太大了跑不动。

        model_args['use_pre']   = False
        model_args['dy_graph']  = True
        model_args['sta_graph'] = True

        self._model_args    = model_args

        # start embedding layer
        self.embedding      = nn.Linear(self._in_feat, self._hidden_dim)

        # time embedding
        self.T_i_D_emb  = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb  = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))



        # Decoupled Spatial Temporal Layer
        self.layers = nn.ModuleList([DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args)])
        for _ in range(self._num_layers - 1):
            self.layers.append(DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args))

        # dynamic and static hidden graph constructor
        if model_args['dy_graph']:
            self.dynamic_graph_constructor  = DynamicGraphConstructor(**model_args)

        # node embeddings
        self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))

        # output layer
        self.out_fc_1   = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2   = nn.Linear(self._output_hidden, model_args['gap'])

        self.out_fc_3=nn.Linear( model_args['seq_len'], model_args['pred_len'])

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def _graph_constructor(self, **inputs):
        E_d = inputs['node_embedding_u']
        E_u = inputs['node_embedding_d']
        if self._model_args['sta_graph']:
            static_graph = [F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)]
        else:
            static_graph = []
        if self._model_args['dy_graph']:
            dynamic_graph   = self.dynamic_graph_constructor(**inputs)
        else:
            dynamic_graph   = []
        return static_graph, dynamic_graph

    def change_time(self,seq_time):
        hour = (seq_time[:, -2:-1, ...] + 0.5) * 23  # 得到第几个小时
        min = (seq_time[:, -1:, ...] + 0.5) * 59  # 得到第几分钟
        hour_index = (hour * 60 + min) / (60 / self._model_args["points_per_hour"])
        day = (seq_time[:, 2:3, ...] + 0.5) * (6 - 0)
        return hour_index,day

    def _prepare_inputs(self, history_data,seq_time):
        num_feat    = self._model_args['num_features']
        # node embeddings，得到两个可训练的向量
        node_emb_u  = self.node_emb_u  # [N, d]
        node_emb_d  = self.node_emb_d  # [N, d]

        hour_index,day=self.change_time(seq_time)

        # time slot embedding，得到时间的Embedding
        time_in_day_feat = self.T_i_D_emb[hour_index.squeeze(1).repeat(1,self._model_args["num_nodes"],1).type(torch.LongTensor)].transpose(1,2)    # [B, L, N, d]
        day_in_week_feat = self.D_i_W_emb[day.squeeze(1).repeat(1,self._model_args["num_nodes"],1).type(torch.LongTensor)].transpose(1,2)       # [B, L, N, d]
        # traffic signals
        history_data = history_data[:, :, :, :num_feat]

        return history_data, node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat

    def forward(self, history_data,adj,**kwargs):
        """Feed forward of D2STGNN.

        Args:
            history_data (Tensor): history data with shape: [B, C, N, L]

        Returns:
            torch.Tensor: prediction data with shape: [B, C, N, L]
        """
        history_data=history_data.transpose(1,-1) #(B,L,C,N)
        seq_time = kwargs.get('seqs_time')
        pred_time = kwargs.get('targets_time')
        # ==================== Prepare Input Data ==================== #
        history_data, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat = self._prepare_inputs(history_data,seq_time=seq_time)

        # ========================= Construct Graphs ========================== #
        static_graph, dynamic_graph = self._graph_constructor(node_embedding_u=node_embedding_u, node_embedding_d=node_embedding_d, history_data=history_data, time_in_day_feat=time_in_day_feat, day_in_week_feat=day_in_week_feat)
        # static_graph就是普通的adj，而dynamic_graph是不同阶次的P^{local}
        # Start embedding layer
        history_data   = self.embedding(history_data)

        dif_forecast_hidden_list = []
        inh_forecast_hidden_list = []

        inh_backcast_seq_res = history_data
        for _, layer in enumerate(self.layers):
            inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden = layer(inh_backcast_seq_res, dynamic_graph, static_graph, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat)
            dif_forecast_hidden_list.append(dif_forecast_hidden)
            inh_forecast_hidden_list.append(inh_forecast_hidden)

        # Output Layer
        dif_forecast_hidden = sum(dif_forecast_hidden_list)
        inh_forecast_hidden = sum(inh_forecast_hidden_list)
        forecast_hidden     = dif_forecast_hidden + inh_forecast_hidden

        # regression layer
        forecast    = self.out_fc_2(F.relu(self.out_fc_1(F.relu(forecast_hidden))))
        forecast    = forecast.transpose(1 ,2).contiguous().view(forecast.shape[0], forecast.shape[2], -1)
        forecast=forecast.unsqueeze(1)
        forecast=self.out_fc_3(forecast)
        return forecast #(B,C,N,L)

class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.estimation_gate= EstimationGate(node_emb_dim=model_args['node_hidden'], time_emb_dim=model_args['time_emb_dim'], hidden_dim=64)
        self.dif_layer      = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inh_layer      = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)

    def forward(self, history_data: torch.Tensor, dynamic_graph: torch.Tensor, static_graph, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat):
        """decouple layer

        Args:
            history_data (torch.Tensor): input data with shape (B, L, N, D)
            dynamic_graph (list of torch.Tensor): dynamic graph adjacency matrix with shape (B, N, k_t * N)
            static_graph (ist of torch.Tensor): the self-adaptive transition matrix with shape (N, N)
            node_embedding_u (torch.Parameter): node embedding E_u
            node_embedding_d (torch.Parameter): node embedding E_d
            time_in_day_feat (torch.Parameter): time embedding T_D
            day_in_week_feat (torch.Parameter): time embedding T_W

        Returns:
            torch.Tensor: the un decoupled signal in this layer, i.e., the X^{l+1}, which should be feeded to the next layer. shape [B, L', N, D].
            torch.Tensor: the output of the forecast branch of Diffusion Block with shape (B, L'', N, D), where L''=output_seq_len / model_args['gap'] to avoid error accumulation in auto-regression.
            torch.Tensor: the output of the forecast branch of Inherent Block with shape (B, L'', N, D), where L''=output_seq_len / model_args['gap'] to avoid error accumulation in auto-regression.
        """

        gated_history_data  = self.estimation_gate(node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data) # 得到X^{dif}
        dif_backcast_seq_res, dif_forecast_hidden = self.dif_layer(history_data=history_data, gated_history_data=gated_history_data, dynamic_graph=dynamic_graph, static_graph=static_graph) # 这部分是扩散部分
        inh_backcast_seq_res, inh_forecast_hidden = self.inh_layer(dif_backcast_seq_res) # 这部分是固定块
        return inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden


class EstimationGate(nn.Module):
    """The estimation gate module."""

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(2 * node_emb_dim + time_emb_dim * 2, hidden_dim)
        self.activation = nn.ReLU()
        self.fully_connected_layer_2 = nn.Linear(hidden_dim, 1)

    def forward(self, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data):
        """Generate gate value in (0, 1) based on current node and time step embeddings to roughly estimating the proportion of the two hidden time series."""

        batch_size, seq_length, _, _ = time_in_day_feat.shape
        estimation_gate_feat = torch.cat([time_in_day_feat, day_in_week_feat, node_embedding_u.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length,  -1, -1), node_embedding_d.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length,  -1, -1)], dim=-1)
        hidden = self.fully_connected_layer_1(estimation_gate_feat)
        hidden = self.activation(hidden)
        # activation
        estimation_gate = torch.sigmoid(self.fully_connected_layer_2(hidden))[:, -history_data.shape[1]:, :, :]
        history_data = history_data * estimation_gate
        return history_data






