import torch
from torch import nn

from layers.STID_related import MultiLayerPerceptron


class STID(nn.Module):
    """
    The implementation of CIKM 2022 short paper
        "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting"
    Link: https://arxiv.org/abs/2208.05233
    """

    def __init__(self, model_args):
        super().__init__()
        self.args=model_args
        # attributes
        self.num_nodes = model_args.num_nodes
        self.embed_dim = model_args.d_model
        self.node_dim=self.embed_dim

        self.input_len = model_args.seq_len
        self.input_dim = model_args.num_features
        self.output_len = model_args.pred_len
        self.num_layer = model_args.num_layer

        self.temp_dim_tid = self.embed_dim
        self.temp_dim_diw = self.embed_dim
        self.time_of_day_size = model_args.time_of_day_size
        self.day_of_week_size = model_args.day_of_week_size

        self.if_time_in_day = model_args.if_T_i_D
        self.if_day_in_week = model_args.if_D_i_W
        self.if_spatial = model_args.if_node

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_time_in_day) + \
            self.temp_dim_diw*int(self.if_day_in_week)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def forward(self, seqs,adj,**kwargs):
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """

        # prepare data
        seqs = seqs.permute(0, 3, 2, 1)#[B, L, N, C]
        input_data = seqs[..., range(self.input_dim)]
        seq_time=kwargs.get('seqs_time')
        pred_time=kwargs.get('targets_time')
        #time(dayofyear, dayofmonth, dayofweek, hourofday, minofhour)


        if self.if_time_in_day:
            hour = (seq_time[:,-2:-1,...]+0.5)*23 # 得到第几个小时
            min = (seq_time[:, -1:, ...] + 0.5) *59  # 得到第几分钟
            hour_index=(hour*60+min)/(60/self.args.points_per_hour)
            time_in_day_emb = self.time_in_day_emb[hour_index[...,-1].squeeze(1).repeat(1,self.args.num_nodes).type(torch.LongTensor)]#(B,N,D)
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            day=(seq_time[:,2:3,...]+0.5)*(6 - 0)
            day_in_week_emb = self.day_in_week_emb[day[...,-1].squeeze(1).repeat(1,self.args.num_nodes).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        # encoding
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden)

        return prediction.permute(0, 3, 2, 1)