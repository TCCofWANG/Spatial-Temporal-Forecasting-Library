import numpy
from layers.GAT_LSTM_related import *
import torch
import torch.nn.functional as F
from torch import nn

class GAT_LSTM(nn.Module):
    def __init__(self, encoder_layers, decoder_layers, pred_len, seq_len, num_nodes, num_features, n_hidden, dropout=0.1, support_len=1, head=1, type='concat'):
        super(GAT_LSTM, self).__init__()
        self.encoder = nn.ModuleList([G_LSTM_layer(seq_len, num_nodes, num_features, num_features, n_hidden, dropout, support_len, head, type) for i in range(encoder_layers)])
        self.decoder = nn.ModuleList([G_LSTM_layer(pred_len, num_nodes, num_features, num_features, n_hidden, dropout, support_len, head, type) for i in range(decoder_layers)])
        self.transform_layer = nn.ModuleList([GAT(num_features, num_features, num_nodes, dropout, support_len) for i in range(pred_len)])
        self.pred_len = pred_len

    def forward(self, x, adj, **kwargs):
        cell_list = []
        if isinstance(adj, numpy.ndarray):
            adj = torch.tensor(adj).to(x.device)
        adj = [adj]
        B, C, N, L = x.shape
        for encoder in self.encoder:
            x, cell = encoder(x, torch.zeros((B, C, N, 1)).to(x.device), adj)
            cell_list.append(cell)

        cell_list = cell_list[::-1]
        top_x = torch.zeros(B, C, N, self.pred_len).to(x.device)
        for i, decoder in enumerate(self.decoder):
            top_x, _ = decoder(top_x, cell_list[i], adj)

        # top_x:B,C,N,L
        # B,C,N->B,C,N,1->F(x)->B,C,N,1
        output = [self.transform_layer[i](top_x[:,:,:,i].unsqueeze(-1), adj) for i in range(self.pred_len)]
        output = torch.concat(output, dim=-1)
        return output