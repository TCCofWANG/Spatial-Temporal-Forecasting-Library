import torch
from torch_utils.graph_process import *
import torch.nn as nn
import numpy as np
from layers.STTNS_related import *

# model input shape:[B, C, N, T]
# model output shape:[B, C, N, T]
class STTNSNet(nn.Module):
    def __init__(self, adj, in_channels, embed_size, time_num,
                 num_layers, T_dim, output_T_dim, heads, dropout, forward_expansion):
        self.num_layers = num_layers
        super(STTNSNet, self).__init__()
        adj=calculate_laplacian_with_self_loop(adj)
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.transformer = Transformer(embed_size, heads, adj, time_num, dropout, forward_expansion) # 主体模型部分
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)
        self.conv3 = nn.Conv2d(embed_size, in_channels, 1)

    def forward(self, x, t,**kwargs):
        t = calculate_laplacian_with_self_loop(t)
        # x[B, C, N, L]
        x = self.conv1(x)  # [B, embed_size, N, T],对特征维度进行升维

        x = x.permute(0,2, 3, 1)  # [B,N,T,embed_size]
        x = self.transformer(x, x, x, t, self.num_layers)  # [B,N, T, embed_size]

        # 预测时间T_dim，转换时间维数(prediction Layers)
        x = x.clone() # [B, N, T, C], C = embed_size
        x = x.permute(0, 2, 1, 3)  # [B, T, N, C]
        x = self.conv2(x)  # [B, out_T_dim, N, C]

        # 将通道降为in_channels
        x = x.permute(0, 3, 2, 1)  # [B, C, N, out_T_dim]
        x = self.conv3(x)  # [B, in_channels, N, out_T_dim]
        out = x
        return out
