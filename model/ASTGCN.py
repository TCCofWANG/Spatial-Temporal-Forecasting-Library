# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 11:21:28 2018

@author: gk
"""
from layers.DGCN_related import ST_BLOCK_0
from torch_utils import graph_process
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, InstanceNorm2d


"""
the parameters:
x-> [batch_num,in_channels,num_nodes,Len],
"""


class ASTGCN_block(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, pred_len,tem_size, K, Kt):
        super(ASTGCN_block, self).__init__()
        self.block1 = ST_BLOCK_0(c_in, c_out, num_nodes, tem_size, K, Kt)
        self.block2 = ST_BLOCK_0(c_out, c_out, num_nodes, tem_size, K, Kt)
        self.final_conv = Conv2d(tem_size, pred_len, kernel_size=(1, 1), padding=(0, 0),
                                 stride=(1, 1), bias=True)
        self.w = Parameter(torch.zeros(c_out,num_nodes, pred_len), requires_grad=True)
        nn.init.xavier_uniform_(self.w)

    def forward(self, x, supports):
        x, _, _ = self.block1(x, supports.clone())
        x, d_adj, t_adj = self.block2(x, supports.clone()) # 堆叠两层一样的block
        x = x.permute(0, 3, 2, 1)
        x = self.final_conv(x).transpose(-1,1) # b,C,N,pred_len
        x = x * self.w # 最终三个分量融合的时候 对应的可学习权重
        return x, d_adj, t_adj


class ASTGCN(nn.Module):
    def __init__(self, c_in, c_out, num_nodes,pred_len, week, day, recent, K, Kt):
        super(ASTGCN, self).__init__()
        self.block_w = ASTGCN_block(c_in, c_out, num_nodes,pred_len, week, K, Kt)
        self.block_d = ASTGCN_block(c_in, c_out, num_nodes,pred_len, day, K, Kt)
        self.block_r = ASTGCN_block(c_in, c_out, num_nodes,pred_len, recent, K, Kt)
        self.bn = BatchNorm2d(c_in, affine=False) # 对特征维度进行归一化，mean、std(c_in) 相当于对一个特征下不同站点和不同时间步间拉成一个标准的正态分布 FIXME 为啥这里不用bn3d呢？

    def forward(self, x_w, x_d, x_r, supports):
        x_w = self.bn(x_w)
        x_d = self.bn(x_d)
        x_r = self.bn(x_r) # 进行标准化
        supports=graph_process.graph_laplace_trans(supports)

        supports=torch.tensor(supports).to(x_w.device)
        x_w, _, _ = self.block_w(x_w, supports) # 分别对不同的周期序列进行相同的处理
        x_d, _, _ = self.block_d(x_d, supports)
        x_r, d_adj_r, t_adj_r = self.block_r(x_r, supports)
        out = x_w + x_d + x_r
        return out





