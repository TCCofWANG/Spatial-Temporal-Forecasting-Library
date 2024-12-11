from layers.STWave_related import *
import math
import torch
import numpy as np
import torch.nn as nn

class STWave(nn.Module):
    def __init__(self, heads,
                 dims, layers, samples,
                 localadj, spawave, temwave,
                 input_len, output_len,**kwargs):
        '''
        Parameters
        ----------
        heads:头数
        dims:每一个头的维度
        layers:Encoder的层数
        samples:
        localadj:根据节点的之间的距离选择出了离节点i较近的节点
        spawave: 正常的adj得到的特征值和特征向量
        temwave: 根据时间序列节点之间的距离构建出的adj的特征值和特征向量
        input_len:输入长度
        output_len：输出预测长度
        '''
        super(STWave, self).__init__()
        self.args=kwargs.get('args')
        features = heads * dims
        I = torch.arange(localadj.shape[0]).unsqueeze(-1)
        localadj = torch.cat([I, torch.from_numpy(localadj)], -1) #不仅仅要和距离最近的点计算Attention还要和自己也计算Attention
        self.input_len = input_len

        self.dual_enc = nn.ModuleList(
            [Dual_Enconder(heads, dims, samples, localadj, spawave, temwave) for i in range(layers)])
        self.adp_f = Adaptive_Fusion(heads, dims)

        self.pre_l = nn.Conv2d(input_len, output_len, (1, 1))
        self.pre_h = nn.Conv2d(input_len, output_len, (1, 1))
        self.pre = nn.Conv2d(input_len, output_len, (1, 1))

        self.start_emb_l = FeedForward([self.args.num_features, features, features])
        self.start_emb_h = FeedForward([self.args.num_features, features, features])
        self.end_emb = FeedForward([features, features, 1])
        self.end_emb_l = FeedForward([features, features, 1])
        self.te_emb = TemEmbedding(features)


    def forward(self,input,adj,**kwargs):
        input=input.transpose(1,-1)#(B,T,N,C)
        XL,XH=disentangle(input)
        xl, xh = self.start_emb_l(XL), self.start_emb_h(XH)# 将特征维度进行升维
        seq_time = kwargs.get('seqs_time')
        pred_time = kwargs.get('targets_time')
        time=torch.concat([seq_time,pred_time],dim=-1) # 将历史时间序列和未来的时间序列进行concat
        # time(dayofyear, dayofmonth, dayofweek, hourofday, minofhour)
        hour = (time[:, -2:-1, ...] + 0.5) * 23  # 得到第几个小时
        min = (time[:, -1:, ...] + 0.5) * 59  # 得到第几分钟
        hour_index = (hour * 60 + min) / (60 / self.args.points_per_hour)
        day_index = (time[:, 2:3, ...] + 0.5) * (6 - 0) #(B,C=1,N=1,L)
        TE=torch.concat([hour_index,day_index],dim=1).squeeze(-2).transpose(-1,-2)#(B,L,2)
        te = self.te_emb(TE)# 对时间进行Embedding
        for enc in self.dual_enc:
            xl, xh = enc(xl, xh, te[:, :self.input_len, :, :])

        hat_y_l = self.pre_l(xl)
        hat_y_h = self.pre_h(xh)
        hat_y = self.adp_f(hat_y_l, hat_y_h, te[:, self.input_len:, :, :])
        hat_y, hat_y_l = self.end_emb(hat_y), self.end_emb_l(hat_y_l)#(B,L,N,C)
        hat_y=hat_y.transpose(1,-1)
        return hat_y


