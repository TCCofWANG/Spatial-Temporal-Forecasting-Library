import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x) # 对特征维度进行标准化
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

class STEmbedding(nn.Module):
    '''
    spatio-temporal embedding
    SE:     [num_vertex, D]
    TE:     [batch_size, num_his + num_pred, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, num_his + num_pred, num_vertex, D]
    '''

    def __init__(self, D, bn_decay):
        super(STEmbedding, self).__init__()
        self.FC_se = FC(
            input_dims=[D, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)

        self.FC_te = FC(
            input_dims=[295, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)  # input_dims = time step per day + days per week=288+7=295

    def forward(self, SE, TE, points_per_hour=12):
        SE=torch.tensor(SE).to(torch.float32).to('cuda') # 节点的Embedding结果(node2Vec的结果)
        # spatial embedding
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.FC_se(SE)# 操作的是特征维度
        # temporal embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7) # 周几
        timeofday = torch.empty(TE.shape[0], TE.shape[1], points_per_hour*24) # 一天中第几个部分(24小时*一个小时有多少记录)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(((TE[..., 2][i]+0.5)*6).to(torch.int64) % 7, 7) # 周几
        for j in range(TE.shape[0]):
            hour=(TE[...,-2]+0.5)*23# 一天中的第几个小时
            min=(TE[...,-1]+0.5)*59#一小时中的第几分钟
            minofday=hour*points_per_hour+min//points_per_hour
            timeofday[j] = F.one_hot(minofday[j].to(torch.int64) % (points_per_hour*24), points_per_hour*24) # 在一天中的第几个
        TE = torch.cat((dayofweek, timeofday), dim=-1) # 在特征维度进行concat
        TE = TE.unsqueeze(dim=2)
        TE = self.FC_te(TE.cuda()) # 操作的是特征维度，进行降维操作
        del dayofweek, timeofday
        return SE + TE

