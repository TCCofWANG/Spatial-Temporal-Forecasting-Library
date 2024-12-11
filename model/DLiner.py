import torch
import torch.nn as nn
import numpy as np

'''该模型测试聚合节点的信息是否有用'''

class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs,adj):
        super(DLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_nodes=configs.num_nodes
        self.num_features=configs.num_features
        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.channels = configs.num_features

        # 特征间使用的FC是共享的
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        # 不同的节点间使用的FC是不共享的
        self.Linear_Seasonal_stack=nn.ModuleList(self.Linear_Seasonal for _ in range(self.num_nodes))
        self.Linear_Trend_stack = nn.ModuleList(self.Linear_Trend for _ in range(self.num_nodes))

        # 将特征映射为1
        self.Linear_Feature=nn.Linear(self.num_features,1)


    def forward(self, x,adj,**kwargs):
        # x: [B,C,N,L]
        B,C,N,L=x.shape
        seasonal_init, trend_init = self.decompsition(x)  # 通过滤波拆解趋势项和季节项
        # 循环不同的节点
        seasonal_output=torch.zeros((B,C,0,self.pred_len),device=x.device)
        trend_output = torch.zeros((B, C, 0, self.pred_len), device=x.device)
        for i in range(len(self.Linear_Seasonal_stack)):
            seasonal_output_ = self.Linear_Seasonal_stack[i](seasonal_init[:,:,i:i+1,...])  # 直接linear，由输入长度映射到输出长度
            trend_output_ = self.Linear_Trend_stack[i](trend_init[:,:,i:i+1,...])           # 直接linear，由输入长度映射到输出长度
            seasonal_output=torch.cat([seasonal_output,seasonal_output_],dim=-2)
            trend_output = torch.cat([trend_output, trend_output_], dim=-2)
        output=seasonal_output+trend_output # [B,C,N,L]
        output=self.Linear_Feature(output.transpose(1,-1)).transpose(1,-1) #(B,C=1,N,L)
        return output  # [B,C,N,L]


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        '''输入输出x:(B,C,N,L)'''
        # padding on the both ends of time series
        front = x[:, :,:, 0:1].repeat(1, 1, 1,(self.kernel_size - 1) // 2)
        end = x[:, : ,:,-1:].repeat(1, 1, 1,(self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)
        B,C,N,L=x.shape
        x=x.reshape(B,-1,L)
        x = self.avg(x)
        x=x.reshape(B,C,N,-1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
