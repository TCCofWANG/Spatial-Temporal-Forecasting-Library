import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from layers.WAVGCRN_related import *
import sys
import pywt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WavGCRN(nn.Module):
    def __init__(self,gcn_depth,num_nodes,predefined_A=None,seq_length=12,out_dim=3,in_dim=3,output_dim=12,
                 list_weight=[0.05, 0.95, 0.95],cl_decay_steps=4000,hidden_size=64,level=1,points_per_hour=12):
        super(WavGCRN, self).__init__()
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.out_dim = out_dim
        self.level = level
        self.points_per_hour = points_per_hour
        self.predefined_A = predefined_A

        self.seq_length = seq_length

        self.hidden_size = hidden_size
        self.in_dim = in_dim

        self.fc_final = nn.Linear(self.hidden_size, self.out_dim)

        dims = [in_dim + self.hidden_size + 1, self.hidden_size]  # +1是因为融入时间步维度

        self.aggC,self.aggH,self.camH,self.camC = [feature_agg(self.hidden_size * 2, self.hidden_size) for i in range(4)]
        self.IDWT_L, self.IDWT_H = [IDWTL(num_nodes, self.hidden_size) for i in range(2)]

        self.gz1D, self.gz2D, self.gr1D, self.gr2D, \
        self.gc1D, self.gc2D, self.gz1AD, self.gz2AD, \
        self.gr1AD, self.gr2AD, self.gc1AD, self.gc2AD, \
        self.gz1_de, self.gz2_de, self.gr1_de, self.gr2_de, \
        self.gc1_de, self.gc2_de = [GCRN(dims, gcn_depth, *list_weight) for i in range(18)]

        self.use_curriculum_learning = True
        self.cl_decay_steps = cl_decay_steps
        self.gcn_depth = gcn_depth

    def preprocessing(self, adj, predefined_A):
        adj = adj + torch.eye(self.num_nodes)
        adj = adj / torch.unsqueeze(adj.sum(-1), -1)
        return [adj, predefined_A]

    def cal_wavelet(self, input, level, cal_type):
        x = input  # b,c,n,l
        wavelet = 'db1'  # 使用Daubechies 1小波

        if cal_type == '1':
            coeffs = pywt.wavedec(x.cpu(), wavelet, level=level, axis=-1)
            xA = torch.tensor(coeffs[0])
            low = coeffs[1:]
            xD = torch.tensor(low[0])

            return xD.to(self.device), xA.to(self.device)  # b,c,n

        elif cal_type == '2':
            output = []
            for i in range(level):
                coeffs = pywt.wavedec(x.cpu(), wavelet, level=1, axis=-1)
                x = torch.tensor(coeffs[0])
                xD = torch.tensor(np.array(coeffs[1:]))
                output.append(xD)
            output.append(x)
            assert len(output) == level + 1

            return [element.contiguous().squeeze(0) for element in output]  # b,c,n

    def interleave_tensors(self, A, B):
        # A和B的shape都为(b, n, c)
        batch_size, num_nodes, features = A.size()

        # 将A和B展开为(b*n, c)
        A_flat = A.view(-1, features)
        B_flat = B.view(-1, features)

        # 创建一个形状为(b*n, 2*c)的空白张量
        C = torch.zeros(batch_size * num_nodes, features * 2)

        # 将A的值放入拼接后的偶数列，B的值放入拼接后的奇数列
        C[:, ::2] = A_flat
        C[:, 1::2] = B_flat

        # 将拼接后的张量重塑回原始形状(b, n, 2*c)
        C = C.view(batch_size, num_nodes, features * 2)

        return C

    def idwtl_layer(self, hidden_state_list):
        a = hidden_state_list[0].contiguous().squeeze()
        for i in range(1, len(hidden_state_list)):
            a = self.interleave_tensors(self.IDWT_L(a) + self.IDWT_L(hidden_state_list[i].contiguous().squeeze()),
                                        self.IDWT_H(a) + self.IDWT_H(hidden_state_list[i].contiguous().squeeze()))
        return a

    def step(self,input,Hidden_State,Cell_State,predefined_A,md,frequncy='D',type='encoder'):

        x = input

        x = x.transpose(1, 2).contiguous()  # b,n,c

        adp = self.preprocessing(torch.tensor(md), predefined_A[0])
        adpT = self.preprocessing(torch.tensor(md).transpose(0, 1), predefined_A[0])

        if type == 'encoder':
            Hidden_State = Hidden_State.view(-1, self.num_nodes, self.hidden_size)
            Cell_State = Cell_State.view(-1, self.num_nodes, self.hidden_size)
            combined = torch.cat((x.to(device), Hidden_State.to(device)), -1)

            if frequncy == 'D':
                z = torch.sigmoid(self.gz1D(combined, adp) + self.gz2D(combined, adpT))
                r = torch.sigmoid(self.gr1D(combined, adp) + self.gr2D(combined, adpT))

                temp = torch.cat((x.to(device), torch.mul(r.to(device), Hidden_State.to(device))), -1)
                Cell_State = torch.tanh(self.gc1D(temp, adp) + self.gc2D(temp, adpT))

            if frequncy == 'AD':
                z = torch.sigmoid(self.gz1AD(combined, adp) + self.gz2AD(combined, adpT))
                r = torch.sigmoid(self.gr1AD(combined, adp) + self.gr2AD(combined, adpT))

                temp = torch.cat((x.to(device), torch.mul(r.to(device), Hidden_State.to(device))), -1)
                Cell_State = torch.tanh(self.gc1AD(temp, adp) + self.gc2AD(temp, adpT))

        elif type == 'decoder':
            Hidden_State = Hidden_State.view(-1, self.num_nodes, self.hidden_size)
            combined = torch.cat((x.to(device), Hidden_State.to(device)), -1)
            z = torch.sigmoid(
                self.gz1_de(combined, adp) + self.gz2_de(combined, adpT))
            r = torch.sigmoid(
                self.gr1_de(combined, adp) + self.gr2_de(combined, adpT))

            temp = torch.cat((x.to(device), torch.mul(r.to(device), Hidden_State.to(device))), -1)
            tmp1=self.gc1_de(temp, adp)
            tmp2=self.gc2_de(temp, adpT)
            Cell_State = torch.tanh(tmp1+tmp2)

        Hidden_State = torch.mul(z.to(device), Hidden_State.to(device)) + torch.mul(
            1 - z.to(device), Cell_State.to(device))

        return Hidden_State, Cell_State

    def forward(self,input, adj ,**kwargs):
        seqs_time=kwargs.get('seqs_time')
        targets_time=kwargs.get('targets_time')
        index=kwargs.get('index')
        batches_seen=index
        x = input
        predefined_A = self.predefined_A
        batch_size = x.size(0)


        hour = (seqs_time[:, -2:-1, ...] + 0.5) * 23  # 得到第几个小时
        min = (seqs_time[:, -1:, ...] + 0.5) * 59  # 得到第几分钟
        hour_index = ((hour * 60 + min) / (60 / self.points_per_hour)).squeeze().type(torch.LongTensor)  # 得到第几个时间步

        T_ds = hour_index[:, ::self.level+1].unsqueeze(-1).unsqueeze(-1) \
            .repeat(1, 1, self.num_nodes, 1).transpose(1, 3)    # 2 = level+1
        f = x

        xD, xAD = self.cal_wavelet(f, self.level, '2')  # 小波变换

        # 获取隐藏单元
        Hidden_State = []
        Cell_State = []
        Hidden_State_1, Cell_State_1 = self.initHidden(batch_size * self.num_nodes, self.hidden_size)
        Hidden_State_2, Cell_State_2 = self.initHidden(batch_size * self.num_nodes, self.hidden_size)

        # Encoder部分
        for i in range(xD.shape[-1]):  # self.seq_length
            x1 = xD[..., i]  # b,c,n
            t = T_ds[..., i]  # b,c,n (此处c=1)
            x1 = torch.cat((x1, t), dim=1)
            Hidden_State_1, Cell_State_1 = self.step(x1, Hidden_State_1, Cell_State_1,
                                                     predefined_A, adj, 'D', 'encoder')
        Hidden_State.append(Hidden_State_1)
        Cell_State.append(Cell_State_1)

        for i in range(xAD.shape[-1]):  # self.seq_length
            x2 = xAD[..., i]  # b,c,n
            t = T_ds[..., i]  # b,c,n (此处c=1)
            x2 = torch.cat((x2, t), dim=1)
            Hidden_State_2, Cell_State_2 = self.step(x2, Hidden_State_2, Cell_State_2,
                                                     predefined_A, adj, 'AD', 'encoder')
        Hidden_State.append(Hidden_State_2)
        Cell_State.append(Cell_State_2)

        Hidden_State = self.idwtl_layer(Hidden_State)
        Cell_State = self.idwtl_layer(Cell_State)

        # 融合部分
        if self.out_dim <= 6.0:
            Hidden_State = 0.3 * self.camH(Hidden_State) + 0.7 * self.aggH(Hidden_State_1, Hidden_State_2)
            Cell_State = 0.3 * self.camC(Cell_State) + 0.7 * self.aggC(Cell_State_1, Cell_State_2)

        else:
            Hidden_State = 0.7 * self.camH(Hidden_State) + 0.3 * self.aggH(Hidden_State_1, Hidden_State_2)
            Cell_State = 0.7 * self.camC(Cell_State) + 0.3 * self.aggC(Cell_State_1, Cell_State_2)

        go_symbol = torch.zeros((batch_size, self.out_dim, self.num_nodes))

        timeofday = ((seqs_time[:, 2:3, ...] + 0.5) * (6 - 0)).squeeze().type(torch.LongTensor) \
            .unsqueeze(-1).unsqueeze(-1) \
            .repeat(1, 1, self.num_nodes, 1).transpose(1, 3)   # 一周的第几天  # B,1,N,L

        decoder_input = go_symbol

        outputs_final = []

        # Decoder部分
        for i in range(self.output_dim):
            decoder_input = torch.cat([decoder_input.to(device), timeofday[..., i].to(device)], dim=1)

            Hidden_State, Cell_State = self.step(decoder_input, Hidden_State, Cell_State,
                                                 predefined_A, adj, None,
                                                 'decoder')

            Hidden_State, Cell_State = Hidden_State.view(-1, self.hidden_size), Cell_State.view(
                -1, self.hidden_size)

            decoder_output = self.fc_final(Hidden_State)

            decoder_input = decoder_output.view(batch_size, self.num_nodes,
                                                self.out_dim).transpose(1, 2)
            outputs_final.append(decoder_output)

            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = timeofday[:, :, :, i].repeat(1, self.out_dim, 1)

        outputs_final = torch.stack(outputs_final, dim=1)

        outputs_final = outputs_final.view(batch_size, self.num_nodes,
                                           self.out_dim,
                                           self.output_dim)
        outputs_final=outputs_final.permute(0, 2, 1, 3) #（B,C,N,L）
        return outputs_final

    def initHidden(self, batch_size, hidden_size):
        Hidden_State = Variable(
                torch.zeros(batch_size, hidden_size))
        Cell_State = Variable(
                torch.zeros(batch_size, hidden_size))

        nn.init.orthogonal_(Hidden_State)
        nn.init.orthogonal_(Cell_State)

        return Hidden_State, Cell_State

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
