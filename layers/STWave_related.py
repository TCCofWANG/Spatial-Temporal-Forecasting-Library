import pywt
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def disentangle(x, w='sym2', j=1):
    if isinstance(x,torch.Tensor):
        x=np.array(x.cpu().detach())
    x = x.transpose(0,3,2,1) # [S,D,N,T]
    coef = pywt.wavedec(x, w, level=j)
    coefl = [coef[0]]
    for i in range(len(coef)-1):
        coefl.append(None)
    coefh = [None]
    for i in range(len(coef)-1):
        coefh.append(coef[i+1])
    xl = pywt.waverec(coefl, w).transpose(0,3,2,1) # 还原时域
    xh = pywt.waverec(coefh, w).transpose(0,3,2,1) # 还原时域
    xl,xh=torch.tensor(xl).cuda(),torch.tensor(xh).cuda()
    return xl, xh

class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemEmbedding(nn.Module):
    def __init__(self, D):
        super(TemEmbedding, self).__init__()
        self.ff_te = FeedForward([295, D, D])

    def forward(self, TE, T=288):
        '''
        TE: [B,T,2]
        return: [B,T,N,D]
        '''
        # temporal embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7).to(TE.device)  # [B,T,7]
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T).to(TE.device)  # [B,T,288]
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 288, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1)  # [B,T,295]
        TE = TE.unsqueeze(dim=2)  # [B,T,1,295]
        TE = self.ff_te(TE)  # [B,T,1,F]

        return TE  # [B,T,N,F]


class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i + 1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L - 1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x


class Sparse_Spatial_Attention(nn.Module):
    def __init__(self, heads, dims, samples, localadj):
        super(Sparse_Spatial_Attention, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims
        self.s = samples
        self.la = localadj

        self.qfc = FeedForward([features, features])
        self.kfc = FeedForward([features, features])
        self.vfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])

        self.ln = nn.LayerNorm(features)
        self.ff = FeedForward([features, features, features], True)
        self.proj = nn.Linear(self.la.shape[1], 1)

    def forward(self, x, spa_eigvalue, spa_eigvec, tem_eigvalue, tem_eigvec):
        '''
        x: [B,T,N,D]
        spa_eigvalue, tem_eigvalue: [D]
        spa_eigvec, tem_eigvec: [N,D]
        return: [B,T,N,D]
        '''
        try:
            tmp=torch.matmul(spa_eigvec, torch.diag_embed(spa_eigvalue)) + torch.matmul(tem_eigvec, # 特征值和特征向量相乘，缩放作用。+等于一个Embedding
                                                                                         torch.diag_embed(tem_eigvalue))
        except:
            tmp=0
        x_ = x +tmp

        Q = self.qfc(x_)
        K = self.kfc(x_)
        V = self.vfc(x_)
        # 切分多头
        Q = torch.cat(torch.split(Q, self.d, -1), 0)
        K = torch.cat(torch.split(K, self.d, -1), 0)
        V = torch.cat(torch.split(V, self.d, -1), 0)

        B, T, N, D = K.shape
        # calculate the sampled Q_K-->为了找出活跃的点（有价值的点）
        K_expand = K.unsqueeze(-3).expand(B, T, N, N, D)
        K_sample = K_expand[:, :, torch.arange(N).unsqueeze(1), self.la, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        Sampled_Nodes = int(self.s * math.log(N, 2))
        M = self.proj(Q_K_sample).squeeze(-1)
        M_top = M.topk(Sampled_Nodes, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(T)[None, :, None],
                   M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) #(B,L,Sampled_Nodes,N)计算的是节点间的相关性

        Q_K /= (self.d ** 0.5)

        attn = torch.softmax(Q_K, dim=-1)

        # copy operation
        cp = attn.argmax(dim=-2, keepdim=True).transpose(-2, -1)

        value = torch.matmul(attn, V) # 得到(B,L,sample_num,D)
        # 根据N与sample_num的关系选择相关性最大的，然后最终的value值就复制相关性最大的点的值
        value=value.unsqueeze(-3).expand(B, T, N, M_top.shape[-1], V.shape[-1])[
                torch.arange(B)[:, None, None, None],
                torch.arange(T)[None, :, None, None],
                torch.arange(N)[None, None, :, None], cp, :].squeeze(-2)
        # 多头恢复
        value = torch.cat(torch.split(value, value.shape[0] // self.h, 0), -1)
        value = self.ofc(value)

        value = self.ln(value)

        return self.ff(value)


class TemporalAttention(nn.Module):
    def __init__(self, heads, dims):
        super(TemporalAttention, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims

        self.qfc = FeedForward([features, features])
        self.kfc = FeedForward([features, features])
        self.vfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])

        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features, features, features], True)

    def forward(self, x, te, Mask=True):
        '''
        x: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        '''
        x += te

        query = self.qfc(x)  # [B,T,N,F]
        key = self.kfc(x)  # [B,T,N,F]
        value = self.vfc(x)  # [B,T,N,F]
        # 切分多头
        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0, 2, 1, 3)  # [k*B,T,N,d]
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0, 2, 3, 1)  # [k*B,N,d,T]
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0, 2, 1, 3)  # [k*B,N,T,d]

        attention = torch.matmul(query, key)  # [k*B,N,T,T]
        attention /= (self.d ** 0.5)  # scaled

        if Mask:
            batch_size = x.shape[0]
            num_steps = x.shape[1]
            num_vertexs = x.shape[2]
            mask = torch.ones(num_steps, num_steps).to(x.device)  # [T,T]
            mask = torch.tril(mask)  # [T,T] 下三角为1
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)  # [1,1,T,T]
            mask = mask.repeat(self.h * batch_size, num_vertexs, 1, 1)  # [k*B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1) * torch.ones_like(attention).to(x.device)  # [k*B,N,T,T]
            attention = torch.where(mask, attention, zero_vec) # 保留下三角的值

        attention = F.softmax(attention, -1)  # [k*B,N,T,T]

        value = torch.matmul(attention, value)  # [k*B,N,T,d]

        value = torch.cat(torch.split(value, value.shape[0] // self.h, 0), -1).permute(0, 2, 1, 3)  # [B,T,N,F]
        value = self.ofc(value)
        value += x

        value = self.ln(value)

        return self.ff(value)


class TemporalConvNet(nn.Module):
    def __init__(self, features, kernel_size=2, dropout=0.2, levels=1):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(features, features, (1, kernel_size), dilation=(1, dilation_size),
                                  padding=(0, padding))
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, xh):
        xh = self.tcn(xh.transpose(1, 3)).transpose(1, 3)
        return xh


class Dual_Enconder(nn.Module):
    def __init__(self, heads, dims, samples, localadj, spawave, temwave):
        super(Dual_Enconder, self).__init__()
        self.temporal_conv = TemporalConvNet(heads * dims)
        self.temporal_att = TemporalAttention(heads, dims)

        self.spatial_att_l = Sparse_Spatial_Attention(heads, dims, samples, localadj)
        self.spatial_att_h = Sparse_Spatial_Attention(heads, dims, samples, localadj)

        spa_eigvalue = torch.from_numpy(spawave[0].astype(np.float32))
        self.spa_eigvalue = nn.Parameter(spa_eigvalue, requires_grad=True)
        self.spa_eigvec = torch.from_numpy(spawave[1].astype(np.float32))

        tem_eigvalue = torch.from_numpy(temwave[0].astype(np.float32))
        self.tem_eigvalue = nn.Parameter(tem_eigvalue, requires_grad=True)
        self.tem_eigvec = torch.from_numpy(temwave[1].astype(np.float32))

    def forward(self, xl, xh, te):
        '''
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        '''
        xl = self.temporal_att(xl, te)  # [B,T,N,F]
        xh = self.temporal_conv(xh)  # [B,T,N,F]

        spa_statesl = self.spatial_att_l(xl, self.spa_eigvalue, self.spa_eigvec.to(xl.device), self.tem_eigvalue,
                                         self.tem_eigvec.to(xl.device))  # [B,T,N,F]
        spa_statesh = self.spatial_att_h(xh, self.spa_eigvalue, self.spa_eigvec.to(xl.device), self.tem_eigvalue,
                                         self.tem_eigvec.to(xl.device))  # [B,T,N,F]
        xl = spa_statesl + xl
        xh = spa_statesh + xh

        return xl, xh


class Adaptive_Fusion(nn.Module):
    def __init__(self, heads, dims):
        super(Adaptive_Fusion, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims

        self.qlfc = FeedForward([features, features])
        self.khfc = FeedForward([features, features])
        self.vhfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])

        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features, features, features], True)

    def forward(self, xl, xh, te, Mask=True):
        '''
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        '''
        xl += te
        xh += te

        query = self.qlfc(xl)  # [B,T,N,F]
        keyh = torch.relu(self.khfc(xh))  # [B,T,N,F]
        valueh = torch.relu(self.vhfc(xh))  # [B,T,N,F]
        # 切分多头
        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0, 2, 1, 3)  # [k*B,N,T,d]
        keyh = torch.cat(torch.split(keyh, self.d, -1), 0).permute(0, 2, 3, 1)  # [k*B,N,d,T]
        valueh = torch.cat(torch.split(valueh, self.d, -1), 0).permute(0, 2, 1, 3)  # [k*B,N,T,d]
        # 时间步Attention
        attentionh = torch.matmul(query, keyh)  # [k*B,N,T,T]

        if Mask:
            batch_size = xl.shape[0]
            num_steps = xl.shape[1]
            num_vertexs = xl.shape[2]
            mask = torch.ones(num_steps, num_steps).to(xl.device)  # [T,T]
            mask = torch.tril(mask)  # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)  # [1,1,T,T]
            mask = mask.repeat(self.h * batch_size, num_vertexs, 1, 1)  # [k*B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1) * torch.ones_like(attentionh).to(xl.device)  # [k*B,N,T,T]
            attentionh = torch.where(mask, attentionh, zero_vec)

        attentionh /= (self.d ** 0.5)  # scaled
        attentionh = F.softmax(attentionh, -1)  # [k*B,N,T,T]

        value = torch.matmul(attentionh, valueh)  # [k*B,N,T,d]

        value = torch.cat(torch.split(value, value.shape[0] // self.h, 0), -1).permute(0, 2, 1, 3)  # [B,T,N,F]
        value = self.ofc(value)
        value = value + xl

        value = self.ln(value)

        return self.ff(value)



