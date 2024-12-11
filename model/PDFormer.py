from layers.PDFormer_related import *
from torch_utils import Get_dtw
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from fastdtw import fastdtw
from functools import partial
import os
from tqdm import tqdm
import scipy.sparse as sp

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
# class Chomp2d(nn.Module):
#     def __init__(self, chomp_size):
#         super(Chomp2d, self).__init__()
#         self.chomp_size = chomp_size
#
#     def forward(self, x):
#         return x[:, :, :x.shape[2] - self.chomp_size, :].contiguous()


class STSelfAttention(nn.Module):
    def __init__(
        self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2, qkv_bias=False,
        attn_drop=0., proj_drop=0., device=torch.device('cpu'), output_dim=1,
    ):
        super().__init__()
        assert dim % (geo_num_heads + sem_num_heads + t_num_heads) == 0
        self.geo_num_heads = geo_num_heads
        self.sem_num_heads = sem_num_heads
        self.t_num_heads = t_num_heads
        self.head_dim = dim // (geo_num_heads + sem_num_heads + t_num_heads)
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.s_attn_size = s_attn_size
        self.t_attn_size = t_attn_size
        self.geo_ratio = geo_num_heads / (geo_num_heads + sem_num_heads + t_num_heads)
        self.sem_ratio = sem_num_heads / (geo_num_heads + sem_num_heads + t_num_heads)
        self.t_ratio = 1 - self.geo_ratio - self.sem_ratio
        self.output_dim = output_dim

        self.pattern_q_linears = nn.ModuleList([
            nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)
        ])
        self.pattern_k_linears = nn.ModuleList([
            nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)
        ])
        self.pattern_v_linears = nn.ModuleList([
            nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)
        ])

        self.geo_q_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_k_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_v_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_attn_drop = nn.Dropout(attn_drop)

        self.sem_q_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_k_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_v_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_attn_drop = nn.Dropout(attn_drop)

        self.t_q_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_patterns, pattern_keys, geo_mask=None, sem_mask=None):
        x,x_patterns,pattern_keys=x.to(device),x_patterns.to(device),pattern_keys.to(device)
        geo_mask=geo_mask.to(device) if isinstance(geo_mask,torch.Tensor) else geo_mask
        sem_mask = sem_mask.to(device) if isinstance(sem_mask, torch.Tensor) else sem_mask

        B, T, N, D = x.shape #TODO 这里算的是时间步间的Attention
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1) # 操作特征维度
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4) # 多头，将特征维度进行拆分
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale # 计算的是时间步间的Attention(B,N,H,L,L)<--融合特征维度
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, int(D * self.t_ratio)).transpose(1, 2) # 相似性矩阵与V相乘-->Attention机制的输出

        geo_q = self.geo_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_k = self.geo_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        for i in range(self.output_dim): # TODO 算的是时间延迟 融入地理邻域的Attention中的key矩阵中
            pattern_q = self.pattern_q_linears[i](x_patterns[..., i])
            pattern_k = self.pattern_k_linears[i](pattern_keys[..., i])
            pattern_v = self.pattern_v_linears[i](pattern_keys[..., i])
            pattern_attn = (pattern_q @ pattern_k.transpose(-2, -1)) * self.scale
            pattern_attn = pattern_attn.softmax(dim=-1)
            geo_k += pattern_attn @ pattern_v # 融入地理邻域的Attention的Key矩阵中
        geo_v = self.geo_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_q = geo_q.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_k = geo_k.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_v = geo_v.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale # TODO 计算的是地理邻域的Attention
        if geo_mask is not None:
            geo_attn.masked_fill_(geo_mask, float('-inf')) # 将地理邻域远于设定的阈值的节点的相似性关系直接mask
        geo_attn = geo_attn.softmax(dim=-1)
        geo_attn = self.geo_attn_drop(geo_attn)
        geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(B, T, N, int(D * self.geo_ratio)) # 得到地理邻域的Attention的输出值

        sem_q = self.sem_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)# TODO 计算的是语义邻域的Attention
        sem_k = self.sem_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_v = self.sem_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_q = sem_q.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_k = sem_k.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_v = sem_v.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale
        if sem_mask is not None:
            sem_attn.masked_fill_(sem_mask, float('-inf'))
        sem_attn = sem_attn.softmax(dim=-1)
        sem_attn = self.sem_attn_drop(sem_attn)
        sem_x = (sem_attn @ sem_v).transpose(2, 3).reshape(B, T, N, int(D * self.sem_ratio)) # 得到语义邻域的Attention

        x = self.proj(torch.cat([t_x, geo_x, sem_x], dim=-1)) # 将三个Attention的结果进行concat，并且进行变化维度
        x = self.proj_drop(x)
        return x


class TemporalSelfAttention(nn.Module):
    def __init__(
        self, dim, dim_out, t_attn_size, t_num_heads=6, qkv_bias=False,
        attn_drop=0., proj_drop=0., device=torch.device('cpu'),
    ):
        super().__init__()
        assert dim % t_num_heads == 0
        self.t_num_heads = t_num_heads
        self.head_dim = dim // t_num_heads
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.t_attn_size = t_attn_size

        self.t_q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, N, D = x.shape
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale

        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)

        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, D).transpose(1, 2)

        x = self.proj(t_x)
        x = self.proj_drop(x)
        return x

class STEncoderBlock(nn.Module):

    def __init__(
        self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, device=torch.device('cpu'), type_ln="pre", output_dim=1,
    ):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = norm_layer(dim)
        self.st_attn = STSelfAttention(
            dim, s_attn_size, t_attn_size, geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads, t_num_heads=t_num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, device=device, output_dim=output_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_patterns, pattern_keys, geo_mask=None, sem_mask=None):
        if self.type_ln == 'pre':
            x = x + self.drop_path(self.st_attn(self.norm1(x), x_patterns, pattern_keys, geo_mask=geo_mask, sem_mask=sem_mask)) #层内的残差连接 类似与Graph Wavenet
            x = x + self.drop_path(self.mlp(self.norm2(x))) # Feedforward
        elif self.type_ln == 'post':# 不进入这部分代码
            x = self.norm1(x + self.drop_path(self.st_attn(x, x_patterns, pattern_keys, geo_mask=geo_mask, sem_mask=sem_mask)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


class PDFormer(nn.Module):
    def __init__(self, config,adj):
        super().__init__()

        self.num_nodes = config.num_nodes
        self.feature_dim = config.num_features
        self.ext_dim = 0
        self.num_batches = config.batch_size
        self.dataset=config.data_name

        self.embed_dim = config.embed_dim
        self.skip_dim = config.skip_dim
        lape_dim = config.lape_dim
        geo_num_heads = config.geo_num_heads
        sem_num_heads = config.sem_num_heads
        t_num_heads = config.t_num_heads
        mlp_ratio = config.mlp_ratio
        qkv_bias = config.qkv_bias
        drop = config.drop
        attn_drop = config.attn_drop
        drop_path = config.drop_path
        self.s_attn_size = config.s_attn_size
        self.t_attn_size = config.t_attn_size
        enc_depth = config.enc_depth
        type_ln = config.type_ln
        self.type_short_path = config.type_short_path

        self.output_dim = config.output_dim
        self.input_window = config.seq_len
        self.output_window = config.pred_len
        add_time_in_day = config.add_time_in_day
        add_day_in_week = config.add_day_in_week
        self.device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
        self.huber_delta = config.huber_delta
        self.quan_delta = config.quan_delta
        self.far_mask_delta = config.far_mask_delta
        self.dtw_delta = config.dtw_delta
        self.config=config
        self.random_flip=True

        get_dtw=Get_dtw.get_dtw(config) # 初始化
        self.dtw_matrix =get_dtw.dtw_matrix # 得到对应的dtw矩阵

        self.adj_mx = adj  # 这里的adj是未处理过的adj
        # sd_mx = data_feature.get('sd_mx')
        sh_mx = self._get_sh_mx()  # 得到节点间的距离

        # if self.type_short_path == "dist":
        #     distances = sd_mx[~np.isinf(sd_mx)].flatten()
        #     std = distances.std()
        #     sd_mx = np.exp(-np.square(sd_mx / std))
        #     self.far_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
        #     self.far_mask[sd_mx < self.far_mask_delta] = 1
        #     self.far_mask = self.far_mask.bool()
        # else:
        sh_mx = sh_mx.T
        self.geo_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
        self.geo_mask[sh_mx >= self.far_mask_delta] = 1
        self.geo_mask = self.geo_mask.bool()
        self.sem_mask = torch.ones(self.num_nodes, self.num_nodes).to(self.device)
        sem_mask = self.dtw_matrix.argsort(axis=1)[:, :self.dtw_delta] #找出语义距离最近的模型
        for i in range(self.sem_mask.shape[0]):
            self.sem_mask[i][sem_mask[i]] = 0
        self.sem_mask = self.sem_mask.bool()

        pattern_keys = get_dtw.pattern_keys  # 得到对应的质心
        self.pattern_keys = torch.from_numpy(pattern_keys).float().to(self.device) # 得到每一个簇质心的坐标 (16=簇个数,3,1)
        self.pattern_embeddings = nn.ModuleList([
            TokenEmbedding(self.s_attn_size, self.embed_dim) for _ in range(self.output_dim)
        ])

        self.enc_embed_layer = DataEmbedding(
            self.feature_dim - self.ext_dim, self.embed_dim, lape_dim, self.adj_mx, drop=drop,
            add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week, device=self.device,
        )

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]
        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(
                dim=self.embed_dim, s_attn_size=self.s_attn_size, t_attn_size=self.t_attn_size, geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads, t_num_heads=t_num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=enc_dpr[i], act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), device=self.device, type_ln=type_ln, output_dim=self.output_dim,
            ) for i in range(enc_depth)
        ])

        self.skip_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1,
            ) for _ in range(enc_depth)
        ])

        self.end_conv1 = nn.Conv2d(
            in_channels=self.input_window, out_channels=self.output_window, kernel_size=1, bias=True,
        )
        self.end_conv2 = nn.Conv2d(
            in_channels=self.skip_dim, out_channels=self.output_dim, kernel_size=1, bias=True,
        )

    '''得到节点间的距离-->耗时很长'''
    def _get_sh_mx(self):
        sh_mx = self.adj_mx.clone()
        print('开始计算节点间的距离')
        sh_cache_path = './datasets/cache/sh_' + self.config.data_name+'_{0}'.format(self.type_short_path) + '.npy'
        if not os.path.exists(sh_cache_path):
            if self.type_short_path == 'hop': # Floyd-Warshall算法
                sh_mx[sh_mx > 0] = 1
                sh_mx[sh_mx == 0] = 511
                for i in range(self.num_nodes):
                    sh_mx[i, i] = 0 # 自己与自己就是0跳
                for k in range(self.num_nodes):
                    for i in range(self.num_nodes):
                        for j in range(self.num_nodes):
                            sh_mx[i, j] = min(sh_mx[i, j], sh_mx[i, k] + sh_mx[k, j], 511)
            else:
                raise print('没有{0}的计算方法'.format(self.type_short_path))
            np.save(sh_cache_path,sh_mx) #将对应的结果进行存下
        sh_mx=np.load(sh_cache_path)
        print('节点间距离计算完成')
        return sh_mx

    def _calculate_normalized_laplacian(self, adj):
        adj = sp.coo_matrix(adj)
        d = np.array(adj.sum(1))
        isolated_point_num = np.sum(np.where(d, 0, 1))
        # print(f"Number of isolated points: {isolated_point_num}")
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        return normalized_laplacian, isolated_point_num


    def _cal_lape(self, adj_mx):
        if isinstance(adj_mx,torch.Tensor):
            adj_mx=np.array(adj_mx.cpu().detach())
        L, isolated_point_num = self._calculate_normalized_laplacian(adj_mx) #计算对应的归一化的拉普拉斯矩阵
        EigVal, EigVec = np.linalg.eig(L.toarray())  # 特征值分解
        idx = EigVal.argsort()  # 根据特征值进行升序排序，表示后面取的是topk的特征向量
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])  # 感觉实部和不取实部的结果应该差不多，啥矩阵的特征值向量一定是实数
        # 以下是取对应的特征值向量
        # 这里为什么要取isolated_point_num？答：这里做的是一个谱聚类的操作，具体原因见：https://www.cnblogs.com/pinard/p/6221564.html
        laplacian_pe = torch.from_numpy(
            EigVec[:, isolated_point_num + 1: self.config.lape_dim + isolated_point_num + 1]).float()
        laplacian_pe.require_grad = False
        return laplacian_pe

    def forward(self, x, adj,**kwargs):
        x=x.to(device)
        laplacian_pe=self._cal_lape(adj) # 得到对应的特征向量
        batch_lap_pos_enc =laplacian_pe.to(self.device)
        if self.random_flip:  # FIXME 这里是为啥 增加随机性？？
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(self.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        lap_mx=batch_lap_pos_enc

        B,C,N,L=x.shape
        x=x.transpose(1,-1) # (B,L,N,C)
        T =  x.shape[1]
        x_pattern_list = []
        for i in range(self.s_attn_size): # 表示的是要滑动多少次窗口 感觉这里不滑动也可以 直接取将整个窗口送到后也可以
            x_pattern = F.pad(
                x[:, :T + i + 1 - self.s_attn_size, :, :self.output_dim], #  这里只取了outputdim 表示如果是预测m个特征维度 输入就是输入m个特征维度
                (0, 0, 0, 0, self.s_attn_size - 1 - i, 0), # (左边(列)，右边(列)，上边(行)，下边(行)，前边(channel)，后边(channel))-->填充的是第二个维度(时间步的维度)
                "constant", 0, #窗口m(1~self.s_attn_size)在序列开始补self.s_attn_size-m个零, FIXME 补0会不会不太合理 而且为啥要在前面补零
            ).unsqueeze(-2)
            x_pattern_list.append(x_pattern)
        x_patterns = torch.cat(x_pattern_list, dim=-2)  # (B, T, N,切分出三个窗口,output_dim) # 这里表示的是历史已知的数据可以被滑动窗口切分成多少个

        x_pattern_list = []
        pattern_key_list = []
        for i in range(self.output_dim): # 对特征维度分别进行Embedding，对质心和切分出的回顾窗口分别进行Embedding
            x_pattern_list.append(self.pattern_embeddings[i](x_patterns[..., i]).unsqueeze(-1)) # 历史回顾窗口的切分的Embedding
            pattern_key_list.append(self.pattern_embeddings[i](self.pattern_keys[..., i]).unsqueeze(-1)) # 质心的Embedding
        x_patterns = torch.cat(x_pattern_list, dim=-1) # 在特征维度进行拼接 ， 对可以切分出多少个时间窗口进行映射
        pattern_keys = torch.cat(pattern_key_list, dim=-1) # 在特征维度进行拼接

        enc = self.enc_embed_layer(x, lap_mx) # Embedding=位置编码+拉普拉斯矩阵的特征向量编码+时间编码(日周期+周周期)+特征数据的普通编码
        skip = 0
        for i, encoder_block in enumerate(self.encoder_blocks):
            enc = encoder_block(enc, x_patterns, pattern_keys, self.geo_mask, self.sem_mask) # backbone
            skip += self.skip_convs[i](enc.permute(0, 3, 2, 1)) # 类似于Graph Wavenet的层间残差连接

        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))
        skip=skip.permute(0, 3, 2, 1)
        return skip.transpose(1,-1) #(B,C,N,L)


