from layers.MegaCRN_related import *
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np

class MegaCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, num_layers=1, cheb_k=3,
                 ycov_dim=5, mem_num=20, mem_dim=64, cl_decay_steps=2000, use_curriculum_learning=True,**kwargs):
        super(MegaCRN, self).__init__()
        args=kwargs.get('args')
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning

        # memory
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.memory = self.construct_memory()

        # encoder-->AGCRN只不过就是将GCN从AGCRN的提出的GCN改为了普通的切比雪夫多项式实现的GCN
        self.encoder = ADCRNN_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.num_layers)

        # deocoder-->AGCRN
        self.decoder_dim = self.rnn_units + self.mem_dim
        self.decoder = ADCRNN_Decoder(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, self.cheb_k,
                                      self.num_layers)

        # output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))


        self.separate_loss = nn.TripletMarginLoss(margin=1.0)
        self.compact_loss = nn.MSELoss()
        self.lamb, self.lamb1 =args.lamb,args.lamb1

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        # Memory库
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)  # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim),
                                         requires_grad=True)  # project to query
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num),
                                          requires_grad=True)  # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num),
                                          requires_grad=True)  # project memory to embedding
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    # 相当于所有的节点与Memory库中的节点进行Attention计算
    def query_memory(self, h_t: torch.Tensor):
        query = torch.matmul(h_t, self.memory['Wq'])  # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)  # alpha: (B, N, M) 得到相似度矩阵
        value = torch.matmul(att_score, self.memory['Memory'])  # (B, N, d) 根据相似度进行加权和
        _, ind = torch.topk(att_score, k=2, dim=-1) # 得到N个节点与Memory库中最相似的两个节点的index
        pos = self.memory['Memory'][ind[:, :, 0]]  # B, N, d 得到每一个点在Memory中最接近的点-->正对
        neg = self.memory['Memory'][ind[:, :, 1]]  # B, N, d 得到负对
        return value, query, pos, neg

    def forward(self, x, adj,**kwargs):
        x=x.transpose(1,-1) #(B,L,N,C)
        B,L,N,C=x.shape
        labels=kwargs.get('targets').transpose(1,-1)
        y_cov=kwargs.get('targets_time').repeat(1,1,N,1).transpose(1,-1)
        batches_seen=kwargs.get("index")

        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [g1, g2] # 得到两个自适应的adj
        # 以下是根据AGCRN模型的层结构，区别在与AGCRN的GCN会根据不同的N有着不同的W，这里没有这个功能
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x, init_state, supports)  # B, T, N, hidden
        h_t = h_en[:, -1, :, :]  # B, N, hidden (last state) 取最后一个时间步的隐藏层

        h_att, query, pos, neg = self.query_memory(h_t)
        h_t = torch.cat([h_t, h_att], dim=-1) #h_att表示的是所有节点与Memory库节点进行Attention计算后的结果

        ht_list = [h_t] * self.num_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.horizon): # 循环预测长度
            h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, supports)
            go = self.proj(h_de)
            out.append(go)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...] # 为了减少误差累计现象
        output = torch.stack(out, dim=1)

        loss1 = self.separate_loss(query, pos.detach(), neg.detach()) # query是根据h_t对特征维度进行映射得到的
        loss2=self.compact_loss(query, pos.detach()) # 希望query和正对接近。正对就是靠query取Attention计算得到的
        loss=self.lamb * loss1 + self.lamb1 * loss2 # 理论上这两个loss是一定为0，这两个loss主要是为了可以更好的更新Memory参数
        output=output.transpose(1,-1) #（B,C,N,L）

        if self.training: # 判断是不是在训练
            return [output, loss]
        else:
            return output





