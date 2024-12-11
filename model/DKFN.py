import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from layers.DFKN_related import FilterLinear

'''Note：这里的邻接矩阵比较特别'''
class DKFN(nn.Module):

    def __init__(self, K, A, num_nodes,num_features,pred_len, Clamp_A=True):
        # GC-LSTM
        '''
        Args:
            K: K-hop graph
            A: adjacency matrix
            num_nodes: 节点维度
            num_features:特征维度
            pred_len:预测的时间长度
            Clamp_A: Boolean value, clamping all elements of A between 0. to 1.
        '''
        super(DKFN, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_size = num_nodes
        self.num_features=num_features
        self.K = K
        self.A_list = []  # Adjacency Matrix List
        # TODO 这里输入的是普通的未处理过的邻接矩阵
        # normalization
        tmp=torch.sum(A, 0)
        tmp=np.where(tmp==0,1e-5,tmp) # 防止为0
        D_inverse = torch.diag(torch.tensor(tmp))
        norm_A = torch.matmul(D_inverse, A)
        A = norm_A

        A_temp = torch.eye(num_nodes, num_nodes)
        for i in range(K):
            A_temp = torch.matmul(A_temp, A)
            if Clamp_A:
                # confine elements of A
                A_temp = torch.clamp(A_temp, max=1.)
            self.A_list.append(A_temp)

        # a length adjustable Module List for hosting all graph convolutions
        self.gc_list = nn.ModuleList([FilterLinear(num_nodes, num_nodes, self.A_list[i], bias=False) for i in range(K)])

        hidden_size = self.num_nodes
        gc_input_size = self.num_nodes * K

        self.fl = nn.Linear(gc_input_size + hidden_size, hidden_size)
        self.il = nn.Linear(gc_input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(gc_input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(gc_input_size + hidden_size, hidden_size)

        # initialize the neighbor weight for the cell state
        self.Neighbor_weight = Parameter(torch.FloatTensor(num_nodes))
        stdv = 1. / math.sqrt(num_nodes)
        self.Neighbor_weight.data.uniform_(-stdv, stdv)

        # RNN
        input_size = self.num_nodes

        self.rfl = nn.Linear(input_size + hidden_size, hidden_size)
        self.ril = nn.Linear(input_size + hidden_size, hidden_size)
        self.rol = nn.Linear(input_size + hidden_size, hidden_size)
        self.rCl = nn.Linear(input_size + hidden_size, hidden_size)

        # addtional vars
        self.c = torch.nn.Parameter(torch.Tensor([1]))


        self.conv_out=nn.Conv1d(self.num_features,self.num_features*pred_len,kernel_size=1,stride=1)

    '''循环一个时间步'''
    def pred_one_step(self, input, Hidden_State, Cell_State, rHidden_State, rCell_State):
        # GC-LSTM(neighbour-dependency)
        # GCN
        x = input #(B,C,N)
        gc = self.gc_list[0](x) # 操作的是节点维度
        for i in range(1, self.K):
            gc = torch.cat((gc, self.gc_list[i](x)), -1) # 在节点维度拼接
        # LSTM
        combined = torch.cat((gc, Hidden_State), -1) # 在节点维度进行concat
        f = torch.sigmoid(self.fl(combined))
        i = torch.sigmoid(self.il(combined))
        o = torch.sigmoid(self.ol(combined))
        C = torch.tanh(self.Cl(combined))
        # 公式8
        NC = torch.mul(Cell_State,
                       torch.mv(Variable(self.A_list[-1], requires_grad=False).cuda(), self.Neighbor_weight)) # torch.mv()表示的是矩阵与向量相乘
        Cell_State = f * NC + i * C # 公式9
        Hidden_State = o * torch.tanh(Cell_State) # 公式10

        # LSTM(self-dependency)-->标准的LSTM
        rcombined = torch.cat((input, rHidden_State), -1)
        rf = torch.sigmoid(self.rfl(rcombined))
        ri = torch.sigmoid(self.ril(rcombined))
        ro = torch.sigmoid(self.rol(rcombined))
        rC = torch.tanh(self.rCl(rcombined))
        rCell_State = rf * rCell_State + ri * rC
        rHidden_State = ro * torch.tanh(rCell_State)

        # Kalman Filtering
        var1, var2 = torch.var(input,dim=(0,-1),keepdim=True), torch.var(gc,dim=(0,-1),keepdim=True) # 测试过这里不同的特征要不要对应不同的分布，效果更好

        pred = (Hidden_State * var1 * self.c + rHidden_State * var2) / (var1 + var2 * self.c) #公式17

        return Hidden_State, Cell_State, gc, rHidden_State, rCell_State, pred

    # def Bi_torch(self, a):
    #     a[a < 0] = 0
    #     a[a > 0] = 1
    #     return a

    def forward(self, inputs,adj,**kwargs):
        '''input:(B,C,N,L)'''
        batch_size = inputs.size(0)
        time_step = inputs.size(-1)
        Hidden_State, Cell_State, rHidden_State, rCell_State = self.initHidden(batch_size)
        for i in range(time_step): # 循环时间步
            Hidden_State, Cell_State, gc, rHidden_State, rCell_State, pred = self.pred_one_step(
                torch.squeeze(inputs[:, :, :,i:i + 1],dim=-1), Hidden_State, Cell_State, rHidden_State, rCell_State)

        B,C,N=pred.shape
        pred=self.conv_out(pred)
        pred=pred.reshape(B,C,N,-1)
        return pred # (B,C,N,L)


    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size,self.num_features, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size,self.num_features ,self.hidden_size).cuda())
            rHidden_State = Variable(torch.zeros(batch_size,self.num_features ,self.hidden_size).cuda())
            rCell_State = Variable(torch.zeros(batch_size,self.num_features ,self.hidden_size).cuda())
            return Hidden_State, Cell_State, rHidden_State, rCell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size,self.num_features, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size,self.num_features ,self.hidden_size))
            rHidden_State = Variable(torch.zeros(batch_size,self.num_features,self.hidden_size))
            rCell_State = Variable(torch.zeros(batch_size,self.num_features,self.hidden_size))
            return Hidden_State, Cell_State, rHidden_State, rCell_State

    def reinitHidden(self, batch_size, Hidden_State_data, Cell_State_data):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            Cell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            rHidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            rCell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            return Hidden_State, Cell_State, rHidden_State, rCell_State
        else:
            Hidden_State = Variable(Hidden_State_data, requires_grad=True)
            Cell_State = Variable(Cell_State_data, requires_grad=True)
            rHidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            rCell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            return Hidden_State, Cell_State, rHidden_State, rCell_State
