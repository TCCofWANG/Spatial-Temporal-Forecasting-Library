from model import *
from torch_utils.load_wave_graph import *
from data.dataset import split_dataset
import torch_utils as tu
from torch_utils import Write_csv,earlystopping
from data.data_process import *
from data.get_data import build_dataloader
import torch
import torch.nn as nn
import numpy as np
import test
import yaml
from datetime import datetime

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
class EXP():
    def __init__(self,args):
        assert args.resume_dir==args.output_dir
        self.agrs=args
        tu.dist.init_distributed_mode(args)  # 初始化分布式训练
        # seed = tu.dist.get_rank() + args.seed  # 0+args.seed
        # tu.model_tool.seed_everything(seed)  # 设置随机数种子，便于可重复实验

        # train_sampler,test_sampler是分布式训练时使用的，未分布式训练时，其都为None
        # get_data
        (adj, self.train_dataloader,self.val_dataloader, self.test_dataloader,
         self.train_sampler,self.val_sampler,self.test_sampler) = build_dataloader(args)
        self.adj=adj # TODO 这里的adj就是普通的adj

        # get_model
        self.build_model(args, adj)  # 得到对应的模型
        self.model.to(device)  # 送入对应的设备中

        self.model = tu.dist.ddp_model(self.model, [args.local_rank])  # 分布式训练模型，好像也没有发挥作用
        if args.dp_mode:
            self.model = nn.DataParallel(self.model)  # 分布训练，单机子多显卡
            print('using dp mode')

        # 模型训练所需的
        criterion = nn.L1Loss()
        self.criterion=criterion

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # 引入权重衰减的Adam
        self.optimizer=optimizer

        # 权重衰减：cos衰减
        lr_optimizer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.end_epoch,eta_min=args.lr / 1000)
        self.lr_optimizer=lr_optimizer

        # 早停机制
        if args.output_dir==None or args.output_dir=='None' or args.output_dir=='none':
            args.output_dir = None
            tu.config.create_output_dir(args)  # 创建输出的目录
            args.resume_dir=args.output_dir

        output_path = os.path.join(args.output_dir,args.model_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        path = os.path.join(output_path, args.data_name + '_best_model.pkl')
        self.output_path = output_path
        self.early_stopping = earlystopping.EarlyStopping(path=path, optimizer=self.optimizer,
                                                          scheduler=self.lr_optimizer, patience=args.patience)
        resume_path = os.path.join(args.resume_dir,args.model_name)
        if not os.path.exists(resume_path):
            raise print('没有找到对应的读取预训练权重的路径')
        resume_path = os.path.join(resume_path, args.data_name + '_best_model.pkl')
        self.resume_path = resume_path

        if args.resume:
            print('加载预训练模型')
            try:
                dp_mode = args.args.dp_mode
            except AttributeError as e:
                dp_mode = True
            # FIXME 自动读取超参数 这里还不完善
            hparam_path = os.path.join(args.output_dir, 'hparam.yaml')
            with open(hparam_path, 'r') as f:
                hparam_dict = yaml.load(f, yaml.FullLoader)
                args.output_dir = hparam_dict['output_dir']

            # 读取最好的权重
            self.load_best_model(path=self.resume_path,args=args, distributed=dp_mode)

    '''建立模型'''
    def build_model(self,args,adj):
        if args.model_name == 'gwnet':
            args.layers=4
            self.model = GWNet(num_nodes=args.num_nodes,in_dim=args.num_features,supports=adj,
                                     out_dim=1,pred_len=args.pred_len, n_hidden=32, kernel_size=2, layers=args.layers, blocks=1,
                                     addaptadj=True)

        elif args.model_name == 'HGCN':
            self.model = H_GCN_wh(num_nodes=args.num_nodes,pred_len=args.pred_len,num_features=args.num_features,supports=adj)

        elif args.model_name == 'STTNS':
            time_num=24*60//args.points_per_hour # 一天中有多少个5分钟
            num_layers=1
            heads=1
            dropout=0
            forward_expansion = 4 # 堆叠多少个时空block
            self.model = STTNSNet(adj, in_channels=args.num_features, embed_size=args.d_model,
                    time_num=time_num, num_layers=num_layers,T_dim=args.seq_len, output_T_dim=args.pred_len,
                    heads=heads, dropout=dropout, forward_expansion=forward_expansion)

        elif args.model_name == 'T_GCN':
            hidden_dim=100
            self.model=TGCN(adj,features=args.num_features,hidden_dim=hidden_dim,pred_len=args.pred_len)

        elif args.model_name == 'SANN':
            args.past_t=1 # 表示的是滞后(卷积时候使用)
            args.dropout=0.0
            self.model=SANN(n_inp=args.num_features, n_out=args.num_features, t_inp=args.seq_len, t_out=args.pred_len, n_points=args.num_nodes,
                            past_t=args.past_t, hidden_dim=args.d_model, dropout=args.dropout)

        elif args.model_name=='DLinear':
            self.model=DLinear(configs=args,adj=adj)

        elif args.model_name == 'DKFN':
            K=3 # 表示的是GCN看几阶邻居
            self.model = DKFN(K=K, A=adj, num_nodes=args.num_nodes,num_features=args.num_features,pred_len=args.pred_len)

        elif args.model_name == "AGCRN":
            self.model = AGCRN(seq_len=args.seq_len, num_nodes=args.num_nodes, num_feature=args.num_features,
                               pred_len=args.pred_len)

        elif args.model_name == "GMAN":
            args.L = 1  # number of STAtt Blocks
            args.K = 8  # number of attention heads
            args.d = 8  # dims of each head attention outputs
            # SE是节点的Node2Vec的表征
            tmp = os.path.join(args.dataset_dir, args.data_name)
            SE_path = os.path.join(tmp, 'SE.txt')

            if not os.path.exists(SE_path):  # 如果不存在文件就直接生成对应的文件
                print(f'由于不存在文件名为{SE_path}的文件，因此调用对应的函数生成对应的文件')
                adj_file = os.path.join(tmp, 'adj.pkl')
                if not os.path.exists(adj_file):
                    adj_file = os.path.join(tmp, 'distance.csv')
                tu.generate_SE.Generate_SE(Adj_file=adj_file, SE_file=SE_path, node_num=args.num_nodes)
            SE = tu.Get_SE.get_SE(datapath=SE_path)
            self.model = GMAN(SE, args, bn_decay=0.1)

        elif args.model_name == "PDFormer":
            args.output_dim=1
            args.embed_dim,args.skip_dim,args.lape_dim = 64,256,8
            args.geo_num_heads,args.sem_num_heads,args.t_num_heads = 4,2,2
            args.mlp_ratio =  4
            args.qkv_bias = True
            args.drop, args.attn_drop,args.drop_path=  0.,0.,0.3
            args.s_attn_size,args.t_attn_size = 3,1
            args.enc_depth = 6
            args.type_ln = "pre"
            args.type_short_path = "hop" # 表示的是节点间的距离使用几阶邻居来衡量
            args.add_time_in_day, args.add_day_in_week = True,True
            args.huber_delta, args.quan_delta, args.far_mask_delta, args.dtw_delta = 2,0.25,7,5
            self.model = PDFormer(config=args,adj=adj)

        elif args.model_name == "STID":
            args.if_T_i_D=True # 是否使用日期编码
            args.if_D_i_W=True # 是否使用周编码
            args.if_node=True # 是否是节点
            args.day_of_week_size=7 # 选取一周的n天
            args.time_of_day_size=args.points_per_hour*24 #一天有几个时间步记录
            args.num_layer=3
            self.model = STID(args)

        elif args.model_name == "ST_Norm":
            args.layers=4
            self.model = ST_Norm(in_dim=args.num_features,pred_len=args.pred_len,
                                 num_nodes=args.num_nodes,tnorm_bool=True,snorm_bool=True,layers=args.layers)

        elif args.model_name=="D2STGNN":
            args.num_hidden,args.node_hidden=32,10
            args.k_s,args.k_t=2,3
            args.gap=3
            args.dropout=0.1
            args.time_emb_dim=10
            args.num_modalities=2
            args.adj=torch.tensor(adj).to(device)
            args.adjs=[(tu.graph_process.transition_matrix(adj).T).to(device), (tu.graph_process.transition_matrix(adj.T).T).to(device)]
            args_dict = vars(args)
            self.model=D2STGNN(**args_dict)

        elif args.model_name=='AFDGCN':
            input_dim,hidden_dim= args.num_features,args.d_model
            output_dim,embed_dim = 1,8
            cheb_k = 2
            horizon,timesteps = args.pred_len,args.seq_len
            num_layers = 1
            heads, kernel_size = 4,5
            self.model = AFDGCN(num_node=args.num_nodes,input_dim=input_dim,hidden_dim=hidden_dim,
                                output_dim=output_dim,embed_dim=embed_dim,cheb_k=cheb_k,horizon=horizon,
                                num_layers=num_layers,heads=heads,timesteps=timesteps,A=adj,kernel_size=kernel_size)

        elif args.model_name=='STWave':
            dataset,_, _=load_data(args)
            train_dataset, _ = split_dataset(dataset, split_rate=0.8)
            layers,samples = 2,1
            heads=8
            dims = 16
            tmp = os.path.join(args.dataset_dir, args.data_name)
            tem_adj_file = os.path.join(tmp, 'tem_adj_file')
            self.localadj, self.spawave, self.temwave = loadGraph(self.adj,tem_adj_file,heads*dims, train_dataset[...,0:1])
            self.model = STWave(heads, dims, layers, samples,
                            self.localadj, self.spawave, self.temwave,
                            args.seq_len, args.pred_len,args=args)

        elif args.model_name=='MegaCRN':
            args.lamb,args.lamb1=0.01,0.01
            args.rnn_units, args.num_rnn_layers=64,1
            args.mem_num,args.mem_dim=20,64
            args.max_diffusion_step,args.cl_decay_steps=3,2000
            args.use_curriculum_learning=True
            self.model =MegaCRN(num_nodes=args.num_nodes, input_dim=args.num_features, output_dim=args.num_features, horizon=args.pred_len,
                    rnn_units=args.rnn_units, num_layers=args.num_rnn_layers, mem_num=args.mem_num, mem_dim=args.mem_dim,
                    cheb_k = args.max_diffusion_step, cl_decay_steps=args.cl_decay_steps, use_curriculum_learning=args.use_curriculum_learning,args=args)

        elif args.model_name=="STGODE":
            args.output_dim = 1
            self.model=STGODE(num_nodes=args.num_nodes,num_features=args.num_features,
                   num_timesteps_input=args.seq_len,num_timesteps_output=args.pred_len,
                   A_sp_hat=adj,args=args)

        elif args.model_name=='PGCN':
            self.model = PGCN(supports=adj,in_dim=args.num_features,out_dim=args.pred_len)

        elif args.model_name=='PMC_GCN':
            time_num = 96  # SH
            num_layers = 3  # Number of ST Block
            heads = 4  # Number of Heads in MultiHeadAttention
            cheb_K = 2  # Order for Chebyshev Polynomials (Eq 2)
            dropout = 0
            forward_expansion = 4  # Dimension of Feed Forward Network: embed_size --> embed_size * forward_expansion --> embed_size
            self.model = PMC_GCN(adj,args.num_features,args.d_model,time_num,num_layers,args.seq_len,args.pred_len,heads,cheb_K,forward_expansion, dropout)

        elif args.model_name == 'STIDGCN':
            args.n_hidden = 32
            args.granularity = 24*args.points_per_hour # 一天多少个采样点，5min采样一次的话那就是288
            self.model = STIDGCN(input_dim=args.num_features, num_nodes=args.num_nodes, channels=args.n_hidden, granularity=args.granularity,
                                 input_len=args.seq_len,output_len=args.pred_len,  points_per_hour=args.points_per_hour, dropout=0.1)

        elif args.model_name == 'TESTAM':
            self.model = TESTAM(args.num_features, args.num_nodes, args.seq_len, dropout=0.3, in_dim=args.num_features, out_dim=args.pred_len, hidden_size=32, layers=3, points_per_hour=args.points_per_hour)

        elif args.model_name == 'WAVGCRN':
            self.model = WavGCRN(batch_size=args.batch_size, gcn_depth=2,num_nodes=args.num_nodes,predefined_A=[adj],seq_length=args.seq_len,out_dim=args.num_features,
                 in_dim=args.num_features,output_dim=args.pred_len,list_weight=[0.05, 0.95, 0.475],cl_decay_steps=2000,
                 hidden_size=64,points_per_hour=args.points_per_hour)

        elif args.model_name == 'DCRNN':
            self.model = DCRNN(adj_mat=adj, batch_size=args.batch_size, enc_input_dim=args.num_features, dec_input_dim=1, max_diffusion_step=2,
                               num_nodes=args.num_nodes,num_rnn_layers=2, rnn_units=args.d_model, seq_len=args.seq_len, output_dim=1, filter_type='dual_random_walk')

        elif args.model_name == 'STGCN':
            args.blocks=[[args.num_features], [64, 16, 64], [64, 16, 64], [128, 128], [args.num_features]]
            args.graph_conv_type='cheb_graph_conv'
            args.act_func='glu'
            args.enable_bias=True
            args.Kt,args.Ks=3,3
            args.droprate=0.5
            args.n_his=args.seq_len
            self.model = STGCN(args=args,adj=adj,blocks= args.blocks,n_vertex=args.num_nodes)


        elif args.model_name == 'GAT_LSTM':
            self.model = GAT_LSTM(encoder_layers=3, decoder_layers=3, pred_len=args.pred_len, seq_len=args.seq_len,
                              num_nodes=args.num_nodes,
                              num_features=args.num_features, n_hidden=args.d_model, dropout=0.1, support_len=1, head=1,
                              type='concat')

        elif args.model_name == 'ASTRformer':
            self.model = ASTRformer(seq_len=args.seq_len,num_nodes=args.num_nodes,in_feature=args.num_features,
                                    d_model=args.d_model,pred_len=args.pred_len,args=args)


        else:
            raise NotImplementedError

    '''一个epoch下的代码'''
    def train_test_one_epoch(self,args,dataloader,adj,save_manager: tu.save.SaveManager,epoch,mode='train',max_iter=float('inf'),**kargs):
        if mode == 'train':
            self.model.train()
            self.optimizer.zero_grad()
        elif mode == 'test' or mode =='val':
            self.model.eval()
        else:
            raise NotImplementedError

        metric_logger = tu.metric.MetricMeterLogger() # 初始化一个字典，记录对应的训练的损失结果

        # Dataloader，只不过为了分布式训练因此多了部分的代码
        for index, unpacked in enumerate(
                metric_logger.log_every(dataloader, header=mode, desc=f'{mode} epoch {epoch}')):
            if index > max_iter:
                break
            seqs, seqs_time,targets,targets_time = unpacked # (B,L,C,N)
            seqs, targets = seqs.cuda().float(), targets.cuda().float()
            seqs_time, targets_time = seqs_time.cuda().float(), targets_time.cuda().float()
            seqs,targets=seqs.permute(0,2,3,1),targets.permute(0,2,3,1)# (B,L,C,N)
            seqs_time, targets_time = seqs_time.permute(0, 2, 3, 1), targets_time.permute(0, 2, 3, 1) #(B,C,N=1,L)
            # TODO 模型的输入和输出的维度都是(B,C,N,L).输出的特征维度默认为1
            self.adj = np.array(self.adj)# 如果不是array，那么送入model的时候第一个维度会被分成两半
            pred = self.model(seqs,self.adj,seqs_time=seqs_time,targets_time=targets_time,targets=targets,mode=mode,index=index,epoch=epoch)  # 输入模型
            if (args.model_name=='MegaCRN' or args.model_name=='TESTAM') and mode=='train':
                pred,loss_part=pred[0],pred[1]

            # 计算损失 TODO 默认计算的是第一个特征维度
            targets=targets[:, 0:1, ...]
            if pred.shape[1]!=1:
                pred=pred[:,0:1,...]

            loss = self.criterion(pred.to(targets.device), targets) # 0表示的是特征只取流量这一个特征(参考DGCN的源代码)
            # 计算MSE、MAE损失
            mse = torch.mean(torch.sum((pred - targets) ** 2, dim=1).detach())
            mae = torch.mean(torch.sum(torch.abs(pred - targets), dim=1).detach())

            metric_logger.update(loss=loss, mse=mse, mae=mae)  # 更新训练记录

            step_logs = metric_logger.values()
            step_logs['epoch'] = epoch
            save_manager.save_step_log(mode, **step_logs)  # 保存每一个batch的训练loss

            if mode == 'train':
                if args.model_name=='MegaCRN' or args.model_name=='TESTAM':
                    loss=loss+loss_part
                loss.backward()
                # 梯度裁剪
                if args.clip_max_norm > 0:  # 裁剪值大于0
                    nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), args.clip_max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        if mode == 'train':
            if args.model_name == 'WAVGCRN':
                self.model.graph_learning()

        epoch_logs = metric_logger.get_finish_epoch_logs()
        epoch_logs['epoch'] = epoch
        save_manager.save_epoch_log(mode, **epoch_logs)  # 保存每一个epoch的训练loss

        return epoch_logs

    def train(self):
        args=self.agrs
        if args.resume!=True:
            tu.config.create_output_dir(args)  # 创建输出的目录
            print('output dir: {}'.format(args.output_dir))
            start_epoch = 0
        else:
            start_epoch=self.start_epoch

        # 以下是保存超参数
        save_manager = tu.save.SaveManager(args.output_dir, args.model_name, 'mse', compare_type='lt', ckpt_save_freq=30)
        save_manager.save_hparam(args)

        max_iter = float('inf')  # 知道满足对应的条件才会停下来

        # 以下开始正式的训练
        for epoch in range(start_epoch, args.end_epoch):
            if tu.dist.is_dist_avail_and_initialized():  # 不进入下面的代码
                self.train_sampler.set_epoch(epoch)
                self.val_sampler.set_epoch(epoch)
                self.test_sampler.set_epoch(epoch)

            tu.dist.barrier()  # 分布式训练，好像也没作用

            # train
            self.train_test_one_epoch(args,self.train_dataloader,self.adj, save_manager, epoch, mode='train')

            self.lr_optimizer.step()  # lr衰减

            # val
            val_logs = self.train_test_one_epoch(args, self.val_dataloader, self.adj, save_manager, epoch, mode='val')

            # test
            test_logs = self.train_test_one_epoch(args,self.test_dataloader,self.adj, save_manager, epoch,mode='test')


            # 早停机制
            self.early_stopping(val_logs['mse'], model=self.model, epoch=epoch)
            if self.early_stopping.early_stop:
                break
        # 训练完成 读取最好的权重
        try:
            dp_mode = args.args.dp_mode
        except AttributeError as e:
            dp_mode = True
        output_path = os.path.join(self.output_path, args.data_name + '_best_model.pkl')
        self.load_best_model(path=output_path, args=args, distributed=dp_mode)


    def ddp_module_replace(self,param_ckpt):
        return {k.replace('module.', ''): v.cpu() for k, v in param_ckpt.items()}

    # TODO 加载最好的模型
    def load_best_model(self, path, args=None, distributed=True):

        ckpt_path = path
        if not os.path.exists(ckpt_path):
            print('路径{0}不存在，模型的参数都是随机初始化的'.format(ckpt_path))
        else:
            ckpt = torch.load(ckpt_path)

            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.lr_optimizer.load_state_dict(ckpt['lr_scheduler'])
            self.start_epoch=ckpt['epoch']

    def test(self):
        args=self.agrs
        try:
            dp_mode = args.args.dp_mode
        except AttributeError as e:
            dp_mode = True

        # 读取最好的权重
        if args.resume:
            self.load_best_model(path=self.resume_path, args=args, distributed=dp_mode)
        star = datetime.now()
        metric_dict=test.test(args,self.model,test_dataloader=self.test_dataloader,adj=self.adj)
        end=datetime.now()
        test_cost_time=(end-star).total_seconds()
        print("test花费了：{0}秒".format(test_cost_time))
        mae=metric_dict['mae']
        mse=metric_dict['mse']
        rmse=metric_dict['rmse']
        mape=metric_dict['mape']


        # 创建csv文件记录训练结果
        if not os.path.isdir('./results/'):
            os.mkdir('./results/')

        log_path = './results/experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR',
                          'batch_size', 'seed', 'MAE', 'MSE', 'RMSE','MAPE','seq_len',
                           'pred_len', 'd_model', 'd_ff','test_cost_time',
                           # 'e_layers', 'd_layers',
                            'info','output_dir']]
            Write_csv.write_csv(log_path, table_head, 'w+')

        time = datetime.now().strftime('%Y%m%d-%H%M%S')  # 获取当前系统时间
        a_log = [{'dataset': args.data_name, 'model': args.model_name, 'time': time,
                  'LR': args.lr,
                  'batch_size': args.batch_size,
                  'seed': args.seed, 'MAE': mae, 'MSE': mse,'RMSE':rmse,"MAPE":mape,'seq_len': args.seq_len,
                  'pred_len': args.pred_len,'d_model': args.d_model, 'd_ff': args.d_ff,
                  'test_cost_time': test_cost_time,
                  # 'e_layers': args.e_layers, 'd_layers': args.d_layers,
                  'info': args.info,'output_dir':args.output_dir}]
        Write_csv.write_csv_dict(log_path, a_log, 'a+')





