
'''该代码是为了部分模型的输入是不同的时间周期准备的，比方说：模型的输入是：近期、日周期、周周期三个部分组成'''
import os
import torch
import torch.nn as nn
import numpy as np
from model import *
import torch_utils as tu
from torch_utils import Write_csv,earlystopping
from data.get_data import *
import datetime
from tqdm import tqdm
from test import test_interval
import yaml


device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
class EXP_interval():
    def __init__(self,args):
        self.agrs=args
        tu.dist.init_distributed_mode(args)  # 初始化分布式训练
        # seed = tu.dist.get_rank() + args.seed  # 0+args.seed
        # tu.model_tool.seed_everything(seed)  # 设置随机数种子，便于可重复实验

        # train_sampler,test_sampler是分布式训练时使用的，未分布式训练时，其都为None
        # get_data
        adj, self.train_dataloader,self.val_dataloader, self.test_dataloader = build_dataloader_interval(args)
        self.train_sampler, self.val_sampler, self.test_sampler=None,None,None
        self.adj=adj

        # get_model
        self.build_model(args)  # 得到对应的模型
        self.model.to(device)  # 送入对应的设备中

        self.model = tu.dist.ddp_model(self.model, [args.local_rank])  # 分布式训练模型，好像也没有发挥作用
        if args.dp_mode:
            self.model = nn.DataParallel(self.model)  # 分布训练，单机子多显卡
            print('using dp mode')

        # 模型训练所需的
        self.criterion = nn.MSELoss()

        self.optimizer= torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # 引入权重衰减的Adam

        # 权重衰减：cos衰减
        self.lr_optimizer = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.end_epoch,eta_min=args.lr / 1000)

        # 早停机制
        if args.output_dir == None or args.output_dir == 'None' or args.output_dir == 'none':
            args.output_dir = None
            tu.config.create_output_dir(args)  # 创建输出的目录
            args.resume_dir = args.output_dir

        output_path = os.path.join(args.output_dir,args.model_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        path = os.path.join(output_path,args.data_name + '_best_model.pkl')
        self.output_path = path
        self.early_stopping = earlystopping.EarlyStopping(path=path,optimizer=self.optimizer, scheduler=self.lr_optimizer,patience=args.patience)
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
            # 自动读取超参数
            hparam_path = os.path.join(args.output_dir, 'hparam.yaml')
            with open(hparam_path, 'r') as f:
                hparam_dict = yaml.load(f, yaml.FullLoader)
                args.output_dir = hparam_dict['output_dir']
            # 读取最好的权重
            self.load_best_model(path=self.resume_path,args=args, distributed=dp_mode)

    '''建立模型'''
    def build_model(self,args):
        if args.model_name == 'DGCN':
            week=args.pred_len*args.num_of_weeks
            day=args.pred_len*args.num_of_days
            recent=args.pred_len*args.num_of_hours
            # K表示的是切比雪夫多项式的阶数，kt表示的是时间卷积的卷积核大小
            self.model=DGCN(c_in=args.num_features,d_model=args.d_model,c_out=args.num_features,pred_len=args.pred_len,num_nodes=args.num_nodes,week=week,day=day,recent=recent,K=3,Kt=3)

        elif args.model_name=='ASTGCN':
            week = args.pred_len * args.num_of_weeks
            day = args.pred_len * args.num_of_days
            recent = args.pred_len * args.num_of_hours
            # K表示的是切比雪夫多项式的阶数，kt表示的是时间卷积的卷积核大小
            self.model = ASTGCN(c_in=args.num_features, c_out=args.num_features,num_nodes=args.num_nodes,pred_len=args.pred_len, week=week,day=day, recent=recent,K=3, Kt=3)

        else:
            raise NotImplementedError

    '''一个epoch下的代码'''
    def train_test_one_epoch(self,args,dataloader,save_manager: tu.save.SaveManager,epoch,mode='train',max_iter=float('inf')):
        if mode == 'train':
            self.model.train()
            self.optimizer.zero_grad()
        elif mode == 'test' or 'val':
            self.model.eval()
        else:
            raise NotImplementedError

        metric_logger = tu.metric.MetricMeterLogger()


        # Dataloader，只不过为了分布式训练因此多了部分的代码
        for index, unpacked in enumerate(
                metric_logger.log_every(dataloader, header=mode, desc=f'{mode} epoch {epoch}')):
            if index > max_iter:
                break
            # TODO 模型的特征输入输出维度(B,C,N,L)
            train_w, train_d, train_r, targets= unpacked
            train_w = train_w.to(device).float()
            train_d = train_d.to(device).float()
            train_r = train_r.to(device).float()
            targets = targets.to(device).float()
            assert (torch.isnan(train_w).any() or torch.isnan(train_d).any()
                    or torch.isnan(train_r).any() or torch.isnan(targets).any())==False
            self.adj=np.array(self.adj) # 如果不是array，那么送入model的时候第一个维度会被分成两半
            pred = self.model(train_w,train_d,train_r,self.adj)  # 输入模型
            # 计算损失
            targets=targets[:,0:1,:,:]
            if pred.shape[1]!=1:
                pred=pred[:,0:1,:,:]
            loss = self.criterion(pred, targets)
            # if torch.isnan(loss):
            #     print()
            # 计算MSE、MAE损失
            mse = torch.mean(torch.sum((pred - targets) ** 2, dim=1).detach())
            mae = torch.mean(torch.sum(torch.abs(pred - targets), dim=1).detach())

            metric_logger.update(loss=loss, mse=mse, mae=mae)  # 更新训练记录

            step_logs = metric_logger.values()
            step_logs['epoch'] = epoch
            save_manager.save_step_log(mode, **step_logs)  # 保存每一个batch的训练loss

            if mode == 'train':
                loss.backward()
                # 梯度裁剪
                if args.clip_max_norm > 0:  # 裁剪值大于0
                    nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), args.clip_max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        epoch_logs = metric_logger.get_finish_epoch_logs()
        epoch_logs['epoch'] = epoch
        save_manager.save_epoch_log(mode, **epoch_logs)  # 保存每一个epoch的训练loss

        return epoch_logs

    '''训练代码'''
    def train(self):
        args=self.agrs

        if args.resume!=True:
            tu.config.create_output_dir(args)  # 创建输出的目录
            print('output dir: {}'.format(args.output_dir))

        # 以下是保存超参数
        save_manager = tu.save.SaveManager(args.output_dir, args.model_name, 'mse', compare_type='lt', ckpt_save_freq=30)
        save_manager.save_hparam(args)
        start_epoch=0
        max_iter = float('inf')  # 知道满足对应的条件才会停下来

        # 以下开始正式的训练
        for epoch in range(start_epoch, args.end_epoch):
            if tu.dist.is_dist_avail_and_initialized():  # 不进入下面的代码
                self.train_sampler.set_epoch(epoch)
                self.val_sampler.set_epoch(epoch)
                self.test_sampler.set_epoch(epoch)

            tu.dist.barrier()  # 分布式训练，好像也没作用

            # train
            self.train_test_one_epoch(args,self.train_dataloader, save_manager, epoch, mode='train')

            self.lr_optimizer.step()  # lr衰减

            # val
            val_logs = self.train_test_one_epoch(args, self.val_dataloader, save_manager, epoch,mode='val')

            # test
            test_logs = self.train_test_one_epoch(args,self.test_dataloader, save_manager, epoch,mode='test')


            # 早停机制
            self.early_stopping(val_logs['mse'], model=self.model, epoch=epoch)
            if self.early_stopping.early_stop:
                break

        # 训练完成 读取最好的权重
        try:
            dp_mode = args.args.dp_mode
        except AttributeError as e:
            dp_mode = True
        self.load_best_model(path=self.output_path,args=args, distributed=dp_mode)


    def ddp_module_replace(self,param_ckpt):
        return {k.replace('module.', ''): v.cpu() for k, v in param_ckpt.items()}

    # TODO 加载最好的模型
    def load_best_model(self,path,args=None, distributed=True):

        resume_path = path
        if not os.path.exists(resume_path):
            print('路径{0}不存在，模型的参数都是随机初始化的'.format(resume_path))

        ckpt = torch.load(resume_path)

        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lr_optimizer.load_state_dict(ckpt['lr_scheduler'])

    '''测试代码'''
    def test(self):
        args=self.agrs
        try:
            dp_mode = args.args.dp_mode
        except AttributeError as e:
            dp_mode = True

        if args.resume:
            # 读取最好的权重
            self.load_best_model(path=self.resume_path,args=args, distributed=dp_mode)
        star=datetime.datetime.now()
        metric_dict=test_interval(args,self.model,test_dataloader=self.test_dataloader,adj=self.adj)
        end=datetime.datetime.now()
        test_cost_time=(end-star).total_seconds()
        print("test花费了：{0}秒".format(test_cost_time))
        mae=metric_dict['mae']
        mse=metric_dict['mse']
        rmse=metric_dict['rmse']
        mape = metric_dict['mape']

        # 创建csv文件记录训练结果
        if not os.path.isdir('./results/'):
            os.mkdir('./results/')

        log_path = './results/experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR',
                          'batch_size', 'seed', 'MAE', 'MSE', 'RMSE','MAPE', 'seq_len',
                           'pred_len', 'd_model', 'd_ff','test_cost_time',
                           # 'e_layers', 'd_layers',
                            'info','output_dir']]
            Write_csv.write_csv(log_path, table_head, 'w+')

        time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')  # 获取当前系统时间
        a_log = [{'dataset': args.data_name, 'model': args.model_name, 'time': time,
                  'LR': args.lr,
                  'batch_size': args.batch_size,
                  'seed': args.seed, 'MAE': mae, 'MSE': mse,'RMSE':rmse,'MAPE':mape,'seq_len': args.seq_len,
                  'pred_len': args.pred_len,'d_model': args.d_model, 'd_ff': args.d_ff,
                  'test_cost_time':test_cost_time,
                  # 'e_layers': args.e_layers, 'd_layers': args.d_layers,
                  'info': args.info,'output_dir':args.output_dir}]
        Write_csv.write_csv_dict(log_path, a_log, 'a+')




