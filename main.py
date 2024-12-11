# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.chdir('')
import io
import sys
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import torch_utils as tu
from torch_utils.set_seed import set_seed
import torch
from exp_.exp import EXP
from exp_.exp_interval import EXP_interval

#————————————————————————————————————————————————————————————————————————————————————————————————————————————
# #########################################!!! Attention !!! #################################################
# Distributed training means running on different machines, non-distributed training means running on one machine. (Multiple cards on one machine does not count as distributed training)
# Note that when you create a new Tensor, if you create it at init time, you should use nn.Parameter() so that the Tensor will be put on the correct cuda.
# Even if the same model, but different dataset, the corresponding output path should be different, otherwise the new code will delete all files under the current path when running (within the train function).
# If there is no batch dimension before inputting the data into the model, then convert its type to numpy, otherwise the first dimension will be split in half (two cards) by default.
# In the build model when the input adj, at this time the adj is already normalized Laplace matrix
#—————————————————————————————————————————————————————————————————————————————————————————————————————————————

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False

'''其他参数在config文件内'''
def add_config(parser):
    # exp_name选择：
    # deep_learning_interval-->'DGCN','ASTGCN'
    # deep_learning-->剩下的模型
    parser.add_argument('--exp_name', default='deep_learning', choices=['deep_learning','deep_learning_interval'], help='模型选用哪一个exp文件')
    parser.add_argument('--train', default=True,type=str2bool,choices=[True,False], help='是否进行训练')
    parser.add_argument('--resume', default=False, type=str2bool, choices=[True, False], help='是否读取预训练权重')
    parser.add_argument('--output_dir',type=str,default=None,help='为None会自根据现有的输出文件自动+1，如果指定就会覆盖现有的输出文件')
    parser.add_argument('--resume_dir', type=str,default=None,help='读取checkpoint的位置')
    parser.add_argument('--dp_mode', type=str2bool, default=False,help='是否在多卡上跑')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)
    # model settings
    parser.add_argument('--model_name', type=str, default='AFDGCN',help=['gwnet','HGCN','STTNS','T_GCN','SANN','DLinear','DKFN','AGCRN'
                                                                          'PDFormer','GMAN','STID','ST_Norm','D2STGNN','AFDGCN','STWave','MegaCRN',
                                                                          'STGODE','FC_LSTM','PGCN','PMC_GCN','STIDGCN','DCRNN',
                                                                          'TESTAM', 'WAVGCRN','STGCN','GAT_LSTM','ASTRformer',
                                                                        'DGCN','ASTGCN'])
    parser.add_argument('--gnn_type', type=str, default='dcn', choices=['dcn', 'gat', 'gcn'])
    parser.add_argument('--data_name', type=str, default='METR-LA', choices=['PeMS-Bay','METR-LA','PEMS04','PEMS08'])

    # dataset settings
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--pred_len', type=int, default=3)
    # 与模型参数有关的设定
    parser.add_argument('--num_features',help='输入模型的特征维度(会自动根据数据集指定)')
    parser.add_argument('--time_features',help='输入模型的时间步特征（会自动根据数据集指定）')
    parser.add_argument('--num_nodes', help='输入模型节点个数(会自动根据数据集指定)')
    parser.add_argument('--d_model',type=int,default=64,help='隐藏层维度1')
    parser.add_argument('--d_ff',type=int,default=128, help='隐藏层维度2')
    parser.add_argument('--num_gcn', type=int, default=10, help='GCN的个数')
    parser.add_argument('--patch_len', type=int, default=3, help='patch_len的长度')
    parser.add_argument('--stride', type=int, default=1, help='stride的长度')

    # 这是exp_interval中的模型的输入数据，输入数据有不同的周期性，窗口大小一般等于预测长度
    parser.add_argument('--num_of_weeks', type=int, default=2,help='表示一次取多少个week周期窗口')
    parser.add_argument('--num_of_days', type=int, default=1,help='表示一次取多少个day周期窗口')
    parser.add_argument('--num_of_hours', type=int, default=2,help='表示一次取多少个hour周期窗口(近期)')

    parser.add_argument('--points_per_hour', type=int, default=12,help='一个小时内有多少个数据采样点(与数据集有关)')

    parser.add_argument('--info', type=str, default='None', help='实验信息')
    return parser

def preprocess_args(args):
    args.pin_memory = False # Dataloader中读数据加速的方式
    return args

if __name__ == '__main__':
    args = tu.config.get_args(add_config) # 获得设定的超参数
    args = preprocess_args(args)
    set_seed(args.seed) # 设置随机数种子

    print(f"|{'=' * 101}|")
    # 使用__dict__方法获取参数字典，之后遍历字典
    for key, value in args.__dict__.items():
        # 因为参数不一定都是str，需要将所有的数据转成str
        print(f"|{str(key):>50s}|{str(value):<50s}|")
    print(f"|{'=' * 101}|")
    print(device)
    if args.exp_name=='deep_learning':
        exp=EXP(args)
    elif args.exp_name=='deep_learning_interval':
        exp=EXP_interval(args)
    else:
        raise print('没有名字为{0}的exp文件'.format(args.exp_name))

    if args.train:
        exp.train()
    with torch.no_grad():
        exp.test()


