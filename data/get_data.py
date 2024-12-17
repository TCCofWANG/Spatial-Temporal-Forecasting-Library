from data.data_process import read_and_generate_dataset,load_data,get_time_features
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from torch.utils import data
from data.dataset import split_dataset,SubwayDataset


'''切分数据集和创建dataloader'''
def build_dataloader(args, test=False):
    '''训练集，验证集，测试集=6:2:2 or 7:1:2，训练集和验证集同分布，但是训练集和测试集不是同分布'''
    dataset,Time,adj = load_data(args)  # 得到所有数据记录和对应的邻接矩阵
    Time=get_time_features(Time) #(total_len,N=1,C=5)，与dataset的形状一样

    total_len, num_nodes, num_features = dataset.shape  # shape(total_len,num_nodes,num_features)
    args.num_nodes = num_nodes  # 得到节点个数
    args.num_features = num_features  # 得到每一个时间步下一个节点对应的特征的个数
    args.time_features = Time.shape[-1]  # 得到时间特征维度
    print('节点个数：', args.num_nodes)
    print('特征维度：', args.num_features)
    print('时间特征维度：', args.time_features)

    if args.data_name == "PeMS-Bay" or args.data_name == "METR-LA":
        # 单单切分训练和测试集（在总长度维度上进行切分）
        train_dataset,test_dataset = split_dataset(dataset,split_rate=0.8)
        Time = Time.reshape(-1, args.time_features, 1)  # (total_len,C=5,N=1)
        train_time_dataset, test_time_dataset = split_dataset(Time, split_rate=0.8)
        # 以下把训练集和验证集切分出来
        # 训练集：验证集：测试集=7:1:2
        train_dataset,val_dataset=split_dataset(train_dataset,split_rate=0.875)
        train_time_dataset, val_time_dataset = split_dataset(train_time_dataset, split_rate=0.875)
    else:
        # 单单切分训练和测试集（在总长度维度上进行切分）
        train_dataset,test_dataset = split_dataset(dataset,split_rate=0.8)
        Time = Time.reshape(-1, args.time_features, 1)  # (total_len,C=5,N=1)
        train_time_dataset, test_time_dataset = split_dataset(Time, split_rate=0.8)
        # 以下把训练集和验证集切分出来
        # 训练集：验证集：测试集=6:2:2
        train_dataset,val_dataset=split_dataset(train_dataset,split_rate=0.75)
        train_time_dataset, val_time_dataset = split_dataset(train_time_dataset, split_rate=0.75)

    # TODO 这里是测试不同的站点服从的分布是一致的(大部分的模型平台这么做)
    train_len=len(train_dataset)
    val_len=len(val_dataset)
    test_len=len(test_dataset)
    # FIXME SZ_metro是最大最小归一化还是标准化
    if args.data_name == "PeMS-Bay" or args.data_name == "METR-LA":
        # 以下两个分别是做标准化
        scaler = StandardScaler(with_mean=True, with_std=True)
        train_dataset = scaler.fit_transform(train_dataset.reshape(-1,num_features)) # 将站点和总时间步融合起来
        val_dataset = scaler.transform(val_dataset.reshape(-1,num_features))
        test_dataset = scaler.transform(test_dataset.reshape(-1,num_features))

        train_dataset = np.transpose(train_dataset.reshape(train_len, num_nodes,num_features),(0,2,1))
        val_dataset = np.transpose(val_dataset.reshape(val_len, num_nodes, num_features),(0,2,1))
        test_dataset = np.transpose(test_dataset.reshape(test_len, num_nodes, num_features),(0,2,1))
        mean = scaler.mean_.reshape(1, num_features,1)
        std = scaler.scale_.reshape(1, num_features,1)

        train_dataset = SubwayDataset(train_dataset, train_time_dataset, args.seq_len, args.pred_len,std=std,mean=mean)
        val_dataset = SubwayDataset(val_dataset, val_time_dataset, args.seq_len, args.pred_len,std=std,mean=mean)
        test_dataset = SubwayDataset(test_dataset, test_time_dataset, args.seq_len, args.pred_len,std=std,mean=mean)

    elif args.data_name=="PEMS04"or args.data_name=="PEMS08"or args.data_name == "SZ_metro":
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_dataset = scaler.fit_transform(train_dataset.reshape(-1,num_features)) #将站点和总时间步融合起来
        val_dataset = scaler.transform(val_dataset.reshape(-1,num_features))
        test_dataset = scaler.transform(test_dataset.reshape(-1,num_features))

        train_dataset = np.transpose(train_dataset.reshape(train_len,num_nodes,num_features),(0,2,1))
        val_dataset = np.transpose(val_dataset.reshape(val_len,num_nodes,num_features),(0,2,1))
        test_dataset = np.transpose(test_dataset.reshape(test_len,num_nodes,num_features),(0,2,1))

        min_values = scaler.data_min_.reshape(1, num_features,1)
        max_values = scaler.data_max_.reshape(1, num_features,1)
        train_dataset = SubwayDataset(train_dataset, train_time_dataset, args.seq_len, args.pred_len, max=max_values,min=min_values)
        val_dataset = SubwayDataset(val_dataset, val_time_dataset, args.seq_len, args.pred_len, max=max_values,min=min_values)
        test_dataset = SubwayDataset(test_dataset, test_time_dataset, args.seq_len, args.pred_len, max=max_values,min=min_values)
    else:
        assert print("数据集规范化未定义")


    if not isinstance(adj, torch.Tensor):
        adj = torch.tensor(adj, dtype=torch.float32)
    adj.cuda()

    train_sampler, val_sampler,test_sampler = None,None,None

    if args.distributed and not test:
        # 以下是分布式训练的部分
        train_sampler = data.DistributedSampler(train_dataset, seed=args.seed)  # 分布式的数据采样器
        train_batch_sampler = data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                                           num_workers=args.num_workers, pin_memory=args.pin_memory)

        val_sampler = data.DistributedSampler(val_dataset, seed=args.seed)
        val_dataloader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler,
                                          num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)

        test_sampler = data.DistributedSampler(test_dataset, seed=args.seed)
        test_dataloader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler,
                                          num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)
    else:
        # 以下是非分布式的训练部分
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                           drop_last=True,
                                           num_workers=args.num_workers,
                                           pin_memory=args.pin_memory)  # pin_memory默认为False，设置为TRUE会加速读取数据

        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                          drop_last=True,
                                          num_workers=args.num_workers, pin_memory=args.pin_memory)

        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                          drop_last=True,
                                          num_workers=args.num_workers, pin_memory=args.pin_memory)
    # FIXME SZ_metro是最大最小归一化还是标准化
    if args.data_name=="PeMS-Bay"or args.data_name=="METR-LA":
        mean,std=np.expand_dims(mean,axis=-1),np.expand_dims(std,axis=-1) #(B=1,C,N,L=1)
        train_dataloader.mean,val_dataloader.mean,test_dataloader.mean=mean,mean,mean
        train_dataloader.std, val_dataloader.std, test_dataloader.std = std, std, std

    elif args.data_name == "PEMS04" or args.data_name == "PEMS08" or args.data_name=="SZ_metro":
        min, max = np.expand_dims(min_values, axis=-1), np.expand_dims(max_values, axis=-1)  # (B=1,C,N,L=1)
        train_dataloader.min, val_dataloader.min, test_dataloader.min = min, min, min
        train_dataloader.max, val_dataloader.max, test_dataloader.max = max, max, max
    return adj, train_dataloader, val_dataloader, test_dataloader, train_sampler, val_sampler, test_sampler






def build_dataloader_interval(args):
    batch_size=args.batch_size
    num_of_weeks, num_of_days=args.num_of_weeks,args.num_of_days
    num_of_hours=args.num_of_hours
    num_for_predict=args.pred_len # 等于预测长度
    points_per_hour=args.points_per_hour
    merge=False # 表示不把训练集和验证集合成一个数据集
    dataset,_,adj = load_data(args)  # 得到所有数据记录和对应的邻接矩阵
    assert np.isnan(dataset).any()==False # 如果原始数据中有nan，这里就报错
    all_data = read_and_generate_dataset(dataset,
                                         num_of_weeks,
                                         num_of_days,
                                         num_of_hours,
                                         num_for_predict,
                                         points_per_hour,
                                         merge)
     #TODO 在此处自定节点个数和特征维度
    args.num_features=all_data['train']['week'].shape[1]
    args.num_nodes = all_data['train']['week'].shape[2]
    print('节点个数：', args.num_nodes)
    print('特征维度：', args.num_features)

    # training set data loader
    train_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['train']['week']),
            torch.Tensor(all_data['train']['day']),
            torch.Tensor(all_data['train']['recent']),
            torch.Tensor(all_data['train']['target'])
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # validation set data loader
    val_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['val']['week']),
            torch.Tensor(all_data['val']['day']),
            torch.Tensor(all_data['val']['recent']),
            torch.Tensor(all_data['val']['target'])
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    # testing set data loader
    test_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['test']['week']),
            torch.Tensor(all_data['test']['day']),
            torch.Tensor(all_data['test']['recent']),
            torch.Tensor(all_data['test']['target'])
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    # 封装一下mean，std属性
    mean=all_data['stats']['target']['mean']
    std = all_data['stats']['target']['std']
    train_loader.mean,val_loader.mean,test_loader.mean=mean,mean,mean # #(B=1,C,N,L=1)
    train_loader.std, val_loader.std, test_loader.std = std, std, std

    return adj,train_loader,val_loader,test_loader








