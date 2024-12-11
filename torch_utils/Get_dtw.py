import torch
from torch.utils.data import DataLoader
from data.dataset import split_dataset,SubwayDataset
from data.data_process import *
import numpy as np
from fastdtw import fastdtw
from tqdm import tqdm
import torch.nn as nn
from tslearn.clustering import TimeSeriesKMeans, KShape
from sklearn.preprocessing import StandardScaler
class get_dtw(nn.Module):
    def __init__(self, config,dtw=True,pattern_keys=True):
        super().__init__()
        # self.parameters_str = \
        #     str(config.data_name) + '_' + str(config.seq_len) + '_' + str(config.pred_len) + '_' \
        #     + str(self.train_rate) + '_' + str(self.part_train_rate) + '_' + str(self.eval_rate) + '_' + str(
        #         self.scaler_type) + '_' \
        #     + str(self.batch_size) + '_' + str(self.load_external) + '_' + str(self.add_time_in_day) + '_' \
        #     + str(self.add_day_in_week) + '_' + str(self.pad_with_last_sample) + '_' + str("".join(self.data_col))
        # self.type_short_path = 'hop'
        # self.cache_file_name = os.path.join('./libcity/cache/',
        #                                     'pdformer_point_based_{}.npz'.format(self.parameters_str))
        self.config=config
        df,Time, _ = load_data(config)
        Time = get_time_features(Time)  # (total_len,N=1,C=5)，与dataset的形状一样
        Time = Time.reshape(-1, config.time_features, 1)  # (total_len,C=5,N=1)
        self.df = df
        self.time=Time
        self.output_dim=config.output_dim
        self.points_per_hour = config.points_per_hour
        self.time_intervals=3600//config.points_per_hour # 一个小时有3600秒除于一个小时有多少个记录点=记录点间秒数差距多少
        if dtw==True:
            self.dtw_matrix = self._get_dtw() # 得到节点间DTW的距离矩阵
        self.points_per_day = 24 * 3600 // self.time_intervals #一天有24*3600秒除于记录点秒数差距=一天内有多少的记录点
        self.cand_key_days=config.cand_key_days = 14
        self.s_attn_size =config.s_attn_size=  3
        self.n_cluster =config.n_cluster=  16
        self.cluster_max_iter=config.cluster_max_iter = 5
        self.cluster_method =config.cluster_method="kshape"
        self.dataset=config.data_name
        if pattern_keys==True:
            self.pattern_keys=self._get_pattern_key() # 得到质心

    '''得到了DTW的距离矩阵'''
    def _get_dtw(self): # FIXME 感觉这里有信息泄露，因为这里会使用测试集的数据
        cache_path = './datasets/cache/dtw_' + self.config.data_name + '.npy'
        if not os.path.exists(cache_path): # 如果不存在对应的npy文件 就自己计算
            print(f'由于不存在路径为{cache_path}对应的文件，因此计算节点间dtw距离')
            df=self.df
            data_mean = np.mean( # TODO 这个是在整个数据集上进行计算节点间的距离
                [df[24 * self.points_per_hour * i: 24 * self.points_per_hour * (i + 1)] # 这里计算的是一天内各个站点对应的特征的平均值
                 for i in range(df.shape[0] // (24 * self.points_per_hour))], axis=0) # data_mean(total,N,C)
            _,self.num_nodes,self.feature=df.shape
            dtw_distance = np.zeros((self.num_nodes, self.num_nodes))
            for i in tqdm(range(self.num_nodes)):
                for j in range(i, self.num_nodes):
                    dtw_distance[i][j], _ = fastdtw(data_mean[:, i, :], data_mean[:, j, :], radius=6) #计算对应的每一个站点天之间的dtw距离
            for i in range(self.num_nodes): # 这里相当于构造一个对称阵
                for j in range(i):
                    dtw_distance[i][j] = dtw_distance[j][i]
            np.save(cache_path, dtw_distance)

        dtw_matrix = np.load(cache_path)
        print('Load DTW matrix from {}'.format(cache_path))
        return dtw_matrix

    def get_seq_traindata(self):
        train_dataset, _ = split_dataset(self.df, split_rate=0.8)  # 单单切分训练和测试集（时间步维度上进行切分）
        train_time_dataset, _ = split_dataset(self.time, split_rate=0.8)

        _, num_nodes, num_features = train_dataset.shape
        scaler = StandardScaler(with_mean=True, with_std=True)
        train_dataset = scaler.fit_transform(train_dataset.reshape(len(train_dataset), -1))  # 将站点和特征融起来
        train_dataset = train_dataset.reshape(len(train_dataset), num_features, num_nodes, )
        train_dataset = SubwayDataset(train_dataset,train_time_dataset, self.config.seq_len, self.config.pred_len)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,drop_last=False)
        self.train_dataset=[]
        for batch_x,_,batch_y,_ in train_dataloader:
            self.train_dataset.append(batch_x)
        self.train_dataset=torch.concat(self.train_dataset,dim=0)
        return self.train_dataset

    def _get_pattern_key(self):
        self.pattern_key_file = os.path.join(  # 这个数据应该是聚类后每一个簇中心对应的数据
            './datasets/cache/', 'pattern_keys_{}_{}_{}_{}_{}_{}'.format(
                self.cluster_method, self.dataset, self.cand_key_days, self.s_attn_size, self.n_cluster,
                self.cluster_max_iter))
        if not os.path.exists(self.pattern_key_file + '.npy'):
            print(f'由于不存在地址为{self.pattern_key_file}的文件，因此计算对应的聚类后的质心数据')

            self.train_dataset=self.get_seq_traindata() # 得到训练集 train_dataset(total_len,Len,dim,N)
            # FIXME 感觉这个维度很有问题
            cand_key_time_steps = self.cand_key_days * self.points_per_day # 14*每一天有多少个记录点 FIXME 这里的14是什么意思？表示的是两周吗？
            pattern_cand_keys = (self.train_dataset[:cand_key_time_steps, :self.s_attn_size, :self.output_dim, :].permute(0,3,1,2) # FIXME 为什么这里要在时间维度上取3
                                 .reshape(-1, self.s_attn_size, self.output_dim)) # TODO 这个是仅仅在训练集上进行聚类,并且训练集是切分了seq_len的
            print("Clustering...")
            if self.cluster_method == "kshape": # 这个是利用自相关来计算进行距离计算，其他与Kmeans相同
                km = KShape(n_clusters=self.n_cluster, max_iter=self.cluster_max_iter).fit(pattern_cand_keys)
            else: # 这个就是Kmeans
                km = TimeSeriesKMeans(n_clusters=self.n_cluster, metric="softdtw", max_iter=self.cluster_max_iter).fit(
                    pattern_cand_keys)
            self.pattern_keys = km.cluster_centers_
            np.save(self.pattern_key_file, self.pattern_keys)
            print("Saved at file " + self.pattern_key_file + ".npy")
        else:
            self.pattern_keys = np.load(self.pattern_key_file + ".npy")  # (16,3,1),16-->簇数，3-->Attention的类别数，1-->特征数
            print("Loaded file " + self.pattern_key_file + ".npy")

        return self.pattern_keys

    # def _load_rel(self):
    #     self.sd_mx = None
    #     super()._load_rel()
    #     self._logger.info('Max adj_mx value = {}'.format(self.adj_mx.max()))
    #     self.sh_mx = self.adj_mx.copy()
    #     if self.type_short_path == 'hop':
    #         self.sh_mx[self.sh_mx > 0] = 1
    #         self.sh_mx[self.sh_mx == 0] = 511
    #         for i in range(self.num_nodes):
    #             self.sh_mx[i, i] = 0
    #         for k in range(self.num_nodes):
    #             for i in range(self.num_nodes):
    #                 for j in range(self.num_nodes):
    #                     self.sh_mx[i, j] = min(self.sh_mx[i, j], self.sh_mx[i, k] + self.sh_mx[k, j], 511)
    #         np.save('{}.npy'.format(self.dataset), self.sh_mx)

    # def _calculate_adjacency_matrix(self):
    #     self._logger.info("Start Calculate the weight by Gauss kernel!")
    #     self.sd_mx = self.adj_mx.copy()
    #     distances = self.adj_mx[~np.isinf(self.adj_mx)].flatten()
    #     std = distances.std()
    #     self.adj_mx = np.exp(-np.square(self.adj_mx / std))
    #     self.adj_mx[self.adj_mx < self.weight_adj_epsilon] = 0
    #     if self.type_short_path == 'dist':
    #         self.sd_mx[self.adj_mx == 0] = np.inf
    #         for k in range(self.num_nodes):
    #             for i in range(self.num_nodes):
    #                 for j in range(self.num_nodes):
    #                     self.sd_mx[i, j] = min(self.sd_mx[i, j], self.sd_mx[i, k] + self.sd_mx[k, j])

    # def get_data(self):
    #     x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
    #     if self.data is None:
    #         self.data = {}
    #         if self.cache_dataset and os.path.exists(self.cache_file_name):
    #             x_train, y_train, x_val, y_val, x_test, y_test = self._load_cache_train_val_test()
    #         else:
    #             x_train, y_train, x_val, y_val, x_test, y_test = self._generate_train_val_test()
    #     self.feature_dim = x_train.shape[-1]
    #     self.ext_dim = self.feature_dim - self.output_dim
    #     self.scaler = self._get_scalar(self.scaler_type,
    #                                    x_train[..., :self.output_dim], y_train[..., :self.output_dim]) # 所有的数据共用训练集的均值和方差
    #     self.ext_scaler = self._get_scalar(self.ext_scaler_type,
    #                                        x_train[..., self.output_dim:], y_train[..., self.output_dim:])
    #     x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim]) # FIXME 这里是所有的站点都服从同一个分布，合理吗？
    #     y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
    #     x_val[..., :self.output_dim] = self.scaler.transform(x_val[..., :self.output_dim])
    #     y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
    #     x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
    #     y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])
    #     if self.normal_external:
    #         x_train[..., self.output_dim:] = self.ext_scaler.transform(x_train[..., self.output_dim:])
    #         y_train[..., self.output_dim:] = self.ext_scaler.transform(y_train[..., self.output_dim:])
    #         x_val[..., self.output_dim:] = self.ext_scaler.transform(x_val[..., self.output_dim:])
    #         y_val[..., self.output_dim:] = self.ext_scaler.transform(y_val[..., self.output_dim:])
    #         x_test[..., self.output_dim:] = self.ext_scaler.transform(x_test[..., self.output_dim:])
    #         y_test[..., self.output_dim:] = self.ext_scaler.transform(y_test[..., self.output_dim:])
    #     train_data = list(zip(x_train, y_train))
    #     eval_data = list(zip(x_val, y_val))
    #     test_data = list(zip(x_test, y_test))
    #     self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
    #         generate_dataloader(train_data, eval_data, test_data, self.feature_name,
    #                             self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample,
    #                             distributed=self.distributed)
    #     self.num_batches = len(self.train_dataloader)
    #     self.pattern_key_file = os.path.join( #ODO 这个数据应该是聚类后每一个簇中心对应的数据 T
    #         './libcity/cache/dataset_cache/', 'pattern_keys_{}_{}_{}_{}_{}_{}'.format(
    #             self.cluster_method, self.dataset, self.cand_key_days, self.s_attn_size, self.n_cluster, self.cluster_max_iter))
    #     if not os.path.exists(self.pattern_key_file + '.npy'):
    #         cand_key_time_steps = self.cand_key_days * self.points_per_day
    #         pattern_cand_keys = x_train[:cand_key_time_steps, :self.s_attn_size, :, :self.output_dim].swapaxes(1, 2).reshape(-1, self.s_attn_size, self.output_dim)
    #         self._logger.info("Clustering...")
    #         if self.cluster_method == "kshape":
    #             km = KShape(n_clusters=self.n_cluster, max_iter=self.cluster_max_iter).fit(pattern_cand_keys)
    #         else:
    #             km = TimeSeriesKMeans(n_clusters=self.n_cluster, metric="softdtw", max_iter=self.cluster_max_iter).fit(pattern_cand_keys)
    #         self.pattern_keys = km.cluster_centers_
    #         np.save(self.pattern_key_file, self.pattern_keys)
    #         self._logger.info("Saved at file " + self.pattern_key_file + ".npy")
    #     else:
    #         self.pattern_keys = np.load(self.pattern_key_file + ".npy") #(16,3,1),16-->簇数，3-->Attention的类别数，1-->特征数？
    #         self._logger.info("Loaded file " + self.pattern_key_file + ".npy")
    #     return self.train_dataloader, self.eval_dataloader, self.test_dataloader






