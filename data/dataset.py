import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

def split_dataset(dataset: np.ndarray, split_rate=0.8):
    '''
    groups=1
    :param dataset: x: (L, N, C)
    :param split_rate:切分数据集的比例
    :return: train: (L, N, C), test: (L, N, C)
    '''
    total_seq_len, num_nodes, _ = dataset.shape
    train_size = int(total_seq_len * split_rate)
    train_dataset, test_dataset = dataset[ 0:train_size, ...],dataset[train_size:, ...]

    return train_dataset, test_dataset


class SubwayDataset(Dataset):
    def __init__(self, dataset,time_dataset, seq_len,pred_len, feature_range=(-1, 1),**kwargs):
        '''
        :param dataset: x:(total_L, N, C)
        :param seq_len: length of split sequence
        :param pred_len:length of pred sequence
        :param feature_range: range of min_max scalar
        '''
        self.feature_range = feature_range
        self.pred_len = pred_len
        self.seq_len=seq_len
        self.mean=kwargs.get("mean")
        self.std=kwargs.get("std")
        self.max_values=kwargs.get("max")
        self.min_values = kwargs.get("min")

        assert len(self.feature_range) == 2 and self.feature_range[1] > self.feature_range[0]

        self.total_seq_len = len(dataset)
        self.dataset=dataset
        _,self.num_features,self.num_nodes=dataset.shape
        self.time_dataset=time_dataset


    def __getitem__(self, item):
        '''
        :param item: index
        :return: x: (C,L,N) and label：(C,N,L)
        '''
        x_end=item+self.seq_len
        y_end=x_end+self.pred_len
        x=self.dataset[item:x_end]
        y=self.dataset[x_end:y_end]
        x_time=self.time_dataset[item:x_end]
        y_time = self.time_dataset[x_end:y_end]
        x,y=torch.FloatTensor(x),torch.FloatTensor(y)
        x_time, y_time = torch.FloatTensor(x_time), torch.FloatTensor(y_time)

        return x,x_time,y,y_time

    def __len__(self):
        return len(self.dataset) - self.seq_len - self.pred_len

    # def transform(self, x: np.ndarray):
    #     '''
    #     :param x: ( L, N, C)
    #     :return: (L, N, C)
    #     groups=1
    #     '''
    #     dim_reduce = True if len(x.shape) == 4 else False
    #     if dim_reduce:
    #         groups, src_seq_len, num_nodes, num_features = x.shape
    #         x = x.reshape((-1, num_nodes, num_features)) # 把group和Len合并
    #     else:
    #         groups = src_seq_len = 1
    #
    #     total_len, num_nodes, num_features = x.shape
    #     # 不同的站点和不同的特征间的最大值和最小值是不一样的
    #     self.min_arr = np.zeros((1, num_nodes, num_features))
    #     self.max_arr = np.zeros((1, num_nodes, num_features))
    #
    #     # 循环各个特征，计算对应的最大最小值
    #     for i in range(num_features):
    #         self.min_arr[:, :, i::num_features] = np.min(x[:, :, i::num_features], axis=0)
    #         self.max_arr[:, :, i::num_features] = np.max(x[:, :, i::num_features], axis=0)
    #
    #     norm_x = (x - self.min_arr) / (self.max_arr - self.min_arr)
    #     min_max_x = norm_x * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
    #
    #     # if dim_reduce:
    #     #     min_max_x = min_max_x.reshape((groups, src_seq_len, num_nodes, num_features))
    #     min_max_x = min_max_x.reshape((src_seq_len, num_nodes, num_features))
    #     return min_max_x
    #
    # '''逆变换'''
    # def inv_transform(self, x: np.ndarray):
    #     '''
    #     :param x: (L, N, C)
    #     :return:
    #     '''
    #     x = (x - self.feature_range[0]) * (self.max_arr - self.min_arr) / (
    #             self.feature_range[1] - self.feature_range[0]) + self.min_arr
    #     return x

    # '''将预处理后的结果进行保存'''
    # def save_min_max_arr(self, output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    #     min_arr_out_path = os.path.join(output_dir, 'min.npy')
    #     max_arr_out_path = os.path.join(output_dir, 'max.npy')
    #     np.save(min_arr_out_path, self.min_arr)
    #     np.save(max_arr_out_path, self.max_arr)






