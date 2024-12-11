import os
import math
import numpy as np
import torch
from tqdm import tqdm
import scipy.sparse as sp
from fastdtw import fastdtw
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import eigsh
def laplacian(W):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    if isinstance(W,torch.Tensor):
        W=np.array(W.detach().cpu())
    d = W.sum(axis=0)
    # Laplacian matrix.
    d = 1 / np.sqrt(d)
    D = sp.diags(d, 0)
    I = sp.identity(d.size, dtype=W.dtype)
    L = I - D * W * D
    return L

def largest_k_lamb(L, k):
    try:
        lamb, U = eigsh(L,k=k,which='LM')
        return (lamb, U)
    except:
        eigenvalues, eigenvectors = np.linalg.eig(L)
        sorted_indices = np.argsort(eigenvalues)  # 返回特征值排序后的索引
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        return (eigenvalues,eigenvectors)

def get_eigv(adj,k):
    L = laplacian(adj)
    eig = largest_k_lamb(L,k)
    return eig

# 根据时间序列的相似性(度量方式DTW距离)来构建adj矩阵
def construct_tem_adj(data, num_node):
    data_mean = np.mean([data[24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0) # 这个是计算一天的均值
    data_mean = data_mean.squeeze().T
    dtw_distance = np.zeros((num_node, num_node))
    for i in tqdm(range(num_node)):
        for j in range(i, num_node):
            dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0] # 计算i节点和j节点的dtw距离
    for i in range(num_node):
        for j in range(i):
            dtw_distance[i][j] = dtw_distance[j][i] # 这个距离是对称的

    nth = np.sort(dtw_distance.reshape(-1))[
        int(np.log2(dtw_distance.shape[0])*dtw_distance.shape[0]):
        int(np.log2(dtw_distance.shape[0])*dtw_distance.shape[0])+1] # NlogN edges 得到阈值
    tem_matrix = np.zeros_like(dtw_distance)
    tem_matrix[dtw_distance <= nth] = 1 # 小于这个阈值的为置为1，大于的为0
    tem_matrix = np.logical_or(tem_matrix, tem_matrix.T) # 逻辑与变为对称矩阵
    return tem_matrix

def loadGraph(adj, temporal_graph, dims, data):
    # calculate spatial and temporal graph wavelets
    adj = adj + np.eye(adj.shape[0])
    if os.path.exists(temporal_graph+".npy"):
        tem_adj = np.load(temporal_graph+".npy")
    else:
        tem_adj = construct_tem_adj(data, adj.shape[0])
        np.save(temporal_graph, tem_adj) # 默认是npy文件，因此不用加.npy的后缀名
    spawave = get_eigv(adj, dims) # 这个是空间的图的adj
    temwave = get_eigv(tem_adj, dims) # 这个是根据时间序列的dtw距离构建的adj

    # derive neighbors
    sampled_nodes_number = int(math.log(adj.shape[0], 2))
    graph = csr_matrix(adj) # 将图稀疏化表示
    dist_matrix = dijkstra(csgraph=graph) # 计算图的最短路径，返回一个二维矩阵
    dist_matrix[dist_matrix==0] = dist_matrix.max() + 10 # 最短路径为0表示可能表示不连通，因此附上一个大值
    localadj = np.argpartition(dist_matrix, sampled_nodes_number, -1)[:, :sampled_nodes_number] # 得到每一行的距离最小的前sampled_nodes_number个点

    return localadj, spawave, temwave
