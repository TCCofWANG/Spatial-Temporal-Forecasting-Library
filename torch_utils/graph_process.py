import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

'''计算过一化的拉普拉斯矩阵:L_{sym}这一个一般用于空域GCN：特征值范围是-1~1'''
def calculate_laplacian_with_self_loop(matrix):
    if not isinstance(matrix,torch.Tensor):
        matrix=torch.tensor(matrix).cuda()
    matrix = matrix.cuda() + torch.eye(matrix.size(0)).cuda()
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian



'''计算归一化的拉普拉斯矩阵：这个一般用于切比雪夫多项式内的:特征值范围也是-1~1'''
def graph_laplace_trans(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]
    if not isinstance(W,np.ndarray):
        W=np.array(W)
    D = np.diag(np.sum(W, axis=1))

    L = D - W # 计算得到拉普拉斯矩阵

    lambda_max = eigs(L, k=1, which='LR')[0].real
    output=(2 * L) / lambda_max - np.identity(W.shape[0])
    output=torch.tensor(output).cuda()
    return output

'''这个一般用于普通谱域实现的GCN中'''
def transition_matrix(adj):
    r"""
    Description:
    -----------
    Calculate the transition matrix `P` proposed in DCRNN and Graph WaveNet.
    P = D^{-1}A = A/rowsum(A)

    Parameters:
    -----------
    adj: np.ndarray
        Adjacent matrix A

    Returns:
    -----------
    P:np.matrix
        Renormalized message passing adj in `GCN`.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    # P = d_mat.dot(adj)
    P = d_mat.dot(adj).astype(np.float32).todense()
    P=torch.tensor(P).cuda()
    return P

'''以下是将邻接矩阵转为tensor格式'''
def graph_to_tensor(adj):
    if isinstance(adj, np.ndarray):
        adj = torch.tensor(adj).float()
    return adj



