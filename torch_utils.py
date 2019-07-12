from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np


def logdet(A):

    """
    There is no batched log-determinant in pytorch, but there is a batched cholesky
    We can use cholesky to calculate the determinant as A = LL^t then
    det(A) = det(L)*det(L^T) = det(L)^2 where L is a lower triangular matrix
    Also remember that: determinant of triangular matrices is the product of the diagonal of the matrix, det(L) = prod(L_ii)

    So in full logdet(L) = 2*sum(log(diag(cholesky(A)))))

    Args: 
        A: symmetric positive definite matrix of shape ..., D x D
    
    Returns:
        log(det(A))
    """
    theval = 2. * torch.sum(torch.log(torch.diagonal(torch.cholesky(A), dim1=-2, dim2=-1)), dim=-1)
    return theval


def init_tensor_gpu_grad(org_tensor, trainable=True, device='cuda'):

    if trainable:
        new_tensor = torch.nn.Parameter(org_tensor.to(device))
    else:
        new_tensor = org_tensor.to(device)
        new_tensor.requires_grad=False
    return new_tensor


def exponential_learning_rate(learning_rate, decay_rate, global_step, decay_steps):

    # https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay
    # Fixed warning: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() 
    # or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
    # https://discuss.pytorch.org/t/clone-and-detach-in-v0-4-0/16861/2
    # remember to convert numpy value to numpy array
    #print ("Updating Learning Rate")
    exp_learn = learning_rate*np.power(decay_rate, global_step/decay_steps)
    
    return exp_learn


def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):

    rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)

    features = np.random.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:,0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:,0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    feats = 10 * np.einsum('ti,tij->tj', features, rotations)

    data = np.random.permutation(np.hstack([feats, labels[:, None]]))

    return torch.tensor(data[:, 0:2]).float(), torch.tensor(data[:, 2], dtype=torch.int64)

def rand_partial_isometry(input_dim, output_dim, stddev=1., seed=0):
    """
    Initialization as in MJJ's code (Johnson et. al. 2016)
    Args:
        m: rows
        n: cols
        stddev: standard deviation
        seed: random seed

    Returns:
        matrix of shape (m, n) with orthonormal columns
    """
    d = max(input_dim, output_dim)
    npr = np.random.RandomState(seed)
    return np.linalg.qr(npr.normal(loc=0, scale=stddev, size=(d, d)))[0][:input_dim,:output_dim]