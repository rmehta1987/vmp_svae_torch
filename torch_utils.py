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
    return new_tensor


def exponential_learning_rate(learning_rate, decay_rate, global_step, decay_steps):

    # https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay

    return torch.tensor(learning_rate*np.power(decay_rate,global_step/decay_steps)).cuda()
