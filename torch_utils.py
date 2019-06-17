from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

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

    return 2. * torch.sum(torch.log(torch.diagonal(torch.cholesky(A), dim1=-2, dim2=-1)), axis=-1)



def init_tensor_gpu_grad(org_tensor, trainable=True, device='cuda'):

    new_tensor = org_tensor.to(device)
    new_tensor.requires_grad = trainable

    return new_tensor
