import numpy as np
import torch
from math import pi


def natural_to_standard(m_o, phi_o, beta, v_hat):
    """
    Converts natural parameters of Normal Inverse Wishart Distirbution to Standard 4 paramter, mean, kappa/beta, covariance, degrees of freedom
    Args:
        m_o: first natural parameter of NIW [beta, v+d+2]  (v = degrees of freedom, d = dim of matrix) --- ???? Double check
        phi_o: second natural parameter of NIW [beta*mu, variance+beta*mu*mu.T]
        beta: parameter of NIW, is also called kappa
        v_hat: degree of freedom parameter of NIW
    
    Returns:
        beta: parameter of NIW, is also called kappa
        m: mean of NIW
        C: variance of NIW
        v_hat = degree of freedom of NIW
    """
    print("\nCalculating Natural to standard")

    # m = torch.divide(phi_o, beta.unsqueeze(-1)) no torch.divide
    m = phi_o / beta.unsqueeze(-1)

    K, D = m.shape
    assert list(beta.shape) == [K,]
    D = int(D)

    C = m_o - _outer(phi_o, m)
    v = v_hat - D - 2
    return beta, m, C, v

def standard_to_natural(beta, m, C, v):
    """
    Args:
        beta: parameter of NIW, is also called kappa
        m: mu of NIW
        C: covariance of NIW
        v: degree of freedom parameter of NIW
    
    Returns:
        m_o: First natural parameter of NIW
        phi_o: Second natural parameter of NIW
        beta: parameter of NIW, also called kappa
        v_hat: degree of freedom parameter of NIW
    """

    print("\nCalculating Standard to Natural")

    K, D = m.shape
    assert list(beta.shape) == [K,]
    D = int(D)

    b = beta.unsqueeze(-1) * m # shape K x D [nb_components x latent_dims]
    A = C + _outer(b, m) # K x D x D 
    v_hat = v + D + 2

    return A, b, beta, v_hat

def _outer(a,b):
    a_ = a.unsqueeze(-1)
    b_ = b.unsqueeze(-2)
    return a_*b_ # element-wise multiplication

def expected_values(niw_standard_params):
    '''
    Args:
        niw_standard_params is a list containing the following parameters:
            beta: parameter of NIW, is also called kappa
            mu: mean of NIW
            C: variance of NIW
            v_hat = degree of freedom of NIW
    '''

    print("\nCalculating Expected Values NIW")
    beta, m, C, v = niw_standard_params
    exp_m = m.clone()
    C_inv = C.inverse()
    C_inv_sym = C_inv + torch.transpose(C_inv,dim0=2,dim1=1) / 2.  # C_inv is of shape NB_components x Latent Dims x Latent Dims
    expected_precision = C_inv_sym*v.unsqueeze(1).unsqueeze(2)
    expected_covariance = expected_precision.inverse()

    return exp_m, expected_covariance