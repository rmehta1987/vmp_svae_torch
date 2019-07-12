from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import matplotlib.pyplot as plt
import numpy as np

import dirichlet, niw
import torch
import torch_utils


"""
Variational Mixture of Gaussians, according to:
  Pattern Matching and Machine Learning (Chapter 10.2)
  Christopher M. Bishop.
  Springer, 2006.
"""


def update_Nk(r_nk):
    # Bishop eq 10.51
    return torch.sum(r_nk, dim=0)


def update_xk(x, r_nk, N_k):
    # Bishop eq 10.52; output shape = (K, D)
    
    x_k = torch.einsum('nk,nd->kd', r_nk, x)
    x_k_normed = x_k / N_k.unsqueeze(1)
    # remove nan values (if N_k == 0)
    return torch.where(torch.isnan(x_k_normed), x_k, x_k_normed)


def update_Sk(x, r_nk, N_k, x_k):
    # Bishop eq 10.53
    
    x_xk = x.unsqueeze(1) - x_k.unsqueeze(0)
    S = torch.einsum('nk,nkde->kde', r_nk, torch.einsum('nkd,nke->nkde', x_xk, x_xk))
    S_normed = S / N_k.unsqueeze(1).unsqueeze(2)
    # remove nan values (if N_k == 0)
    return torch.where(torch.isnan(S_normed), S, S_normed)


def update_alphak(alpha_0, N_k):
    # Bishop eq 10.58
    return torch.add(alpha_0, N_k)


def update_betak(beta_0, N_k):
    # Bishop eq 10.60
    return torch.add(beta_0, N_k,)

##### Start here again
def update_mk(beta_0, m_0, N_k, x_k, beta_k):
    # Bishop eq 10.61
   
    if len(beta_0.shape) == 1:
        beta_0 = torch.reshape(beta_0, (-1, 1))

    Nk_xk = N_k.unsqueeze(1) * x_k
    beta0_m0 = beta_0 * m_0
    return (beta0_m0 + Nk_xk) / beta_k.unsqueeze(1)


def update_Ck(C_0, x_k, N_k, m_0, beta_0, beta_k, S_k):
    # Bishop eq 10.62
    
        C = C_0 + N_k.unsqueeze(1).unsqueeze(2)* S_k
        Q0 = x_k - m_0
        q = torch.einsum('kd,ke->kde', Q0, Q0)
        return C + torch.einsum('k,kde->kde', (beta_0 * N_k) / beta_k, q)


def update_vk(v_0, N_k):
    # Bishop eq 10.63
    return (v_0 + N_k + 1).clone()


def compute_expct_mahalanobis_dist(x, beta_k, m_k, P_k, v_k):
    # Bishop eq 10.64
    # output shape: (N, K)

    _, D = x.shape

    dist = x.unsqueeze(1) - m_k.unsqueeze(0)  # shape=(N, K, D)
    m = torch.einsum('k,nk->nk', v_k,
                    torch.einsum('nkd,nkd->nk', dist,
                            torch.einsum('kde,nke->nkd', P_k, dist)))
    return torch.add(m, torch.reshape(D/beta_k, (1, -1)))   # shape=(1, K)

def compute_expct_log_det_prec(v_k, P_k):
    # Bishop eq 10.65
    
        log_det_P = torch_utils.logdet(P_k)

        K, D, _ = P_k.shape
        D_log_2 = float(D) * torch.log(2.)

        i = torch.arange(D, dtype=torch.float32).unsqueeze(0)
        sum_digamma = torch.sum(torch.digamma(0.5 * (v_k.unsqueeze(1) + 1. + i)), dim=1)

        return (sum_digamma + D_log_2 + log_det_P).clone()


def compute_log_pi(alpha_k):
    # Bishop eq 10.66
    
    alpha_hat = torch.sum(alpha_k)
    return torch.subtract(torch.digamma(alpha_k), torch.digamma(alpha_hat))


def compute_rnk(expct_log_pi, expct_log_det_cov, expct_dev):

    log_rho_nk = expct_log_pi + 0.5 * expct_log_det_cov - 0.5 * expct_dev

    # for numerical stability: subtract largest log p(z=k) for each k
    rho_nk_save = torch.exp(log_rho_nk - torch.reshape(torch.max(log_rho_nk, dim=1), (-1, 1)))

    # normalize
    rho_n_sum = torch.sum(rho_nk_save, dim=1)  # shape = (N,)
    return rho_nk_save / rho_n_sum.unsqueeze(1)


def e_step(x, alpha_k, beta_k, m_k, P_k, v_k, name='e_step'):
    """
    Variational E-update: update local parameters
    Args:
        x: data
        alpha_k: Dirichlet parameter
        beta_k: NW param, variance of mean
        m_k: NW param, mean
        P_k: NW param, precision
        v_k: NW param, degrees of freedom

    Returns:
        responsibilities and mixture coefficients
    """
    expct_dev = compute_expct_mahalanobis_dist(x, beta_k, m_k, P_k, v_k)  # Bishop eq 10.64
    expct_log_det_cov = compute_expct_log_det_prec(v_k, P_k)              # Bishop eq 10.65
    expct_log_pi = compute_log_pi(alpha_k)                                # Bishop eq 10.66
    r_nk = compute_rnk(expct_log_pi, expct_log_det_cov, expct_dev)        # Bishop eq 10.49

    return r_nk, torch.exp(expct_log_pi)

'''
def e_step_missing_data(x, alpha_k, beta_k, m_k, P_k, v_k, missing_data_mask, name='e_step_imp'):
    """
    Variational E-update: update local parameters ignoring missing data.
    Args:
        x: data
        alpha_k: Dirichlet parameter
        beta_k: NW param; variance of mean
        m_k: NW param; mean
        P_k: NW param: precision
        v_k: NW param: degrees of freedom
        missing_data_mask: binary matrix of shape (N, D) indicating missing values

    Returns:
        responsibilities and mixture coefficients
    """
    with tf.name_scope(name):
        expct_dev = compute_dev_missing_data(x, beta_k, m_k, P_k, v_k, missing_data_mask)
        expct_log_det_cov = compute_expct_log_det_prec(v_k, P_k)
        expct_log_pi = compute_log_pi(alpha_k)
        r_nk = compute_rnk(expct_log_pi, expct_log_det_cov, expct_dev)

        return r_nk, tf.exp(expct_log_pi)
'''

def m_step(x, r_nk, alpha_0, beta_0, m_0, C_0, v_0):
    """
    Variational M-update: Update global parameters
    Args:
        x: data
        r_nk: responsibilities
        alpha_0: prior Dirichlet parameters
        beta_0: prior NiW; controls variance of mean
        m_0: prior of mean
        C_0: prior Covariance
        v_0: prior degrees of freedom

    Returns:
        posterior parameters as well as data statistics
    """

    N_k = update_Nk(r_nk)                                     # Bishop eq 10.51
    x_k = update_xk(x, r_nk, N_k)                             # Bishop eq 10.52
    S_k = update_Sk(x, r_nk, N_k, x_k)                        # Bishop eq 10.53

    alpha_k = update_alphak(alpha_0.cuda(), N_k)                     # Bishop eq 10.58
    beta_k = update_betak(beta_0.cuda(), N_k)                        # Bishop eq 10.60
    m_k = update_mk(beta_0.cuda(), m_0.cuda(), N_k, x_k, beta_k)            # Bishop eq 10.61
    C_k = update_Ck(C_0.cuda(), x_k, N_k, m_0.cuda(), beta_0.cuda(), beta_k, S_k)  # Bishop eq 10.62
    v_k = update_vk(v_0.cuda(), N_k)                                 # Bishop eq 10.63

    return alpha_k, beta_k, m_k, C_k, v_k, x_k, S_k
