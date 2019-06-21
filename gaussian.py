import numpy as np
import torch
from math import pi

from torch_utils import logdet


def natural_to_standard(eta1, eta2):
    """ Convert natural parameters to standard normal parameters mu and sigma
        Args: 
            eta1: natural parameter one of normal distribution mu*precision
            eta2: natural parameter two of normal distribution -0.5*precision

        Returns:
            mu: mean parameter of normal distribution [Shape: Batch size x Latent Dims]
            sigma: variance of normal distirbution [Shape: Batch size x Latent Dims x Latent Dims ]
    """  

    sigma = torch.inverse(-2 * eta2)  # eta2 = 1/[2*sigma] => 2*eta2 = 1/[sigma] => inverse(1/sigma) = sigma
    mu = sigma@eta1.unsqueeze(2)  # eta1 = mu * 1/sigma => sigma * mu(1/sigma) = mu
    mu = torch.reshape(mu, eta1.shape)
    return mu, sigma 

def standard_to_natural(mu, sigma):
    """
    Args:
        mu: 10x6
        sigma: 10x6x6
    """
    # if len(mu.get_shape()) != 2 or len(sigma.get_shape()) != 3:
    #     raise NotImplementedError("standard_to_natural is not implemented for this case.")
    eta_2 = -0.5 * sigma.inverse()  # shape = (nb_components, latent_dim, latent_dim)
    eta_1 = -2 * (eta_2@mu.unsqueeze(-1))  # shape = (nb_components, latent_dim)
    eta_1 = torch.reshape(eta_1, mu.shape)

    return (eta_1, eta_2)

def log_probability_nat(x, eta1, eta2, weights=None):

    """
    Computes log N(x|eta1, eta2)
    Args:
        x: means
        eta1: natural parameter one of normal distribution mu*precision
        eta2: natural parameter two of normal distribution -0.5*precision
        weights: mixture proportions

    Returns:
        Normalized log-probability
    """

    N, D = x.shape

    # distinguish between 1 and k>1 components...
    if len(eta1.shape) != 3:
        shape = eta1.shape
        raise AssertionError("eta1 must be of shape (N,K,D). Its shape is %s." % str(shape))

    # Calculate Gaussian Log Probability of natural parameters, or log of exponential family
    # see https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf pg 4 (8.20)

    # General formula for Gaussians:  1/sqrt(2*pi)^D  * exp(eta1 * x - 0.5*eta2*(x**2) - 0.5*eta2*(mu**2) - log det(sigma))
    # sigma = -2 * eta2
    
    logprob = torch.einsum('nd,nkd->nk', x, eta1) # eta1*x
    logprob += torch.einsum('nkd,nd->nk', torch.einsum('nd,nkde->nke', x, eta2), x) # x * (eta2*x)
    logprob -= D/2. * torch.log(torch.tensor(2.*pi).to('cuda'))  # log(1/sqrt(2*pi))

    # add dimension for further computations
    eta1 = eta1.unsqueeze(3) # Dims should now be: N x K x D x 1
    # inv(eta2)*eta1 => mu => mu * eta1 = eta2*(mu**2) and 1/4 comes from formula above, we are including it now
    logprob += 1./4 * torch.einsum('nkdi,nkdi->nk', eta2.inverse()@eta1, eta1)

    logprob += 0.5*logdet(-2.*eta2 + 1e-20 + torch.eye(D).to('cuda'))

    if weights is not None:
        logprob += torch.log(weights).unsqueeze(0)
    
    # log sum exp trick
    max_logprob = torch.max(logprob, dim=1, keepdim=True)[0]
    normalizer = max_logprob + torch.log(torch.sum(torch.exp(logprob-max_logprob), dim=1, keepdim=True))

    return logprob - normalizer


def log_probability_nat_per_samp(x_samps, eta1, eta2):
    """
    Args:
        x_samps: matrix of shape (minibatch_size, nb_components, nb_samps, latent_dims)
        eta1: 1st natural parameter for Gaussian distr; shape: (size_minibatch, nb_components, latent_dim)
        eta2: 2nd natural parameter for Gaussian distr; shape: (size_minibatch, nb_components, latent_dim, latent_dim)

    Returns:
        1/S sum^S_{s=1} log N(x^(s)|eta1, eta2) of shape (N, K, S)
    """
    # same as above, but x consists of S samples for K components: x.shape = (N, K, S, D)
    # todo: merge with above function (above is the same but normalised)

    N, K, S, D = x_samps.shape
    assert list(eta1.shape) == [N, K, D]
    assert list(eta2.shape) == [N, K, D, D]

    # -1\2 (sigma^(-1) * x) * x + sigma^(-1)*mu*x
    log_normal = torch.einsum('nksd,nksd->nks', torch.einsum('nkij,nksj->nksi', eta2, x_samps), x_samps)
    log_normal += torch.einsum('nki,nksi->nks', eta1, x_samps)

    # 1/4 (-2 * sigma * (sigma^(-1) * mu)) sigma^(-1) * mu = -1/2 mu sigma^(-1) mu; shape = N, K, 1
    log_normal += torch.tensor(1.0 / 4).cuda() * torch.einsum('nkdi,nkd->nki', eta2.inverse()@eta1.unsqueeze(-1), eta1)
    log_pi = torch.tensor(np.log(2 * np.pi)).cuda()
    log_normal -= torch.tensor(D/2.).cuda() * log_pi

    # + 1/2 log |sigma^(-1)|
    log_sigma = logdet(-2.0 * eta2 + 1e-20 * torch.eye(D).cuda())
    log_normal += torch.tensor(1.0 / 2).cuda() * log_sigma.unsqueeze(2)

    return log_normal


