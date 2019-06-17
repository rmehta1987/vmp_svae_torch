import torch
import numpy as np
from models import vae, nrm
import re

def e_step(phi_enc,phi_nrm, nb_samples, seed=0):
    """

    Args:
        phi_enc: encoded data; Base Measure Natural Parameters [In our case, Dirichlet, so alpha]
        phi_nrm: parameters of recognition NRM-MM (Base measure parameters and cluster assignments)
        nb_samples: number of ties to sample from q(x|z,y)
        seed: random seed

    Returns:

    """

    #Assuming Dirichlet Base Measure
    alpha_phi1 = phi_enc

    #unpack cluster assignment and parameter of cluster
    alpha_phi2, z_phi2 = unpack_recognition_nrm(phi_nrm)


    # compute log q(z|y, phi)
    log_z_given_y_phi, dbg = compute_log_z_given_y(alpha_phi1, alpha_phi2, eta2_phi2, pi_phi2)