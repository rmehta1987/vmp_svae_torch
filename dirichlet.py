from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def expected_log_pi(dir_standard_param):
    torch.subtract(torch.digamma(dir_standard_param),
                           torch.digamma(torch.sum(dir_standard_param, dim=-1, keepdim=True)))


def standard_to_natural(alpha):
    return alpha - 1


def natural_to_standard(alpha_nat):
    return alpha_nat + 1