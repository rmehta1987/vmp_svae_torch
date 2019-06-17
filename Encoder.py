import torch
import numpy as np
from torch import nn

class Encoder(nn.Module):
    def __init__(self, layerspecs):
        super(Encoder, self).__init__()

        self.layers = layerspecs # The encoder layers

    def encode(self, x):
        ''' 
        Creates a sequential based on the layers in self.layers
        Has to specialized layers one that returns standard Gaussian parameteres and
        the natural parameters
        '''
        modules = []
        for i, (hidden_units, actlayer) in enumerate(self.layers):
            
            if actlayer is 'standard':
                the_act_layer = Standard_Activation()
            elif actlayer is 'natparam':
                the_act_layer = Natural_Parameter_Activation()
            else:
                the_act_layer = actlayer
            if i == 1:
                modules.append(nn.Linear(self.input_dim,hidden_units))
                modules.append(the_act_layer)
                prev_units = hidden_units
            elif i == len(self.layers)-1:
                modules.append(nn.Linear(prev_units,hidden_units*2))
                modules.append(the_act_layer)
            else:
                modules.append(nn.Linear(prev_units,hidden_units))
                modules.append(the_act_layer)


        self.net = nn.Sequential(*modules)
        return self.net(x)


    def forward(self, x):
        
        self.input_dim = x.shape[-1]
        self.output_dim, self.the_type = self.layers[-1]

        # Ravel Inputs so shape is: (M, K, D) -> (M*K, D)
        x = torch.reshape(x, (-1, self.input_dim))
        mu, var = self.encode(x)

        # Create a res-net like shortcut
        # Why do we need do this (obviously for initalization, but why not just use regular) ?!?!
        orthonormal_cols = rand_partial_isometry(self.input_dim, self.output_dim, 1., seed=seed)
        W = torch.from_numpy(orthonormal_cols).float().cuda()
        W.requires_grad = True

        b1 = torch.zeros_like(output_dim).cuda()
        b1.requires_grad = True

        out_res = torch.add(torch.matmul(x, W), b1)

        # need to create shortcut for second output since Gaussian
        b2 = torch.zeros_like(output_dim).cuda()
        b2.requires_grad = True

        if self.the_type == 'standard':
            a = torch.tensor(1., dtype=torch.float32)
        elif self.the_type == 'natparam':
            a = torch.tensor(-0.5, dtype=torch.float32)
        else:
            raise NotImplementedError
        
        out_res = (out_res, a*torch.log1p(torch.exp(b2)))

        # unravel output: (M*K, D) -> (M, K, D)
        output_shape = list(x.shape[:-1])
        output_shape.append(self.output_dim)

        outputs = (torch.reshape(torch.add(mu,out_res[0]),output_shape), torch.reshape(torch.add(var,out_res[1]),output_shape))
        
        return outputs
    
    def rand_partial_isometry_(input_dim, output_dim, seed=0):
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
        d = max(m, n)
        npr = np.random.RandomState(seed)
        return np.linalg.qr(npr.normal(loc=0, scale=stddev, size=(d, d)))[0][:m,:n]

class Standard_Activation(nn.Module):
    def __init__(self):
        super(Standard_Activation, self).__init__()
    
    def forward(self, input):
        raw_1, raw_2 = torch.chunk(input, 2, axis=-1)
        mean = raw_1
        var = torch.nn.Softplus(raw_2)
        return mean, var
    
class Natural_Parameter_Activation(nn.Module):
    def __init__(self):
        super(Natural_Parameter_Activation, self).__init__()
    
    def forward(self, input):
        raw_1, raw_2 = torch.chunk(input, 2, axis=-1)
        eta1 = raw_1
        eta2 = -1./2 * torch.nn.softplus(raw_2)
        return eta1, eta2


