import torch
import numpy as np
from torch import nn
from torch_utils import init_tensor_gpu_grad
from torch.nn import functional as F
from torch_utils import rand_partial_isometry

class Encoder(nn.Module):
    def __init__(self, layerspecs, input_dim=784):
        super(Encoder, self).__init__()
        self.layers = layerspecs  # The encoder layers
        self.input_dim = input_dim
        self.output_dim, self.the_type = self.layers[-1]
        self.net = self.initalize(self.layers)
        self.res_model = ResNetShort(self.input_dim, self.output_dim, the_type=self.the_type)

    def initalize(self, layers):
        modules = []
        for i, (hidden_units, actlayer) in enumerate(self.layers):
            
            if actlayer is 'standard':
                the_act_layer = Standard_Activation()
            elif actlayer is 'natparam':
                the_act_layer = Natural_Parameter_Activation()
            else:
                the_act_layer = actlayer
            if i == 0:
                modules.append(nn.Linear(self.input_dim,hidden_units))
                modules.append(the_act_layer)
                prev_units = hidden_units
            elif i == len(self.layers)-1:
                modules.append(nn.Linear(prev_units,hidden_units*2))
                modules.append(the_act_layer)
            else:
                modules.append(nn.Linear(prev_units,hidden_units))
                modules.append(the_act_layer)
 
        return nn.Sequential(*modules)
    
    def encode(self, x):
        ''' 
        Creates a sequential based on the layers in self.layers
        Has to specialized layers one that returns standard Gaussian parameteres and
        the natural parameters
        '''
        return self.net(x)

    def forward(self, x):
   
        # Ravel Inputs so shape is: (M, K, D) -> (M*K, D)
        input_shape = list(x.shape)
        x = torch.reshape(x, (-1, self.input_dim))
        mu, var = self.encode(x)

        # unravel output: (M*K, D) -> (M, K, D)
        output_shape = input_shape[:-1]
        output_shape.append(self.output_dim)
        
        out_res, out_res2 = self.res_model(x)
        outputs = (torch.reshape(torch.add(mu,out_res),output_shape), torch.reshape(torch.add(var,out_res2),output_shape))
        
        return outputs
    


class Standard_Activation(nn.Module):
    def __init__(self):
        super(Standard_Activation, self).__init__()
    
    def forward(self, input):
        raw_1, raw_2 = torch.chunk(input, 2, dim=-1)
        mean = raw_1
        var = F.softplus(raw_2)
        return mean, var
    
class Natural_Parameter_Activation(nn.Module):
    def __init__(self):
        super(Natural_Parameter_Activation, self).__init__()
    
    def forward(self, input):
        raw_1, raw_2 = torch.chunk(input, 2, dim=-1)
        eta1 = raw_1
        eta2 = -1./2 * F.softplus(raw_2)
        return eta1, eta2

class ResNetShort(nn.Module):
    def __init__(self, input_dim, output_dim, the_type='standard'):
        super(ResNetShort, self).__init__()
        # Initalize resnet shortcut
        # Create a res-net like shortcut
        # Why do we need do this (obviously for initalization, but why not just use regular) ?!?!
        orthonormal_cols = rand_partial_isometry(input_dim, output_dim, 1.)
        self.W = init_tensor_gpu_grad(torch.from_numpy(orthonormal_cols).float(), trainable=True, device='cuda')
        self.b1 = init_tensor_gpu_grad(torch.zeros(output_dim),trainable=True, device='cuda')
        # need to create shortcut for second output since Gaussian
        self.b2 = init_tensor_gpu_grad(torch.zeros(output_dim),trainable=True, device='cuda')

        if the_type == 'standard':
            self.a = torch.tensor(1.).float().to('cuda')
        elif the_type == 'natparam':
            self.a = torch.tensor(-0.5).float().to('cuda')
        else:
            raise NotImplementedError
    
    def forward(self, input):
        
        return torch.matmul(input, self.W) + self.b1, self.a*torch.log1p(torch.exp(self.b2))


