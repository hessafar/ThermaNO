import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# pi = np.pi

class FNN(nn.Module):
    def __init__(self, layers_unit, activation):
        super(FNN, self).__init__()
        # self.layers = layers_unit
        n_layers = len(layers_unit)
        self.activation = activation
        self.layers = nn.ModuleList([nn.Linear(layers_unit[i], layers_unit[i + 1]) for i in range(n_layers - 2)])
        self.layer_output = nn.Linear(layers_unit[-2], layers_unit[-1])
    # def forward(self, input):
    #     z = input
    #     for i in range(self.n_layers - 2):
    #         y = self.linears[i](z)
    #         z = self.activation(y)
    #     z = self.linears[-1](z)
    #     return z
    
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.layer_output(x)


class vDeepONet(nn.Module):
    def __init__(self, layers_tr, activation_tr, layers_br, activation_br,
                       scaling_fac_tr, scaling_bias_tr, scaling_fac_br, scaling_bias_br,
                       rescale_fac, rescale_bias):
        super(vDeepONet, self).__init__()
        self.scaling_fac_br = scaling_fac_br
        self.scaling_bias_br = scaling_bias_br
        self.scaling_fac_tr = scaling_fac_tr
        self.scaling_bias_tr = scaling_bias_tr
        self.rescale_fac = rescale_fac
        self.rescale_bias = rescale_bias

        
        self.branch = FNN(layers_unit=layers_br, activation=activation_br)
        self.trunk = FNN(layers_unit=layers_tr, activation=activation_tr)
        # self.root = FNN(layers=[3 * layers_br[-1], layers_br[-1], layers_br[-1], 1], activation=activation_tr)
        # self.root = FNN(layers=[3 * layers_br[-1], 1], activation=activation_tr)
        
        # self.root = FNN(layers_unit=[layers_br[-1], layers_br[-1], layers_br[-1], 1], activation=activation_tr)
        # self.root = FNN(layers_unit=[3*layers_br1[-1], 1], activation=activation_tr)
        # self.root1 = FNN(layers_unit=[layers_br[-1], 1], activation=activation_tr)
        # self.root2 = FNN(layers_unit=[layers_br[-1], 1], activation=activation_tr)
        # self.root3 = FNN(layers_unit=[layers_br[-1], 1], activation=activation_tr)
        # self.root = FNN(layers_unit=[3*layers_br[-1], int(0.5*layers_br[-1]), 1], activation=activation_tr)

        
        # self.b = nn.parameter.Parameter(
        #     torch.tensor(0.0))  # Adding extra bios. The model also works without extra bios.
        

    def normalizer_br(self, input_br):
        return self.scaling_fac_br*input_br + self.scaling_bias_br
    def normalizer_tr(self, input_tr):
        return self.scaling_fac_tr*input_tr + self.scaling_bias_tr
    def renormalizer(self, input):
        return self.rescale_fac*input + self.rescale_bias


    def forward(self, input_br, input_tr):  # Original
        out_scaling_br = self.normalizer_br(input_br)
        out_scaling_tr = self.normalizer_tr(input_tr)
        out_br = self.branch(out_scaling_br)
        out_tr = self.trunk(out_scaling_tr)
        
        out_mul = torch.einsum("ij,ij->i", out_tr, out_br).view(-1,1) # Use when no root net is employed.
        
        ref_val = 0.0
        out_reg = ref_val + F.softplus(out_mul)
        

        return out_mul