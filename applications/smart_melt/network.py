import numpy as np
import torch
import torch.nn as nn
import time
from matplotlib import pyplot as plt
# import vtk
# from vtk.util.numpy_support import numpy_to_vtk
import h5py
import os
from config import device
from models.vdeeponet import vDeepONet
from models.endeeponet import enDeepONet
from utils.helpers import from_numpy_to_torch, Sin

class Temperature_distribution(nn.Module):
    def __init__(self, input_br, input_tr, output, input_br_test, input_tr_test, output_test, scaling_dict,
                 layers_br, layers_tr):
        
        super(Temperature_distribution, self).__init__()
        
        self.input_br = from_numpy_to_torch(input_br, False)
        self.input_tr = from_numpy_to_torch(input_tr, False)
        self.output = from_numpy_to_torch(output, False)
        self.input_br_test = from_numpy_to_torch(input_br_test, False)
        self.input_tr_test = from_numpy_to_torch(input_tr_test, False)
        self.output_test = from_numpy_to_torch(output_test, False)

        indices = torch.randperm(self.input_tr.shape[0])
        self.input_br =  self.input_br[indices]#[:150000]
        self.input_tr = self.input_tr[indices]#[:150000]
        self.output = self.output[indices]#[:150000]

        scaling_fac_br = scaling_dict["scaling_fac_br"]
        scaling_bias_br = scaling_dict["scaling_bias_br"]
        scaling_fac_tr = scaling_dict["scaling_fac_tr"]
        scaling_bias_tr = scaling_dict["scaling_bias_tr"]
        rescale_fac = scaling_dict["rescale_fac"]
        rescale_bias = scaling_dict["rescale_bias"]
        
        scaling_fac_br = from_numpy_to_torch(scaling_fac_br, False)
        scaling_bias_br = from_numpy_to_torch(scaling_bias_br, False)
        scaling_fac_tr = from_numpy_to_torch(scaling_fac_tr, False)
        scaling_bias_tr = from_numpy_to_torch(scaling_bias_tr, False)
        rescale_fac = from_numpy_to_torch(rescale_fac, False)
        rescale_bias = from_numpy_to_torch(rescale_bias, False)
        


        # self.activation = nn.Tanh()
        # self.activation = nn.ReLU()
        self.activation = Sin()
        self.network = enDeepONet(layers_br=layers_br, activation_br=self.activation,
                                layers_tr=layers_tr, activation_tr=self.activation,
                                scaling_fac_br=scaling_fac_br, scaling_fac_tr=scaling_fac_tr,
                                scaling_bias_br=scaling_bias_br, scaling_bias_tr=scaling_bias_tr,
                                rescale_fac=rescale_fac, rescale_bias=rescale_bias).to(device)
        

        self.mse = nn.MSELoss()

        self.optimizer = None

        self.total_loss = 0.0
        self.epo = 0
        self.pde_loss = []
        self.data_loss = []
        self.bc_loss = []
        self.in_neu = False
        self.in_diri = False
        self.first_epoch = True

    def set_optimizer(self, optimizer, lr):
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        if optimizer == "LBFGS":
            # self.optimizer = torch.optim.LBFGS(self.network.parameters(), lr=1.0, max_iter=20, max_eval=30,
            #                                    history_size=50, tolerance_grad=1e-05,
            #                                    tolerance_change=0.5 * np.finfo(float).eps,
            #                                    line_search_fn="strong_wolfe")
            self.optimizer = torch.optim.LBFGS(self.network.parameters(), lr=lr, max_iter=25, max_eval=25,
                                               history_size=50, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

 
    def compute_data_loss(self, input_br, input_tr, labeled_data):
        u = self.network(input_br, input_tr)
        return self.mse(u, labeled_data)

    def closure(self):
        # reset gradients to zero:
        self.optimizer.zero_grad()

        # loss calculations
 

        loss_data = self.compute_data_loss(self.input_br, self.input_tr, self.output)
        
        self.total_loss = loss_data

        
        # derivative with respect to net's weights:
        self.total_loss.backward()

        return self.total_loss

    def train_netwrok(self, num_epochs, output_freq, tolerance):
        # self.network.train()
        # self.optimizer.step(self.closure)
        self.network.train()
        loss_old = 100.
        
        if self.first_epoch:
            self.first_epoch = False

        for epoch in range(num_epochs):
            self.epo += 1
            self.optimizer.step(self.closure)
            loss_test = self.compute_data_loss(self.input_br_test, self.input_tr_test, self.output_test)
            if epoch % output_freq == 0:
                print("Iteration: {:}, Total_Loss: {:0.6e}, Test_Loss: {:0.6e}"
                      .format(epoch, self.total_loss, loss_test))
            # if abs(self.total_loss - loss_old)/self.total_loss < 1e-15: 
            if abs(self.total_loss) < tolerance or abs(self.total_loss - loss_old)/self.total_loss < 1e-15: 
                print("No more improvement in the loss! Training terminated!")
                break
            loss_old = self.total_loss



        print("Iteration: {:}, Total_Loss: {:0.6e}, Test_Loss: {:0.6e}"
                      .format(epoch, self.total_loss, loss_test))
        # print("Iteration: {:}, Total_Loss: {:0.6e}, PDE_Loss: {:0.6e}, BC_Loss: {:0.6e}, IC_Loss: {:0.6e}"
        #               .format(epoch, self.total_loss, self.current_loss_PDE, self.current_loss_BC1+self.current_loss_BC2+self.current_loss_BC3+self.current_loss_BC4, self.current_loss_IC))
        #return self.pde_loss, self.data_loss, self.bc_loss
