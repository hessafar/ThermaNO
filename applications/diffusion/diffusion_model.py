import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from models.endeeponet import enDeepONet
from config import global_params, device, laser_config, material_params, C_T, T_0
from utils.helpers import Sin, from_numpy_to_torch


mini_batch = global_params["mini_batch"]
batch_size = global_params["batch_size"]
batch_size_ini = global_params["batch_size_ini"]
batch_size_neu = global_params["batch_size_neu"]
batch_size_diri = global_params["batch_size_diri"]
coeff = laser_config["coeff"]
P0_laser = laser_config["power"]
r_laser = laser_config["radius"]
depth_laser = laser_config["depth"]
H = laser_config["domainH"]
rho = material_params["density"]
kappa = material_params["conductivity"]
cp = material_params["cp"]

class Diffusion(nn.Module):
    def __init__(self, input_tr, input_br, input_c_laser,
                        ic_input_tr, ic_input_br,
                        bc_data, ic_value, layers_tr, layers_br1,
                        scaling_dict):
        super(Diffusion, self).__init__()

        self.ic_value = T_0

        scaling_fac_br1 = scaling_dict["scaling_fac_br1"]
        scaling_bias_br1 = scaling_dict["scaling_bias_br1"]
        scaling_fac_tr = scaling_dict["scaling_fac_tr"]
        scaling_bias_tr = scaling_dict["scaling_bias_tr"]
        rescale_fac = scaling_dict["rescale_fac"]
        rescale_bias = scaling_dict["rescale_bias"]

        scaling_fac_br1 = from_numpy_to_torch(scaling_fac_br1, False)
        scaling_bias_br1 = from_numpy_to_torch(scaling_bias_br1, False)
        scaling_fac_tr = from_numpy_to_torch(scaling_fac_tr, False)
        scaling_bias_tr = from_numpy_to_torch(scaling_bias_tr, False)
        rescale_fac = from_numpy_to_torch(rescale_fac, False)
        rescale_bias = from_numpy_to_torch(rescale_bias, False)

        self.num_bc = 6 #len(bc_dict)
                
        # self.data_total = [input_tr, input_br1, input_c_laser, ic_input_tr, ic_input_br1, input_tr_bc,  input_br1_bc, bc_info]
        data_total_cp = [input_tr, input_br, input_c_laser] #, ic_input_tr, ic_input_br1, input_tr_bc,  input_br1_bc, bc_info]
        data_total_ini = [ic_input_tr, ic_input_br, ic_value]
        data_total_bc_neu = [bc_data[0], bc_data[1], bc_data[2]]
        data_total_bc_diri = [bc_data[3], bc_data[4]]

        self.data_total_cp = [
        from_numpy_to_torch(data_total_cp[0], True),
        from_numpy_to_torch(data_total_cp[1], False),
        from_numpy_to_torch(data_total_cp[2], False)
        ]
        
        self.data_total_ini = [
            from_numpy_to_torch(data_total_ini[0], True),
            from_numpy_to_torch(data_total_ini[1], False),
            from_numpy_to_torch(data_total_ini[2], False)
        ]
        
        self.data_total_bc_neu = [
            from_numpy_to_torch(data_total_bc_neu[0], True),
            from_numpy_to_torch(data_total_bc_neu[1], False),
            from_numpy_to_torch(data_total_bc_neu[2], False)
        ]
        
        self.data_total_bc_diri = [
            from_numpy_to_torch(data_total_bc_diri[0], True),
            from_numpy_to_torch(data_total_bc_diri[1], False)
        ]

        self.data_cp = (
        self.data_total_cp[0].contiguous(),
        self.data_total_cp[1].contiguous(),
        self.data_total_cp[2].contiguous()
        )
        
        self.data_ini = (
            self.data_total_ini[0].contiguous(),
            self.data_total_ini[1].contiguous(),
            self.data_total_ini[2].contiguous()
        )
        
        self.data_bc_neu = (
            self.data_total_bc_neu[0].contiguous(),
            self.data_total_bc_neu[1].contiguous(),
            self.data_total_bc_neu[2].contiguous()
        )
        
        self.data_bc_diri = (
            self.data_total_bc_diri[0].contiguous(),
            self.data_total_bc_diri[1].contiguous()
        )

        # Pre-calculate slice indices
        self.batches_cp = [(i, i+batch_size) 
                        for i in range(0, len(self.data_cp[0]), batch_size)]
        
        self.batches_ini = [(i, i+batch_size_ini)
                        for i in range(0, len(self.data_ini[0]), batch_size_ini)]
        
        self.batches_bc_neu = [(i, i+batch_size_neu)
                            for i in range(0, len(self.data_bc_neu[0]), batch_size_neu)]
        
        self.batches_bc_diri = [(i, i+batch_size_diri)
                            for i in range(0, len(self.data_bc_diri[0]), batch_size_diri)]


        # self.input_tr_bc = torch.tensor(input_bc[:, 0:dim + 1], dtype=torch.float32, requires_grad=True).to(device)
        # self.input_br_bc = torch.tensor(input_bc[:, dim + 1:-1], dtype=torch.float32, requires_grad=False).to(device)
        # self.f_t_bc = torch.tensor(input_bc[:, -1], dtype=torch.float32, requires_grad=False).to(device)
        # self.bc_type = torch.tensor(bc_type, dtype=torch.float32, requires_grad=False).to(device)

        self.lambda_IC = 1.0 #nn.Parameter(torch.tensor(initial_lambda_IC))
        self.lambda_BC = 1.0 #nn.Parameter(torch.tensor(initial_lambda_BC))
        self.lambda_PDE = 1.0 #nn.Parameter(torch.tensor(initial_lambda_PDE))

        self.current_loss_PDE = 0.0
        self.current_loss_BC = 0.0
        self.current_loss_IC = 0.0

        # initial_lambda_ic = 1.0
        # initial_lambda_boundary = 1.0
        # self.lambda_ic = nn.Parameter(torch.tensor(initial_lambda_ic))
        # self.lambda_boundary = nn.Parameter(torch.tensor(initial_lambda_boundary))
        self.activation = nn.Tanh()
        self.activation1 = nn.ReLU()
        self.activation = Sin()
        self.network = enDeepONet(layers_tr=layers_tr, activation_tr=self.activation,
                                layers_br=layers_br1, activation_br=self.activation,
                                scaling_fac_tr=scaling_fac_tr, scaling_bias_tr=scaling_bias_tr,
                                scaling_fac_br=scaling_fac_br1, scaling_bias_br=scaling_bias_br1,
                                rescale_fac=rescale_fac, rescale_bias=rescale_bias).to(device)
        self.mse = nn.MSELoss()

        self.optimizer = None
        self.scheduler = None

        self.total_loss = 0.0
        self.epo = 0
        self.first_epoch = True
        self.pde_loss = []
        self.data_loss = []
        self.bc_loss = []


    def set_optimizer(self, optimizer, lr):
        if optimizer == "Adam":
            # self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, betas=(0.8, 0.999))
            # self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
            self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr, weight_decay=0.01)
            # self.scheduler = ExponentialLR(self.optimizer, gamma=0.85)
        if optimizer == "LBFGS":
            # self.optimizer = torch.optim.LBFGS(self.network.parameters(), lr=1.0, max_iter=20, max_eval=30,
            #                                    history_size=50, tolerance_grad=1e-05,
            #                                    tolerance_change=0.5 * np.finfo(float).eps,
            #                                    line_search_fn="strong_wolfe")
            self.optimizer = torch.optim.LBFGS(self.network.parameters(), lr=lr, max_iter=20, max_eval=25,
                                               history_size=50, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

    def compute_PDE_loss(self, input_br, input_tr, f_t):
        # alpha = 0.01
        u = self.network(input_br, input_tr)
        grad_u_i = torch.autograd.grad(u, input_tr, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        grad_u_t = grad_u_i[:, 0]
        grad_u_x = grad_u_i[:, 1]
        grad_u_y = grad_u_i[:, 2]
        grad_u_z = grad_u_i[:, 3]
        
        # cond = (11.82 + 0.0106*temp) / C_k
        # c_p = 330.9 + 0.563*temp  - 4.015e-4*(temp)**2 + 9.465e-8*(temp)**3 + \
        #         (H_l / (T_l - T_s)) * (1 + torch.tanh(((T_l - T_s) - abs(temp-T_l) - abs(temp-T_s))/1.0))
        # c_p = 330.9 + 0.563*temp  - 4.015e-4*(temp)**2 + 9.465e-8*(temp)**3 + \
        #         0.25*(H_l / (T_l - T_s)) * (torch.exp(-(((temp - 1690.5)/(T_l - T_s))**2)))
        
        # ks = (9.248e-2 + 1.571e-4*(u*C_T))*100
        # kl = (1.241e-1 + 3.279e-5*(u*C_T))*100
        # T_s = 1600
        # T_l = 1700
        # ep1 = 1e-2
        # ep2 = 1e-2
        # ep3 = 1e-3
        # cond = 1.*ks*(0.5*(1 - torch.tanh(ep1*(u*C_T-T_l)))) + 1*kl*(0.5*(1 + torch.tanh(ep2*(u*C_T-T_l)))) + \
        #     0.05*ks*(1 + torch.tanh(ep3*((T_l-T_s) - torch.abs(u*C_T-T_l) - torch.abs(u*C_T-T_s))))


        # For the variable conductivity use the following 4 lines
        # cond = (234.887443886408 + 0.018367198 * (u*C_T-273) + 225.10088 * torch.tanh(0.0118783 * ((u*C_T-273) - 1543.815)))
        cond1 = 25.0 #(234.887443886408 + 0.018367198 * (u*C_T-273) + 225.10088 * torch.tanh(0.0118783 * ((u*C_T-273) - 1543.815)))
        cond2 = 0.3*cond1 #0.5*cond1
        x1 = 0.8
        x2 = 1.2
        L = 1.0
        beta = 30.0
        cond = cond1 + 0.5*(cond2-cond1) * (torch.tanh(beta*(input_tr[:,1]-x1)/L) + torch.tanh(beta*(x2-input_tr[:,1])/L))
        cond = cond*torch.ones_like(u)
        k_grad_u_x = cond[:,0]*grad_u_x
        k_grad_u_y = cond[:,0]*grad_u_y
        k_grad_u_z = cond[:,0]*grad_u_z

        # For vthe constant ariable conductivity use the following 3 lines
        # k_grad_u_x = kappa*grad_u_x
        # k_grad_u_y = kappa*grad_u_y
        # k_grad_u_z = kappa*grad_u_z
        
 
        grad_u_xx = torch.autograd.grad(k_grad_u_x, input_tr, grad_outputs=torch.ones_like(grad_u_x),
                                        create_graph=True)[0][:, 1]
        grad_u_yy = torch.autograd.grad(k_grad_u_y, input_tr, grad_outputs=torch.ones_like(grad_u_y),
                                        create_graph=True)[0][:, 2]
        grad_u_zz = torch.autograd.grad(k_grad_u_z, input_tr, grad_outputs=torch.ones_like(grad_u_z),
                                        create_graph=True)[0][:, 3]

        P_x = coeff * f_t[:, 2] * torch.exp(-3*(((input_tr[:,1] - f_t[:, 0])/r_laser) ** 2 + ((input_tr[:,2] - f_t[:, 1])/ r_laser) ** 2 )) * torch.exp(
        -3*(input_tr[:,3] - H) ** 2 / depth_laser ** 2) # Laser power at the given time and location

        c_p = (446.337248 + 0.14180844 * (u*C_T-273) - 61.431671211432 * torch.exp(
                    -0.00031858431233904*((u*C_T-273)-525)**2) + 1054.9650568*torch.exp(-0.00006287810196136*((u*C_T-273)-1545)**2))

        # res2 = 1e-3*(grad_u_xx + grad_u_yy + grad_u_zz)/(rho*c_p*1e-9)
        # q = P_x / (rho*c_p*C_T*1e-9)
        # res = (grad_u_t - res2 - q)

        # For the variable properties use the following res
        res = (grad_u_t - (1e-3*(grad_u_xx + grad_u_yy + grad_u_zz)/(rho*c_p*1e-9)) - (P_x / (rho*c_p*C_T*1e-9)))
        
        # For the constant  properties use the following res
        # res = (grad_u_t - (1e-3*(grad_u_xx + grad_u_yy + grad_u_zz)/(rho*cp*1e-9)) - (P_x / (rho*cp*C_T*1e-9)))

        null = torch.zeros_like(res)

        return self.mse(res, null)

    def compute_BC_loss_neu(self, input_br, input_tr, normals):
        
        u = self.network(input_br, input_tr)
        grad_u_i = torch.autograd.grad(u, input_tr, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        grad_u_n = grad_u_i*normals
        null = torch.zeros_like(grad_u_n)
        # print("test size", grad_u_i.shape, normals.shape, grad_u_n.shape, null.shape)
        return self.mse(grad_u_n, null)

    def compute_BC_loss_diri(self, input_br, input_tr):
        
        u = self.network(input_br, input_tr)
        value = self.ic_value*torch.ones_like(u)
        return self.mse(u, value)
    
    def compute_IC_loss(self, input_br, input_tr, u_0):

        u = self.network(input_br, input_tr)
        # u0 = u_0 * torch.ones_like(u)
        return self.mse(u, u_0)

    def create_mini_batches(arr1, arr2, arr3, batch_size):
        for i in range(0, len(arr1), batch_size):
            yield (arr1[i:i + batch_size], arr2[i:i + batch_size], arr3[i:i + batch_size])

    def create_shuffled_batches(self, dataset_length, batch_size):
        """Create shuffled batch indices for one epoch"""
        indices = torch.randperm(dataset_length)
        return [indices[i:i+batch_size] for i in range(0, dataset_length, batch_size)]
    
    def create_one_shuffled(self, dataset_length, batch_size):
        indices = torch.randperm(dataset_length)[:batch_size]
        return indices

    def closure(self):
        # reset gradients to zero:
        self.optimizer.zero_grad()

        # loss calculations
        loss_PDE = self.compute_PDE_loss(self.input_br1, self.input_tr, self.input_c_laser)
        loss_BC = 0.0
        for i in range(self.num_bc):
            loss_BC += self.compute_BC_loss(self.input_br1_bc[i], self.input_tr_bc[i], self.bc_info[i])
        loss_IC = self.compute_IC_loss(self.ic_input_br1, self.ic_input_tr, self.ic_value)
        
        # L2 regularization (weight decay)
        # l2_lambda = 0.00001  # Adjust the regularization strength
        # regularization_loss = 0
        # for param in self.network.parameters():
        #     regularization_loss += torch.norm(param)**2
        # self.total_loss = loss_PDE + self.lambda_boundary*loss_BC + self.lambda_ic*loss_IC + l2_lambda * regularization_loss
        
        # Loss without regularization
        self.total_loss =  self.lambda_PDE*loss_PDE + self.lambda_BC*loss_BC + self.lambda_IC*loss_IC

        self.current_loss_PDE = loss_PDE
        self.current_loss_BC = loss_BC #[0] + loss_BC[1] + loss_BC[2] + loss_BC[3]
        self.current_loss_IC = loss_IC

        self.pde_loss.append(loss_PDE.item())
        self.data_loss.append(loss_IC.item())
        self.bc_loss.append(loss_BC.item())
        
        # derivative with respect to net's weights:
        self.total_loss.backward()

        return self.total_loss

    def train_netwrok(self, num_epochs, output_freq, tolerance):
        self.network.train()
        loss_old = 100.
        
        if mini_batch == 0:
            pass
            # self.input_tr = from_numpy_to_torch(self.data_total[0], True)
            # self.input_br1 = from_numpy_to_torch(self.data_total[1], False)
            # self.input_c_laser = from_numpy_to_torch(self.data_total[2], False)
            # self.ic_input_tr = from_numpy_to_torch(self.data_total[3], True)
            # self.ic_input_br1 = from_numpy_to_torch(self.data_total[4], False)
            # self.input_tr_bc = from_numpy_to_torch(self.data_total[5], True)
            # self.input_br1_bc = from_numpy_to_torch(self.data_total[6], False)
            # self.bc_info = from_numpy_to_torch(self.data_total[7], False)

            # if self.first_epoch:
            #     pde_0 = self.compute_PDE_loss(self.input_br1, self.input_tr, self.input_c_laser).item()
            #     bc_0 = 0.0
            #     for i in range(self.num_bc):
            #         bc_0 += self.compute_BC_loss(self.input_br1_bc[i], self.input_tr_bc[i], self.bc_info[i]).item()
                
                
            #     self.lambda_PDE = 1.0/pde_0 if pde_0 >= 1e-15 else 1.0
            #     self.lambda_BC = 1.0/bc_0 if bc_0 >= 1e-15 else 1.0
                
            #     self.first_epoch = False

            # for epoch in range(num_epochs):
            #     self.epo += 1
            #     self.optimizer.step(self.closure)
            #     if epoch % output_freq == 0:
            #         print("Iteration: {:}, Total_Loss: {:0.6e}, PDE_Loss: {:0.6e}, BC_Loss: {:0.6e}, IC_Loss: {:0.6e}"
            #             .format(epoch, self.total_loss, self.current_loss_PDE, self.current_loss_BC, self.current_loss_IC))
            #     if abs(self.current_loss_PDE) < tolerance or abs(self.total_loss - loss_old)/self.total_loss < 1e-15: 
            #         print("No more improvement in the accuracy! Training terminated!")
            #         break
            #     loss_old = self.total_loss
        else: #using mini batches
            
            # dataset_cp = TensorDataset(*self.data_total_cp)
            # dataset_ini = TensorDataset(*self.data_total_ini)
            # dataset_bc_neu = TensorDataset(*self.data_total_bc_neu)
            # dataset_bc_diri = TensorDataset(*self.data_total_bc_diri)
            # dataloader_cp = DataLoader(dataset_cp, batch_size=batch_size, shuffle=True)
            # dataloader_ini = DataLoader(dataset_ini, batch_size=batch_size_ini, shuffle=True)
            # dataloader_bc_neu = DataLoader(dataset_bc_neu, batch_size=batch_size_neu, shuffle=True)
            # dataloader_bc_diri = DataLoader(dataset_bc_diri, batch_size=batch_size_diri, shuffle=True)
            
            # input_tr, input_br, input_c_laser = next(iter(dataloader_cp))
            # input_tr_bc_neu, input_br_bc_neu, normals = next(iter(dataloader_bc_neu))
            # input_tr_bc_diri, input_br_bc_diri = next(iter(dataloader_bc_diri))
            
            # Recently commented
            # cp_start, cp_end = self.batches_cp[0]
            # bc_neu_start, bc_neu_end = self.batches_bc_neu[0]
            # bc_diri_start, bc_diri_end = self.batches_bc_diri[0]
            # input_tr, input_br, input_c_laser = (
            #         self.data_cp[0][cp_start:cp_end],
            #         self.data_cp[1][cp_start:cp_end],
            #         self.data_cp[2][cp_start:cp_end]
            #         )
            # input_tr_bc_neu, input_br_bc_neu, normals = (
            #     self.data_bc_neu[0][bc_neu_start:bc_neu_end],
            #     self.data_bc_neu[1][bc_neu_start:bc_neu_end],
            #     self.data_bc_neu[2][bc_neu_start:bc_neu_end]
            #      )

            # input_tr_bc_diri, input_br_bc_diri = (
            #     self.data_bc_diri[0][bc_diri_start:bc_diri_end],
            #     self.data_bc_diri[1][bc_diri_start:bc_diri_end]
            #     )

            # Compute weights
            cp_batches      = self.create_one_shuffled(len(self.data_cp[0]), batch_size)
            ini_batches     = self.create_one_shuffled(len(self.data_ini[0]), batch_size_ini)
            bc_neu_batches  = self.create_one_shuffled(len(self.data_bc_neu[0]), batch_size_neu)
            bc_diri_batches = self.create_one_shuffled(len(self.data_bc_diri[0]), batch_size_diri)
            input_tr = self.data_cp[0][cp_batches]
            input_br = self.data_cp[1][cp_batches]
            input_c_laser = self.data_cp[2][cp_batches]
            input_tr_bc_neu = self.data_bc_neu[0][bc_neu_batches]
            input_br_bc_neu = self.data_bc_neu[1][bc_neu_batches]
            normals = self.data_bc_neu[2][bc_neu_batches]
            input_tr_bc_diri = self.data_bc_diri[0][bc_diri_batches]
            input_br_bc_diri = self.data_bc_diri[1][bc_diri_batches]
            input_tr_ic = self.data_ini[0][ini_batches]
            input_br_ic = self.data_ini[1][ini_batches]
            output_ic = self.data_ini[2][ini_batches]
            
            if self.first_epoch:
                pde_0 = self.compute_PDE_loss(input_br, input_tr, input_c_laser).item()
                ic_0 = self.compute_IC_loss(input_br_ic, input_tr_ic, output_ic).item()
                bc_0 = 0.0
                for _ in range(2):
                    bc_0 += self.compute_BC_loss_neu(input_br_bc_neu, input_tr_bc_neu, normals).item()
                    bc_0 += self.compute_BC_loss_diri(input_br_bc_diri, input_tr_bc_diri).item()
                
                
                self.lambda_PDE = 1.0/pde_0 if pde_0 >= 1e-15 else 1.0
                self.lambda_IC = 1.0/ic_0 if ic_0 >= 1e-15 else 1.0
                self.lambda_BC = 1.0/bc_0 if bc_0 >= 1e-15 else 1.0
                
                self.first_epoch = False

            for epoch in range(num_epochs):
                running_loss = 0.0
                running_loss_PDE = 0.0
                running_loss_IC = 0.0
                running_loss_BC = 0.0
                self.optimizer.zero_grad()

                # cp_batches      = self.create_shuffled_batches(len(self.data_cp[0]), batch_size)
                # ini_batches     = self.create_shuffled_batches(len(self.data_ini[0]), batch_size_ini)
                # bc_neu_batches  = self.create_shuffled_batches(len(self.data_bc_neu[0]), batch_size_neu)
                # bc_diri_batches = self.create_shuffled_batches(len(self.data_bc_diri[0]), batch_size_diri)

                cp_batches      = self.create_one_shuffled(len(self.data_cp[0]), batch_size)
                ini_batches     = self.create_one_shuffled(len(self.data_ini[0]), batch_size_ini)
                bc_neu_batches  = self.create_one_shuffled(len(self.data_bc_neu[0]), batch_size_neu)
                bc_diri_batches = self.create_one_shuffled(len(self.data_bc_diri[0]), batch_size_diri)
                
                # Define closure for the LBFGS optimizer
                def closure():
                    nonlocal running_loss
                    nonlocal running_loss_PDE
                    nonlocal running_loss_IC
                    nonlocal running_loss_BC
                    running_loss = 0.0  # Reset running loss for this step
                    running_loss_PDE = 0.0
                    running_loss_IC = 0.0
                    running_loss_BC = 0.0
                    self.optimizer.zero_grad()  # Clear previous gradients

                    input_tr = self.data_cp[0][cp_batches]
                    input_br = self.data_cp[1][cp_batches]
                    input_c_laser = self.data_cp[2][cp_batches]
                    l_PDE = self.compute_PDE_loss(input_br, input_tr, input_c_laser)
                    loss_PDE = self.lambda_PDE*l_PDE
                    running_loss_PDE = loss_PDE 

                    input_tr_ic = self.data_ini[0][ini_batches]
                    input_br_ic = self.data_ini[1][ini_batches]
                    output_ic = self.data_ini[2][ini_batches]
                    l_IC = self.compute_IC_loss(input_br_ic, input_tr_ic, output_ic)
                    loss_IC = self.lambda_IC*l_IC
                    running_loss_IC = loss_IC

                    input_tr_bc_neu = self.data_bc_neu[0][bc_neu_batches]
                    input_br_bc_neu = self.data_bc_neu[1][bc_neu_batches]
                    normals = self.data_bc_neu[2][bc_neu_batches]
                    l_BC_neu = self.compute_BC_loss_neu(input_br_bc_neu, input_tr_bc_neu, normals)
                    loss_BC_neu = self.lambda_BC*l_BC_neu
                    
                    input_tr_bc_diri = self.data_bc_diri[0][bc_diri_batches]
                    input_br_bc_diri = self.data_bc_diri[1][bc_diri_batches]
                    l_BC_diri = self.compute_BC_loss_diri(input_br_bc_diri, input_tr_bc_diri)
                    loss_BC_diri = self.lambda_BC*l_BC_diri
                    running_loss_BC = loss_BC_neu + loss_BC_diri
                    self.total_loss = loss_PDE + loss_IC + loss_BC_neu + loss_BC_diri
                    self.total_loss.backward()

                                       
                    
                    # # Compute interior loss (PDE residual)
                    # # for input_tr, input_br, input_c_laser in dataloader_cp:
                    # # for start, end in self.batches_cp:
                    # #     input_tr = self.data_cp[0][start:end]
                    # #     input_br = self.data_cp[1][start:end]
                    # #     input_c_laser = self.data_cp[2][start:end]
                    # for batch_indices in cp_batches:
                    #     input_tr = self.data_cp[0][batch_indices]
                    #     input_br = self.data_cp[1][batch_indices]
                    #     input_c_laser = self.data_cp[2][batch_indices]
                        
                        
                    #     loss_PDE = self.lambda_PDE*self.compute_PDE_loss(input_br, input_tr, input_c_laser)
                    #     loss_PDE.backward()
                    #     running_loss_PDE += loss_PDE.item()

                    # # for input_tr_ic, input_br_ic in dataloader_ini:
                    # # for start, end in self.batches_ini:
                    # #     input_tr_ic = self.data_ini[0][start:end]
                    # #     input_br_ic = self.data_ini[1][start:end]
                    # for batch_indices in ini_batches:
                    #     input_tr_ic = self.data_ini[0][batch_indices]
                    #     input_br_ic = self.data_ini[1][batch_indices]

                    #     loss_IC = self.lambda_IC*self.compute_IC_loss(input_br_ic, input_tr_ic, self.ic_value)
                    #     loss_IC.backward()
                    #     running_loss_IC+= loss_IC.item()

                    # # for input_tr_bc_neu, input_br_bc_neu, normals in dataloader_bc_neu:
                    # # for start, end in self.batches_bc_neu:
                    # #     input_tr_bc_neu = self.data_bc_neu[0][start:end]
                    # #     input_br_bc_neu = self.data_bc_neu[1][start:end]
                    # #     normals = self.data_bc_neu[2][start:end]
                    # for batch_indices in bc_neu_batches:
                    #     input_tr_bc_neu = self.data_bc_neu[0][batch_indices]
                    #     input_br_bc_neu = self.data_bc_neu[1][batch_indices]
                    #     normals = self.data_bc_neu[2][batch_indices]

                    #     loss_BC = self.lambda_BC*self.compute_BC_loss_neu(input_br_bc_neu, input_tr_bc_neu, normals)
                    #     loss_BC.backward()
                    #     running_loss_BC+= loss_BC.item()
                        
                    # # for input_tr_bc_diri, input_br_bc_diri in dataloader_bc_diri:
                    # # for start, end in self.batches_bc_diri:
                    # #     input_tr_bc_diri = self.data_bc_diri[0][start:end]
                    # #     input_br_bc_diri = self.data_bc_diri[1][start:end]
                    # for batch_indices in bc_diri_batches:
                    #     input_tr_bc_diri = self.data_bc_diri[0][batch_indices]
                    #     input_br_bc_diri = self.data_bc_diri[1][batch_indices]

                    #     loss_BC = self.lambda_BC*self.compute_BC_loss_diri(input_br_bc_diri, input_tr_bc_diri)
                    #     loss_BC.backward()
                    #     running_loss_BC+= loss_BC.item()
                    # running_loss = running_loss_PDE + running_loss_BC + running_loss_IC
                    # self.total_loss = running_loss
                    return self.total_loss  # Return the accumulated loss for LBFGS step

                # Perform the optimization step using the closure function
                self.optimizer.step(closure)

                # If you want the actual (unweighted) PDE, IC, and BC losses from the closure,
                # store them after forward:
                
                # if epoch > 0 and epoch % 500 == 0:
                #     # with torch.no_grad():
                #     self.optimizer.zero_grad()
                #     input_tr = self.data_cp[0][cp_batches]
                #     input_br = self.data_cp[1][cp_batches]
                #     input_c_laser = self.data_cp[2][cp_batches]
                #     input_tr_bc_neu = self.data_bc_neu[0][bc_neu_batches]
                #     input_br_bc_neu = self.data_bc_neu[1][bc_neu_batches]
                #     normals = self.data_bc_neu[2][bc_neu_batches]
                #     input_tr_bc_diri = self.data_bc_diri[0][bc_diri_batches]
                #     input_br_bc_diri = self.data_bc_diri[1][bc_diri_batches]
                #     input_tr_ic = self.data_ini[0][ini_batches]
                #     input_br_ic = self.data_ini[1][ini_batches]
                #     output_ic = self.data_ini[2][ini_batches]
                #     l_PDE = self.compute_PDE_loss(input_br, input_tr, input_c_laser).item()
                #     l_IC  = self.compute_IC_loss(input_br_ic, input_tr_ic, output_ic).item()
                #     l_bc1 = self.compute_BC_loss_neu(input_br_bc_neu, input_tr_bc_neu, normals).item()
                #     l_bc2 = self.compute_BC_loss_diri(input_br_bc_diri, input_tr_bc_diri).item()
                #     l_bc = l_bc1 + l_bc2
                #     if l_PDE >= 1e-15:
                #         self.lambda_PDE = 1.0/l_PDE #else 1.0
                #     if l_IC >= 1e-15:
                #         self.lambda_IC = 1.0/l_IC  #else 1.0
                #     if l_bc >= 1e-15:
                #         self.lambda_BC = 1.0/l_bc  #else 1.0


                running_loss = self.total_loss
                # if epoch % 500 and self.scheduler is not None:
                #     self.scheduler.step()
                if epoch % output_freq == 0:
                    print("Iteration: {:}, Total_Loss: {:0.6e}, PDE_Loss: {:0.6e}, BC_Loss: {:0.6e}, IC_Loss: {:0.6e}"
                        .format(epoch, self.total_loss, running_loss_PDE, running_loss_BC, running_loss_IC))
                if abs(running_loss_PDE) < tolerance or abs(running_loss - loss_old)/running_loss < 1e-15: 
                    print("No more improvement in the accuracy! Training terminated!")
                    break
                loss_old =running_loss

        print("Iteration: {:}, Total_Loss: {:0.6e}, PDE_Loss: {:0.6e}, BC_Loss: {:0.6e}, IC_Loss: {:0.6e}"
                    .format(epoch, self.total_loss, self.current_loss_PDE, self.current_loss_BC, self.current_loss_IC))
        # print("Iteration: {:}, Total_Loss: {:0.6e}, PDE_Loss: {:0.6e}, BC_Loss: {:0.6e}, IC_Loss: {:0.6e}"
        #               .format(epoch, self.total_loss, self.current_loss_PDE, self.current_loss_BC1+self.current_loss_BC2+self.current_loss_BC3+self.current_loss_BC4, self.current_loss_IC))
        return self.pde_loss, self.data_loss, self.bc_loss
