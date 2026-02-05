import numpy as np
import torch
from config import device, Laser, default_dtype, random_seed, T1, T_0, C_T, random_seed, global_params, is_single_track
from utils import data_gen, helpers, io_utils
from applications.diffusion.bc_fuctions import BoundaryCondition
from applications.diffusion.diffusion_model import Diffusion
from matplotlib import pyplot as plt
import time
import os

dim = global_params["dimension"]

def run_case():
    #set_default_dtype(torch.float32)
    print(default_dtype, device)
    helpers.set_default_dtype1(default_dtype)
    laser = Laser()
    print("final time", laser.time_final)

    # path1 = np.array([0])
    # path2 = np.array([1])
    # path3 = np.array([2])
    path1 = ([0]) #--> 1
    path2 = ([0, 1]) #--> 2
    path3 = ([0]) #--> 3
    path4 = ([0]) #--> 4
    path5 = ([0]) #--> 5
    path6 = ([0]) #--> 6
    # path_total = [path1, path2, path3, path4, path5, path6]
    
    # path_total = 40*[path1]
    # path_test = 6*[path1]
    path_total = 40*[path1]
    path_test = 5*[path1]
    
    grid = [100, 50, 50]
    # grid = [100, 75, 50]
    # grid = [400, 300, 200]
    
    # path_total = [path1, path2, path3, path4, path5, path6]#, path7, path8, path9, path10]
    # path_total = generate_path(n_section=3, n_path=6)
    # path_test = [path1, path2, path3, path4, path5, path6]
    # path_test = [path3]
    # path_total = generate_path(4, 1)
    path = path_total
    # coeff = 5 / (np.pi * laser.r_laser ** 3)
    
    # random_seed = 4000
    torch.manual_seed(random_seed)
    ##### Define Scaling ###########
    X_low =  np.array([0.0, 0.0, 0.0, 0.0])#, dtype=np.float32)   # Lower bounds for [time, x, y]
    X_high =  np.array([laser.time_final, laser.L, laser.W, laser.H])#, dtype=np.float32)   # Higher bounds for [time, x, y]
    scaling_fac_tr = 2.0 / (X_high - X_low)
    scaling_bias_tr = -2*X_low/(X_high - X_low) - 1
    # scaling_fac_tr = 1.0 / (X_high - X_low)
    # scaling_bias_tr = -X_low/(X_high - X_low)
    
    # p_low = 0.
    # p_high = laser.P0*coeff  #/ (rho*cp*C_T*1e-9)
    # p_high = laser.L  #/ (rho*cp*C_T*1e-9)

    # for sine functions
    # p_low = 130.0
    # p_high = 175.0

    # for gp functions
        
    if not is_single_track:
        p_low = 0.
        p_high = laser.L
        scaling_fac_br = np.array([1.0 / (p_high - p_low)]*laser.n_sensors)
        scaling_bias_br = np.array([-p_low/(p_high - p_low)]*laser.n_sensors)
        # scaling_fac_br = np.array([2 / (p_high - p_low)]*laser.n_sensors)
        # scaling_bias_br = np.array([-2*p_low/(p_high - p_low) - 1]*laser.n_sensors)
    else:
        p_low = 120.0
        p_high = 180.0
        scaling_fac_br = np.array([1.0 / (p_high - p_low)]*laser.n_sensors_p)
        scaling_bias_br = np.array([-p_low/(p_high - p_low)]*laser.n_sensors_p)





    # scaling_fac_br = np.array([2 / (p_high - p_low)]*laser.n_sensors_p)#, dtype=np.float32)
    # scaling_bias_br = np.array([-2*p_low/(p_high - p_low) - 1]*laser.n_sensors_p)#, dtype=np.float32)
    
    # scaling_fac_br = np.array([1.0 / (p_high)]*laser.n_sensors_p) #, dtype=np.float32)
    # scaling_bias_br = np.array([0.0]*laser.n_sensors_p) #, dtype=np.float32)

    T_low = laser.T_0
    T_high = T1
    rescale_fac = np.array([(T_high - T_low)/2.0])#, dtype=np.float32)
    rescale_bias = np.array([(T_high + T_low)/2.0])#, dtype=np.float32)
    # rescale_fac = np.array([(T_high - T_low)], dtype=np.float32)
    # rescale_bias = np.array([(T_low)], dtype=np.float32)
    scaling_dict= {"scaling_fac_br1": scaling_fac_br,
                   "scaling_bias_br1": scaling_bias_br,
                   "scaling_fac_tr": scaling_fac_tr,
                   "scaling_bias_tr": scaling_bias_tr,
                   "rescale_fac": rescale_fac,
                   "rescale_bias": rescale_bias}
    ################################
    # print(coeff,  laser.P0, laser.P0*coeff)
    # exit()

    first_trained = 1
    input_file_available = 0
    continue_training = 0
    restore_model = 0

    single_path_test = 0
    multi_path_test = 0
    multi_time_test = 1
    timehistory_test = 0
    
    generate_new_test = 1
    print_profile = 0
    
    print("Number of sensors for multi path:", laser.n_sensors)
    num_n = 64 #for varP
    if not is_single_track:
        layers_br1 = [laser.n_sensors, num_n, num_n, num_n, num_n, num_n]
    else:
        layers_br1 = [laser.n_sensors_p, num_n, num_n, num_n, num_n, num_n]#, num_n, num_n]#, num_n, num_n]#, num_n, num_n]
    layers_tr = [dim+1, num_n, num_n, num_n, num_n, num_n]

    fixed_ic = True
    
    optimizer = "Both"
    # optimizer = "Adam"
    results_directory = "results/run4" # 3 or 5
    max_epoch = 701

    # optimizer = "LBFGS"
    # results_directory = "results/run4" # 3 or 5
    # max_epoch = 800
    
    num_t = 100 # 50 for varP
    points_per_t = 400 # 800 for varP
    num_ic = 25000
    num_bc = 6
    tolerance = 1.5e-4 #2.65e-4 #2.20e-4
    # tolerance = 2.5e-4 # for LBFGS
    # tolerance = #1.2e-4 # for Adam
    

    # loaded_model = "saved_model_PIDEEP_3d_multi_track1.pth"
    # print("Loading the saved model...")
    # model1 = torch.load(loaded_model)
    # model1.eval()
    # print("Model loaded!")
    # input_ic, input_ic1 = generate_initial_state_points(laser=laser, points_per_t=40000, path=path)
    # input_tr = from_numpy_to_torch(input_ic, True)
    # input_br = from_numpy_to_torch(input_ic1, False)
    # T_ini = model1.network(input_br, input_tr).to("cpu").detach().numpy()
    # print("DONE!")
    # exit()
        
    print(f"num_ic is {num_ic}")
    print(f"tolerance is {tolerance}")
    print(f"Results are written in {results_directory}")
    print(f"The optimizer is {optimizer}")
    if first_trained:
        if not input_file_available:
            print("Generating collocation points...")
            input_cp = data_gen.generate_collocation_points(laser=laser, num_t=num_t, points_per_t=points_per_t, path=path)
            print("Generating data points...")
            if fixed_ic:
                input_ic = data_gen.generate_IC_points(laser=laser, num_x=points_per_t, path=path)
                ic_val = input_ic[2]
            else:
                input_ic = data_gen.generate_IC_points_state(laser=laser, num_x=num_ic, path=path)
                ic_val = input_ic[2]
            
            print(f"Trunk input shape: {input_cp[0].shape}, Branch1 input shape: {input_cp[1].shape}, Laser loc input shape: {input_cp[2].shape}")
            print(f"Trunk input shape: {input_ic[0].shape}, Branch1 input shape: {input_ic[1].shape}, output shape: {input_ic[2].shape}")
            print("--------------------------------------------------------------")
            # print(len(input_cp[0]), len(input_cp[1]), len(input_cp[2]))
            # print(len(input_ic[0]), len(input_ic[1]))
            # print(input_cp[1])
            # print(input_cp[2])
            
            
            location = [0.0, laser.L, 0.0, laser.W, 0.0, laser.H]
            n_direction = [1, 1, 2, 2, 3, 3]
            # location = [0.0, laser.H]
            # n_direction = [1, 1]
            neu = "Neumann"
            diri = "Dirichlet"
            bc_type = [neu, neu, neu, neu, diri, neu]

            val = [T_0, T_0, T_0, T_0, T_0, T_0]
            # num_t = 100
            points_per_t = 20  #previously 20
            neumann_tr, neumann_br, bc_normals = [], [], []
            dirichlet_tr, dirichlet_br = [], []
            for i in range(num_bc):
                print(f"Generating data for the boundary condition no.{i+1} with the type {bc_type[i]}")
                if bc_type[i] == neu:
                    # type_bc --> Dirichlet = 0, Neumann = 1
                    bc = BoundaryCondition(flag=i+1, location=location[i], type_bc=1, n_direction=n_direction[i], value=1)
                    bc_info = bc.generatePoints(laser=laser, num_t=num_t, points_per_t=points_per_t, path=path, location_bc=location[i])
                    neumann_tr.append(bc_info["data"][0])
                    neumann_br.append(bc_info["data"][1])
                    bc_normals.append(bc_info["normals"])
                    print(f'Trunk input shape: {bc_info["data"][0].shape}, Branch1 input shape: {bc_info["data"][1].shape}, normals shape: {bc_info["normals"].shape}')
                    print("--------------------------------------------------------------")
                else:
                    bc = BoundaryCondition(flag=i+1, location=location[i], type_bc=0, n_direction=n_direction[i], value=val[i])
                    bc_info = bc.generatePoints(laser=laser, num_t=num_t, points_per_t=points_per_t, path=path, location_bc=location[i])
                    dirichlet_tr.append(bc_info["data"][0])
                    dirichlet_br.append(bc_info["data"][1])
                    print(f'Trunk input shape: {bc_info["data"][0].shape}, Branch1 input shape: {bc_info["data"][1].shape}')
                    print("--------------------------------------------------------------")
            neumann_tr = np.array(neumann_tr).reshape(-1, 4)
            if not is_single_track:
                neumann_br = np.array(neumann_br).reshape(-1, laser.n_sensors)
            else:
                neumann_br = np.array(neumann_br).reshape(-1, laser.n_sensors_p)
            bc_normals = np.array(bc_normals).reshape(-1, 4)
            dirichlet_tr = np.array(dirichlet_tr).reshape(-1, 4)
            if not is_single_track:
                dirichlet_br = np.array(dirichlet_br).reshape(-1, laser.n_sensors)
            else:
                dirichlet_br = np.array(dirichlet_br).reshape(-1, laser.n_sensors_p)
            print(f"Total Neumann BC Trunk and Branch input and normals shapes are: {neumann_tr.shape, neumann_br.shape, bc_normals.shape}")
            print(f"Total Dirichlet BC Trunk and Branch input shapes are: {dirichlet_tr.shape, dirichlet_br.shape}")
            
            #### Shuffling #######
            # np.random.seed(128)
            # permutation_cp = np.random.permutation(len(input_cp[0]))
            # permutation_ic = np.random.permutation(len(input_ic[0]))
            # perm_bc_neu = np.random.permutation(len(neumann_tr))
            # perm_bc_diri = np.random.permutation(len(dirichlet_tr))

            # input_cp[0] = input_cp[0][permutation_cp]
            # input_cp[1] = input_cp[1][permutation_cp]
            # input_cp[2] = input_cp[2][permutation_cp]
            # input_ic[0] = input_ic[0][permutation_ic]
            # input_ic[1] = input_ic[1][permutation_ic]
            # neumann_tr = neumann_tr[perm_bc_neu]
            # neumann_br = neumann_br[perm_bc_neu]
            # bc_normals = bc_normals[perm_bc_neu]
            # dirichlet_tr = dirichlet_tr[perm_bc_diri]
            # dirichlet_br = dirichlet_br[perm_bc_diri]
            ######################
            
            
            bc_data = [neumann_tr, neumann_br, bc_normals, dirichlet_tr, dirichlet_br]
            
            # print(bc_normals)
                    
            # write_hdf_cp(filename="inputData_temp_2d", cp_points=input_cp, ic_points=input_ic, data_points=bc_dict, parameters=1.0)
        else:
            print("Reading input data from file...")
            # input_cp, input_ic, bc_dict = read_hdf_cp("inputData_temp_2d", num_data=num_bc)
        
        # input_cp = torch.from_numpy(input_cp).float().to(device)
        # input_ic = torch.from_numpy(input_ic).float().to(device)

        print("Training is performed on", device)
        # exit()
        saved_model = results_directory + "/saved_model_PIDEEP_3d_multi_track.pth"
        model = Diffusion(input_tr=input_cp[0], input_br=input_cp[1], input_c_laser=input_cp[2],
                          ic_input_tr=input_ic[0], ic_input_br=input_ic[1],
                          bc_data=bc_data, ic_value=ic_val, layers_tr=layers_tr, layers_br1=layers_br1,
                          scaling_dict=scaling_dict)
        print("Training starts on", device)
        start_time = time.perf_counter()
        if optimizer == "Adam":
            model.set_optimizer(optimizer=optimizer, lr=1.0e-3)
            l_pde, l_data, l_bc = model.train_netwrok(num_epochs=max_epoch, output_freq=100, tolerance=tolerance) # For Adam
        if optimizer == "LBFGS":
            model.set_optimizer(optimizer=optimizer, lr=1.0)
            l_pde, l_data, l_bc = model.train_netwrok(num_epochs=max_epoch, output_freq=10, tolerance=tolerance)  # For LBFGS
        if optimizer == "Both":
            model.set_optimizer(optimizer="Adam", lr=1.0e-3)
            l_pde, l_data, l_bc = model.train_netwrok(num_epochs=max_epoch, output_freq=100, tolerance=tolerance) # For Adam
            model.set_optimizer(optimizer="LBFGS", lr=1.0)
            l_pde, l_data, l_bc = model.train_netwrok(num_epochs=max_epoch, output_freq=10, tolerance=tolerance)  # For LBFGS

        end_time = time.perf_counter()
        torch.save(model, saved_model)
        print("The training took {} seconds!".format(end_time-start_time))

    if continue_training:
        print("The training continues...")
        saved_model = results_directory + "/saved_model_PIDEEP_3d_multi_track.pth"
        load_model = saved_model
        model = torch.load(load_model)
        start_time = time.perf_counter()
        if optimizer == "Adam":
            model.set_optimizer(optimizer=optimizer, lr=1.0e-3)
            l_pde, l_data, l_bc = model.train_netwrok(num_epochs=max_epoch, output_freq=100, tolerance=tolerance) # For Adam
        if optimizer == "LBFGS":
            model.set_optimizer(optimizer=optimizer, lr=1.0)
            l_pde, l_data, l_bc = model.train_netwrok(num_epochs=max_epoch, output_freq=10, tolerance=tolerance)  # For LBFGS
        if optimizer == "Both":
            model.set_optimizer(optimizer="Adam", lr=1.0e-3)
            l_pde, l_data, l_bc = model.train_netwrok(num_epochs=max_epoch, output_freq=100, tolerance=tolerance) # For Adam
            model.set_optimizer(optimizer="LBFGS", lr=1.0)
            l_pde, l_data, l_bc = model.train_netwrok(num_epochs=max_epoch, output_freq=10, tolerance=tolerance)  # For LBFGS

        torch.save(model, saved_model)
        end_time = time.perf_counter()
        print("The training took {} seconds!".format(end_time-start_time))

    if restore_model:
        print("Loading the saved model...")
        saved_model = results_directory + "/saved_model_PIDEEP_3d_multi_track.pth"
        load_model = saved_model
        model = torch.load(load_model, weights_only=False)
        model.eval()
        print("Model loaded!")

    if first_trained or continue_training:
        epoch = np.linspace(1, len(l_pde), len(l_pde))
        l_pde = np.array(l_pde)
        l_data = np.array(l_data)
        l_bc = np.array(l_bc)
        fig, ax = plt.subplots()
        ax.plot(epoch, l_pde, label="Loss_PDE")
        # ax.set_title("l_pde")
        # plt.show()
        ax.plot(epoch, l_data, label="Loss_Data")
        # ax.set_title("l_data")
        # plt.show()
        ax.plot(epoch, l_bc, label="Loss_BC")
        plt.legend(loc="upper right")
        plt.yscale('log')
        # ax.set_title("l_bc")
        # plt.show()
        # exit()
    
    # Testing
    
    if single_path_test:
        outfile = "output_PIDeep_single_path"
        path = path_test[0]
        print("Creating input data for testing...")
        n_grids = 100

        if generate_new_test:
            input_tr, input_br = data_gen.generate_test_points_plane(laser=laser, time=0.99*laser.time_final, path=path, location=laser.H, plane="xy", grid=grid)
            io_utils.write_hdf_testdata(filename="testdata_xy_temp", trunk_net=input_tr, branch_net=input_br)
        else:
            input_tr, input_br = io_utils.read_hdf_testdata(filename="testdata_xy_temp")
        input_tr = helpers.from_numpy_to_torch(input_tr, True) #torch.from_numpy(input_tr).to(device)
        input_br = helpers.from_numpy_to_torch(input_br, False) #torch.from_numpy(input_br).to(device)
        print("Starting evaluating...")
        u_pred = model.network(input_br, input_tr).to("cpu")
        print("Evaluation ends!")
        u_pred = u_pred.detach()
        u_pred = u_pred.numpy().reshape(101, -1) # Use it when using generate_test_points_plane
        u_pred = u_pred.reshape((u_pred.shape[0], u_pred.shape[1], 1))
        u_pred = u_pred*C_T
        print(u_pred.shape)
        io_utils.write_hdf_2d(filename=outfile, data=u_pred, laser=laser, plane="xy", grid=grid)

        # T =  u_pred
        # T_melt = 1618
        # dx = laser.L / 200
        # dy = laser.H / 50
        # mp_depth = 0.0
        # for i in range(T.shape[0]-1):
        #     if T[i,-1,-1] < T_melt and T[i+1,-1,-1] > T_melt:
        #         mp_low = i*dx + dx*(T_melt - T[i,-1,-1]) / (T[i+1,-1,-1] - T[i,-1,-1])
        #     if T[i,-1,-1] > T_melt and T[i+1,-1,-1] < T_melt:
        #         mp_high = i*dx + dx*(T_melt - T[i,-1,-1]) / (T[i+1,-1,-1] - T[i,-1,-1])
        #     for j in range(T.shape[1]-1):
        #         if T[i,j,-1] < T_melt and T[i,j+1,-1] > T_melt:
        #             depth_temp = H - (j*dy + dy*(T_melt -T[i,j,-1]) / (T[i,j+1,-1] - T[i,j,-1]))
        #             if depth_temp > mp_depth: mp_depth = depth_temp
        
        # print(f"Melt pool starting X: {mp_low}")
        # print(f"Melt pool ending X: {mp_high}")
        # print(f"Melt pool length: {mp_high - mp_low}")
        # print(f"Melt pool depth: {mp_depth*1e3}")

    if multi_path_test:
        outfile_xy = results_directory + "/output_PIDeep_multi_track_xy"
        outfile_xz = results_directory + "/output_PIDeep_multi_track_xz"
        if os.path.exists(outfile_xy+".h5"):
            os.remove(outfile_xy+".h5")
        if os.path.exists(outfile_xz+".h5"):
            os.remove(outfile_xz+".h5")
        # time1 = 2*0.00495
        time1=0.99*laser.time_final
        # time1=5*0.99*laser.time_final/6.0
        # locations = [0.5, 0.725, 0.95]
        locations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        # locations = [0.5 + laser.h_spacing_test[0], 0.5 + laser.h_spacing_test[1], 0.5 + laser.h_spacing_test[2],
        #              0.5 + laser.h_spacing_test[3], 0.5 + laser.h_spacing_test[4], 0.5 + laser.h_spacing_test[5]]
        # locations = [0.95, 0.725, 0.95, 0.5, 0.725, 0.5]
        # locations = [0.95, 0.725, 0.95, 0.5, 0.725, 0.5]
        # path1 = ([0, 1, 2]) --> 1
        # path2 = ([0, 2, 1]) --> 2
        # path3 = ([1, 0, 2]) --> 3
        # path4 = ([1, 2, 0]) --> 4
        # path5 = ([2, 0, 1]) --> 5
        # path6 = ([2, 1, 0]) --> 6
        for p in range(len(path_test)):
            path = path_test[p]
            print(f"Creating input data for testing path{p+1}...")
            n_grids = 100

            if generate_new_test:
                # time=0.99*laser.time_final # befor changing to time=time1
                input_tr_xy, input_br_xy = data_gen.generate_test_points_plane(laser=laser, time=time1, path=path, path_id=p, location=laser.H, plane="xy", grid=grid)
                input_tr_xz, input_br_xz = data_gen.generate_test_points_plane(laser=laser, time=time1, path=path, path_id=p, location=locations[p], plane="xz", grid=grid)
                # write_hdf_testdata(filename="testdata_xy_temp", trunk_net=input_tr, branch_net=input_br)
            else:
                input_tr, input_br = io_utils.read_hdf_testdata(filename="testdata_xy_temp")
            

            u_pred = data_gen.run_model(model, input_tr_xy, input_br_xy, grid)
            io_utils.write_hdf_2d_path(filename=outfile_xy, parameters=1, data=u_pred, grid=grid, laser=laser, path_group_name=f"path_{p+1}", data_group_name=f"ts")

            u_pred = data_gen.run_model(model, input_tr_xz, input_br_xz, grid)
            io_utils.write_hdf_2d_path(filename=outfile_xz, parameters=1, data=u_pred, grid=grid, laser=laser, path_group_name=f"path_{p+1}", data_group_name=f"ts")


            # T =  u_pred
            # T_melt = 1618
            # dx = laser.L / 200
            # dy = laser.H / 50
            # mp_depth = 0.0
            # mp_low = 0.0
            # mp_high = 0.0
            # for i in range(T.shape[0]-1):
            #     if T[i,-1,-1] < T_melt and T[i+1,-1,-1] > T_melt:
            #         mp_low = i*dx + dx*(T_melt - T[i,-1,-1]) / (T[i+1,-1,-1] - T[i,-1,-1])
            #     if T[i,-1,-1] > T_melt and T[i+1,-1,-1] < T_melt:
            #         mp_high = i*dx + dx*(T_melt - T[i,-1,-1]) / (T[i+1,-1,-1] - T[i,-1,-1])
            #     for j in range(T.shape[1]-1):
            #         if T[i,j,-1] < T_melt and T[i,j+1,-1] > T_melt:
            #             depth_temp = H - (j*dy + dy*(T_melt -T[i,j,-1]) / (T[i,j+1,-1] - T[i,j,-1]))
            #             if depth_temp > mp_depth: mp_depth = depth_temp
            
            # print(f"Melt pool starting X: {mp_low}")
            # print(f"Melt pool ending X: {mp_high}")
            # print(f"Melt pool length: {mp_high - mp_low}")
            # print(f"Melt pool depth: {mp_depth*1e3}")
   
    if multi_time_test:
        # times = np.linspace(start=0.0, stop=laser.time_final, num=6)
        times = [0.002, 0.004, 0.006, 0.008, 0.01]
        outfile_xy = results_directory + "/output_PIDeep_multi_track_xy"
        outfile_xz = results_directory + "/output_PIDeep_multi_track_xz"
        if os.path.exists(outfile_xy+".h5"):
            os.remove(outfile_xy+".h5")
        if os.path.exists(outfile_xz+".h5"):
            os.remove(outfile_xz+".h5")
        
        # locations = [0.5, 0.725, 0.95]
        locations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        # locations = [0.5 + laser.h_spacing_test[0], 0.5 + laser.h_spacing_test[1], 0.5 + laser.h_spacing_test[2],
        #              0.5 + laser.h_spacing_test[3], 0.5 + laser.h_spacing_test[4], 0.5 + laser.h_spacing_test[5]]
        # locations = [0.95, 0.725, 0.95, 0.5, 0.725, 0.5]
        # locations = [0.95, 0.725, 0.95, 0.5, 0.725, 0.5]
        # path1 = ([0, 1, 2]) --> 1
        # path2 = ([0, 2, 1]) --> 2
        # path3 = ([1, 0, 2]) --> 3
        # path4 = ([1, 2, 0]) --> 4
        # path5 = ([2, 0, 1]) --> 5
        # path6 = ([2, 1, 0]) --> 6
        for p in range(len(path_test)):
            path = path_test[p]
            print(f"Creating input data for testing path{p+1}...")
            n_grids = 100

            if generate_new_test:
                # time=0.99*laser.time_final # befor changing to time=time1
                input_tr_xy, input_br_xy = data_gen.generate_test_points_plane(laser=laser, time=times[p], path=path, path_id=p, location=laser.H, plane="xy", grid=grid)
                input_tr_xz, input_br_xz = data_gen.generate_test_points_plane(laser=laser, time=times[p], path=path, path_id=p, location=locations[p], plane="xz", grid=grid)
                # write_hdf_testdata(filename="testdata_xy_temp", trunk_net=input_tr, branch_net=input_br)
            else:
                input_tr, input_br = io_utils.read_hdf_testdata(filename="testdata_xy_temp")
            

            u_pred = data_gen.run_model(model, input_tr_xy, input_br_xy, grid)
            io_utils.write_hdf_2d_path(filename=outfile_xy, parameters=1, data=u_pred, grid=grid, laser=laser, path_group_name=f"path_{p+1}", data_group_name=f"ts")

            u_pred = data_gen.run_model(model, input_tr_xz, input_br_xz, grid)
            io_utils.write_hdf_2d_path(filename=outfile_xz, parameters=1, data=u_pred, grid=grid, laser=laser, path_group_name=f"path_{p+1}", data_group_name=f"ts")

            # T =  u_pred
            # T_melt = 1618
            # dx = laser.L / 200
            # dy = laser.H / 50
            # mp_depth = 0.0
            # for i in range(T.shape[0]-1):
            #     if T[i,-1,-1] < T_melt and T[i+1,-1,-1] > T_melt:
            #         mp_low = i*dx + dx*(T_melt - T[i,-1,-1]) / (T[i+1,-1,-1] - T[i,-1,-1])
            #     if T[i,-1,-1] > T_melt and T[i+1,-1,-1] < T_melt:
            #         mp_high = i*dx + dx*(T_melt - T[i,-1,-1]) / (T[i+1,-1,-1] - T[i,-1,-1])
            #     for j in range(T.shape[1]-1):
            #         if T[i,j,-1] < T_melt and T[i,j+1,-1] > T_melt:
            #             depth_temp = H - (j*dy + dy*(T_melt -T[i,j,-1]) / (T[i,j+1,-1] - T[i,j,-1]))
            #             if depth_temp > mp_depth: mp_depth = depth_temp
            
            # print(f"Melt pool starting X: {mp_low}")
            # print(f"Melt pool ending X: {mp_high}")
            # print(f"Melt pool length: {mp_high - mp_low}")
            # print(f"Melt pool depth: {mp_depth*1e3}")

    if timehistory_test:
        outfile = "output_PIDeep_timehistory"
        num_timesteps = 10
        if os.path.exists(outfile+".h5"):
            # Delete the file
            os.remove(outfile+".h5")
        for p in range(len(path_test)):
            path = path_test[p]
            print(f"Creating input data for testing path{p+1}...")
            for ts in range(num_timesteps):
                n_grids = 100
                time_ = (ts+1)*0.99*laser.time_final / num_timesteps
                if generate_new_test:
                    input_tr, input_br = data_gen.generate_test_points_timehistory(laser=laser, time=time_, path=path, location=laser.H, plane="xy", n_grids=n_grids)
                    # input_tr, input_br = generate_test_points(laser=laser, time=time_, path=path, location=laser.H, plane="xy", n_grids=n_grids)
                    io_utils.write_hdf_testdata(filename="testdata_xy_temp", trunk_net=input_tr, branch_net=input_br)
                else:
                    input_tr, input_br = io_utils.read_hdf_testdata(filename="testdata_xy_temp")
                input_tr = torch.from_numpy(input_tr).float().to(device)
                input_br = torch.from_numpy(input_br).float().to(device)
                print("Starting evaluating...")
                u_pred = model.network(input_br, input_tr).to("cpu")
                print("Evaluation ends!")
                u_pred = u_pred.detach()
                u_pred = u_pred.numpy().reshape(201, -1) # Use it when using generate_test_points_plane
                u_pred = u_pred.reshape((u_pred.shape[0], u_pred.shape[1], 1))
                u_pred = u_pred*C_T
                print(u_pred.shape)
                io_utils.write_hdf_2d_path(filename=outfile, parameters=1, data=u_pred, laser=laser, path_group_name=f"path_{p+1}", data_group_name=f"ts_{ts+1}")
    
    
    
    print("Program finished!")    