import numpy as np
import torch
import torch.nn as nn
import time
from matplotlib import pyplot as plt
# import vtk
# from vtk.util.numpy_support import numpy_to_vtk
import h5py
import os
from config import device, default_dtype
from applications.smart_melt.network import Temperature_distribution
from utils import helpers


T_ref = 2000.0
class H5explorer:
    def __init__(self):
        self.path = []
        self.timesteps = []
        self.data_address = []
        self.geo_dic = {}
    
    def __call__(self, name, obj):
        if isinstance(obj, h5py.Group):
            if "path" in name:
                for subgroup_name, item in obj.items():
                    if (name not in self.path) and (isinstance(item, h5py.Group)): self.path.append(name)
                    if "Temp" not in subgroup_name:
                        self.timesteps.append(int(subgroup_name))
                    if isinstance(item, h5py.Dataset):           
                        
                        self.data_address.append(f"{name}/{subgroup_name}")
            if "Grid" in name:
                self.geo_dic["dims"] = obj.attrs["dims"]
                self.geo_dic["spacing"] = obj.attrs["spacing"]

            if "simulation" in name:
                self.geo_dic["dt_coeff"] = obj.attrs["dt_coeff"]
                

def generate_training_data_old(source, num_path, num_profile_per_path, num_sample_per_profile):
    np.random.seed(128)
    read_file_h5 = "output_path.h5"
    explore = H5explorer()
    with h5py.File(read_file_h5, "r") as h5file:
        h5file.visititems(explore)
        number_ts = len(explore.timesteps)
        data = np.array([h5file[explore.data_address[i]] for i in range(number_ts)])
        x_coord = np.array(h5file["structuredGrid/X"])
        y_coord = np.array(h5file["structuredGrid/Y"])

        total_profile_per_path = int(number_ts / len(explore.path))   # Number of total time steps per path
        
        num_samples = num_path * num_profile_per_path * num_sample_per_profile
        coord_value = np.full((num_samples, 2), np.NAN) # for x and y coordinates
        temperature_label = np.full((num_samples, 1), np.NAN) # labeld temperature associated with coordinates x and y
        surface_profile = np.empty((0, x_coord.shape[0])) # Stores surface temperatures for each training set

        counter = 0
        for path in range(num_path):
            profile_index = path*total_profile_per_path + np.random.choice(total_profile_per_path, size=num_profile_per_path, replace=False)
            for profile in range(num_profile_per_path):
                data_address = explore.data_address[profile_index[profile]]
                data = h5file[data_address]
                profile_temp = np.tile(data[0, -1, :], (num_sample_per_profile, 1)) # data shape: [0, y_index, x_index]
                surface_profile = np.vstack((surface_profile, profile_temp))
                total_nodes = data.shape[0]*data.shape[1]*data.shape[2]
                rand_nodes = np.random.choice(total_nodes, size=num_sample_per_profile, replace=False)
                all_indices = list(np.ndindex(data.shape))
                for i in range(num_sample_per_profile):
                    coord_value[counter, 0] = x_coord[all_indices[rand_nodes[i]][2]]
                    coord_value[counter, 1] = y_coord[all_indices[rand_nodes[i]][1]]
                    temperature_label[counter, 0] = data[all_indices[rand_nodes[i]][0], all_indices[rand_nodes[i]][1], 
                                                all_indices[rand_nodes[i]][2]]  # data shape: [0, y_index, x_index]
                    counter += 1
        print(surface_profile.shape, coord_value.shape, temperature_label.shape)
        
        return surface_profile, coord_value, temperature_label


def generate_test_data(source, total_path, used_path, num_profile_per_path, num_sample_per_profile):
    np.random.seed(128)
    read_file_h5 = "output_path.h5"
    explore = H5explorer()
    with h5py.File(read_file_h5, "r") as h5file:
        h5file.visititems(explore)
        number_ts = len(explore.timesteps)
        data = np.array([h5file[explore.data_address[i]] for i in range(number_ts)])
        x_coord = np.array(h5file["structuredGrid/X"])
        y_coord = np.array(h5file["structuredGrid/Y"])

        total_profile_per_path = int(number_ts / len(explore.path))   # Number of total time steps per path
        
        num_samples = (total_path - used_path) * num_profile_per_path * num_sample_per_profile
        coord_value = np.full((num_samples, 2), np.NAN) # for x and y coordinates
        temperature_label = np.full((num_samples, 1), np.NAN) # labeld temperature associated with coordinates x and y
        surface_profile = np.empty((0, x_coord.shape[0])) # Stores surface temperatures for each training set

        counter = 0
        for path in range(used_path, total_path):
            profile_index = path*total_profile_per_path + np.random.choice(total_profile_per_path, size=num_profile_per_path, replace=False)
            for profile in range(num_profile_per_path):
                data_address = explore.data_address[profile_index[profile]]
                data = h5file[data_address]
                profile_temp = np.tile(data[0, -1, :], (num_sample_per_profile, 1)) # data shape: [0, y_index, x_index]
                surface_profile = np.vstack((surface_profile, profile_temp))
                total_nodes = data.shape[0]*data.shape[1]*data.shape[2]
                rand_nodes = np.random.choice(total_nodes, size=num_sample_per_profile, replace=False)
                all_indices = list(np.ndindex(data.shape))
                for i in range(num_sample_per_profile):
                    coord_value[counter, 0] = x_coord[all_indices[rand_nodes[i]][2]]
                    coord_value[counter, 1] = y_coord[all_indices[rand_nodes[i]][1]]
                    temperature_label[counter, 0] = data[all_indices[rand_nodes[i]][0], all_indices[rand_nodes[i]][1], 
                                                all_indices[rand_nodes[i]][2]]  # data shape: [0, y_index, x_index]
                    counter += 1
        print(surface_profile.shape, coord_value.shape, temperature_label.shape)
        
        return surface_profile, coord_value, temperature_label


def generate_test_data_section(source, index_section):
    np.random.seed(128)
    read_file_h5 = "output_path_test.h5"
    explore = H5explorer()
    with h5py.File(read_file_h5, "r") as h5file:
        h5file.visititems(explore)
        number_ts = len(explore.timesteps)
        data = np.array([h5file[explore.data_address[i]] for i in range(number_ts)])
        x_coord = np.array(h5file["structuredGrid/X"])
        y_coord = np.array(h5file["structuredGrid/Y"])

        temp = h5file["path2/19000/Temperature"]

        surface_profile = np.tile(temp[0, -1, :], (x_coord.shape[0], 1)) # data shape: [0, y_index, x_index]

        coord_value = np.full((x_coord.shape[0], 2), np.NAN) # for x and y coordinates
        temperature_label = np.full((x_coord.shape[0], 1), np.NAN) # labeld temperature associated with coordinates x and y

        for i in range(x_coord.shape[0]):
            coord_value[i, 0] = x_coord[i]
            coord_value[i, 1] = y_coord[index_section]
            temperature_label[i, 0] = temp[0, index_section, i]  # data shape: [0, y_index, x_index]

                
        return surface_profile, coord_value, temperature_label




def generate_training_data(scenarios, skip):
    read_file_h5 = "/home/hessafar/codes/cuda_laser/Results_datadriven/laser_distance.h5"
    group_x = "structuredGrid/X"
    group_y = "structuredGrid/Y"
    temperature_list = []
    distance_list = []
    num_sc = len(scenarios)
    ######### Reading data from the related file ############
    with h5py.File(read_file_h5, "r") as h5file:
        x_coord = np.array(h5file[group_x])
        y_coord = np.array(h5file[group_y])
        laser_time = h5file["time"][:]
        laser_time = laser_time[::40]
        if skip != 0:
            x_coord = x_coord[0::skip]
            y_coord = y_coord[0::skip]
        for i in range(num_sc):
            index = scenarios[i]
            group_T = f"path{index}/end" + f"/Temperature_xy"
            temperature = np.array(h5file[group_T]) / T_ref
            temperature = temperature.reshape(temperature.shape[1], temperature.shape[2])
            if skip != 0:
                temperature = temperature[0::skip, 0::skip]
            temperature_list.append(temperature)
            distance = h5file[f'path_{index}'][:]
            distance = distance[::40]
            distance_list.append(distance)
    #########################################################

    imax = x_coord.shape[0]
    jmax = y_coord.shape[0]
    input_tr = []
    input_br = []
    output = []
    for sc in range(num_sc):
        distance_per_scenario = np.tile(distance_list[sc], (imax*jmax, 1))
        input_br.append(distance_per_scenario)
        for i in range(imax):
            for j in range(jmax):
                input_tr.append([x_coord[i], y_coord[j]])
                output.append(temperature_list[sc][i, j])
    
    input_tr = np.array(input_tr)
    input_br = np.array(input_br).reshape(-1,distance_list[0].shape[0])
    output = np.array(output).reshape(-1,1)

    return input_tr, input_br, output



def generate_training_data_transient(scenarios, times, skip):
    read_file_h5 = "/home/hessafar/codes/cuda_laser/Results_datadriven/laser_distance.h5"
    transient_source = "/home/hessafar/codes/cuda_laser/Results_datadriven/output_path.h5"
    group_x = "structuredGrid/X"
    group_y = "structuredGrid/Y"
    temperature_data = []
    
    distance_list = []
    num_sc = len(scenarios)
    time_steps = [f"{time:0.6f}" for time in times]
    
    ######### Reading data from the related file for transient temperature and grid points ############
    with h5py.File(transient_source, "r") as h5file:
        x_coord = np.array(h5file[group_x])
        y_coord = np.array(h5file[group_y])
        if skip != 0:
            x_coord = x_coord[0::skip]
            y_coord = y_coord[0::skip]
        for i in range(num_sc):
            scenario_data = []
            for ts in time_steps:
                index = scenarios[i]
                group_T = f"path{index}/{ts}/Temperature_xy"
                temperature = np.array(h5file[group_T]) / T_ref
                temperature = temperature.reshape(temperature.shape[1], temperature.shape[2])
                if skip != 0:
                    temperature = temperature[0::skip, 0::skip]
                scenario_data.append(temperature)
            temperature_data.append(scenario_data)
    ####################################################################################################
    
    ### IMPORTANT: The temperature_data will be in the form of temperature_data[scenario][time] ########
    
    ######### Reading data from the related file for laser properties ###############
    with h5py.File(read_file_h5, "r") as h5file:
        laser_time = h5file["time"][:]
        laser_time = laser_time[::40]
        for i in range(num_sc):
            index = scenarios[i]
            distance = h5file[f'path_{index}'][:]
            distance = distance[::40]
            distance_list.append(distance)
    #####################################################################################

    imax = x_coord.shape[0]
    jmax = y_coord.shape[0]
    input_tr = []
    input_br = []
    output = []
    for sc in range(num_sc):
        distance_per_scenario = np.tile(distance_list[sc], (imax*jmax*len(time_steps), 1))
        input_br.append(distance_per_scenario)
        for idx, time in enumerate(time_steps):
            for i in range(imax):
                for j in range(jmax):
                    input_tr.append([float(time), x_coord[i], y_coord[j]])
                    output.append(temperature_data[sc][idx][i, j])
    
    input_tr = np.array(input_tr)
    input_br = np.array(input_br).reshape(-1,distance_list[0].shape[0])
    output = np.array(output).reshape(-1,1)

    return input_tr, input_br, output

def generate_training_data_power_transient(scenarios, times, skip):
    # file_power = "/home/hessafar/codes/cuda_laser/gp_functions_40samples_sm.npz"
    file_power = "/home/hessafar/codes/cuda_laser/square_wave_40samples.npz"
    transient_source = "/home/hessafar/codes/cuda_laser/Results_datadriven/output_path.h5"
    group_x = "structuredGrid/X"
    group_y = "structuredGrid/Y"
    temperature_data = []
    
    num_sc = len(scenarios)
    time_steps = [f"{time:0.6f}" for time in times]
    
    ######### Reading data from the related file for transient temperature and grid points ############
    with h5py.File(transient_source, "r") as h5file:
        x_coord = np.array(h5file[group_x])
        y_coord = np.array(h5file[group_y])
        if skip != 0:
            x_coord = x_coord[0::skip]
            y_coord = y_coord[0::skip]
        for i in range(num_sc):
            scenario_data = []
            for ts in time_steps:
                index = scenarios[i]
                group_T = f"path{index}/{ts}/Temperature_xy"
                temperature = np.array(h5file[group_T]) / T_ref
                temperature = temperature.reshape(temperature.shape[1], temperature.shape[2])
                if skip != 0:
                    temperature = temperature[0::skip, 0::skip]
                scenario_data.append(temperature)
            temperature_data.append(scenario_data)
    ####################################################################################################
    
    ### IMPORTANT: The temperature_data will be in the form of temperature_data[scenario][time] ########
    
    ######### Reading data from the related file for laser properties ###############
    data = np.load(file_power)
    modulated_power = data["y_value"]
    #####################################################################################

    imax = x_coord.shape[0]
    jmax = y_coord.shape[0]
    input_tr = []
    input_br = []
    output = []
    for sc in range(num_sc):
        distance_per_scenario = np.tile(modulated_power[sc], (imax*jmax*len(time_steps), 1))
        input_br.append(distance_per_scenario)
        for idx, time in enumerate(time_steps):
            for i in range(imax):
                for j in range(jmax):
                    input_tr.append([float(time), x_coord[i], y_coord[j]])
                    output.append(temperature_data[sc][idx][i, j])
    
    input_tr = np.array(input_tr)
    input_br = np.array(input_br).reshape(-1,modulated_power[0].shape[0])
    output = np.array(output).reshape(-1,1)

    return input_tr, input_br, output

def generate_training_data_power_modulation(scenarios, times, skip):
    # file_power = "/home/hessafar/codes/cuda_laser/gp_functions_40samples_sm.npz"
    file_power = "/home/hessafar/codes/cuda_laser/modulations_40samples.npz"
    transient_source = "/home/hessafar/codes/cuda_laser/Results_datadriven/output_path_long.h5"
    group_x = "structuredGrid/X"
    group_y = "structuredGrid/Y"
    temperature_data = []
    
    num_sc = len(scenarios)
    time_steps = [f"{time:0.6f}" for time in times]
    
    ######### Reading data from the related file for transient temperature and grid points ############
    with h5py.File(transient_source, "r") as h5file:
        x_coord = np.array(h5file[group_x])
        y_coord = np.array(h5file[group_y])
        if skip != 0:
            x_coord = x_coord[0::skip]
            y_coord = y_coord[0::skip]
        for i in range(num_sc):
            scenario_data = []
            for ts in time_steps:
                index = scenarios[i]
                group_T = f"path{index}/{ts}/Temperature_xy"
                temperature = np.array(h5file[group_T]) / T_ref
                temperature = temperature.reshape(temperature.shape[1], temperature.shape[2])
                if skip != 0:
                    temperature = temperature[0::skip, 0::skip]
                scenario_data.append(temperature)
            temperature_data.append(scenario_data)
    ####################################################################################################
    
    ### IMPORTANT: The temperature_data will be in the form of temperature_data[scenario][time] ########
    
    ######### Reading data from the related file for laser properties ###############
    data = np.load(file_power)
    p_overshoot = data["p_overshoot"]
    duration = data["duration"]
    t_0 = data["t_0"]
    print(p_overshoot.shape, duration.shape, t_0.shape)
    #####################################################################################

    imax = x_coord.shape[0]
    jmax = y_coord.shape[0]
    input_tr = []
    input_br = []
    output = []
    for sc_id, sc in enumerate(scenarios):
        # print(sc)
        val = np.array([p_overshoot[sc], duration[sc], t_0[sc]])
        distance_per_scenario = np.tile(val, (imax*jmax*len(time_steps), 1))
        input_br.append(distance_per_scenario)
        for idx, time in enumerate(time_steps):
            for i in range(imax):
                for j in range(jmax):
                    input_tr.append([float(time), x_coord[i], y_coord[j]])
                    output.append(temperature_data[sc_id][idx][i, j])
    
    input_tr = np.array(input_tr)
    input_br = np.array(input_br).reshape(-1,3) # (3 corresponds with 3 modulation parameters (p_overshoot, duration, t_0))
    output = np.array(output).reshape(-1,1)

    return input_tr, input_br, output

def generate_training_data_power_modulation_meltpool(scenarios, modulations, melt_pool_data):
      
    ######### Reading data from the related file for transient temperature and grid points ############
    with h5py.File(melt_pool_data, "r") as h5file:
        group_times = f"times/time"
        times = np.array(h5file[group_times])
        
        scenario_data = []
        for i, sc in enumerate(scenarios):
            group_sc = f"path{sc}/height"
            mp_depth = np.array(h5file[group_sc]) / 0.25
            # group_sc = f"path{sc}/dheight"
            # mp_depth = np.array(h5file[group_sc]) / 0.0012
            scenario_data.append(mp_depth)
           
    ####################################################################################################
    
    ### IMPORTANT: The scenario_data is in the form of scenario_data[scenario][time] ########
    
    ######### Reading data from the related file for laser properties ###############
    data = np.load(modulations)
    p_overshoot = data["p_overshoot"]
    duration = data["duration"]
    t_0 = data["t_0"]
    # print(p_overshoot.shape, duration.shape, t_0.shape)
    #####################################################################################

    timesteps_per_sc = times.shape[0]
    input_tr = []
    input_br = []
    output = []
    for sc_id, sc in enumerate(scenarios):
        val = np.array([p_overshoot[sc-1], duration[sc-1], t_0[sc-1]])
        modulations_per_scenario = np.tile(val, timesteps_per_sc)
        input_br.append(modulations_per_scenario)
        for idx, time in enumerate(times):
            input_tr.append(time)
            output.append(scenario_data[sc_id][idx])
    
    input_tr = np.array(input_tr).reshape(-1,1)
    input_br = np.array(input_br).reshape(-1,3) # (3 corresponds with 3 modulation parameters (p_overshoot, duration, t_0))
    output = np.array(output).reshape(-1,1)
    ind = np.where(input_tr >= 0.0015)[0]
    input_tr = input_tr[ind]
    input_br = input_br[ind]
    output = output[ind]

    return input_tr, input_br, output

def generate_training_data_power_modulation_meltpool_xloc(scenarios, modulations, melt_pool_data):
      
    ######### Reading data from the related file for transient temperature and grid points ############
    with h5py.File(melt_pool_data, "r") as h5file:
        group_times = f"times/time"
        times = np.array(h5file[group_times])
        
        scenario_data = []
        x_loc_data = []
        for i, sc in enumerate(scenarios):
            group_loc = f"path{sc}/x_loc"
            temp_loc = np.array(h5file[group_loc])
            x_loc_data.append(temp_loc)
            
            # group_sc = f"path{sc}/height"
            group_sc = f"path{sc}/width"
            mp_depth = np.array(h5file[group_sc]) #/ 0.25
            
            dDepth = np.zeros_like(mp_depth)
            for j in range(1, mp_depth.shape[0]-1):
                if abs(temp_loc[j+1] - temp_loc[j-1]) > 1e-9:
                    dDepth[j] = (mp_depth[j+1] - mp_depth[j-1]) / (temp_loc[j+1] - temp_loc[j-1])
                else:
                    dDepth[j] = 0.0
            dDepth[0] = dDepth[1]
            dDepth[-1] = dDepth[-2]
            # group_sc = f"path{sc}/dheight"
            # mp_depth = np.array(h5file[group_sc]) / 0.0012
            scenario_data.append(mp_depth)
            # scenario_data.append(dDepth)

           
    ####################################################################################################
    
    ### IMPORTANT: The scenario_data is in the form of scenario_data[scenario][time] ########
    
    ######### Reading data from the related file for laser properties ###############
    data = np.load(modulations)
    # p_overshoot = data["p_overshoot"]
    # duration = data["duration"]
    # t_0 = data["t_0"]

    dP1 = data["dP1"]
    dP2 = data["dP2"]
    t0 = data["t0"]
    dt1 = data["dt1"]
    dt2 = data["dt2"]
    dt3 = data["dt2"]
    # print(p_overshoot.shape, duration.shape, t_0.shape)
    #####################################################################################

    timesteps_per_sc = times.shape[0]
    input_tr = []
    input_br = []
    output = []
    for sc_id, sc in enumerate(scenarios):
        # val = np.array([p_overshoot[sc-1], duration[sc-1], t_0[sc-1]])
        val = np.array([dP1[sc-1], dP2[sc-1], t0[sc-1], dt1[sc-1], dt2[sc-1], dt3[sc-1]])
        modulations_per_scenario = np.tile(val, timesteps_per_sc)
        input_br.append(modulations_per_scenario)
        for idx, time in enumerate(times):
            input_tr.append(x_loc_data[sc_id][idx])
            output.append(scenario_data[sc_id][idx])
    
    input_tr = np.array(input_tr).reshape(-1,1)
    input_br = np.array(input_br).reshape(-1,6) # (3 corresponds with 3 modulation parameters (p_overshoot, duration, t_0))
    output = np.array(output).reshape(-1,1)
    ind = np.where(input_tr >= 0.0012)[0]
    input_tr = input_tr[ind]
    input_br = input_br[ind]
    output = output[ind]

    return input_tr, input_br, output


def run_case():
    random_seed = 42
    print(default_dtype, device)
    helpers.set_default_dtype1(default_dtype)
    torch.manual_seed(random_seed)
    
    # file_power = "/home/hessafar/codes/cuda_laser/modulations_125samples_tri.npz"
    # transient_source = "/home/hessafar/codes/cuda_laser/Results_datadriven/meltpool_125_tri.h5"
    
    file_power = "/home/hessafar/codes/cuda_laser/modulations_125samples_6params.npz"
    # file_power = "profile_samples/modulations_125samples_6params.npz"
    transient_source = "/home/hessafar/codes/cuda_laser/Results_datadriven/meltpool_125_6params.h5"
    
    train_scenarios = np.arange(1,1024) #[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 18, 20]
    # times =  np.linspace(0.00201, 0.004, 200)
    # input_tr, input_br, output = generate_training_data(scenarios=train_scenarios, skip=0)
    # input_tr, input_br, output = generate_training_data_transient(scenarios=train_scenarios, times=times, skip=0)
    times =  [0.001, 0.002, 0.003, 0.004]
    # input_tr, input_br, output = generate_training_data_power_transient(scenarios=train_scenarios, times=times, skip=0)
    # input_tr, input_br, output = generate_training_data_power_modulation(scenarios=train_scenarios, times=times, skip=0)
    input_tr, input_br, output = generate_training_data_power_modulation_meltpool_xloc(scenarios=train_scenarios,
                                                                                  modulations=file_power,
                                                                                  melt_pool_data=transient_source)
    out_max = 1.2*(output).max()
    out_min = (output).min()
    print(f"The shape of Trunk net input: {input_tr.shape}")
    print(f"The shape of Brach net input: {input_br.shape}")
    print(f"The shape of output: {output.shape}")
    print("Maximum of the output:", out_max, out_min)
    output = output / out_max
    
    test_scenario = [36]
    times =  [0.004]
    # input_tr_test, input_br_test, output_test = generate_training_data(scenarios=test_scenario, skip=0)
    # input_tr_test, input_br_test, output_test = generate_training_data_transient(scenarios=test_scenario, times=times, skip=0)
    # input_tr_test, input_br_test, output_test = generate_training_data_power_transient(scenarios=test_scenario, times=times, skip=0)
    # input_tr_test, input_br_test, output_test = generate_training_data_power_modulation(scenarios=test_scenario, times=times, skip=0)
    input_tr_test, input_br_test, output_test = generate_training_data_power_modulation_meltpool_xloc(scenarios=test_scenario,
                                                                                                 modulations=file_power,
                                                                                                 melt_pool_data=transient_source)
    
    output_test = output_test / out_max

    # results_directory = "results/smart_melt/"
    results_directory = "results/run1/"
    tolerance = 6.051e-7
    n_sensors = 6
    num_n = 100
    layers_br = [n_sensors, num_n, num_n, num_n, num_n, num_n, num_n]#, num_n, num_n]#, num_n, num_n]
    layers_tr = [1, num_n, num_n, num_n, num_n, num_n, num_n]#, num_n, num_n]

    saved_model = results_directory + "saved_model_smartMelt.pth"

    restore_model = 0
    # first_trained = 0
    # continue_training = 0

    melt_pool = 1

    ##### Define Scaling ###########
    # X_low =  np.array([0.00201, 0.0, 0.0])   # Lower bounds for [time, x, y]
    # X_high =  np.array([0.004, 2e-3, 1e-3])   # Higher bounds for [time, x, y]
    X_low =  np.array([input_tr.min()])   # Lower bounds for [time, x, y]
    X_high =  np.array([input_tr.max()])   # Higher bounds for [time, x, y]
    scaling_fac_tr = 2.0 / (X_high - X_low)
    scaling_bias_tr = -2*X_low/(X_high - X_low) - 1
    # scaling_fac_tr = 1.0 / (X_high - X_low)
    # scaling_bias_tr = -X_low/(X_high - X_low)

    # T_low = input_br.min()
    # T_high = input_br.max()
    # scaling_fac_br = np.array([2 / (T_high - T_low)]*n_sensors)
    # scaling_bias_br = np.array([-2*T_low/(T_high - T_low) - 1]*n_sensors)
    # scaling_fac_br = np.array([1.0 / (T_high - T_low)]*n_sensors)
    # scaling_bias_br = np.array([-T_low/(T_high - T_low)]*n_sensors)

    # power_overshoot = rng.uniform(10, 40.0, size=num_samples)
    # duration = rng.uniform(0.0001, 0.0004, size=num_samples)
    # starting_time = rng.uniform(0.0025, 0.0030, size=num_samples)

    # X_low =  np.array([10.0, 0.0001, 0.0025])
    # X_high =  np.array([40.0, 0.0004, 0.0030])
    # X_low =  np.array([10.0, 0.0001, 0.0023])
    # X_high =  np.array([40.0, 0.0012, 0.0030])
    # X_low =  np.array([10.0, 2e-6, 0.0023])
    # X_high =  np.array([25.0, 2e-4, 0.0027])
    X_low =  np.array([20.0, -25.0, 0.0025, 2e-6, 2e-6, 2e-6])
    X_high =  np.array([40.0, -10.0, 0.0027, 5e-4, 5e-4, 5e-4])
    scaling_fac_br = 2.0 / (X_high - X_low)
    scaling_bias_br = -2*X_low/(X_high - X_low) - 1


    # T_low = laser.T_0
    # T_high = T1
    rescale_fac = 1.0 #np.array([(T_high - T_low)/2.0])
    rescale_bias = 1.0 #np.array([(T_high + T_low)/2.0])
    # rescale_fac = np.array([(T_high - T_low)], dtype=np.float32)
    # rescale_bias = np.array([(T_low)], dtype=np.float32)
    scaling_dict= {"scaling_fac_br": scaling_fac_br,
                    "scaling_bias_br": scaling_bias_br,
                    "scaling_fac_tr": scaling_fac_tr,
                    "scaling_bias_tr": scaling_bias_tr,
                    "rescale_fac": rescale_fac,
                    "rescale_bias": rescale_bias}
    ################################



    if not restore_model:

        model = Temperature_distribution(input_br=input_br, input_tr=input_tr, output=output,
                                input_br_test=input_br_test, input_tr_test=input_tr_test, output_test=output_test,
                                scaling_dict=scaling_dict, layers_br=layers_br, layers_tr=layers_tr)
        
        print("Training starts on", device)
        start_time = time.perf_counter()
        # optimizer = "Adam"
        # model.set_optimizer(optimizer=optimizer, lr=1.0e-3)
        # model.train_netwrok(num_epochs=10000, output_freq=100, tolerance=tolerance) # For Adam
        optimizer = "LBFGS"
        model.set_optimizer(optimizer=optimizer, lr=1.0)
        model.train_netwrok(num_epochs=1000, output_freq=10, tolerance=tolerance)  # For LBFGS
        end_time = time.perf_counter()
        print("The training took {} seconds!".format(end_time-start_time))
        torch.save(model, saved_model)


    # if continue_training:
    #     pass

    if restore_model:
        print("Loading the saved model...")
        load_model = saved_model
        model = torch.load(load_model, weights_only=False)
        model.eval()
        print("Model loaded!")
    
    # input_br_test = helpers.from_numpy_to_torch(input_br_test, False)
    # input_tr_test = helpers.from_numpy_to_torch(input_tr_test, False)
    if not melt_pool:
        T_pred = model.network(input_br_test, input_tr_test).to("cpu").detach().numpy().reshape(201,51)
        output_test = output_test.reshape(201,51)
        np.savez(results_directory+"T_pred.npz", T_pred=T_pred, T_ref=output_test)
    else:
        test_scenario = np.arange(10,20)
        for i,sc in enumerate(test_scenario):
            # file_power = "/home/hessafar/codes/cuda_laser/modulations_125samples.npz"
            # transient_source = "/home/hessafar/codes/cuda_laser/Results_datadriven/meltpool_125.h5"
            print(sc)
            input_tr_test, input_br_test, output_test = generate_training_data_power_modulation_meltpool_xloc(scenarios=[sc],
                                                                                                         modulations=file_power,
                                                                                                         melt_pool_data=transient_source)
            input_br_test = helpers.from_numpy_to_torch(input_br_test, False)
            input_tr_test = helpers.from_numpy_to_torch(input_tr_test, False)
            depth_pred = model.network(input_br_test, input_tr_test).to("cpu").detach().numpy().reshape(-1,1)
            output_test = output_test.reshape(-1,1)
            pred = depth_pred
            ref = output_test
            # pred = 0.0012*np.cumsum(depth_pred)*0.5
            # ref = 0.0012*np.cumsum(output_test)*0.5
            np.savez(results_directory+f"depth_pred_{i}.npz", depth_pred=pred, depth_ref=ref)

    print("Simulation successfully terminated!")
    
 #-------------------------------------------------------------------------------------------------------------------   
    print("Optimization begins.....")
    # transient_source = "/home/hessafar/codes/cuda_laser/Results_datadriven/meltpool_40.h5"
    with h5py.File(transient_source, "r") as h5file:
        group_times = f"times/time"
        times = np.array(h5file[group_times])
    ind = np.where(times >= 0.0015)[0]
    times = times[ind]
    times = np.linspace(0.0012, 0.0023, 100)
    dx = np.zeros_like(times)
    for i in range(1, times.shape[0]-1):
        dx[i] = times[i+1] - times[i-1]
    counter = 0

    # def compute_depth(p_overshoot, duration, t_0) -> np.ndarray:
    def compute_depth(dP1, dP2, t0, dt1, dt2, dt3) -> np.ndarray:
        # val = np.array([p_overshoot, duration, t_0])
        nonlocal counter
        counter += 1
        val = np.array([dP1, dP2, t0, dt1, dt2, dt3])
        input_br = np.tile(val, (times.shape[0],1))#.reshape(-1,3)
        input_tr = times.reshape(-1,1)
        input_br = helpers.from_numpy_to_torch(input_br, False)
        input_tr = helpers.from_numpy_to_torch(input_tr, False)
        depth_pred = model.network(input_br, input_tr).to("cpu").detach().numpy().reshape(-1,)

        
        # depth_pred[1:-1] = np.cumsum(out_max*depth_pred[1:-1]*dx[1:-1])
        # depth_pred[0] = depth_pred[1]
        # depth_pred[-1] = depth_pred[-2]

        # depth_pred = np.cumsum(0.0012*depth_pred)*0.5
        return depth_pred*out_max
    
    def objective(params):
        # a, b, c = params
        a, b, c, d, e, f = params
        depth_profile = compute_depth(a, b, c, d, e, f)
        # depth_profile = abs(depth_profile)
        # return np.var(depth_profile)
        return np.mean((depth_profile - depth_profile[0])**2)
        
    
    from scipy.optimize import minimize
    from scipy.optimize import differential_evolution
        
       
    # initial_guess = [15.0, 0.0002, 0.003]  # starting point for [a, b, c]
    # bounds = [(10.0, 40.0), (0.0001, 0.0004), (0.0025, 0.0030)]  # adjust as needed
    # bounds = [(X_low[0], X_high[0]), (X_low[1], X_high[1]), (X_low[2], X_high[2])]  # adjust as needed
    bounds = [(X_low[0], X_high[0]), (X_low[1], X_high[1]), (X_low[2], X_high[2]),
              (X_low[3], X_high[3]), (X_low[4], X_high[4]), (X_low[5], X_high[5])]  # adjust as needed
    
    # result = minimize(objective, initial_guess, bounds=bounds, method='TNC')

    # print("Optimal parameters:")
    # print("a =", result.x[0])
    # print("b =", result.x[1])
    # print("c =", result.x[2])
    # print("Minimum variance:", result.fun)

    result = differential_evolution(objective, bounds)
    # print(f"Best (a, b, c): [[{result.x[0]}], [{result.x[1]}], [{result.x[2]}]]")

    print(f"dP1 = [{result.x[0]}]")
    print(f"dP2 = [{result.x[1]}]")
    print(f"t0 = [{result.x[2]}]")
    print(f"dt1 = [{result.x[3]}]")
    print(f"dt2 = [{result.x[4]}]")
    print(f"dt3 = [{result.x[5]}]")
    print(f"{counter=}")


    # num_s = 10
    # p_search = np.linspace(X_low[0], X_high[0], num_s)
    # d_search = np.linspace(X_low[1], X_high[1], num_s)
    # t_search = np.linspace(X_low[2], X_high[2], num_s)
    # cr_min = 100.0
    # i_min, j_min, k_min = 0, 0, 0
    # iter = 0

    # for i in range(num_s):
    #     for j in range(num_s):
    #         for k in range(num_s):
    #             iter += 1
    #             depth = compute_depth(p_search[i], d_search[j], t_search[k])
    #             criterion = np.var(depth)
    #             # criterion = depth.sum()
    #             if criterion < cr_min:
    #                 cr_min = criterion
    #                 i_min, j_min, k_min = i, j, k

    # print("Optimal parameters:", p_search[i_min], d_search[j_min], t_search[k_min])

    var1 = np.var(compute_depth(result.x[0], result.x[1], result.x[2],
                                result.x[3], result.x[4], result.x[5]))
    # var2 = np.var(compute_depth(p_search[i_min], d_search[j_min], t_search[k_min]))

    print(var1)
    # print(var2)
    
    # val = np.array([p_search[i_min], d_search[j_min], t_search[k_min]])
    # val = np.array([result.x[0], result.x[1], result.x[2]])
    # input_br = np.tile(val, (times.shape[0],1))#.reshape(-1,3)
    # input_tr = times.reshape(-1,1)
    # input_br = helpers.from_numpy_to_torch(input_br, False)
    # input_tr = helpers.from_numpy_to_torch(input_tr, False)
    # depth_pred = model.network(input_br_test, input_tr_test).to("cpu").detach().numpy().reshape(-1,1)
    depth_pred = compute_depth(result.x[0], result.x[1], result.x[2],
                               result.x[3], result.x[4], result.x[5])
    depth_pred_1 = 1#compute_depth(p_search[i_min], d_search[j_min], t_search[k_min])
    np.savez(results_directory+f"depth_opt.npz", x_loc=times, depth_opt=depth_pred, depth_opt_1=depth_pred_1)