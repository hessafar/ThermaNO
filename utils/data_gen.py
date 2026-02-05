import numpy as np
import h5py
from utils.helpers import from_numpy_to_torch
from config import global_params, random_seed, C_T, is_single_track, T_0
from scipy.interpolate import interp1d
# import sobol_seq
from scipy.stats import qmc


dim = global_params["dimension"]
gp_file = global_params["gp_file"]
gp_file_test = global_params["gp_file_test"]


def get_laser_center(laser, path, time, path_id):
    time_line = laser.time_final / laser.n_lines  # Time needed for each line to complete
    if time % time_line == 0:
        line_no = int(time / time_line) - 1 # The current line no. corresponding to the given time
    else:
        line_no = int(time / time_line)  # The current line no. corresponding to the given time
    
    # for constant starting point
    # x0_line = laser.X_start[0] + path[line_no] * laser.h_spacing[path_id] * laser.jump_direction[0]  # x0 in the current line no.
    # y0_line = laser.X_start[1] + path[line_no] * laser.h_spacing[path_id] * laser.jump_direction[1]  # y0 in the current line no.

    # for non-constant starting point
    x0_line = laser.x1[path_id] + path[line_no] * laser.h_spacing[path_id] * laser.jump_direction[0]  # x0 in the current line no.
    y0_line = laser.x2[path_id] + path[line_no] * laser.h_spacing[path_id] * laser.jump_direction[1]  # y0 in the current line no.

    time_in_section = time - line_no * time_line  # Time spent in the current line no.
    x_laser = x0_line + laser.v_laser * laser.laser_direction[0] * time_in_section
    y_laser = y0_line + laser.v_laser * laser.laser_direction[1] * time_in_section
    # print(time_final, time_line, line_no, x0_line, y0_line, time_in_section)
    return (x_laser, y_laser)

def discretize_input_function(laser, path, path_id):#, time): #, x, y, z):
    lp = []
    
    ts = 0
    for line_order in range(laser.n_lines):
        # for constant starting point
        # x_laser = laser.X_start[0] + path[line_order]*laser.h_spacing[path_id] * laser.jump_direction[0]
        # y_laser = laser.X_start[1] + path[line_order]*laser.h_spacing[path_id] * laser.jump_direction[1]

        # for non-constant starting point
        x_laser = laser.x1[path_id] + path[line_order]*laser.h_spacing[path_id] * laser.jump_direction[0]
        y_laser = laser.x2[path_id] + path[line_order]*laser.h_spacing[path_id] * laser.jump_direction[1]

        lp.append([x_laser, y_laser])
        for _ in range(laser.line_total_ts):
            ts += 1
            x_laser += laser.v_laser * laser.laser_direction[0] * laser.dt
            y_laser += laser.v_laser * laser.laser_direction[1] * laser.dt
            lp.append([x_laser, y_laser])
    # lp.append([x_laser_t, y_laser_t])
    
    lp = np.array(lp).reshape(-1,)
    
    # P_x = coeff * laser.P0 * np.exp(-3*(((x - c_laser)/laser.a_laser) ** 2 + ((y - y_c)/laser.b_laser) ** 2)) * np.exp(
    #                         -3*(z - laser.H) ** 2 / laser.depth_laser ** 2) # Laser power at the given time and location
    
    # print(laser.line_process_time * n_lines)
    return lp

def discretize_input_function_test(laser, path, path_id):#, time): #, x, y, z):
    
    lp = []
    ts = 0
    for line_order in range(laser.n_lines):
        # for constant starting point
        # x_laser = laser.X_start[0] + path[line_order]*laser.h_spacing_test[path_id] * laser.jump_direction[0]
        # y_laser = laser.X_start[1] + path[line_order]*laser.h_spacing_test[path_id] * laser.jump_direction[1]

        # for non-constant starting point
        x_laser = laser.x1[path_id] + path[line_order]*laser.h_spacing_test[path_id] * laser.jump_direction[0]
        y_laser = laser.x2_test[path_id] + path[line_order]*laser.h_spacing_test[path_id] * laser.jump_direction[1]

        lp.append([x_laser, y_laser])
        for _ in range(laser.line_total_ts):
            ts += 1
            x_laser += laser.v_laser * laser.laser_direction[0] * laser.dt
            y_laser += laser.v_laser * laser.laser_direction[1] * laser.dt
            lp.append([x_laser, y_laser])
    # lp.append([x_laser_t, y_laser_t])
    lp = np.array(lp).reshape(-1,)
    
    return lp

def get_all_gps(laser, num_paths): #get all gp functions for the branch net sensors
    #### Loading GP function for laser power samples from the saved file ######
    data = np.load(gp_file)
    t_n = data["x_value"]  # corresponding time for the samples
    y_samples_loaded = data["y_value"]  # all laser power samples
    ###########################################################################
    lp = []
    for i in range(num_paths):
        gp_function = y_samples_loaded[i,:] # The laser power sample for the m-th path
        gp_function_interpolator = interp1d(t_n.ravel(), gp_function, kind='cubic', fill_value="extrapolate")
        sensors = np.linspace(0, laser.time_final, laser.n_sensors_p)
        lp.append(gp_function_interpolator(sensors).reshape(-1,))
    lp = np.array(lp).reshape(num_paths, -1)
    return lp

def get_all_gps_test(laser, num_paths, const_power=False): #get all gp functions for the branch net sensors
    #### Loading GP function for laser power samples from the saved file ######
    data = np.load(gp_file_test)
    t_n = data["x_value"]  # corresponding time for the samples
    y_samples_loaded = data["y_value"]  # all laser power samples
    ###########################################################################
    lp = []
    for i in range(num_paths):
        gp_function = y_samples_loaded[i,:] # The laser power sample for the m-th path
        gp_function_interpolator = interp1d(t_n.ravel(), gp_function, kind='cubic', fill_value="extrapolate")
        sensors = np.linspace(0, laser.time_final, laser.n_sensors_p)
        if not const_power:
            lp.append(gp_function_interpolator(sensors).reshape(-1,))
        else:
            lp.append(laser.P0*np.ones((laser.n_sensors_p,)))
    lp = np.array(lp).reshape(num_paths, -1)
    return lp

def discretize_gp_function(laser, path_no):#, time): #, x, y, z):
    
    #### Loading GP function for laser power samples from the saved file ######
    data = np.load(gp_file)
    t_n = data["x_value"]  # corresponding time for the samples
    y_samples_loaded = data["y_value"]  # all laser power samples
    ###########################################################################
    gp_function = y_samples_loaded[path_no, :] # The laser power sample for the m-th path
    gp_function_interpolator = interp1d(t_n.ravel(), gp_function, kind='cubic', fill_value="extrapolate")
    sensors = np.linspace(0, laser.time_final, laser.n_sensors_p)
    lp = gp_function_interpolator(sensors).reshape(-1,)

    return lp

def cluster_transform(xi, center, width=0.1, gamma=5):
    """Cluster x around center."""
    return center + width * np.tanh(gamma * (2*xi - 1))

def generate_collocation_points(laser, num_t, points_per_t, path):
    num_path = len(path)
    if is_single_track: gp_functions = get_all_gps(laser=laser, num_paths=num_path)
    # n_lines = path.shape[0]
    adap_perc = 0.5
    adap_perc_laser = 0.0
    samples = num_t * points_per_t
    ppt_adap = int(adap_perc * points_per_t)
    ppt_adap_laser = int(adap_perc_laser * points_per_t)
    samples_l = num_t * ppt_adap
    samples_laser = num_t * ppt_adap_laser
    # samples_l = int(adap_perc * samples)
    low_bound = np.array([0.0, 0.0, 0.0, 0.0])   # Lower bounds for [time, x, y]
    high_bound = np.array([laser.time_final, laser.L, laser.W, laser.H]) # Higher bounds for [time, x, y]
    low_bound_space = np.array([0.0, 0.0, 0.0])   # Lower bounds for [time, x, y]
    high_bound_space = np.array([laser.L, laser.W, laser.H]) # Higher bounds for [time, x, y]
    # tn = np.linspace(0.0, 0.999, num_t)
    tn = np.linspace(0.0, laser.time_final*0.999, num_t)

    #### Loading GP function for laser power samples from the saved file ######
    data = np.load(gp_file)
    t_n = data["x_value"]  # corresponding time for the samples
    y_samples_loaded = data["y_value"]  # all laser power samples
    ###########################################################################
    input_tr = []
    input_br1 = []
    input_c_laser = []
    for m in range(num_path):
        if is_single_track:
            gp_function = y_samples_loaded[m, :] # The laser power sample for the m-th path
            gp_function_interpolator = interp1d(t_n.ravel(), gp_function, kind='cubic', fill_value="extrapolate")

        coll_coord = np.full((samples, dim+1), 0.0)
        coll_coord_l = np.full((samples_l, dim+1), 0.0)
        coll_coord_laser = np.full((samples_laser, dim+1), 0.0)
        
        ##### Sampling random collocation points within the geometry ####
        # sampler = qmc.Sobol(d=dim+1, scramble=True, seed = random_seed+m)
        sampler = qmc.Sobol(d=dim, scramble=True, seed = random_seed+m)
        sobol_sample = sampler.random(n=samples)
        # coll_coord = qmc.scale(sobol_sample, low_bound, high_bound)
        coll_coord[:,1:] = qmc.scale(sobol_sample, low_bound_space, high_bound_space)

        # for i in range (samples):
        #     seed = random_seed + i + 10*m
        #     # coll_coord[i, :], _ = sobol_seq.i4_sobol(dim+1, seed)
        #     temp_val, _ = sobol_seq.i4_sobol(dim, seed)
        #     coll_coord[i, 1] = temp_val[0]
        #     coll_coord[i, 2] = temp_val[1]
        #     coll_coord[i, 3] = temp_val[2]

        #### Sampling on a uniform grid in time #####
        for i in range(num_t):
            coll_coord[i*points_per_t:(i+1)*points_per_t, 0] = tn[i]
        #############################################
        # coll_coord = coll_coord*(high_bound - low_bound) + low_bound
        #################################################################
        
        ##### Clustering for non-uniform thermal conductivity############
        width_cluster = 0.3
        gamma_cluster = 2
        sampler = qmc.Sobol(d=dim, scramble=True, seed = random_seed+m)
        sobol_sample = sampler.random(n=samples_l)
        coll_coord_l[:,1:] = qmc.scale(sobol_sample, low_bound_space, high_bound_space)
        coll_coord_l[:,1] = cluster_transform(sobol_sample[:,1], center=1.0, width=width_cluster, gamma=gamma_cluster)
        for i in range(num_t):
            coll_coord_l[i*ppt_adap:(i+1)*ppt_adap, 0] = tn[i]
        
        coll_coord = np.vstack((coll_coord, coll_coord_l))
        #################################################################


        ######### Sampling adaptive collocation points in the vicinity of the laser #######
        # sampler = qmc.Sobol(d=dim, scramble=True, seed = random_seed+m)
        # sobol_sample = sampler.random(n=samples_laser)
        # for t in range(num_t):
        #     for i in range (ppt_adap_laser):
        #         ind = t*ppt_adap_laser + i
        #         x_laser_t, y_laser_t = get_laser_center(laser=laser, path=path[m], time=tn[t], path_id=m)
                
        #         # coll_coord_laser[:,1:] = qmc.scale(sobol_sample, low_bound_space, high_bound_space)
        #         # seed = 6*random_seed + i + 11*m
        #         # temp_val, _ = sobol_seq.i4_sobol(dim, seed)
        #         coll_coord_laser[ind, 0] = tn[t]
        #         R = sobol_sample[ind, 0] * 0.95 * laser.r_laser
        #         theta = sobol_sample[ind, 1] * 2 * np.pi
        #         phi = sobol_sample[ind, 2] * 0.5 * np.pi

        #         coll_coord_laser[ind, 1] = R * np.sin(phi) * np.cos(theta) + x_laser_t
        #         coll_coord_laser[ind, 2] = R * np.sin(phi) * np.sin(theta) + y_laser_t
        #         coll_coord_laser[ind, 3] = laser.H - R * np.cos(phi)

        # coll_coord = np.vstack((coll_coord, coll_coord_laser)) 
        ###################################################################################
        input_tr.append(coll_coord)
        
        for n in range(samples + samples_l + samples_laser):
            if not is_single_track:
                row_br1 = discretize_input_function(laser=laser, path=path[m], path_id=m) #, coll_coord[n, 0])
                p_t = [laser.P0]
                input_br1.append(row_br1)
            else:
                p_t = gp_function_interpolator(np.array([coll_coord[n, 0]])) # input should be an array (here with one element), p_t is also an array.
                input_br1.append(gp_functions[m, :])

            x_laser_t, y_laser_t = get_laser_center(laser=laser, path=path[m], time=coll_coord[n, 0], path_id=m)
            # input_br1.append(gp_functions[m, :])
            input_c_laser.append([x_laser_t, y_laser_t, p_t[0]])
            
    
    input_tr = np.array(input_tr)
    input_br1 = np.array(input_br1)
    input_c_laser = np.array(input_c_laser)
    input_tr = input_tr.reshape((-1, dim + 1))
    if not is_single_track:
        input_br1 = input_br1.reshape((-1, laser.n_sensors))
    else:
        input_br1 = input_br1.reshape((-1, laser.n_sensors_p))
    input_c_laser = input_c_laser.reshape((-1, 3))
    return [input_tr, input_br1, input_c_laser]

def generate_IC_points(laser, num_x, path):
    num_path = len(path)
    if is_single_track: gp_functions = get_all_gps(laser=laser, num_paths=num_path)
    # low_bound = np.array([0.0, 0.0, 0.0, 0.0])   # Lower bounds for [time, x, y]
    # high_bound = np.array([laser.time_final, laser.L, laser.W, laser.H]) # Higher bounds for [time, x, y]
    low_bound = np.array([0.0, 0.0, 0.0])   # Lower bounds for [time, x, y]
    high_bound = np.array([laser.L, laser.W, laser.H]) # Higher bounds for [time, x, y]

    input_tr = []
    input_br1 = []
    T0 = []
    for m in range(num_path):
        coll_coord = np.full((num_x, dim+1), 0.0)
        temp = np.full((num_x,), T_0)
        sampler = qmc.Sobol(d=dim, scramble=True, seed = random_seed+m)
        sobol_sample = sampler.random(n=num_x)
        coll_coord[:,1:] = qmc.scale(sobol_sample, low_bound, high_bound)
        coll_coord[:,0] = 0.0
        T0.append(temp)

        # for i in range (num_x):
        #     seed = random_seed + i + 10*m
        #     temp_val, _ = sobol_seq.i4_sobol(dim, seed)
        #     coll_coord[i, :] = [0, temp_val[0], temp_val[1], temp_val[2]]
        #     T0.append(T_0)
        # coll_coord = coll_coord*(high_bound - low_bound) + low_bound
        input_tr.append(coll_coord)
        for n in range(num_x):
            if not is_single_track:
                row_br1 = discretize_input_function(laser=laser, path=path[m], path_id=m)
                input_br1.append(row_br1)
            else:
                input_br1.append(gp_functions[m,:])

    input_tr = np.array(input_tr)
    input_br1 = np.array(input_br1)
    input_tr = input_tr.reshape((-1, dim + 1))
    if not is_single_track:
        input_br1 = input_br1.reshape((-1, laser.n_sensors))
    else:
        input_br1 = input_br1.reshape((-1, laser.n_sensors_p))
    T0 = np.array(T0).reshape(-1,1)
    return [input_tr, input_br1, T0]

def sample_near_target(arr, target_position, sigma=1.0, size=1):
    """
    Randomly samples elements from a 3D array, favoring elements close to the target position.
    
    Parameters:
    - arr: 3D numpy array to sample from.
    - target_position: Tuple (x, y, z) indicating the target element's indices.
    - sigma: Spread of the Gaussian kernel (larger sigma samples farther elements).
    - size: Number of samples to return.
    
    Returns:
    - Array of sampled elements.
    """
    shape = arr.shape
    x0, y0, z0 = target_position
    
    # Generate grid of indices
    i, j, k = np.indices(shape)
    
    # Calculate squared distances from target
    dist_sq = (i - x0)**2 + (j - y0)**2 + (k - z0)**2
    
    # Compute Gaussian weights
    weights = np.exp(-dist_sq / (2 * sigma**2))
    weights /= weights.sum()  # Normalize to sum to 1
    
    # Flatten the weights and sample indices
    flat_indices = np.arange(weights.size)
    selected_flat_indices = np.random.choice(flat_indices, size=size, p=weights.ravel())
    
    # Convert flat indices to 3D coordinates
    selected_positions = np.unravel_index(selected_flat_indices, shape)
    
    return selected_positions, arr[selected_positions]

def generate_IC_points_state(laser, num_x, path):
    file = "output_np.npz"
    data = np.load(file)
    data = data["temperature"]  #shpae = (nz, ny, nx) = (401, 301, 201), dx=5e-6 m or 5e-3 mm
    # data = data[::4,::4,::4]
    target = (50, 25, 66)  # Target position at the center


    dx = dy = dz = 5e-3
    # dx = dy = dz = 0.02
    # x = np.random.randint(0, data.shape[2], num_x)
    # y = np.random.randint(0, data.shape[1], num_x)
    # z = np.random.randint(0, data.shape[0], num_x)



    num_path = len(path)
    if is_single_track: gp_functions = get_all_gps(laser=laser, num_paths=num_path)
    low_bound = np.array([0.0, 0.0, 0.0, 0.0])   # Lower bounds for [time, x, y]
    high_bound = np.array([laser.time_final, laser.L, laser.W, laser.H]) # Higher bounds for [time, x, y]

    input_tr = []
    input_br1 = []
    T0 = []

    for m in range(num_path):
        # indices, _ = sample_near_target(data, target, sigma=30.0, size=num_x)
        x = np.random.randint(0, data.shape[2], num_x) # indices[2] #np.random.randint(0, data.shape[2], num_x)
        y = np.random.randint(0, data.shape[1], num_x) # indices[1] #np.random.randint(0, data.shape[1], num_x)
        z = np.random.randint(0, data.shape[0], num_x) # indices[0] #np.random.randint(0, data.shape[0], num_x)
        coll_coord = np.full((num_x, dim+1), 0.0)
        for i in range (num_x):
            coll_coord[i, :] = [0, x[i]*dx, y[i]*dy, z[i]*dz]
            T0.append(data[z[i], y[i], x[i]]/C_T)
        # coll_coord = coll_coord*(high_bound - low_bound) + low_bound
        input_tr.append(coll_coord)
        for n in range(num_x):
            if not is_single_track:
                row_br1 = discretize_input_function(laser=laser, path=path[m], path_id=m)
                input_br1.append(row_br1)
            else:
                input_br1.append(gp_functions[m,:])

    input_tr = np.array(input_tr)
    input_br1 = np.array(input_br1)
    input_tr = input_tr.reshape((-1, dim + 1))
    if not is_single_track:
        input_br1 = input_br1.reshape((-1, laser.n_sensors))
    else:
        input_br1 = input_br1.reshape((-1, laser.n_sensors_p))
    T0 = np.array(T0).reshape(-1,1)
    return [input_tr, input_br1, T0]

def generate_IC_points_state_sobol(laser, num_x, path):
    file = "output_np.npz"
    data = np.load(file)
    data = data["temperature"]  #shpae = (nz, ny, nx) = (401, 301, 201), dx=5e-6 m or 5e-3 mm
    data = data[::4,::4,::4]
    target = (50, 25, 66)  # Target position at the center


    dx = dy = dz = 5e-3
    # x = np.random.randint(0, data.shape[2], num_x)
    # y = np.random.randint(0, data.shape[1], num_x)
    # z = np.random.randint(0, data.shape[0], num_x)



    num_path = len(path)
    if is_single_track: gp_functions = get_all_gps(laser=laser, num_paths=num_path)
    low_bound = np.array([0.0, 0.0, 0.0, 0.0])   # Lower bounds for [time, x, y]
    high_bound = np.array([laser.time_final, laser.L, laser.W, laser.H]) # Higher bounds for [time, x, y]

    input_tr = []
    input_br1 = []
    T0 = []

    for m in range(num_path):
        indices, _ = sample_near_target(data, target, sigma=30.0, size=num_x)
        x = indices[2] #np.random.randint(0, data.shape[2], num_x)
        y = indices[1] #np.random.randint(0, data.shape[1], num_x)
        z = indices[0] #np.random.randint(0, data.shape[0], num_x)
        coll_coord = np.full((num_x, dim+1), 0.0)
        for i in range (num_x):
            coll_coord[i, :] = [0, x[i]*dx, y[i]*dy, z[i]*dz]
            T0.append(data[z[i], y[i], x[i]]/C_T)
        # coll_coord = coll_coord*(high_bound - low_bound) + low_bound
        input_tr.append(coll_coord)
        for n in range(num_x):
            if not is_single_track:
                row_br1 = discretize_input_function(laser=laser, path=path[m], path_id=m)
                input_br1.append(row_br1)
            else:
                input_br1.append(gp_functions[m,:])

    input_tr = np.array(input_tr)
    input_br1 = np.array(input_br1)
    input_tr = input_tr.reshape((-1, dim + 1))
    if not is_single_track:
        input_br1 = input_br1.reshape((-1, laser.n_sensors))
    else:
        input_br1 = input_br1.reshape((-1, laser.n_sensors_p))
    T0 = np.array(T0).reshape(-1,1)
    return [input_tr, input_br1, T0]

def generate_test_points_plane(laser, time, path, path_id, location, plane, grid):
    time_final =  laser.time_final
    # if is_single_track: gp_functions = get_all_gps_test(laser=laser, num_paths=6)
    if is_single_track: gp_functions = get_all_gps_test(laser=laser, num_paths=6, const_power=True)
    nx = grid[0]
    ny = grid[1]
    nz = grid[2]
    dx = laser.L / nx
    dy = laser.W / ny
    dz = laser.H / nz
    
    count = 0
    input_tr = []
    input_br1 = []
    if plane == "xy":
        x = 0.0
        y = 0.0
        z = location
        for i in range(nx+1):
            x = i * dx
            for j in range(ny+1):
                y = j * dy
                input_tr.append([time, x, y, z])
                if not is_single_track:
                    row_br1 = discretize_input_function_test(laser=laser, path=path, path_id=path_id)
                    input_br1.append(row_br1)
                else:
                    input_br1.append(gp_functions[path_id,:])
                # count += 1
                # if count % 100 == 0: print(count)
    if plane == "xz":
        x = 0.0
        y = location
        z = 0.0
        for i in range(nx+1):
            x = i * dx
            for k in range(nz+1):
                z = k * dz
                input_tr.append([time, x, y, z])
                if not is_single_track:
                    row_br1 = discretize_input_function_test(laser=laser, path=path, path_id=path_id)
                    input_br1.append(row_br1)
                else:
                    input_br1.append(gp_functions[path_id,:])
    # if plane == "yz":
    #     for j in range(n_grids+1):
    #         y = j * dy
    #         for k in range(n_grids+1):
    #             z = k * dz
    #             row = np.hstack(([time, location, y, z], discretize_input_function(laser, path, time, location, y, z)))
    #             input.append(row)
    #             count += 1
    #             if count % 100 == 0: print(count)
    input_tr = np.array(input_tr)
    input_br1 = np.array(input_br1)
    input_tr = input_tr.reshape((-1, dim + 1))
    if not is_single_track:
        input_br1 = input_br1.reshape((-1, laser.n_sensors))
    else:
        input_br1 = input_br1.reshape((-1, laser.n_sensors_p))
    return input_tr, input_br1

def generate_test_points_3d(laser, time, path):
    time_final =  laser.time_final
    dx = 2e-2*laser.L
    dy = 2e-2*laser.W
    dz = 2e-2*laser.H
    x = 0.0
    y = 0.0
    z = 0.0
    input = []
    count = 0
    for i in range(51):
        x = i * dx
        for j in range(51):
            y = j * dy
            for k in range(51):
                z = k * dz
                row = np.hstack(([time, x, y, z], discretize_input_function(laser, path, time, x, y, z)))
                input.append(row)
                count += 1
                if count % 100 == 0: print(count)
    input = np.array(input)
    input.reshape((-1, dim + laser.n_sensors + 2))
    input_tr = input[:, 0:dim + 1]
    input_br = input[:, dim + 1:-1]
    return input_tr, input_br

def generate_path(n_section, n_path):
    # self.n_section = n_section
    np.random.seed(20)

    paths = []
    for i in range(n_path):
        temp = np.random.choice(n_section, size=n_section, replace=False)
        # print(temp, paths)
        while any((temp == x).all() for x in paths):
            temp = np.random.choice(n_section, size=n_section, replace=False)

        paths.append(temp)
    return np.array(paths)


def generate_heating_profile(laser, time, path, location, n_grids):
    time_final =  laser.time_final
    dx = laser.L / n_grids
    dy = laser.W / n_grids
    dz = laser.H / n_grids
    x = 0.0
    y = 0.0
    z = 0.0
    count = 0
    input = []
    for i in range(n_grids+1):
        x = i * dx
        for j in range(n_grids+1):
            y = j * dy
            lp = discretize_input_function(laser, path, time, x, y, location)
            input.append(lp[-1])
            count += 1
            if count % 100 == 0: print(count)
    
    input = np.array(input)
    # input.reshape((-1, n_grids+1))
    
    return input


def run_model(model, input_tr, input_br, grid):
    input_tr = from_numpy_to_torch(input_tr, False) #torch.from_numpy(input_tr).to(device)
    input_br = from_numpy_to_torch(input_br, False) #torch.from_numpy(input_br).to(device)
    print("Starting evaluating...")
    u_pred = model.network(input_br, input_tr).to("cpu")
    print("Evaluation ends!")
    u_pred = u_pred.detach()
    u_pred = u_pred.numpy().reshape(grid[0]+1, -1) # Use it when using generate_test_points_plane
    u_pred = u_pred.reshape((u_pred.shape[0], u_pred.shape[1], 1))
    u_pred = u_pred*C_T
    return u_pred

def generate_initial_state_points(laser, points_per_t, path):
    num_path = len(path)
    samples = points_per_t
    ppt_adap = int(0.05* points_per_t)
    
    low_bound = np.array([0.0, 0.0, 0.0])   # Lower bounds for [time, x, y]
    high_bound = np.array([laser.L, laser.W, laser.H]) # Higher bounds for [time, x, y]
    low_bound_z = np.array([0.0, 0.0, 0.85*laser.H])   # Lower bounds for [time, x, y]


    input_tr = []
    for m in range(num_path):
        coll_coord = np.full((samples, dim+1), 0.0)
        coll_coord_l = np.full((ppt_adap, dim+1), 0.0)
        
        ##### Sampling random collocation points within the geometry ####
        sampler = qmc.Sobol(d=dim, scramble=True, seed = random_seed+m)
        sobol_sample = sampler.random(n=samples)
        coll_coord[:,1:] = qmc.scale(sobol_sample, low_bound, high_bound)
        coll_coord[:,0] = 0.999*laser.time_final
        # for i in range (samples):
        #     seed = random_seed + i + 10*m
        #     # coll_coord[i, :], _ = sobol_seq.i4_sobol(dim+1, seed)
        #     temp_val, _ = sobol_seq.i4_sobol(dim, seed)
        #     coll_coord[i, 0] = 0.999
        #     coll_coord[i, 1] = temp_val[0]
        #     coll_coord[i, 2] = temp_val[1]
        #     coll_coord[i, 3] = temp_val[2]
        # coll_coord = coll_coord*(high_bound - low_bound) + low_bound

        ##### Sampling random collocation points for top plate ####
        ppt_z = 400
        coll_coord_z = np.full((ppt_z, dim+1), 0.0)
        sampler = qmc.Sobol(d=dim, scramble=True, seed = random_seed+m)
        sobol_sample = sampler.random(n=ppt_z)
        coll_coord_z[:,1:] = qmc.scale(sobol_sample, low_bound_z, high_bound)
        coll_coord_z[:,0] = 0.999*laser.time_final
        # for i in range (ppt_z):
        #     seed = random_seed + i + 40*(m+1)
        #     # coll_coord[i, :], _ = sobol_seq.i4_sobol(dim+1, seed)
        #     temp_val, _ = sobol_seq.i4_sobol(dim, seed)
        #     coll_coord_z[i, 0] = 0.999
        #     coll_coord_z[i, 1] = temp_val[0]
        #     coll_coord_z[i, 2] = temp_val[1]
        #     coll_coord_z[i, 3] = temp_val[2]
        # #############################################
        # coll_coord_z = coll_coord_z*(high_bound - low_bound_z) + low_bound_z
        coll_coord = np.vstack((coll_coord, coll_coord_z))
        #################################################################


        ######### Sampling adaptive collocation points in the vicinity of the laser #######
        # for ind in range (ppt_adap):
        #     # print(ind)
        #     x_laser_t = 1.5
        #     y_laser_t = 0.725
        #     seed = 6*random_seed + ind + 11*m
        #     temp_val1, _ = sobol_seq.i4_sobol(dim, seed)
        #     coll_coord_l[ind, 0] = 0.999*laser.time_final
        #     R = temp_val1[0] * 1.0 * laser.r_laser
        #     theta = temp_val1[1] * 2 * pi
        #     phi = temp_val1[2] * 0.5 * pi

        #     coll_coord_l[ind, 1] = R * np.sin(phi) * np.cos(theta) + x_laser_t
        #     coll_coord_l[ind, 2] = R * np.sin(phi) * np.sin(theta) + y_laser_t
        #     coll_coord_l[ind, 3] = laser.H - R * np.cos(phi)

        # coll_coord = np.vstack((coll_coord, coll_coord_l)) 
        ###################################################################################
        input_tr.append(coll_coord)
        print(coll_coord.shape)

        input_br1 = []
        for n in range(samples):
            row_br1 = discretize_input_function(laser, path[m]) #, coll_coord[n, 0])
            x_laser_t, y_laser_t = get_laser_center(laser=laser, path=path[m], time=coll_coord[n, 0])
            input_br1.append(row_br1)
            # input_c_laser.append([x_laser_t, y_laser_t])

        
    input_tr = np.array(input_tr)
    input_tr = input_tr.reshape((-1, dim + 1))
    input_br1 = np.array(input_br1)
    input_br1 = input_br1.reshape((-1, laser.n_sensors))

    
    return input_tr, input_br1