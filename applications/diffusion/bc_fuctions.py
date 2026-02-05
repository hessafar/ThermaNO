import numpy as np
from config import random_seed, global_params, is_single_track
from utils.data_gen import get_all_gps, discretize_input_function
# import sobol_seq
from scipy.stats import qmc

dim = global_params["dimension"]

class BoundaryCondition:

    def __init__(self, flag, location, type_bc, n_direction, value):
        self.info = {"flag": flag,
                     "location": location,
                     "type": type_bc,
                     "n_direction": n_direction,
                     "value": value,
                     "normals": None,
                     "data": None
                     }

    def generatePoints(self, laser, num_t, points_per_t, path, location_bc):
        time_final = laser.time_final
        num_path = len(path)
        if is_single_track: gp_functions = get_all_gps(laser=laser, num_paths=num_path)
        
        samples = num_t * points_per_t
        low_bound =  np.array([0.0, 0.0, 0.0, 0.0])#, dtype=np.float32)   # Lower bounds for [time, x, y]
        high_bound =  np.array([laser.time_final, laser.L, laser.W, laser.H])#, dtype=np.float32) # Higher bounds for [time, x, y]

        input_tr = []
        input_br1 = []
        bc_normals = []
        # bc_type = []
        # count = 0
        for m in range(num_path):
            bc_coord = np.full((samples, dim+1), 0.0)
            sampler = qmc.Sobol(d=dim, scramble=True, seed = random_seed+m)
            sobol_sample = sampler.random(n=samples)
            # for i in range (samples):
            #     seed = random_seed + i + 10*m
            #     temp_val, _ = sobol_seq.i4_sobol(dim+1, seed)
            #     temp_val = temp_val*(high_bound - low_bound) + low_bound
            if self.info["n_direction"] == 1:
                low_bound =  np.array([0.0, 0.0, 0.0])
                high_bound =  np.array([laser.time_final, laser.W, laser.H])
                scaled_sample = qmc.scale(sobol_sample, low_bound, high_bound)
                # bc_coord[i, :] = [temp_val[0], location_bc, temp_val[2], temp_val[3]]
                bc_coord[:, 0] = scaled_sample[:, 0]
                bc_coord[:, 1] = location_bc*np.ones((samples,))
                bc_coord[:, 2] = scaled_sample[:, 1]
                bc_coord[:, 3] = scaled_sample[:, 2]
                bc_normals.append(samples*[0, 1, 0, 0])
            elif self.info["n_direction"] == 2:
                low_bound =  np.array([0.0, 0.0, 0.0])
                high_bound =  np.array([laser.time_final, laser.L, laser.H])
                scaled_sample = qmc.scale(sobol_sample, low_bound, high_bound)
                # bc_coord[i, :] = [temp_val[0], temp_val[1], location_bc, temp_val[3]]
                bc_coord[:, 0] = scaled_sample[:, 0]
                bc_coord[:, 1] = scaled_sample[:, 1]
                bc_coord[:, 2] = location_bc*np.ones((samples,))
                bc_coord[:, 3] = scaled_sample[:, 2]
                bc_normals.append(samples*[0, 0, 1, 0])
            else:
                low_bound =  np.array([0.0, 0.0, 0.0])
                high_bound =  np.array([laser.time_final, laser.L, laser.W])
                scaled_sample = qmc.scale(sobol_sample, low_bound, high_bound)
                # bc_coord[i, :] = [temp_val[0], temp_val[1], temp_val[2], location_bc]
                bc_coord[:, 0] = scaled_sample[:, 0]
                bc_coord[:, 1] = scaled_sample[:, 1]
                bc_coord[:, 2] = scaled_sample[:, 2]
                bc_coord[:, 3] = location_bc*np.ones((samples,))
                bc_normals.append(samples*[0, 0, 0, 1])
            # bc_coord = bc_coord*(high_bound - low_bound) + low_bound
            input_tr.append(bc_coord)

            for _ in range(samples):
                if not is_single_track:
                    row_br1 = discretize_input_function(laser=laser, path=path[m], path_id=m)#, bc_coord[n, 0],  bc_coord[n, 1], bc_coord[n, 2])
                    input_br1.append(row_br1)
                else:
                    input_br1.append(gp_functions[m,:])
        input_tr = np.array(input_tr)
        input_br1 = np.array(input_br1)
        bc_normals = np.array(bc_normals)
        input_tr = input_tr.reshape((-1, dim + 1))
        if not is_single_track:
            input_br1 = input_br1.reshape((-1, laser.n_sensors))
        else:
            input_br1 = input_br1.reshape((-1, laser.n_sensors_p))
        bc_normals = bc_normals.reshape((-1, dim + 1))
        data = [input_tr, input_br1]
        self.info.update({"data": data})
        self.info.update({"normals": bc_normals})
        return self.info
  