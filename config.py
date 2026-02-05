import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

default_dtype = torch.float32
random_seed = 4000  #42 works pretty fast but the tolerance should be smaller 8.5e-5
np.random.seed(random_seed)

# Scaling the real properties
rho = 8351.910158  # Density, unit: kg/m3
cp = 500.0  # Specific heat capacity, unit: J/(kg K)
k = 15.0  # Theramal conductivity, unit: W/(m K)
T_l = 1723.0  # Liquidus temperature (K)
T_s = 1658.0  # Solidus temperature (K)
H_l = 280000  # Phase change enthalpy (J/kg)
alpha = k / (rho*cp)

C_T = 1800. # for varP: 1750
P0_laser = 150
T_0 = 300. / C_T #300.0 #/ C_T
T1 = 1850. / C_T #600.0

C_L = 1 #4e-3
C_H = 1 #1.0e-3
# C_Wi = 2.0e-3

C_t = 1 #4e-2
C_alpha = C_L**2/C_t
# C_alpha = alpha / 1e-3  #1e-2 #C_L**2/C_t
# C_t = C_L**2 / C_alpha

C_v = C_L / C_t
# alpha = alpha / C_alpha
# C_rho = rho #0.25*rho**2
# C_M = C_rho * C_L**3
C_W = 1 #C_M * C_L**2 / C_t**3
# C_k = C_W / (C_L * C_T)
# C_rhocp = C_k / C_alpha

# rhocp_hat = rho * cp / C_rhocp

aa = 0.06640008
bb = 458.9848
cc = -142787.368
h_s = aa*(T_s**2) + bb*T_s + cc
h_l = 769.856*T_l - 211597.432
m_h = (h_l - h_s)/(T_l - T_s)
b_h = h_s - T_s*m_h


L = 2.0 #e-3 / C_L # Length - Corresponding to x
W = 1.0#e-3 #/ C_Wi # Width - Corresponding to y
H = 1.0 #e-3 / C_L
L_eff = L - 1
offset_laser = 0.5
d_laser = 0.9 #e-3 / C_L  # Diameter of the laser (m)
r_laser = 0.5 * d_laser
depth_laser = 0.5 #400e-6 / C_L
P0_hat = P0_laser # / C_W
pi = np.pi
coeff = 6 * np.sqrt(3.0) / (pi * np.sqrt(pi) * r_laser ** 3)

is_single_track = True

class Laser:
    L = L #/ C_L # Length - Corresponding to x
    W = W #/ C_Wi # Width - Corresponding to y
    H = H #/ C_L # Height - Corresponding to z
    d_laser = d_laser  # Diameter of the laser (m)
    r_laser = 0.5 * d_laser  # Raduis of the laser (m)
    a_laser = r_laser
    b_laser = r_laser/0.5
    v_laser = 100 #100 # 0.1 / C_v # Velocity of the laser (mm/s)
    # c_laser = 0.2 * L  # Center of the laser (m)
    depth_laser = depth_laser
    P0 = P0_hat #/ rhocp_hat
    # alpha = 0.01
    # T_bc = 0.1
    T_0 = 0.0 #300.0 / C_T

    num_func = 40
    h_spacing = np.random.uniform(0.20*d_laser, 0.6*d_laser, num_func)
    h_spacing_test = np.array([0.25*d_laser, 0.30*d_laser, 0.35*d_laser, 0.4*d_laser, 0.45*d_laser, 0.50*d_laser])
    x1 = num_func*[0.5]
    x2 = num_func*[0.5] #h_spacing + 0.5
    x2_test = h_spacing_test + 0.5
    X_start = (0.5, 0.5)
    hatch_spacing = 0.25*d_laser
    jump_direction = (0, 1)
    laser_direction = (1, 0)
    n_lines = 1
    line_length = 1.0
    line_process_time = (line_length) / v_laser
    line_total_ts = 20 #int(time_final / dt)
    time_final = n_lines * line_process_time
    # sc1 = np.array([0, 1, 2])
    # sc2 = np.array([2, 0, 1])
    # sc3 = np.array([1, 0, 2])
    # sc4 = np.array([0])
    # scenarios = np.array([sc4])
    # line_order = 
    # path_final_ts = int(path_process_time / dt)


    #time_final = line_length / v_laser
    num_grid_i = 40
    num_grid_j = 40
    dx = L / num_grid_i
    dy = W / num_grid_j
    dt = line_process_time / line_total_ts
    # n_sensors = (num_grid_i + 1) * (num_grid_j + 1) * ts_final
    # n_sensors = (num_grid_i + 1) * (ts_final)
    n_sensors =  2*(line_total_ts + 1)*n_lines
    n_sensors_p = 50 # number of sensors for variable laser power


# Laser configuration (example values; use your original values)
laser_config = {
    "coeff": coeff,
    "power": P0_laser,
    "radius": r_laser,
    "depth": depth_laser,
    "domainH": H
}

material_params = {
    "density": rho,
    "conductivity": k,
    "cp": cp
}

# Network configuration
net_config = {
    "layers_tr": [4, 64, 64, 64],
    "layers_br": [20, 64, 64],
    "activation": "tanh",  # you can later map this string to torch.tanh or a custom activation.
}

# Scaling parameters (example; use your actual values)
# scaling_dict = {
#     "scaling_fac_tr": np.array([1.0, 1.0, 1.0, 1.0]),
#     "scaling_bias_tr": np.array([0.0, 0.0, 0.0, 0.0]),
#     "scaling_fac_br": np.array([1.0] * laser_config["n_sensors"]),
#     "scaling_bias_br": np.array([0.0] * laser_config["n_sensors"]),
#     "rescale_fac": np.array([1.0]),
#     "rescale_bias": np.array([0.0]),
# }

# Training parameters
# training_config = {
#     "num_t": 6 * 50,
#     "points_per_t": 400,
#     "num_epochs": 1000,
#     "lr": 1e-3,
#     "tolerance": 1.75e-4,
#     "optimizer": "Adam",  # or "LBFGS"
# }

global_params = {
    "dimension": 3,
    "mini_batch": True,
    "batch_size": 20000,
    "batch_size_ini": 4000,
    "batch_size_neu": 20000,
    "batch_size_diri": 10000,
    # "gp_file": "sine_samples_sobol.npz",
    "gp_file": "gp_functions_40samples.npz",
    "gp_file_test": "sine_samples_sobol_test.npz"
}

# "gp_file": "gp_functions.npz"