import numpy as np
import h5py
import os


def write_vtk(filename, data, laser, n_grids, plane): # This function is not used anymore!

    dx = laser.L / n_grids
    dy = laser.W / n_grids
    dz = laser.H / n_grids

    # Convert the NumPy array to a VTK array
    vtk_data_array = numpy_to_vtk(num_array=data.ravel(order='F'), deep=1, array_type=vtk.VTK_FLOAT)

    # Define the scalar name
    vtk_data_array.SetName("Temperature")

    # Create a VTK image data object
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(data.shape)

    # Set the grid spacing (dx, dy, dz)
    # vtk_image.SetSpacing(dx, dy, dz)
    if plane =="xy":
        vtk_image.SetSpacing(dx, dy*C_Wi/C_L, 1)
    if plane =="xz":
        vtk_image.SetSpacing(dx, 1, dz*C_H/C_L)
    if plane =="yz":
        vtk_image.SetSpacing(1, dy*C_Wi/C_L, dz*C_H/C_L)

    # Set the origin (optional)
    origin_x, origin_y, origin_z = 0.0, 0.0, 0.0
    vtk_image.SetOrigin(origin_x, origin_y, origin_z)

    # Attach the VTK data array to the image data object
    vtk_image.GetPointData().SetScalars(vtk_data_array)

    # Write the VTK image data to a .vtk file
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(filename)
    writer.SetInputData(vtk_image)
    writer.Write()

def write_hdf_2d(filename, data, laser, plane, grid):
    dim = data.shape
    file_h5 = filename + ".h5"
    file_xdmf = filename + ".xdmf"
    nx = grid[0]
    ny = grid[1]
    nz = grid[2]
    with h5py.File(file_h5, "w") as h5file:

        # param_group = h5file.create_group("SimulationParameters")
        # key_l = list(parameters.keys())
        # for i in range (len(key_l)):
        #     param_group.attrs[key_l[i]] = parameters[key_l[i]]
        dx = laser.L / nx
        dy = laser.W / ny
        dz = laser.H / nz
        # origin = (0.0, 0.0, 0.0)
        if plane =="xy":
            spacing_x = dx
            spacing_y = dy
            spacing_z = 1.0
            origin = (0.0, 0.0, laser.H)
        # if plane =="xz":
        #     spacing_x = dx*C_L
        #     spacing_y = 1.0
        #     spacing_z = dz*C_H
        #     origin = (0.0, 0.5*C_Wi, 0.0)
        # if plane =="yz":
        #     spacing_x = 1.0
        #     spacing_y = dy*C_Wi
        #     spacing_z = dz*C_H
        #     origin = (0.5*C_L, 0.0, 0.0)
        # Create a group for the structured grid
        grid_group = h5file.create_group("StructuredGrid")
        
        # Add grid dimensions,origin, and spacing as attributes
        grid_group.attrs["dims"] = dim
        grid_group.attrs["orign"] = origin
        grid_group.attrs["spacing"] = (spacing_x, spacing_y, spacing_z)
        
        # Create datasets for each coordinate axis
        x = np.linspace(origin[0], origin[0] + (dim[0] - 1) * spacing_x, dim[0])
        y = np.linspace(origin[1], origin[1] + (dim[1] - 1) * spacing_y, dim[1])
        z = np.linspace(origin[2], origin[2] + (dim[2] - 1) * spacing_z, dim[2])
        
        grid_group.create_dataset("X", data=x)
        grid_group.create_dataset("Y", data=y)
        grid_group.create_dataset("Z", data=z)
       
        # Create a dataset for the temperature data
        temperature_dataset = grid_group.create_dataset("Temperature", data=data.T)
        temperature_dataset.attrs["Unit"] = "Kelvin"  # Add unit attribute if needed
    
        
    xdmf_content = f"""<?xml version="1.0" ?>
        <Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
        <Domain>
            <Grid Name="StructuredGrid" GridType="Uniform">
            <Topology TopologyType="3DRectMesh" Dimensions="{dim[2]} {dim[1]} {dim[0]}"/>
            <Geometry GeometryType="VxVyVz">
                <DataItem Format="HDF" Dimensions="{dim[0]}" NumberType="Float" Precision="4">
                {file_h5}:/StructuredGrid/X
                </DataItem>
                <DataItem Format="HDF" Dimensions="{dim[1]}" NumberType="Float" Precision="4">
                {file_h5}:/StructuredGrid/Y
                </DataItem>
                <DataItem Format="HDF" Dimensions="{dim[2]}" NumberType="Float" Precision="4">
                {file_h5}:/StructuredGrid/Z
                </DataItem>
            </Geometry>
            <Attribute Name="Temperature" AttributeType="Scalar" Center="Node">
                <DataItem Dimensions="{dim[2]} {dim[1]} {dim[0]}" NumberType="Float" Precision="4" Format="HDF">
                {file_h5}:/StructuredGrid/Temperature
                </DataItem>
            </Attribute>
            </Grid>
        </Domain>
        </Xdmf>
        """

    
    with open(file_xdmf, "w") as xdmf_file:
        xdmf_file.write(xdmf_content)

def write_hdf_cp(filename, cp_points, ic_points, data_points, parameters):
    file_h5 = filename + ".h5"
    with h5py.File(file_h5, "w") as h5file:

        param_group = h5file.create_group("TrainingParameters")
        param_group.attrs["network"] = "DeepONet"
        param_group.attrs["laser"] = "Heat source"
        
        cp_group = h5file.create_group("CollocationPoints")
        input_dataset = cp_group.create_dataset("Value", data=cp_points)

        ic_group = h5file.create_group("InitialConditionPoints")
        input_dataset = ic_group.create_dataset("Value", data=ic_points)

        data_group = [None]*len(data_points)
        for i in range(len(data_points)):
            data_group[i] = h5file.create_group("DataPoints_{}".format(i+1))
            for key, value in data_points[i].items():
                if isinstance(value, np.ndarray):
                    data_group[i].create_dataset(key, data=value)
                elif isinstance(value, (int, float, str)):
                    data_group[i].attrs[key] = value
                else:
                    raise TypeError(f"Unsupported data type: {type(value)} for key: {key}")

def write_hdf_testdata(filename, branch_net, trunk_net):
    file_h5 = filename + ".h5"
    with h5py.File(file_h5, "w") as h5file:

        data_group = h5file.create_group("InputData")
        
        input_dataset = data_group.create_dataset("input_trunk", data=trunk_net)
        input_dataset = data_group.create_dataset("input_branch", data=branch_net)
        # h5file.create_dataset("input_branch", data=branch_net)

def read_hdf_cp(filename, num_data):
    file_h5 = filename + ".h5"
    with h5py.File(file_h5, 'r') as h5file:
        input_cp = h5file["CollocationPoints/Value"][:]
        input_ic = h5file["InitialConditionPoints/Value"][:]

        bc_dict = []
        for i in range(num_data):
            data_group = h5file["DataPoints_{}".format(i+1)]
            tmp_dict = {"location": data_group.attrs["location"],
                        "type": data_group.attrs["type"],
                        "n_direction": data_group.attrs["n_direction"],
                        "value": data_group.attrs["value"],
                        "data": h5file["DataPoints_{}/data".format(i+1)][:]}

            bc_dict.append(tmp_dict)

    return input_cp, input_ic, bc_dict

def read_hdf_testdata(filename):
    file_h5 = filename + ".h5"
    with h5py.File(file_h5, 'r') as h5file:
        # List all groups
        input_tr = h5file["InputData/input_trunk"][:]
        input_br = h5file["InputData/input_branch"][:]
 
        return input_tr, input_br

def write_hdf_2d_path(filename, parameters, data, grid, laser, path_group_name, data_group_name):
    dim = data.shape
    file_h5 = filename + ".h5"
    nx = grid[0]
    ny = grid[1]
    nz = grid[2]
    with h5py.File(file_h5, "a") as h5file:

        # path_group_name = "path{}".format(path_number)
        grid_group_name = "structuredGrid"
        # param_group_name = "simulationParameters"
        if path_group_name in h5file:
            path_group = h5file[path_group_name]
        else:
            path_group = h5file.create_group(path_group_name)
        
        # time_group = path_group.create_group("{}".format(time))
        data_droup =  path_group.create_group(data_group_name)
        # if grid_group_name not in h5file:
        #     param_group = h5file.create_group(param_group_name)
        #     key_l = list(parameters.keys())
        #     for i in range (len(key_l)):
        #         param_group.attrs[key_l[i]] = parameters[key_l[i]]
        
        spacing_x = laser.L / nx
        spacing_y = laser.H / ny
        spacing_z = 1.0
        origin = (0.0, 0.0, laser.H)

        if grid_group_name not in h5file:
            grid_group = h5file.create_group(grid_group_name)
            
            # Add grid dimensions,origin, and spacing as attributes
            grid_group.attrs["dims"] = dim
            grid_group.attrs["orign"] = origin
            grid_group.attrs["spacing"] = (spacing_x, spacing_y, spacing_z)

            # Create datasets for each coordinate axis
            x = np.linspace(origin[0], origin[0] + (dim[0] - 1) * spacing_x, dim[0])
            y = np.linspace(origin[1], origin[1] + (dim[1] - 1) * spacing_y, dim[1])
            z = np.linspace(origin[2], origin[2] + (dim[2] - 1) * spacing_z, dim[2])
            
            grid_group.create_dataset("X", data=x)
            grid_group.create_dataset("Y", data=y)
            grid_group.create_dataset("Z", data=z)

        # Create a dataset for the temperature data
        temperature_dataset = data_droup.create_dataset("Temperature", data=data.T)
        temperature_dataset.attrs["Unit"] = "Kelvin"  # Add unit attribute if needed
