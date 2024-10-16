### Some helper functions for handling data

### Simon Daubner (simon.daubner@kit.edu)
### Department of Mechanical Engineering
### Karlsruhe Institute of Technology

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import os

#%% Input
def read_image_stack_pgm(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".pgm"):
            filepath = os.path.join(folder_path, filename)
            image = imageio.imread(filepath)
            images.append(image)
    return np.array(images)


def add_voxel_sphere(array, center_x, center_y, center_z, radius):
    """
    Create a voxelized representation of a sphere in 3D array based on
    given midpoint and radius in terms of pixel resolution.
    """
    nx, ny, nz = array.shape
    x, y, z = np.ogrid[:nx, :ny, :nz]

    distance_squared = (x - center_x + 0.5)**2 + (y - center_y + 0.5)**2 + (z - center_z + 0.5)**2
    mask = distance_squared <= radius**2
    array[mask] = 1


def create_voxelized_sphere(Radius):
    """
    Create a voxelized representation of a sphere as a 3D array based on
    a Radius given in terms of pixel resolution.

    Parameters:
        Radius (int): Sphere radius given in pixels.

    Returns:
        numpy.ndarray: 3D array where values of 1 represent the sphere.
    """
    Nx = 2*Radius+20
    Ny = Nx
    Nz = Nx
    array = np.zeros((Nx,Ny,Nz))
    add_voxel_sphere(array, Nx/2, Ny/2, Nz/2, Radius)
    return array


def create_fcc_cube(pixels, overlap=0.0):
    """
    Create a voxelized FCC unit cell structure in a cube with given
    pixel resolution and overlap of the spheres.

    Parameters:
        pixels (int): Cube/Array side length given in pixels.
        overlap (float): Overlap of neighbouring spheres given in percent.
                     1 corresponds to radius = distance between midpoints.

    Returns:
        numpy.ndarray: 3D array where values of 1 represent the FCC structure.
    """
    # Initialize a 3D numpy array filled with zeros
    cube = np.zeros((pixels, pixels, pixels), dtype=int)

    # Calculate the center and radius
    center = 0.5*pixels
    radius = 0.25*np.sqrt(2)*pixels/(1-0.5*overlap)

    # Add half-spheres centered on each face of the cube
    # We have 6 centers, a list of three center positions with a pos and neg sign
    for axis in range(3):
        for sign in [-1, 1]:
            center_pos = [center] * 3
            center_pos[axis] = center + sign * (center)
            add_voxel_sphere(cube, *center_pos, radius)

    # Add quarter-spheres at each corner of the cube
    for corner in [(0, 0, 0), (0, 0, pixels), (0, pixels, 0), (0, pixels, pixels),
                   (pixels, 0, 0), (pixels, 0, pixels), (pixels, pixels, 0), (pixels, pixels, pixels)]:
        add_voxel_sphere(cube, *corner, radius)

    return cube


def theoretical_fcc_metrics(a, overlap):
    # Notation consistent with https://en.wikipedia.org/wiki/Spherical_cap
    if overlap < (1-np.cos(np.pi/6))*2:
        radius = 0.25*np.sqrt(2)*a/(1-0.5*overlap)
        h = 0.5*radius*overlap
        cap_radius = np.sqrt(2*radius*h - h*h)
        cap_volume = np.pi/3*h*h*(3*radius-h)
        cap_area = 2*np.pi*radius*h

        volume = 4*4/3*np.pi*radius**3 - 48*cap_volume
        volume_fraction = volume/(a**3)

        surface = 4*4*np.pi*radius**2 - 48*cap_area
        specific_surface = surface/(a**3)
    else:
        raise ValueError("Overlap must be smaller than 26.8%!")

    return volume_fraction, specific_surface, cap_radius


#%% Field modification
def solveTwoPhaseWithoutCurvature(array, eps=4, convergence = 0.01, potential = 'well', stabilize = 0.0):
    """
    Compute phase-field evolution based on Allen-Cahn equation.
    Field values are in [0,1] and represent the volume fraction.
    Curvature effects are removed from evolution equation such that shape is preserved.
    Set dx=1 and mobility M=1 in de-dimensionalized equation for solution.

    Parameters:
        array (numpy.ndarray): 2D/3D voxel data of phase as a binary array.
        timesteps (int): Number of timesteps for Euler-Forward scheme
        eps (float): Epsilon scales interfacial with. Typically in range [3,6]
        potential (string): Can be either 'well' or 'obstacle'
    """

    # Define threshold close to zero to avoid division by zero
    zero = 1e-15
    # Stable timestep for dx=1 and M=1
    dt = 0.0025

    if array.ndim == 2:
        [nx,ny] = np.shape(array)

        # Initialize phi field which has dimensions Nx+2 and Ny+2. Boundary values of array
        # are copied into ghost cells which are necessary to impose boundary conditions.
        field = np.concatenate((np.reshape(array[0,:],(1,ny)),array,np.reshape(array[-1,:],(1,ny))),axis=0)
        field = np.concatenate((np.reshape(field[:,0],(nx+2,1)),field,np.reshape(field[:,-1],(nx+2,1))),axis=1)

        # Construct slices for better readability
        # x-1: left,   x+1: right
        # y-1: bottom, y+1: top
        center = np.s_[1:-1,1:-1]
        left   = np.s_[ :-2,1:-1]
        right  = np.s_[2:  ,1:-1]
        bottom = np.s_[1:-1, :-2]
        top    = np.s_[1:-1,2:  ]

        # Terminate loop if either
        # 10'000 steps have been computed or
        # ratio of F_pot/F_grad has converged to one
        it = 1
        converged = False
        while it<10001 and not converged:
            norm2 = 0.25 * ((field[right] - field[left])**2) + 0.25 * ((field[top] - field[bottom])**2)
            F_grad = eps*np.sum(norm2)

            # As we wil divide by norm2, we need to take care of small values
            bulk = np.where(norm2 <= zero)
            norm2[bulk] = 1.0

            eLe = (0.25 * ((field[right] - field[left])**2) * (field[right] - 2*field[center] + field[left]  )
                 + 0.25 * ((field[top] - field[bottom])**2) * (field[top]   - 2*field[center] + field[bottom])
                 + 0.125 * (field[right] - field[left]) * (field[top] - field[bottom])
                         * (field[2:,2:] + field[:-2, :-2] - field[:-2,2:] - field[2:,:-2]) )

            laplace = field[right] - 2*field[center] + field[left] + field[top] - 2*field[center] + field[bottom]

            # Assemble derivatives of gradient and potential terms
            if potential == "well":
                field[center] += dt * 2*(eps*(stabilize*laplace + (1.0-stabilize)*eLe/norm2) - 9/eps*field[center]*(1-field[center])*(1-2*field[center]))
                F_pot  = 9/eps*np.sum((field[center]**2) * ((1-field[center])**2))

            elif potential == "obstacle":
                field[1:-1,1:-1] += dt * (2*eps*(stabilize*laplace + (1.0-stabilize)*eLe/norm2) - 16/eps/np.pi**2 * (1-2*field[center]))
                field = np.maximum(0.0, np.minimum(field, 1.0))
                F_pot  = 16/eps/(np.pi**2) * np.sum(field[center] * (1-field[center]))

            else:
                raise ValueError("Choose well or obstacle as potential term!")

            # Isolate boundary conditions
            field[0,:] = field[1,:]
            field[-1,:] = field[-2,:]
            field[:,0] = field[:,1]
            field[:,-1] = field[:,-2]

            it += 1
            converged = np.abs(F_pot/F_grad-1.0)<convergence

    elif array.ndim == 3:
        [nx,ny,nz] = np.shape(array)

        # Initialize phi field which has dimensions Nx+2, Ny+2 and Nz+2.
        field = np.concatenate((np.reshape(array[0,:,:],(1,ny,nz)),array,np.reshape(array[-1,:,:],(1,ny,nz))),axis=0)
        field = np.concatenate((np.reshape(field[:,0,:],(nx+2,1,nz)),field,np.reshape(field[:,-1,:],(nx+2,1,nz))),axis=1)
        field = np.concatenate((np.reshape(field[:,:,0],(nx+2,ny+2,1)),field,np.reshape(field[:,:,-1],(nx+2,ny+2,1))),axis=2)

        # Construct slices for better readability
        # x-1: left,   x+1: right
        # y-1: bottom, y+1: top
        # z-1: back,   z+1: front
        center = np.s_[1:-1,1:-1,1:-1]
        left   = np.s_[ :-2,1:-1,1:-1]
        right  = np.s_[2:  ,1:-1,1:-1]
        bottom = np.s_[1:-1, :-2,1:-1]
        top    = np.s_[1:-1,2:  ,1:-1]
        back   = np.s_[1:-1,1:-1, :-2]
        front  = np.s_[1:-1,1:-1,2:  ]

        # Terminate loop if either
        # 10'000 steps have been computed or
        # ratio of F_pot/F_grad has converged to one
        it = 1
        converged = False
        while it<10001 and not converged:
            norm2 = ( 0.25 * ((field[right] - field[left])**2) 
                     +0.25 * ((field[top]   - field[bottom])**2)
                     +0.25 * ((field[front] - field[back])**2) )
            F_grad = eps*np.sum(norm2)

            # As we wil divide by norm2, we need to take care of small values
            bulk = np.where(norm2 <= zero)
            norm2[bulk] = 1.0 

            eLe = (  0.25 * ((field[right] - field[left])**2) * (field[right] - 2*field[center] + field[left]  )
                   + 0.25 * ((field[top] - field[bottom])**2) * (field[top]   - 2*field[center] + field[bottom])
                   + 0.25 * ((field[front] - field[back])**2) * (field[front] - 2*field[center] + field[back]  )
                   + 0.125 * (field[right] - field[left]) * (field[top] - field[bottom]) * (field[2:,2:,1:-1] + field[:-2,:-2,1:-1] - field[:-2,2:,1:-1] - field[2:,:-2,1:-1])
                   + 0.125 * (field[right] - field[left]) * (field[front] - field[back]) * (field[2:,1:-1,2:] + field[:-2,1:-1,:-2] - field[:-2,1:-1,2:] - field[2:,1:-1,:-2])
                   + 0.125 * (field[top] - field[bottom]) * (field[front] - field[back]) * (field[1:-1,2:,2:] + field[1:-1,:-2,:-2] - field[1:-1,:-2,2:] - field[1:-1,2:,:-2]) )

            laplace = ( field[right] - 2*field[center] + field[left]
                       +field[top]   - 2*field[center] + field[bottom]
                       +field[front] - 2*field[center] + field[back]  )

            # Assemble derivatives of gradient and potential terms
            if potential == "well":
                field[center] += dt * 2*( eps*(stabilize*laplace + (1.0-stabilize)*eLe/norm2)
                                                 -9/eps*field[center]*(1-field[center])*(1-2*field[center]) )
                F_pot  = 9/eps*np.sum((field[center]**2) * ((1-field[center])**2))

            elif potential == "obstacle":
                field[center] += dt * (2*eps*(stabilize*laplace + (1.0-stabilize)*eLe/norm2) - 16/eps/np.pi**2 * (1-2*field[center]))
                field = np.maximum(0.0, np.minimum(field, 1.0))
                F_pot  = 16/eps/(np.pi**2) * np.sum(field[center] * (1-field[center]))

            else:
                raise ValueError("Choose well or obstacle as potential term!")

            # Isolate boundary conditions
            field[ 0,:,:] = field[ 1,:,:]
            field[-1,:,:] = field[-2,:,:]
            field[:, 0,:] = field[:, 1,:]
            field[:,-1,:] = field[:,-2,:]
            field[:,:, 0] = field[:,:, 1]
            field[:,:,-1] = field[:,:,-2]

            it += 1
            converged = np.abs(F_pot/F_grad-1.0)<convergence
    else:
        raise ValueError("Array must be 2D or 3D!")

    print(f"Converged in {it-1} steps. F_pot/F_grad={(F_pot/F_grad):.4f}")

    return field[center]


def extract_inner_features(labelled_array):
    initial_labels = np.unique(labelled_array).size
    if initial_labels < 3:
        raise ValueError("Input array should be labelled array with more than 3 phases!")

    # Find all features which are in contact with domain boundary
    boundary_labels = np.unique(labelled_array[0,:,:])
    boundary_labels = np.concatenate((boundary_labels, np.unique(labelled_array[-1,:,:])))
    boundary_labels = np.concatenate((boundary_labels, np.unique(labelled_array[:,0,:])))
    boundary_labels = np.concatenate((boundary_labels, np.unique(labelled_array[:,-1,:])))
    boundary_labels = np.concatenate((boundary_labels, np.unique(labelled_array[:,:,0])))
    boundary_labels = np.concatenate((boundary_labels, np.unique(labelled_array[:,:,-1])))
    boundary_labels = np.unique(boundary_labels)

    mask_boundary_labels = np.isin(labelled_array, boundary_labels)
    labelled_array[mask_boundary_labels] = 0
    print(f"{np.unique(labelled_array).size} of initial {initial_labels} labels remaining.")


def relabel_random_order(array):
    remaining_labels = np.unique(array)
    new_labels = np.arange(len(remaining_labels))
    # Zero should be kept where it is
    np.random.shuffle(new_labels[1:])

    # Create a mapping from old labels to new shuffled labels
    label_mapping = dict(zip(remaining_labels, new_labels))

    # Vectorized relabeling using np.vectorize for efficiency
    relabel_function = np.vectorize(lambda x: label_mapping[x])

    return relabel_function(array)


#%% Write output
def write_dict_to_txt(dictionary, filename, delimiter="\t"):
    """
    Write a dictionary to a text file.

    Parameters:
        dictionary (dict): The dictionary to be written.
        filename (str): The name of the file to write to.
        delimiter (str): The delimiter to use between fields (default is "\t").

    Returns:
        None
    """
    with open(filename, "w") as txtfile:
        # Write header
        txtfile.write(delimiter.join(dictionary.keys()) + "\n")
        # Write data
        for i in range(len(next(iter(dictionary.values())))):
            row = delimiter.join(str(dictionary[key][i]) for key in dictionary)
            txtfile.write(row + "\n")


def export_to_vtk(array, filename="output.vtk", spacing=(1.0, 1.0, 1.0)):
    """
    Export a 3D numpy array to VTK format for visualization in VisIt or ParaView.

    Parameters:
        array (numpy.ndarray): The 3D numpy array.
        filename (str): The output VTK file name.
        spacing (tuple): The voxel size for each axis (dx, dy, dz).
    """
    # Create a structured grid from the array
    grid = pv.ImageData()

    grid.dimensions = np.array(array.shape) + 1
    grid.spacing = spacing
    grid.origin = np.zeros(3)
    grid.cell_data["values"] = array.flatten(order="F")  # Fortran order flattening
    grid.save(filename)


def export_histogram(data, bins, range=(0,1), density=True, filename="histogram.txt"):
    hist, bin_edges = np.histogram(data, bins=bins, range=range, density=density)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_data = np.column_stack((midpoints, hist))

    np.savetxt(filename, hist_data, fmt="%.6f", delimiter="\t", header="bins\t distribution")


#%% Plotting
def plotField2D(field, title, dpi=100):
    [nx,ny] = field.shape
    plt.figure(figsize=(5, 5), dpi=dpi)
    plt.imshow(field, cmap='Greys', origin='lower', extent=[0, nx, 0, ny])
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.title(title)
    plt.show()


def plot_connectivity(phase, feature, title=None, dpi=100):
    """
    Create a 3D plot based on voxel data stored in a NumPy array.

    Parameters:
        phase (numpy.ndarray): 3D voxel data of phase as a binary array.
        feature (numpy.ndarray): 3D voxel data of feature as a binary array.
    """
    # Get the coordinates of all voxels with a value of 1 (occupied voxels)
    x, y, z = np.where(feature)
    xx = x+0.5
    yy = y+0.5
    zz = z+0.5

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the voxels of phase in blue with opacity 0.05
    ax.voxels(phase, facecolors='blue', edgecolor='none',alpha=0.05)

    # Plot connected feature with red spheres
    ax.scatter(xx, yy, zz, c='r', marker='o',alpha=0.5)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.show()
