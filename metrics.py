### Functions for microstructure metrics

### Simon Daubner (s.daubner@imperial.ac.uk)
### Dyson School of Design Engineering
### Imperial College London

import numpy as np
import taufactor as tau
import porespy as ps
import torch

from scipy.ndimage import label, generate_binary_structure, convolve, find_objects
from skimage import measure


def volume_fraction(array, grayscale_value):
    total_voxels = array.size
    matching_voxels = np.count_nonzero(array == grayscale_value)
    vol_fraction = matching_voxels / total_voxels
    return vol_fraction


def phase_fraction(phase_field):
    return np.sum(phase_field)/(phase_field.size)


def label_periodic(field, grayscale_value, neighbour_structure, periodic, debug=False):
    # Initialize phi field whith enlarged dimensions in periodic directions. Boundary values of
    # array are copied into ghost cells which are necessary to impose boundary conditions.
    padx = int(periodic[0])
    pady = int(periodic[1])
    padz = int(periodic[2])
    mask = np.pad(field, ((padx, padx), (pady, pady), (padz, padz)), mode='wrap')
    labeled_mask, num_labels = label(mask==grayscale_value, structure=neighbour_structure)
    count = 1
    for k in range(100):
        # Find indices where labels are different at the boundaries and create swaplist
        swap_list = np.zeros((1,2))
        if periodic[0]:
            # right x
            indices = np.where((labeled_mask[0,:,:]!=labeled_mask[-2,:,:]) & (labeled_mask[0,:,:]!=0) & (labeled_mask[-2,:,:]!=0))
            additional_swaps = np.column_stack((labeled_mask[0,:,:][indices], labeled_mask[-2,:,:][indices]))
            swap_list = np.row_stack((swap_list,additional_swaps))
            # left x
            indices = np.where((labeled_mask[1,:,:]!=labeled_mask[-1,:,:]) & (labeled_mask[1,:,:]!=0) & (labeled_mask[-1,:,:]!=0))
            additional_swaps = np.column_stack((labeled_mask[1,:,:][indices], labeled_mask[-1,:,:][indices]))
            swap_list = np.row_stack((swap_list,additional_swaps))
        if periodic[1]:
            # top y
            indices = np.where((labeled_mask[:,0,:]!=labeled_mask[:,-2,:]) & (labeled_mask[:,0,:]!=0) & (labeled_mask[:,-2,:]!=0))
            additional_swaps = np.column_stack((labeled_mask[:,0,:][indices], labeled_mask[:,-2,:][indices]))
            swap_list = np.row_stack((swap_list,additional_swaps))
            # bottom y
            indices = np.where((labeled_mask[:,1,:]!=labeled_mask[:,-1,:]) & (labeled_mask[:,1,:]!=0) & (labeled_mask[:,-1,:]!=0))
            additional_swaps = np.column_stack((labeled_mask[:,1,:][indices], labeled_mask[:,-1,:][indices]))
            swap_list = np.row_stack((swap_list,additional_swaps))
        if periodic[2]:
            # front z
            indices = np.where((labeled_mask[:,:,0]!=labeled_mask[:,:,-2]) & (labeled_mask[:,:,0]!=0) & (labeled_mask[:,:,-2]!=0))
            additional_swaps = np.column_stack((labeled_mask[:,:,0][indices], labeled_mask[:,:,-2][indices]))
            swap_list = np.row_stack((swap_list,additional_swaps))
            # back z
            indices = np.where((labeled_mask[:,:,1]!=labeled_mask[:,:,-1]) & (labeled_mask[:,:,1]!=0) & (labeled_mask[:,:,-1]!=0))
            additional_swaps = np.column_stack((labeled_mask[:,:,1][indices], labeled_mask[:,:,-1][indices]))
            swap_list = np.row_stack((swap_list,additional_swaps))
        swap_list = swap_list[1:,:]
        # Sort swap list columns to ensure consistent ordering
        swap_list = np.sort(swap_list, axis=1)

        # Remove duplicates from swap_list
        swap_list = np.unique(swap_list, axis=0)
        # print(f"swap_list contains {swap_list.shape[0]} elements.")
        if (swap_list.shape[0]==0):
            break
        for i in range(swap_list.shape[0]):
            index = swap_list.shape[0] - i -1
            labeled_mask[labeled_mask == swap_list[index][1]] = swap_list[index][0]
        count += 1
    if(debug):
        print(f"Did {count} iterations for periodic labelling.")
    dim = labeled_mask.shape
    return labeled_mask[padx:dim[0]-padx,pady:dim[1]-pady,padz:dim[2]-padz], np.unique(labeled_mask).size-1


def find_spanning_labels(labelled_array, axis):
    """
    Find labels that appear on both ends along given axis

    Returns:
        set: Labels that appear on both ends of the first axis.
    """
    if axis == "x":
        front = np.s_[0,:,:]
        end   = np.s_[-1,:,:]
    elif axis == "y":
        front = np.s_[:,0,:]
        end   = np.s_[:,-1,:]
    elif axis == "z":
        front = np.s_[:,:,0]
        end   = np.s_[:,:,-1]
    else:
        raise ValueError("Axis should be x, y or z!")

    first_slice_labels = np.unique(labelled_array[front])
    last_slice_labels = np.unique(labelled_array[end])
    spanning_labels = set(first_slice_labels) & set(last_slice_labels)
    spanning_labels.discard(0)  # Remove the background label if it exists
    return spanning_labels


def extract_through_feature(array, grayscale_value, axis, periodic=[False,False,False], connectivity=0, debug=False):
    if array.ndim != 3:
        print(f"Expected a 3D array, but got an array with {array.ndim} dimension(s).")
        return None

    # Compute volume fraction of given grayscale value
    vol_phase = volume_fraction(array, grayscale_value)

    # Define a list of connectivities to loop over
    connectivities_to_loop_over = [connectivity] if connectivity else range(1, 4)
    through_feature = []
    through_feature_fraction = np.zeros(len(connectivities_to_loop_over))

    # Compute the largest interconnected features depending on given connectivity
    count = 0
    for conn in connectivities_to_loop_over:
        # connectivity 1 = cells connected by sides (4/6 neighbours)
        # connectivity 2 = cells connected by sides & edges (8/14 neighbours)
        # connectivity 3 = cells connected by sides & edges & corners (8/26 neighbours)
        neighbour_structure = generate_binary_structure(3,conn)
        # Label connected components in the mask with given neighbour structure
        if any(periodic):
            labeled_mask, num_labels = label_periodic(array, grayscale_value, neighbour_structure, periodic, debug=debug)
        else:
            labeled_mask, num_labels = label(array == grayscale_value, structure=neighbour_structure)
        if(debug):
            print(f"Found {num_labels} labelled regions. For connectivity {conn} and grayscale {grayscale_value}.")

        through_labels = find_spanning_labels(labeled_mask,axis)
        spanning_network = np.isin(labeled_mask, list(through_labels))

        through_feature.append(spanning_network)
        through_feature_fraction[count] = volume_fraction(spanning_network,1)/vol_phase
        count += 1

    return through_feature, through_feature_fraction


def largest_interconnected_feature(array, grayscale_value, periodic=[False,False,False], batch_size=0, connectivity=0):
    if array.ndim != 3:
        print(f"Expected a 3D array, but got an array with {array.ndim} dimension(s).")
        return None

    # Compute volume fraction of given grayscale value
    vol_phase = volume_fraction(array, grayscale_value)

    # Define a list of connectivities to loop over
    connectivities_to_loop_over = [connectivity] if connectivity else range(1, 4)
    largest_feature = []
    largest_feature_size = np.zeros(len(connectivities_to_loop_over))

    # Compute the largest interconnected features depending on given connectivity
    count = 0
    for conn in connectivities_to_loop_over:
        # connectivity 1 = cells connected by sides (4 neighbours)
        # connectivity 2 = cells connected by sides & edges (12 neighbours)
        # connectivity 3 = cells connected by sides & edges & corners (26 neighbours)
        neighbour_structure = generate_binary_structure(3,conn)
        # Label connected components in the mask with given neighbour structure
        if any(periodic):
            labeled_mask, num_labels = label_periodic(array, grayscale_value, neighbour_structure, periodic)
        else:
            labeled_mask, num_labels = label(array == grayscale_value, structure=neighbour_structure)
        # print(f"Found {num_labels} labelled regions. For connectivity {conn} and grayscale {grayscale_value}.")
        # If too many labels are found, some presorting is necessary for runtime
        # Batch of 500 labels will be summed to find index range with possibly largest feature
        if (batch_size>0) & (num_labels > batch_size):
            if any(periodic):
                print("Batching is currently not implemented for periodic arrays.")
                return None
            print(f"Found {num_labels} labeled regions. For connectivity {conn} and grayscale {grayscale_value}.")

            batch_size = int(batch_size + np.ceil((num_labels%batch_size)/np.floor_divide(num_labels,batch_size)))
            print(f"Divide work in {int(np.ceil(num_labels/batch_size))} batches.")
            batch = 0
            idx = 0
            fractions = np.zeros(int(np.ceil(num_labels/batch_size)))
            while batch < num_labels:
                matching_voxels = np.count_nonzero((batch < labeled_mask) & (labeled_mask < batch+batch_size+1))
                fractions[idx] = matching_voxels/labeled_mask.size/vol_phase
                # Validation
                # print(f"batch {batch/batch_size}: {fractions[idx]} of total")
                batch += batch_size
                idx += 1
            sort_batchs = np.argsort(-fractions)
            print(f"Select batch {sort_batchs[0]}: {fractions[sort_batchs[0]]} of total")
            # print(f"Largest feature might be larger than returned value. But definitely smaller than {fractions[sort_batchs[0]]}.")
            idx = 0
            volume_id = np.zeros(batch_size)
            for label_id in range(sort_batchs[0]*batch_size+1, (sort_batchs[0]+1)*batch_size+1):
                volume_id[idx] = volume_fraction(labeled_mask,label_id)/vol_phase
                idx += 1
            sort_idx = np.argsort(-volume_id)
            largest_feature.append(labeled_mask == (sort_idx[0]+sort_batchs[0]*batch_size+1))
        else:
            # Sort features by volume fraction
            volume_id = np.zeros(num_labels)
            idx = 0
            for label_id in np.unique(labeled_mask)[1:]:
                volume_id[idx] = volume_fraction(labeled_mask,label_id)/vol_phase
                idx += 1
            sort_idx = np.argsort(-volume_id)
            largest_feature.append(labeled_mask == np.unique(labeled_mask)[1:][sort_idx[0]])
        largest_feature_size[count] = volume_id[sort_idx[0]]
        count += 1

    return largest_feature, largest_feature_size


def specific_surface_area_marching(array, voxel_size=1.0):
    """
    Compute the surface area of a 3D phase represented by 1 entries in a binary array.

    Parameters:
        binary_array (numpy.ndarray): The 3D binary array with 1 entries for the phase of interest.
        voxel_size (float): The size of each voxel (optional, default is 1.0).

    Returns:
        float: The specific surface area of the phase in 1/length unit based on specified voxel size.
    """
    # Use marching cubes to extract the surface mesh
    vertices, faces, _, _ = measure.marching_cubes(array, 0.5, method='lewiner')
    surface_area = measure.mesh_surface_area(vertices, faces)

    volume = np.prod(array.shape)*voxel_size
    specific_surface = surface_area/volume

    return specific_surface


def specific_surface_area_porespy(array, voxel_size=1.0, smooth=None):
    if smooth:
        smoothing = ps.tools.ps_round(smooth, array.ndim, smooth=False)
        surface_area = ps.metrics.region_surface_areas(array.astype(int), voxel_size=voxel_size, strel=smoothing)
    else:
        surface_area = ps.metrics.region_surface_areas(array.astype(int), voxel_size=voxel_size)
    volume = np.prod(array.shape)*voxel_size**3
    specific_surface = surface_area[0]/volume

    return specific_surface


def smooth_with_convolution(array):
    strel = generate_binary_structure(3,1)
    smooth = convolve(array, strel, mode='nearest') / np.sum(strel)

    return smooth


def specific_surface_area(array, dx=1.0, dy=1.0, dz=1.0, smooth=None):
    """
    Compute the surface area of a 3D phase represented by phase fraction [0,1] in an array.

    Parameters:
        float_array (numpy.ndarray): The 3D float array with phase fraction for the phase of interest.
        dx, dy, dz (float): The side lengths of each voxel (optional, default is 1.0).

    Returns:
        float: The specific surface area of the phase in 1/length unit based on specified voxel size.
    """

    if smooth:
        mask = smooth_with_convolution(array)
    else:
        mask = array

    norm2 = (np.gradient(mask, axis=0)/dx)**2
    norm2 += (np.gradient(mask, axis=1)/dy)**2
    if array.ndim == 3:
        norm2 += (np.gradient(mask, axis=2)/dz)**2

    # Calculate the surface area as the integral over |grad phi|
    surface_area = np.sum(np.sqrt(norm2)) #*dx*dy*dz

    # Norm the calculated surface to the box volume
    volume = np.prod(array.shape) #*dx*dy*dz
    specific_surface = surface_area/volume

    return specific_surface


def crop_area_of_interest(tensor, labels):
    indices = torch.nonzero(torch.isin(tensor, labels), as_tuple=True)
    min_idx = [torch.min(idx).item() for idx in indices]
    max_idx = [torch.max(idx).item() for idx in indices]

    # Slice the tensor to the bounding box
    # Make sure to stay inside the bounds of total array
    sub_tensor = tensor[max(min_idx[0]-2,0):min(max_idx[0]+3,tensor.shape[0]),
                        max(min_idx[1]-2,0):min(max_idx[1]+3,tensor.shape[1]),
                        max(min_idx[2]-2,0):min(max_idx[2]+3,tensor.shape[2])]
    return sub_tensor


def specific_surface_areas_torch(labelled_array, dx=1.0, dy=1.0, dz=1.0, device='cpu'):
    """
    Compute the surface areas of all labelled phases represented by individual integers.
    """
    volume = np.prod(labelled_array.shape) #*dx*dy*dz
    labels = np.unique(labelled_array)[1:]
    surfaces = np.zeros(labels.size)

    tensor = torch.tensor(labelled_array, dtype=torch.float32)
    # Move tensor to the specified device (CPU or GPU)
    tensor = tensor.to(device)
    # Hard coded Gaussian kernel
    kernel = torch.tensor([[[0.0115, 0.0279, 0.0115],
                            [0.0279, 0.0679, 0.0279],
                            [0.0115, 0.0279, 0.0115]],

                           [[0.0279, 0.0679, 0.0279],
                            [0.0679, 0.1653, 0.0679],
                            [0.0279, 0.0679, 0.0279]],

                           [[0.0115, 0.0279, 0.0115],
                            [0.0279, 0.0679, 0.0279],
                            [0.0115, 0.0279, 0.0115]]],
                            dtype=torch.float32, device=device)
    kernel = kernel/torch.sum(kernel)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    for i, label in enumerate(labels):
        sub_tensor = crop_area_of_interest(tensor, label)
        # Create binary mask for the label within the slice
        mask = (sub_tensor == label).float()
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.conv3d(mask, kernel, padding='same')
        mask = mask.squeeze()

        grad = torch.gradient(mask, spacing=(dx,dy,dz))
        norm2 = grad[0].pow(2) + grad[1].pow(2)
        if labelled_array.ndim == 3:
            norm2 += grad[2].pow(2)
        surface_area = torch.sum(torch.sqrt(norm2)).item()

        surfaces[i] = surface_area / volume

    return surfaces


def tortuosity(array, run_on='cpu'):
    # https://github.com/tldr-group/taufactor/blob/main/README.md
    # https://www.sciencedirect.com/science/article/pii/S2352711016300280

    # create a solver object with loaded image
    s = tau.Solver(array, device=run_on)
    # call solve function
    # s.solve(verbose='per_iter', iter_limit=30000, conv_crit=1e-3)
    s.solve(iter_limit=20000, conv_crit=1e-3)

    return s.tau


def multiphaseTortuosity(array,phases):
    # https://github.com/tldr-group/taufactor/blob/main/README.md
    # https://www.sciencedirect.com/science/article/pii/S2352711016300280
    # phases must be a dictionary containing the labels and
    # respective conductivity/diffusivity of phases.
    # e.g. phases = {255:1, 0:0.123}

    # create a solver object with loaded image
    s = tau.MultiPhaseSolver(array, cond=phases, device='cpu')
    # call solve function
    s.solve()
    # s.solve(iter_limit=1000, conv_limit = 0.01)

    return s
