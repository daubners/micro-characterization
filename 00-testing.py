import matplotlib.pyplot as plt
import numpy as np
import taufactor.metrics as tau
import scipy
import torch
import tifffile
import porespy as ps
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max

import metrics
import data
import time

from scipy.ndimage import label, generate_binary_structure, convolve

path_to_tif = "data-paper/battery-structures/nmc-1-cal-withcbd-w099-binarized.tif"
tif_file = tifffile.imread(path_to_tif)
ms_array = np.array(tif_file)
ms_array = ms_array[20:-20,20:-20,20:-20]
print("Stack shape:", ms_array.shape)

labels = {"pore":0, "NMC":1, "CBD":2}
px = 398e-9 # pixel resolution in m

volume_fraction = {}

for key, value in labels.items():
    # Compute volume fraction of full datset
    volume_fraction[key] = metrics.volume_fraction(ms_array, value)
    print(f"Volume fraction of {key}: {volume_fraction[key]:.4f}")

NMC = ms_array==1

distance = ndi.distance_transform_edt(NMC)
coords = peak_local_max(distance, footprint=np.ones((3, 3, 3)), labels=NMC)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=NMC)

snow_out = ps.filters.snow_partitioning(NMC, r_max=4, sigma=0.4)
print(snow_out)

data.export_to_vtk(labels, "NMC_watershed.vtk")
data.export_to_vtk(snow_out.regions, "NMC_snow_partitioning.vtk")

# surface = {}
# for key, value in labels.items():
#     surface[key] = metrics.specific_surface_area((structure == value).astype(float), dx=px, dy=px, dz=px)
#     print(f"Specific surface area of {key}: {surface[key]*1e-6:.5f} [1/µm]")


# for key1, value in labels.items():
#     for key2, value in labels.items():
#         if key1 != key2:
#             for key3, value in labels.items():
#                 if key3 != key2 and key3 != key1:
#                     area12 = 0.5*(surface[key1] + surface[key2] - surface[key3])
#                     area13 = 0.5*(surface[key1] + surface[key3] - surface[key2])
#                     area23 = 0.5*(surface[key2] + surface[key3] - surface[key1])
#                     print(f"Specific surface area of {key1}-{key2}:  {area12*1e-6:.5f} [1/µm]")
#                     print(f"Specific surface area of {key1}-{key3}:  {area13*1e-6:.5f} [1/µm]")
#                     print(f"Specific surface area of {key2}-{key3}:  {area23*1e-6:.5f} [1/µm]")
#             break
#     break

# array = data.create_voxelized_sphere(200)
# timer = []
# timer.append(time.time())

# center = np.s_[1:-1,1:-1,1:-1]
# left   = np.s_[ :-2,1:-1,1:-1]
# right  = np.s_[2:  ,1:-1,1:-1]
# bottom = np.s_[1:-1, :-2,1:-1]
# top    = np.s_[1:-1,2:  ,1:-1]
# back   = np.s_[1:-1,1:-1, :-2]
# front  = np.s_[1:-1,1:-1,2:  ]

# strel = np.zeros((3,3,3))
# strel[0,1,1] = 1
# strel[2,1,1] = 1
# strel[1,0,1] = 1
# strel[1,2,1] = 1
# strel[1,1,0] = 1
# strel[1,1,2] = 1
# strel[1,1,1] = -6
# timer = []
# timer.append(time.time())
# laplace = ( array[right] - 2*array[center] + array[left]
#            +array[top]   - 2*array[center] + array[bottom]
#            +array[front] - 2*array[center] + array[back]  )
# timer.append(time.time())
# laplace2 = convolve(array, strel, mode='nearest')[center]
# timer.append(time.time())
# laplace3 = scipy.signal.fftconvolve(array, strel, mode='valid')
# timer.append(time.time())

# print(timer[1]-timer[0])
# print(timer[2]-timer[1])
# print(timer[3]-timer[2])

# print(np.sum(laplace-laplace2))
# print(np.sum(laplace-laplace3))

# radii = [100]


# rad1i 
# vol_fraction = {'theo':np.zeros(len(radii)), 'num':np.zeros(len(radii))}

# methods = ['theo', 'faces', 'marching', 'conv_marching', 'porespy', 'gradient', 'conv_gradient']
# area = {method: np.zeros(len(radii)) for method in methods}
# times = {method: np.zeros(len(radii)) for method in methods[1:]}

# count = 0
# for Radius in radii:
#     sharp_field = data.create_voxelized_sphere(Radius)

#     # Optional plotting
#     # title = 'Sphere with {} pixel radius'.format(Radius)
#     # data.plotField2D(1-sharp_field[:,:,Radius+10], title, dpi=200)

#     vol_fraction['theo'][count] = 4/3*np.pi*Radius**3/(sharp_field.size)
#     vol_fraction['num'][count] = metrics.volume_fraction(sharp_field, 1)

#     timer = []
#     area['theo'][count] = 4*np.pi*Radius**2/(np.prod(sharp_field.shape)*dx**3)
#     timer.append(time.time())
#     area['faces'][count] = 3*tau.surface_area(sharp_field, phases=[1]).item()
#     timer.append(time.time())
#     area['marching'][count] = metrics.specific_surface_area_marching(sharp_field)
#     timer.append(time.time())
#     smooth_field = metrics.smooth_with_convolution(sharp_field)
#     area['conv_marching'][count] = metrics.specific_surface_area_marching(smooth_field)
#     timer.append(time.time())
#     area['porespy'][count]  = metrics.specific_surface_area_porespy(sharp_field)
#     timer.append(time.time())
#     area['gradient'][count] = metrics.specific_surface_area(sharp_field)
#     timer.append(time.time())
#     area['conv_gradient'][count] = metrics.specific_surface_area(sharp_field, smooth=1)
#     timer.append(time.time())

#     for i, method in enumerate(methods[1:]):
#         times[method][count] = timer[i+1]-timer[i]

#     print(f"Finished radius = {Radius}.")
#     count = count +1

# colors = ['black', 'red', 'blue', 'purple', 'orange', 'lime', 'green']
# line_styles = ['-', '-', '-.', ':', '--', '-', '-.','-']

# fig, ax = plt.subplots(figsize=(10, 4))
# for i, method in enumerate(methods[1:]):
#     ax.loglog(radii, times[method], label=method, color=colors[i+1], linestyle=line_styles[i+1])

# ax.set_xlabel('cells per radius')
# ax.set_ylabel('Computation time [s]')
# ax.set_title('Computation time in s')
# ax.legend()
# ax.grid()

# plt.show()