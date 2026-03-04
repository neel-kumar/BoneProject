from pathlib import Path
import time
import cv2
import numpy as np
import re
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, maximum_filter
from skimage.morphology import thin

from solid import *
from solid.utils import *

def scout(filename, obj, to_stl = False):
    scad_render_to_file(obj, filename + '.scad')
    if to_stl:
        print("to stl")
        subprocess.run([r'C:\Program Files\OpenSCAD\openscad.exe', '-o', filename + '.stl', './' + filename + '.scad'])

def render_2d_mat(mat, t_vert, cube_size=0.1):
    cells = None
    for i, row in enumerate(mat):
        for j, val in enumerate(row):
            if val:
                c = translate([j * cube_size, i * cube_size, t_vert])(cube(cube_size))
                cells = c if cells is None else cells + c
    return cells

def render_3d_mat(mat, cube_size=0.1):
    positions = np.argwhere(mat)
    if len(positions) == 0:
        return cube(0)
    
    cells = None
    for z, y, x in positions:
        c = translate([x * cube_size, y * cube_size, z * cube_size])(cube(cube_size))
        cells = c if cells is None else cells + c
    return cells

def max_thickness(bone, skeleton, voxel_size_mm) -> float:
    edt = distance_transform_edt(bone)          # shape same as bone, float64

    # Sample EDT only at skeletal locations
    skeletal_radii = edt[skeleton.astype(bool)]  # 1-D array of radii in voxels

    if skeletal_radii.size == 0:
        return 0.0

    max_radius_voxels = skeletal_radii.max()
    max_thickness_mm  = 2.0 * max_radius_voxels * voxel_size_mm
    return max_thickness_mm

start = time.time()

voxeld = np.load('s01_voxel.npy')
# sub = voxeld[80:180, 750:800, 800:900]
sub = voxeld[80:180, 750:850, 850:950]
# sub = voxeld[550:650, 700:800, 800:900]
sub = sub[::2, ::2, ::2]
print(sub.shape)
print(f'true cnt: {sub.sum()}')
#
# # ske = skeletonize(sub)
# # print(ske.shape)
# # print(f'true cnt: {ske.sum()}')
# #
# # dist = distance_transform_edt(sub)
# # local_max = (dist == maximum_filter(dist, size=2))
# # medial = sub & local_max
# # print(f'true cnt: {medial.sum()}')
# #
# ske_2d = np.zeros_like(sub, dtype=bool)
#
# # for z in range(sub.shape[0]):
# #     slice_2d = sub[z]
# #     if slice_2d.any():
# #         ske_2d[z] = skeletonize(slice_2d)
#
# for z in range(sub.shape[0]):
#     slice_2d = sub[z]
#     if slice_2d.any():
#         ske_2d[z] = thin(slice_2d, max_num_iter=3)
# print(f'true cnt: {ske_2d.sum()}')

# scout('../scad/ske7.3', render_3d_mat(medial));
# scout('../scad/ske6.7', render_3d_mat(ske_2d));
scout('../scad/voxel7', render_3d_mat(sub));

end = time.time()
print(f"Elapsed: {end - start:.2f} seconds")
