import time
import subprocess
import numpy as np
from scipy.ndimage import *
from skimage.morphology import skeletonize

from solid import *

def scout(filename, obj, to_stl=False):
    scad_render_to_file(obj, filename + '.scad')
    if to_stl:
        print("to stl")
        subprocess.run([r'C:\Program Files\OpenSCAD\openscad.exe', '-o', filename + '.stl', './' + filename + '.scad'])

def render_3d_mat(mat, cube_size=0.1):
    positions = np.argwhere(mat)
    if len(positions) == 0:
        return cube(0)
    cells = None
    for z, y, x in positions:
        c = translate([x * cube_size, y * cube_size, z * cube_size])(cube(cube_size))
        cells = c if cells is None else cells + c
    return cells


def render_2d_mat(mat, t_vert, cube_size=0.1):
    cells = None
    for i, row in enumerate(mat):
        for j, val in enumerate(row):
            if val:
                c = translate([j * cube_size, i * cube_size, t_vert])(cube(cube_size))
                cells = c if cells is None else cells + c
    return cells

STRUCT26 = np.ones((3, 3, 3), dtype=bool)
STRUCT6  = np.zeros((3, 3, 3), dtype=bool)
STRUCT6[1, 1, 0] = STRUCT6[1, 1, 2] = True
STRUCT6[1, 0, 1] = STRUCT6[1, 2, 1] = True
STRUCT6[0, 1, 1] = STRUCT6[2, 1, 1] = True
STRUCT6[1, 1, 1] = True

def largest_connected_component(mat):
    labeled, n = label(mat, structure=STRUCT26)
    if n == 0:
        return np.zeros_like(mat, dtype=bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # exclude background
    largest_label = counts.argmax()
    return labeled == largest_label

# voxeld = np.load('s01_voxel.npy')
# # sub = voxeld[80:180, 700:800, 800:900]
# # sub = voxeld[550:650, 700:800, 800:900]
# sub = voxeld[80:180, 750:850, 850:950]
# sub = sub[::2, ::2, ::2]
#
# np.save('s01_voxel_8.npy', sub);
# sub = (largest_connected_component(sub))
# print(sub.shape)
# ske = skeletonize(sub)
# print(ske.shape)
# print(f'true cnt: {ske.sum()}')
#
# scout('../scad/test', render_3d_mat(ske));

start = time.time()

sub = np.load('s01_sub_ske_full.npy')
print(sub.shape)
scout('../scad/ske_full', render_3d_mat(sub));
print(f'true cnt: {sub.sum()}')

print(f"elapsed: {time.time() - start:.2f} seconds")
