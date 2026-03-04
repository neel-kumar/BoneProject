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

start = time.time()

voxeld = np.load('s01_voxel.npy')
# sub = voxeld[80:180, 700:800, 800:900]
sub = voxeld[550:650, 700:800, 800:900]
sub = sub[::2, ::2, ::2]
print(sub.shape)
print(f'true cnt: {sub.sum()}')



end = time.time()
print(f"Elapsed: {end - start:.2f} seconds")
