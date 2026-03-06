from datetime import datetime
from pathlib import Path
import time
import cv2
import numpy as np
import re
import glob
import sys

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

def slice_mid(mat, dim = [100,100]):
    cy, cx = mat.shape[0] // 2, mat.shape[1] // 2
    return mat[cy-dim[0]:cy+dim[0], cx-dim[1]:cx+dim[1]]

# First argument is the directory name containing the CT Scan
# Specify the directory name without the /, since this string is used to construct the output file name.
in_dir = sys.argv[1]
print(f"Processing directory {in_dir}")

thresh_val = 85

start = time.time()

# cur_vert = 0
# voxel_size = 0.1
# objs = []
voxelized = []
for file in sorted(glob.glob(f"{in_dir}/*/*rec[0-9]*.bmp")):
    print(f"{datetime.now()} - Loading file: {file}")
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    binary = img > thresh_val
    voxelized.append(binary)

voxelized = np.stack(voxelized)
print(voxelized.shape)
print(f"Saving file: {in_dir}_voxel.npy")
np.save(f"{in_dir}_voxel.npy", voxelized)
# scout('../scad/slices', obj)
end = time.time()
print(f"Elapsed: {end - start:.2f} seconds")
