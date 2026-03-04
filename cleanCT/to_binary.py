from pathlib import Path
import time
import cv2
import numpy as np
import re

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

in_dir = 'C:/Users/dumbo/code/BoneProject/micro_CT/micro_CT/micro_CT/S.01/A_Rec/'
out_dir = 'C:/Users/dumbo/code/BoneProject/cleanCT/bmp/'
thresh_val = 85

start = time.time()

directory = Path(in_dir)
# cur_vert = 0
# voxel_size = 0.1
# objs = []
voxelized = []
for file in directory.iterdir():  
    if not file.is_file() or Path(file).suffix != '.bmp' or not bool(re.match(r'A__rec\d{8}$', Path(file).stem)):
        continue
    print(Path(file).stem)

    filen = Path(file).stem + '.bmp'
    path = in_dir + filen
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    binary = img > thresh_val
    voxelized.append(binary)

voxelized = np.stack(voxelized)
print(voxelized.shape)
np.save('s01_voxel.npy', voxelized)
# scout('../scad/slices', obj)
end = time.time()
print(f"Elapsed: {end - start:.2f} seconds")
