# two image:
# check old for s/e/v -open, and if shape point
# check new/current to check for simple point


# PRIMARY thinning:
# -each iteration is 3 scans
#
# 1st: unmarked s-open is marked if it is a shape point, if not a shape or simple point it is deleted, else unmarked
#
# 2nd: e-open point can never be a shape point (can never be marked), fix fig 6 drilling, so unmarked, e-open, simple points satisfying Condition 3 are deleted
#
# 3rd: unmarked v-open points that are simple points are deleted


# FINAL thinning:
# -1 iteration
# -1 scan
#
# all black points deleted if erodable (see Def 6)

# // look at 8-subfield for parallelization later

import time
import subprocess
import numpy as np
from scipy.ndimage import label
from solid import *

def scout(filename, obj, to_stl=False):
    st = time.time()

    scad_render_to_file(obj, filename + '.scad')
    print('to openscad: ' + filename + '.scad')
    if to_stl:
        print("to stl")
        subprocess.run([r'C:\Program Files\OpenSCAD\openscad.exe', '-o', filename + '.stl', './' + filename + '.scad'])

    print(f"scout elapsed: {time.time() - st:.2f} seconds")

def render_3d_mat(mat, cube_size=0.1):
    positions = np.argwhere(mat)
    if len(positions) == 0:
        return cube(0)
    cells = None
    for z, y, x in positions:
        c = translate([x * cube_size, y * cube_size, z * cube_size])(cube(cube_size))
        cells = c if cells is None else cells + c
    return cells

def get_neighbors(p, mat):
    z, y, x = p
    return mat[z-1:z+2, y-1:y+2, x-1:x+2]

def get_s_points(p, mat):
    z, y, x = p
    return {
        'N': mat[z, y-1, x], 'S': mat[z, y+1, x],
        'E': mat[z, y, x+1], 'W': mat[z, y, x-1],
        'T': mat[z-1, y, x], 'B': mat[z+1, y, x]
    }

def is_simple(p, current_mat):
    # if p is a (26, 6) simple point
    n_star = get_neighbors(p, current_mat).copy()
    n_star[1, 1, 1] = 0
    if np.sum(n_star) == 0: return False
    
    s_vals = [n_star[1,1,0], n_star[1,1,2], n_star[1,0,1], n_star[1,2,1], n_star[0,1,1], n_star[2,1,1]]
    if all(s_vals): return False
    
    _, n_26 = label(n_star, structure=np.ones((3,3,3)))
    if n_26 != 1: return False
    
    white_18 = np.zeros((3,3,3), dtype=bool)
    for dz, dy, dx in np.ndindex((3,3,3)):
        if (abs(dz-1)+abs(dy-1)+abs(dx-1) <= 2) and not n_star[dz, dy, dx]:
            white_18[dz, dy, dx] = True
    struct6 = np.array([[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]])
    _, n_6 = label(white_18, structure=struct6)
    return n_6 == 1

def is_s_open(p, old_mat):
    # s-open if any s-neighbor is white.
    return any(v == 0 for v in get_s_points(p, old_mat).values())

def is_shape_point(p, old_mat):
    # condition 2: surface-like shape point
    s = get_s_points(p, old_mat)
    for a, d in [('T','B'), ('N','S'), ('W','E')]:
        if s[a] == 0 and s[d] == 1:
            return True 
    return False

def thick(p, old_mat):
    # def 5: thick
    s = get_s_points(p, old_mat)
    black_count = sum(s.values())
    return 1 if black_count >= 3 else 0

def thin_3d_saha(voxels, final_thinning = True):
    MAXINT = 1000000
    img = np.where(voxels > 0, 0, -MAXINT)
    img = np.pad(img, 2, constant_values=-MAXINT)
    
    iter_n = 1
    while True:
        changed = False
        thr = -MAXINT + iter_n
        old_img_mask = (img >= thr-1) 
        
        # 1: s-open points
        for p_idx in np.argwhere(img == 0):
            p = tuple(p_idx)
            if is_s_open(p, old_img_mask):
                if is_shape_point(p, old_img_mask):
                    img[p] = iter_n 
                elif is_simple(p, img >= 0):
                    img[p] = thr 
                    changed = True

        # 2: e-open points
        for p_idx in np.argwhere(img == 0):
            p = tuple(p_idx)
            if is_simple(p, img >= 0): 
                img[p] = thr
                changed = True

        # 3: v-open points
        for p_idx in np.argwhere(img == 0):
            p = tuple(p_idx)
            if is_simple(p, img >= 0):
                img[p] = thr
                changed = True

        if not changed: break
        iter_n += 1

    if final_thinning:
        # final-thinning: thick slanted surfaces
        final_old_mask = (img >= 0)
        for p_idx in np.argwhere(img >= 0):
            p = tuple(p_idx)
            # def 6: erodable if simple, conditions 4, 5, or 6
            if is_simple(p, img >= 0) and thick(p, final_old_mask):
                img[p] = -MAXINT + iter_n
            
    return img[2:-2, 2:-2, 2:-2] >= 0

start = time.time()

voxeld = np.load('s01_voxel.npy')
sub = voxeld[80:180, 750:850, 850:950]
sub = sub[::2, ::2, ::2]
print('input extracted')

labeled, n = label(sub, structure=np.ones((3,3,3)))
if n > 0:
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    sub = (labeled == counts.argmax())

ske = thin_3d_saha(sub)
print(f'true cnt: {ske.sum()}')
scout('../scad/ske8.1', render_3d_mat(ske))
# np.save('s01_8_skeleton.npy', skeleton);

print(f"elapsed: {time.time() - start:.2f} seconds")
