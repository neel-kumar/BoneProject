import time
import subprocess
import numpy as np
from scipy.ndimage import label
from skimage.morphology import skeletonize
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

def largest_connected_component(mat):
    labeled, n = label(mat, structure=np.ones((3,3,3)))
    if n == 0:
        return np.zeros_like(mat, dtype=bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest_label = counts.argmax()
    return labeled == largest_label

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

def get_middle_plane(p, pair_name, mat):
    z, y, x = p
    if pair_name in ['T', 'B']: return mat[z, y-1:y+2, x-1:x+2]
    if pair_name in ['N', 'S']: return mat[z-1:z+2, y, x-1:x+2]
    return mat[z-1:z+2, y-1:y+2, x]

def is_simple(p, current_mat):
    # Checks if p is a (26, 6) simple point
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
    return any(v == 0 for v in get_s_points(p, old_mat).values())

def is_shape_point(p, old_mat):
    # Condition 2: Surface-like shape point
    s = get_s_points(p, old_mat)
    for a, d in [('T','B'), ('N','S'), ('W','E')]:
        if s[a] == 0:
            m_plane = get_middle_plane(p, a, old_mat)
            if np.sum(m_plane) >= 2:
                return True
    return False

def satisfies_condition_3(p, current_mat, old_mat):
    # Condition 3: prevents drilling by checking 2D connectivity in coordinate planes
    for pair in [('T','B'), ('N','S'), ('W','E')]:
        m_plane = get_middle_plane(p, pair[0], current_mat).copy()
        m_plane[1, 1] = 0
        _, n_comp = label(m_plane, structure=np.ones((3,3)))
        if n_comp != 1: return False
        s_in_m = [m_plane[0,1], m_plane[2,1], m_plane[1,0], m_plane[1,2]]
        if all(s_in_m): return False
    return True

def thick(p, old_mat):
    # Definition 5: thick(a,d,p)
    s = get_s_points(p, old_mat)
    black_count = sum(s.values())
    return 1 if black_count >= 3 else 0

def thin_3d_saha(voxels, final_thinning=True):
    MAXINT = 1000000
    img = np.where(voxels > 0, 0, -MAXINT)
    img = np.pad(img, 2, constant_values=-MAXINT)
    iter_n = 1
    while True:
        changed = False
        thr = -MAXINT + iter_n
        old_img_mask = (img >= thr-1) 
        
        # 1st: unmarked s-open is marked if it is a shape point, if not a shape or simple point it is deleted, else unmarked
        for p_idx in np.argwhere(img == 0):
            p = tuple(p_idx)
            if is_s_open(p, old_img_mask):
                if is_shape_point(p, old_img_mask):
                    img[p] = iter_n 
                elif is_simple(p, img >= 0):
                    img[p] = thr 
                    changed = True

        # 2nd: e-open point can never be a shape point (can never be marked), fix fig 6 drilling, so unmarked, e-open, simple points satisfying Condition 3 are deleted
        for p_idx in np.argwhere(img == 0):
            p = tuple(p_idx)
            if is_simple(p, img >= 0) and satisfies_condition_3(p, img >= 0, old_img_mask): 
                img[p] = thr
                changed = True

        # 3rd: unmarked v-open points that are simple points are deleted
        for p_idx in np.argwhere(img == 0):
            p = tuple(p_idx)
            if is_simple(p, img >= 0):
                img[p] = thr
                changed = True

        if not changed: break
        iter_n += 1

    if final_thinning:
        # all black points deleted if erodable (see Def 6)
        final_old_mask = (img >= 0)
        for p_idx in np.argwhere(img >= 0):
            p = tuple(p_idx)
            if is_simple(p, img >= 0) and thick(p, final_old_mask):
                img[p] = -MAXINT + iter_n
            
    return img[2:-2, 2:-2, 2:-2] >= 0

start = time.time()

voxeld = np.load('s01_voxel.npy')
sub = voxeld[80:180, 750:850, 850:950]
sub = sub[::2, ::2, ::2]
sub = largest_connected_component(sub)
ske = thin_3d_saha(sub)
print(f'true cnt: {ske.sum()}')
scout('../scad/ske8.1', render_3d_mat(ske))
# np.save('s01_8_skeleton.npy', skeleton);

print(f"elapsed: {time.time() - start:.2f} seconds")
