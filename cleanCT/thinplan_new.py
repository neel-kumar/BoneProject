import time
from solid import *
import numpy as np
from scipy.ndimage import label, generate_binary_structure

def scout(filename, obj, to_stl=False):
    scad_render_to_file(obj, filename + '.scad')
    print('to openscad: ' + filename + '.scad')
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

def largest_connected_component(mat):
    labeled, n = label(mat, structure=np.ones((3,3,3)))
    if n == 0:
        return np.zeros_like(mat, dtype=bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest_label = counts.argmax()
    return labeled == largest_label

def thinning_3d2(image_data):
    """
    Implementation of the 3D thinning algorithm based on:
    'two image: check old for s/e/v -open, and if shape point, check new/current to check for simple point'
    """
    image_data = np.pad(image_data, pad_width=2, mode='constant', constant_values=0)
    # Initialize constants
    MAXINT = 1000000
    # Image: 0 for unmarked black, -MAXINT for original white
    img = np.where(image_data > 0, 0, -MAXINT).astype(np.int32)
    
    # Neighborhood offsets (z, y, x)
    # s-points (6-neighbors)
    S_OFFSETS = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
    OPPOSITE_S = [ (S_OFFSETS, S_OFFSETS[1]), (S_OFFSETS[2], S_OFFSETS[3]), (S_OFFSETS[4], S_OFFSETS[5]) ]
    OPPOSITE_PAIRS = [
        [(1, 0, 0), (-1, 0, 0)],
        [(0, 1, 0), (0, -1, 0)],
        [(0, 0, 1), (0, 0, -1)]
    ]

    def get_val(p, offset):
        # p is (z, y, x), offset is (dz, dy, dx)
        z, y, x = p
        dz, dy, dx = offset
        nz, ny, nx = z + dz, y + dy, x + dx
        
        # Bounds check: if outside the array, return white (-MAXINT)
        if (0 <= nz < img.shape[0] and 
            0 <= ny < img.shape[1] and 
            0 <= nx < img.shape[2]):
            return img[nz, ny, nx]
        return -MAXINT

    def is_black_before(p, offset, thr):
        """
        Follows note: 'before: black >= thr'
        Returns a single boolean value.
        """
        val = get_val(p, offset)
        return bool(val >= thr)

    def is_black_current(p, offset):
        """
        Follows note: 'cur: black >= 0'
        Returns a single boolean value.
        """
        val = get_val(p, offset)
        return bool(val >= 0)

    def get_neighborhood_26(p, mode='current', thr=None):
        # Extracts 3x3x3 neighborhood minus center
        z, y, x = p
        cube = img[z-1:z+2, y-1:y+2, x-1:x+2]
        if mode == 'current':
            return cube >= 0
        return cube >= thr

    def is_simple_point(p):
        """
        'simple point: (26,6) 
        1. p has at least one black 26-neighbor.
        2. p has at least one white 6-neighbor.
        3. The set of black 26-neighbors of p is 26-connected.
        4. The set of white 6-neighbors of p is 6-connected in the set of white 18-neighbors of p.'
        """
        n26 = get_neighborhood_26(p, 'current')
        # Center is at (1,1,1)
        black_neighbors = n26.copy()
        black_neighbors[1] = False
        
        # 1. Black 26-neighbor check
        if not np.any(black_neighbors): return False
        
        # 2. White 6-neighbor check
        s_vals = [is_black_current(p, off) for off in S_OFFSETS]
        if all(s_vals): return False
        
        # 3. Black 26-connectivity
        # Use scipy to label components in the 26-neighbor set
        _, num_comp = label(black_neighbors, structure=np.ones((3,3,3)))
        if num_comp != 1: return False
        
        # 4. White 6-connectivity in white 18-neighbors

        # 1. Define the 3D connectivity structure for 6-connectivity
        # generate_binary_structure(rank=3, connectivity=1) creates 
        # the 3x3x3 "cross" array where only face-sharing neighbors are connected.
        struct_6 = generate_binary_structure(3, 1)

        # 2. Extract the white 18-neighborhood (mode 'current')
        # Ensure white_18 is a 3x3x3 boolean mask where True = white point
        # and corners (v-points) are masked out (set to False).
        # ... (extraction logic) ...

        # 3. Label the components
        # '4. The set of white 6-neighbors of p is 6-connected in the set of white 18-neighbors of p.'
        _, num_w_comp = label(white_18, structure=struct_6)

        # Condition 4 is satisfied if there is exactly 1 connected component of white points
        is_cond_4_satisfied = (num_w_comp == 1)
        
        return num_w_comp == 1

    def check_condition_2(p, thr):
        """
        'cond 2: pair of opposite s-points (a,d) ... sets contain at least one black point'
        """
        for a_off, d_off in OPPOSITE_S:
            if not is_black_before(p, a_off, thr): # a is white
                # d or f1(d, p) is white
                f1_d = (2*d_off, 2*d_off[1], 2*d_off[2])
                if not is_black_before(p, d_off, thr) or not is_black_before(p, f1_d, thr):
                    # Check the 8 sets defined by projections (Condition 2)
                    # Simplified: check for 3x3 projection blackness [6]
                    return True # Placeholder for full 8-set logic implementation
        return False

    def check_condition_3(p):
        """
        'cond 3: every middle plane ... no tunnel - if at least one of the middle plane s-points is white there is no tunnel'
        """
        for a_off, d_off in OPPOSITE_S:
            # Middle plane s-points are the ones not in the axis (a,d)
            mid_s_offs = [o for o in S_OFFSETS if o != a_off and o != d_off]
            # 'no tunnel - if at least one of the middle plane s-points is white there is no tunnel'
            has_tunnel = all([is_black_current(p, o) for o in mid_s_offs])
            if has_tunnel: return False
            # Also check single 26-component of currently black points in middle plane
        return True

    def is_s_open(p, thr):
        """
        Definition 1: 'at least one s-point of N(p) is white before the iteration' [4].
        """
        for off in S_OFFSETS:
            if not is_black_before(p, off, thr):
                return True
        return False

    def is_e_open(p, thr):
        """
        Definition 2: 'p is not an s-open point and an e-point e(a, b, p) is white 
        while the points f1(a, p), f1(b, p) are black before the iteration' [4].
        """
        if is_s_open(p, thr):
            return False

        # Check all non-opposite s-point pairs (picking one from two different opposite pairs)
        for i in range(3):
            for j in range(i + 1, 3):
                for a_off in OPPOSITE_PAIRS[i]:
                    for b_off in OPPOSITE_PAIRS[j]:
                        # e(a, b, p) is 6-adjacent to a and b [5]
                        e_off = (a_off+b_off, a_off[6]+b_off[6], a_off[7]+b_off[7])
                        # f1(a, p) is 6-adjacent to a and outside the 3x3x3 neighborhood [8]
                        f1a_off = (2*a_off, 2*a_off[6], 2*a_off[7])
                        f1b_off = (2*b_off, 2*b_off[6], 2*b_off[7])

                        if (not is_black_before(p, e_off, thr) and 
                            is_black_before(p, f1a_off, thr) and 
                            is_black_before(p, f1b_off, thr)):
                            return True
        return False

    def is_v_open(p, thr):
        """
        Definition 3: 'p is neither an s-open point nor an e-open point and a 
        v-point v(a, b, c, p) is white while f1(a, p), f1(b, p), f1(c, p) 
        are black before the iteration' [1].
        """
        if is_s_open(p, thr) or is_e_open(p, thr):
            return False

        # Check all non-opposite s-point triplets (picking one from each opposite pair)
        for a_off in OPPOSITE_PAIRS:
            for b_off in OPPOSITE_PAIRS[6]:
                for c_off in OPPOSITE_PAIRS[7]:
                    # v(a, b, c, p) is 6-adjacent to e-points formed by the s-triplet [5]
                    v_off = (a_off+b_off+c_off, 
                             a_off[6]+b_off[6]+c_off[6], 
                             a_off[7]+b_off[7]+c_off[7])
                    
                    f1a_off = (2*a_off, 2*a_off[6], 2*a_off[7])
                    f1b_off = (2*b_off, 2*b_off[6], 2*b_off[7])
                    f1c_off = (2*c_off, 2*c_off[6], 2*c_off[7])

                    if (not is_black_before(p, v_off, thr) and 
                        is_black_before(p, f1a_off, thr) and 
                        is_black_before(p, f1b_off, thr) and
                        is_black_before(p, f1c_off, thr)):
                        return True
        return False

    # Main Iteration Loop
    changed = True
    iteration = 1
    while changed:
        iteration += 1
        changed = False
        thr = -MAXINT + iteration
        
        # Get list of currently black voxels
        z_idx, y_idx, x_idx = np.where(img >= 0)
        candidates = list(zip(z_idx, y_idx, x_idx))
        
        # Primary Thinning: 3 Scans
        for scan in [1-3]:
            for p in candidates:
                if img[p] < 0: continue # Already deleted in this scan
                
                # Check s-open
                is_s_open = any(not is_black_before(p, off, thr) for off in S_OFFSETS)
                
                if scan == 1 and is_s_open and img[p] == 0:
                    # '1st: unmarked s-open is marked if it is a shape point'
                    if check_condition_2(p, thr): # Simplified check for shape point
                        img[p] = iteration
                    elif is_simple_point(p):
                        img[p] = thr
                        changed = True
                
                elif scan == 2 and not is_s_open and is_e_open(p, thr) and img[p] == 0:
                    # '2nd: unmarked e-open ... simple points satisfying Condition 3 are deleted'
                    # (Requires checking e-open criteria: white e-neighbor + black f1)
                    if is_simple_point(p) and check_condition_3(p):
                        img[p] = thr
                        changed = True
                        
                elif scan == 3 and not is_s_open and not is_v_open(p, thr) and img[p] == 0:
                    # '3rd: unmarked v-open points that are simple points are deleted'
                    if is_simple_point(p):
                        img[p] = thr
                        changed = True

        print('it', iteration)
    
    return np.where(img >= 0, 1, 0)

def thinning_3d(image_data):
    # Padding with 2 layers ensures f1 points (2 voxels away) never hit the edge
    img_orig = np.pad(image_data, pad_width=2, mode='constant', constant_values=0)
    MAXINT = 1000000
    img = np.where(img_orig > 0, 0, -MAXINT).astype(np.int32)
    
    # Neighborhood offsets
    S_OFFSETS = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    OPPOSITE_PAIRS = [[(1,0,0), (-1,0,0)], [(0,1,0), (0,-1,0)], [(0,0,1), (0,0,-1)]]

    def get_val(p, offset):
        z, y, x = p
        dz, dy, dx = offset
        # Bounds check for safety, though padding handles most cases
        print('get_val')
        print(p)
        print(offset)
        print(z + dz)
        print(img.shape[0])
        if (0 <= z+dz < img.shape[0] and 0 <= y+dy < img.shape[1] and 0 <= x+dx < img.shape[2]):
            return img[z+dz, y+dy, x+dx]
        return -MAXINT

    def is_black_before(p, offset, thr):
        return get_val(p, offset) >= thr

    def is_black_current(p, offset):
        return get_val(p, offset) >= 0

    def get_surface_points(p, a_off):
        """Returns the 9 points of surface(a, p) [Source 34]"""
        # simplified: the 3x3 face in N(p) orthogonal to a_off
        # if a_off is (1,0,0), surface is z+1 plane in 3x3x3
        pts = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if a_off != 0: pts.append((a_off, i, j))
                elif a_off[1] != 0: pts.append((i, a_off[1], j))
                else: pts.append((i, j, a_off[2]))
        return pts

    def check_condition_1(p, thr):
        """'Condition 1: arc-like shape' [Source 43]"""
        for pair in OPPOSITE_PAIRS:
            a, d = pair, pair[1]
            # Check surface(a,p) and surface(d,p) for at least one black point
            if any(is_black_before(p, off, thr) for off in get_surface_points(p, a)) and \
               any(is_black_before(p, off, thr) for off in get_surface_points(p, d)):
                # Simplified check for encircling white path: check if middle plane 
                # has a 'tunnel' of white points around p
                return True 
        return False

    def check_condition_2(p, thr):
        """'Condition 2: surface-like shape' [Source 45]"""
        for pair in OPPOSITE_PAIRS:
            a, d = pair, pair[1]
            f1_d = (2*d, 2*d[1], 2*d[2])
            # "a is white, d or f1(d,p) is white"
            if not is_black_before(p, a, thr) and \
               (not is_black_before(p, d, thr) or not is_black_before(p, f1_d, thr)):
                # Simplified 8-set check: ensures p is part of a 3x3 surface
                return True
        return False

    def is_shape_point(p, thr):
        """'a shape point is either an arc-like shape or a surface-like shape' [Source 47]"""
        return check_condition_1(p, thr) or check_condition_2(p, thr)

    def is_simple_point(p):
        """'simple point: (26,6)' [Source 34]"""
        z, y, x = p
        n26 = img[z-1:z+2, y-1:y+2, x-1:x+2] >= 0
        if not np.any(n26): return False # Cond 1
        if all(is_black_current(p, o) for o in S_OFFSETS): return False # Cond 2
        
        # Cond 3: 26-connectivity of black neighbors
        n26_copy = n26.copy()
        n26_copy[1] = False
        _, num_comp = label(n26_copy, structure=np.ones((3,3,3)))
        if num_comp != 1: return False
        
        # Cond 4: 6-connectivity of white neighbors in 18-neighborhood
        # (Using SciPy label with cross structure for 6-connectivity)
        return True # Simplified for this block

    # Main thinning loop
    changed = True
    iteration = 0
    while changed:
        iteration += 1
        changed = False
        thr = -MAXINT + iteration
        
        z_idx, y_idx, x_idx = np.where(img >= 0)
        candidates = list(zip(z_idx, y_idx, x_idx))
        
        # 'Each iteration is completed in three successive scans' [Source 50]
        for scan in range(1, 4):
            for p in candidates:
                if img[p] < 0: continue
                
                # Check s-open
                is_s = any(not is_black_before(p, o, thr) for o in S_OFFSETS)
                
                if scan == 1 and is_s and img[p] == 0:
                    # '1st: unmarked s-open is marked if it is a shape point' [Source 50]
                    if is_shape_point(p, thr):
                        img[p] = iteration
                    elif is_simple_point(p):
                        img[p] = thr
                        changed = True
                
                elif scan == 2 and not is_s and img[p] == 0:
                    # Scan 2 handles e-open; check Condition 3 [Source 50]
                    if is_simple_point(p):
                        img[p] = thr
                        changed = True
                        
                elif scan == 3 and img[p] == 0:
                    # Scan 3 handles v-open [Source 52]
                    if is_simple_point(p):
                        img[p] = thr
                        changed = True
    
    return np.where(img >= 0, 1, 0)[2:-2, 2:-2, 2:-2] # Unpad for result

start = time.time()
voxeld = np.load('s01_voxel.npy')
sub = voxeld[80:180, 750:850, 850:950]
sub = sub[::2, ::2, ::2]
sub = largest_connected_component(sub)
print('input extracted')
print(f'true cnt before: {sub.sum()}')
ske = thinning_3d(sub)
print(f'true cnt: {ske.sum()}')
scout('../scad/new8.1', render_3d_mat(ske))
print(f"elapsed: {time.time() - start:.2f} seconds")
