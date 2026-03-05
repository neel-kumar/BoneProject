import time
import subprocess
from collections import deque
import numpy as np
from scipy.ndimage import label
from solid import *
import json

MAXINT = 1000000
S_OFFS = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
N26_OFFS = []
for dz in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dz == 0 and dy == 0 and dx == 0: continue
            N26_OFFS.append((dz, dy, dx))
                

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

def count_ones_3x3(mat, p):
    """Count non-zero values in the 3x3x3 neighborhood around point p (including p itself)."""
    z, y, x = p
    nb = mat[max(0, z-1):z+2, max(0, y-1):y+2, max(0, x-1):x+2]
    return int((nb != 0).sum())

def classify_skeleton_point(mat, p):
    """Classify a skeleton point as 'rod' or 'surface' using PCA on its 26-neighborhood.

    Collects positions of all 26-connected skeleton neighbors relative to p, then
    applies SVD to find the two largest singular values (s0 >= s1 >= s2).
    A high s1/s0 ratio means neighbors are coplanar -> surface.
    A low ratio means neighbors are colinear -> rod.
    Works for any orientation, including slanted rods and plates.
    """
    z, y, x = p
    neighbors = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                nz, ny, nx = z + dz, y + dy, x + dx
                if 0 <= nz < mat.shape[0] and 0 <= ny < mat.shape[1] and 0 <= nx < mat.shape[2]:
                    if mat[nz, ny, nx]:
                        neighbors.append([dz, dy, dx])

    if len(neighbors) < 5 and not cond2_len(p, mat, 1, min8=3):
        return 'rod'
    else:
        return 'surface'

    pts = np.array(neighbors, dtype=float)
    pts -= pts.mean(axis=0)
    _, s, _ = np.linalg.svd(pts, full_matrices=False)

    # s[1]/s[0] near 0 -> colinear (rod); near 1 -> coplanar (surface)
    if s[0] > 0 and s[1] / s[0] > 0.5:
        return 'surface'
    return 'rod'

def render_3d_mat_color2(mat, cube_size=0.1):
    binary = mat
    positions = np.argwhere(binary)
    if len(positions) == 0:
        return cube(0)
    cells = None
    for z, y, x in positions:
        kind = classify_skeleton_point(binary, (z, y, x))
        if kind == 'rod':
            c = color([0.4, 0.5, 0.8])(cube(cube_size))  # light blue - rod/curve
        else:
            c = color([0.8, 0.4, 0.4])(cube(cube_size))  # light red - surface/plate
        c = translate([x * cube_size, y * cube_size, z * cube_size])(c)
        cells = c if cells is None else cells + c
    return cells

def render_3d_mat_color3(mat, cube_size=0.1, min8 = 5):
    positions = np.argwhere(mat)
    if len(positions) == 0:
        return cube(0)
    
    # Pad to safely check neighbors
    padded = np.pad(mat, 1, constant_values=0)
    cells = None
    
    for z, y, x in positions:
        p_pad = (z + 1, y + 1, x + 1)

        # if mat[z,y,x] > 0:
        if cond2_len(p_pad, padded, 1, min8 = min8):
            # V-points: less red
            c = color([0.8, 0.4, 0.4])(cube(cube_size))
        elif cond1(p_pad, padded, 1):
            # S-points: Blue
            c = color([0.4, 0.5, 0.8])(cube(cube_size))
        else:
            # Interior: Gray
            c = color([0.5, 0.5, 0.5])(cube(cube_size))
            
        c = translate([x * cube_size, y * cube_size, z * cube_size])(c)
        cells = c if cells is None else cells + c
        
    return cells

def classify_mat(mat, min8 = 5):
    positions = np.argwhere(mat)
    padded = np.pad(mat, 1, constant_values=0)
    ret = np.zeros_like(mat, dtype=int)
    # 1 is surface, 2 is arc

    for z, y, x in positions:
        p_pad = (z + 1, y + 1, x + 1)
        if cond2_len(p_pad, padded, 1, min8 = min8):
            ret[z,y,x] = 1

    Z, Y, X = ret.shape
    for z, y, x in positions:
        p_pad = (z + 1, y + 1, x + 1)
        if ret[z,y,x] != 1:
            if not cond1(p_pad, padded, 1):
                for dz, dy, dx in N26_OFFS:
                    nz, ny, nx = z+dz, y+dy, x+dx
                    if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X:
                        if ret[nz, ny, nx] == 1:
                            ret[z,y,x] = -1

    for z, y, x in positions:
        if ret[z,y,x] == -1:
            ret[z,y,x] = 1

    for z, y, x in positions:
        p_pad = (z + 1, y + 1, x + 1)
        if ret[z,y,x] == 0:
            ret[z,y,x] = 2
        
    return ret

def expand_labels(img, cla, fi):
    """
    Reconstruct the original binary volume from the thinning-encoded img
    and propagate skeleton labels (1=surface, 2=arc) to fill the full structure.

    Reverses the onion-layer erosion process encoded in img by thin_3d_saha:
      img[p] > 0            -> skeleton point marked at iteration img[p]
      img[p] == -MAXINT + i -> point eroded at iteration i  (1 <= i <= fi)
      img[p] == -MAXINT - 1 -> background / final-thinning deleted

    For each layer restored (from innermost outward), labels are propagated
    via BFS from already-labeled interior voxels into the newly added voxels.

    Parameters
    ----------
    img : ndarray
        Thinning-encoded volume (unpadded output of thin_3d_saha).
    cla : ndarray, same shape as img
        Skeleton classification from classify_mat: 0=bg, 1=surface, 2=arc.
    fi  : int
        Maximum iteration number returned by thin_3d_saha.

    Returns
    -------
    result : ndarray, same shape, dtype int8
        0 = background, 1 = surface/plate, 2 = arc/rod.
    """
    result = np.zeros(img.shape, dtype=np.int8)

    # Seed the labeled skeleton core
    result[img > 0] = cla[img > 0].astype(np.int8)

    Z, Y, X = img.shape

    # Restore layers from innermost (closest to skeleton) outward
    for i in range(fi, 0, -1):
        thr = -MAXINT + i
        layer_coords = np.argwhere(img == thr)

        if len(layer_coords) == 0:
            continue

        layer_set = set(map(tuple, layer_coords))

        # BFS: seed layer voxels adjacent to already-labeled interior voxels
        queue = deque()
        queued = set()

        for coord in map(tuple, layer_coords):
            z, y, x = coord
            for dz, dy, dx in N26_OFFS:
                nz, ny, nx = z + dz, y + dy, x + dx
                if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X:
                    lbl = int(result[nz, ny, nx])
                    if lbl > 0:
                        queue.append((coord, lbl))
                        queued.add(coord)
                        break  # first labeled neighbor wins as seed

        # Flood-fill labels through the layer (first-in wins, Voronoi-like)
        labeled = set()
        while queue:
            coord, lbl = queue.popleft()
            if coord in labeled:
                continue
            labeled.add(coord)
            result[coord] = lbl
            z, y, x = coord
            for dz, dy, dx in N26_OFFS:
                nb = (z + dz, y + dy, x + dx)
                if nb in layer_set and nb not in labeled and nb not in queued:
                    queue.append((nb, lbl))
                    queued.add(nb)

        # Fallback: any layer voxel not reachable from labeled interior
        for coord in map(tuple, layer_coords):
            if coord not in labeled:
                result[coord] = 1  # default to surface

    return result


def color_class(mat, cube_size=0.1):
    positions = np.argwhere(mat > 0)
    
    cells = None
    for z, y, x in positions:
        if mat[z,y,x] == 1:
            # V-points: less red
            c = color([0.8, 0.4, 0.4])(cube(cube_size))
        else: # mat[z,y,x] == 2:
            # S-points: Blue
            c = color([0.4, 0.5, 0.8])(cube(cube_size))
            
        c = translate([x * cube_size, y * cube_size, z * cube_size])(c)
        cells = c if cells is None else cells + c
        
    return cells


def render_3d_mat_color(mat, fi, cube_size=0.1):
    thr = -MAXINT + fi
    positions = np.argwhere(mat >= thr)
    if len(positions) == 0:
        return cube(0)
    
    # Pad to safely check neighbors
    padded = np.pad(mat, 1, constant_values=-MAXINT)
    cells = None
    old_img = (padded >= thr-1) 
    
    for z, y, x in positions:
        p_pad = (z + 1, y + 1, x + 1)

        # if mat[z,y,x] > 0:
        if cond2_len(p_pad, padded, thr):
            # V-points: less red
            c = color([0.8, 0.4, 0.4])(cube(cube_size))
        elif cond1(p_pad, padded, thr):
            # S-points: Blue
            c = color([0.4, 0.5, 0.8])(cube(cube_size))
        elif mat[z,y,x] == thr and is_s_open(p_pad, old_img):
            # red
            c = color([1, 0, 0])(cube(cube_size))
        elif mat[z,y,x] > 0:
            # E-points: Green
            c = color([0.5, 0.7, 0.5])(cube(cube_size))
        else:
            # Interior: Gray
            c = color([0.5, 0.5, 0.5])(cube(cube_size))
            
        c = translate([x * cube_size, y * cube_size, z * cube_size])(c)
        cells = c if cells is None else cells + c
        
    return cells

# saha

S_PTS = [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]

# Three unordered pairs of opposite s-point offsets (dz, dy, dx)
OPPOSITE_PAIRS = [
    ((-1,0,0), (1,0,0)),   # Top / Bottom
    ((0,-1,0), (0,1,0)),   # North / South
    ((0,0,-1), (0,0,1)),   # West / East
]

# d must be from {pB, pS, pW} for Definition 5 (thick)
_THICK_D_CANDIDATES = {(-1, 0, 0), (0, -1, 0), (0, 0, -1)}

def is_s_open(s, mat):
    # print('is_s_open')
    # print(s)
    z, y, x = s
    for sp in S_PTS:
        dz,dy,dx = sp
        # print((z+dz,y+dy,x+dx), mat[(z+dz,y+dy,x+dx)])
        if mat[(z+dz,y+dy,x+dx)] == 0:
            return True

def surface(a_off, s):
    z, y, x = s
    dz, dy, dx = a_off
    positions = []
    
    # Each surface contains exactly nine points [2].
    # These points form a 3x3 plane orthogonal to the axis of a_off [1].
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if dz != 0:
                # Offset is along Z-axis: Surface is in the (z+dz) plane, vary Y and X
                positions.append((dz, i, j))
            elif dy != 0:
                # Offset is along Y-axis: Surface is in the (y+dy) plane, vary Z and X
                positions.append((i, dy, j))
            else:
                # Offset is along X-axis: Surface is in the (x+dx) plane, vary Z and Y
                positions.append((i, j, dx))
                
    return positions

def check_6closed(white_em_offsets):
    """
    Generates a list of valid 6-closed paths that encircle p and
    checks if any are fully contained in white_em_offsets.
    
    Assumes a Top/Bottom pair (axis index 0), where encirclement 
    requires PN (0, 1, 0) and PE (0, 0, 1) [Source 44].
    """
    # 1. Nomenclature Mapping (Offsets relative to p) [Source 28, 33]
    PN, PS, PE, PW = (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
    PNE, PSE, PSW, PNW = (0, 1, 1), (0, -1, 1), (0, -1, -1), (0, 1, -1)
    
    # f-extensions for the 'bulged' 10-point paths [Source 30, 33]
    f1S, f2SE, f2SW = (0, -2, 0), (0, -2, 1), (0, -2, -1)
    f1W, f2WN, f2WS = (0, 0, -2), (0, 1, -2), (0, -1, -2)

    # 2. Define Valid 6-Closed Paths
    # All paths include PN and PE to satisfy the encirclement rule [Source 44].
    valid_circuits = [
        # Path A: Standard 8-point ring in the Middle Plane (M)
        [PN, PNE, PE, PSE, PS, PSW, PW, PNW],
        
        # Path B: 10-point ring 'bulging' into the South extension
        # (Replaces PS with f2SE -> f1S -> f2SW)
        [PN, PNE, PE, PSE, f2SE, f1S, f2SW, PSW, PW, PNW],
        
        # Path C: 10-point ring 'bulging' into the West extension
        # (Replaces PW with f2WS -> f1W -> f2WN)
        [PN, PNE, PE, PSE, PS, PSW, f2WS, f1W, f2WN, PNW],
        
        # Path D: 12-point ring bulging into both extensions
        [PN, PNE, PE, PSE, f2SE, f1S, f2SW, PSW, f2WS, f1W, f2WN, PNW]
    ]

    # 3. Check membership
    white_set = set(white_em_offsets)
    for i, circuit in enumerate(valid_circuits):
        # A set is valid if every offset in the circuit is currently white
        if all(offset in white_set for offset in circuit):
            # print(f"Path {i} found in white_em_offsets")
            return True
            
    return False

def cond1(p, img, thr):
    """
    Strict implementation of Condition 1: Arc-like shape.
    Requires a 6-closed path of white points in the Extended Middle Plane 
    encircling p [Source 43, 44].
    """
    # Define s-neighbors and nomenclature [Source 28, 31]
    PT, PN, PE = (-1, 0, 0), (0, 1, 0), (0, 0, 1) # Top, North, East
    PS, PB, PW = (0, -1, 0), (1, 0, 0), (0, 0, -1) # South, Bottom, West
    
    # Opposite pairs of s-neighbors (dz, dy, dx)
    OPPOSITE_PAIRS = [
        ((-1,0,0), (1,0,0)), # Top/Bottom
        ((0,1,0), (0,-1,0)), # North/South
        ((0,0,1), (0,0,-1))  # East/West
    ]

    def is_white(offset):
        # 'white before the iteration': value < thr [Source 37]
        nb_pt = (p[0]+offset[0], p[1]+offset[1], p[2]+offset[2])
        if any(c < 0 or c >= img.shape[i] for i, c in enumerate(nb_pt)):
            return True
        return img[nb_pt] < thr

    for a_off, d_off in OPPOSITE_PAIRS:
        # 1. Surface Check: both surfaces must contain at least one black point [Source 43, 45]
        surf_a = surface(a_off, p)
        surf_d = surface(d_off, p)
        if not (any(not is_white(off) for off in surf_a) and 
                any(not is_white(off) for off in surf_d)):
            continue

        # 2. Build Extended Middle Plane (EM) offsets [Source 32, 33]
        # Identify 'other' 4 s-neighbors in the middle plane
        others = [o for o in [PT, PB, PN, PS, PE, PW] if o != a_off and o != d_off]
        
        # Base Middle Plane (8 points)
        em_offsets = others.copy() # s-points (4)
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                if others[i] != tuple(-np.array(others[j])): # non-opposite
                    # e-points (4)
                    em_offsets.append(tuple(np.array(others[i]) + np.array(others[j])))

        # Extended Additions (7 points) using PS, PB, PW directions [Source 31, 33]
        targets = [o for o in others if o in [PS, PB, PW]]
        for x in targets:
            # f1 functions
            em_offsets.append(tuple(2 * np.array(x))) 
            for y in others:
                if x != tuple(-np.array(y)) and x != y: # non-opposite
                    # f2 functions
                    em_offsets.append(tuple(2 * np.array(x) + np.array(y)))
        # f3 function for the pair in targets
        if len(targets) == 2:
            em_offsets.append(tuple(2 * np.array(targets[0]) + 2 * np.array(targets[1])))

        # 3. Identify white points in EM and find 6-closed paths
        white_em = [o for o in set(em_offsets) if is_white(o)]
        if len(white_em) < 4: continue

        # Build connectivity graph for white_em points (6-adjacency)
        # Check for 6-closed paths using BFS/DFS or labeling
        # Strict Rule: 'A 6-closed path... encircles p if it contains two points 
        # of EM(a,d,p) intersection {PT, PN, PE}' [Source 44]
        
        # Test intersection
        essential_pts = [o for o in white_em if o in [PT, PN, PE]]
        if len(essential_pts) >= 2:
            # Verify if these points are part of a 6-closed white loop
            # (Simplified check: if white points form a circuit in the plane)
            if check_6closed(white_em):
                return True 
                
    return False

def e_p(a, b):
    """e(a, b, p) is 6-adjacent to s-neighbors a and b [Source 29]"""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def v_p(a, b, c):
    """v(a, b, c, p) is 6-adjacent to e-neighbors e(a,b,p), e(b,c,p), e(c,a,p) [Source 29]"""
    return (a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2])

def f1_p(a):
    """f1(a, p) is 6-adjacent to s-neighbor a and outside N(p) [Source 30]"""
    return (2 * a[0], 2 * a[1], 2 * a[2])

def is_e_open(p, mat):
    """
    Definition 2: p is not an s-open point and an e-point e(a, b, p) is white
    while the points f1(a, p), f1(b, p) are black before the iteration.
    """
    if is_s_open(p, mat):
        return False

    def is_black(off):
        return mat[(p[0]+off[0], p[1]+off[1], p[2]+off[2])]

    for i in range(3):
        for j in range(i + 1, 3):
            for a_off in OPPOSITE_PAIRS[i]:
                for b_off in OPPOSITE_PAIRS[j]:
                    if (not is_black(e_p(a_off, b_off)) and
                            is_black(f1_p(a_off)) and
                            is_black(f1_p(b_off))):
                        return True
    return False

def is_v_open(p, mat):
    """
    Definition 3: 'p is neither an s-open point nor an e-open point and a
    v-point v(a, b, c, p) is white while f1(a, p), f1(b, p), f1(c, p)
    are black before the iteration' [1].
    """
    if is_s_open(p, mat) or is_e_open(p, mat):
        return False

    def is_black(off):
        return mat[(p[0]+off[0], p[1]+off[1], p[2]+off[2])]

    # Check all triplets: one s-point from each of the three opposite pairs
    for a_off in OPPOSITE_PAIRS[0]:
        for b_off in OPPOSITE_PAIRS[1]:
            for c_off in OPPOSITE_PAIRS[2]:
                if (not is_black(v_p(a_off, b_off, c_off)) and
                        is_black(f1_p(a_off)) and
                        is_black(f1_p(b_off)) and
                        is_black(f1_p(c_off))):
                    return True
    return False

def cond2(p, img, thr):
    """
    'cond 2: pair of opposite s-points (a,d) ... sets contain at least one black point'
    Follows Source [3] exactly.
    """
    def is_black_before(p, off, thr):
        pt = (p[0]+off[0], p[1]+off[1], p[2]+off[2])
        if 0 <= pt[0] < img.shape[0] and 0 <= pt[1] < img.shape[1] and 0 <= pt[2] < img.shape[2]:
            return img[pt] >= thr
        return False  # Treat points outside bounds as white/background

    # Requirement: d must be from the set {pB, pS, pW} [Source 45]
    # pW = (-1,0,0), pS = (0,-1,0), pB = (0,0,-1) — corrected from (1,0,0) which was pE
    D_CANDIDATES = {(-1, 0, 0), (0, -1, 0), (0, 0, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)}
    # D_CANDIDATES = {(-1, 0, 0), (0, -1, 0), (0, 0, -1)}

    for i in range(3):
        pair = OPPOSITE_PAIRS[i]

        # Identify which element is d (in D_CANDIDATES) and which is a
        if pair[0] in D_CANDIDATES:
            d, a = pair[0], pair[1]
        elif pair[1] in D_CANDIDATES:
            d, a = pair[1], pair[0]
        else:
            continue  # skip if neither is a valid d

        # Identify other four s-points as pairs (b, e) and (c, f)
        pair_be = OPPOSITE_PAIRS[(i + 1) % 3]
        pair_cf = OPPOSITE_PAIRS[(i + 2) % 3]
        b, e = pair_be
        c, f = pair_cf
        
        # Step 1: Check initial white-point criteria [Source 45]
        # 'a is white' AND 'd or f1(d, p) is white' (before iteration: value < thr)
        if not is_black_before(p, a, thr) and \
           (not is_black_before(p, d, thr) or not is_black_before(p, f1_p(d), thr)):
            
            # Step 2: Check each of the 8 sets manually [Source 45]
            # Each set MUST contain at least one black point before the iteration.
            
            # 1. {e(a,b,p), b, e(b,d,p)}
            if not any(is_black_before(p, off, thr) for off in [e_p(a,b), b, e_p(b,d)]): continue
            
            # 2. {e(a,c,p), c, e(c,d,p)}
            if not any(is_black_before(p, off, thr) for off in [e_p(a,c), c, e_p(c,d)]): continue
            
            # 3. {e(a,e,p), e, e(d,e,p)}
            if not any(is_black_before(p, off, thr) for off in [e_p(a,e), e, e_p(d,e)]): continue
            
            # 4. {e(a,f,p), f, e(d,f,p)}
            if not any(is_black_before(p, off, thr) for off in [e_p(a,f), f, e_p(d,f)]): continue
            
            # 5. {v(a,b,c,p), e(b,c,p), v(b,c,d,p)}
            if not any(is_black_before(p, off, thr) for off in [v_p(a,b,c), e_p(b,c), v_p(b,c,d)]): continue
            
            # 6. {v(a,b,f,p), e(b,f,p), v(b,d,f,p)}
            if not any(is_black_before(p, off, thr) for off in [v_p(a,b,f), e_p(b,f), v_p(b,d,f)]): continue
            
            # 7. {v(a,c,e,p), e(c,e,p), v(c,d,e,p)}
            if not any(is_black_before(p, off, thr) for off in [v_p(a,c,e), e_p(c,e), v_p(c,d,e)]): continue
            
            # 8. {v(a,e,f,p), e(e,f,p), v(d, e, f, p)}
            if not any(is_black_before(p, off, thr) for off in [v_p(a,e,f), e_p(e,f), v_p(d,e,f)]): continue
            
            # If all 8 sets passed for this pair (a, d)
            return True
            
    return False

def cond2_len(p, img, thr, min8 = 7):
    """
    'cond 2: pair of opposite s-points (a,d) ... sets contain at least one black point'
    Follows Source [3] exactly.
    """
    def is_black_before(p, off, thr):
        pt = (p[0]+off[0], p[1]+off[1], p[2]+off[2])
        if 0 <= pt[0] < img.shape[0] and 0 <= pt[1] < img.shape[1] and 0 <= pt[2] < img.shape[2]:
            return img[pt] >= thr
        return False  # Treat points outside bounds as white/background

    # Requirement: d must be from the set {pB, pS, pW} [Source 45]
    # pW = (-1,0,0), pS = (0,-1,0), pB = (0,0,-1) — corrected from (1,0,0) which was pE
    D_CANDIDATES = {(-1, 0, 0), (0, -1, 0), (0, 0, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)}
    # D_CANDIDATES = {(-1, 0, 0), (0, -1, 0), (0, 0, -1)}

    for i in range(3):
        pair = OPPOSITE_PAIRS[i]

        # Identify which element is d (in D_CANDIDATES) and which is a
        if pair[0] in D_CANDIDATES:
            d, a = pair[0], pair[1]
        elif pair[1] in D_CANDIDATES:
            d, a = pair[1], pair[0]
        else:
            continue  # skip if neither is a valid d

        # Identify other four s-points as pairs (b, e) and (c, f)
        pair_be = OPPOSITE_PAIRS[(i + 1) % 3]
        pair_cf = OPPOSITE_PAIRS[(i + 2) % 3]
        b, e = pair_be
        c, f = pair_cf
        
        # Step 1: Check initial white-point criteria [Source 45]
        # 'a is white' AND 'd or f1(d, p) is white' (before iteration: value < thr)
        if not is_black_before(p, a, thr) and (not is_black_before(p, d, thr) or not is_black_before(p, f1_p(d), thr)):
            sets = [
                    [e_p(a,b), b, e_p(b,d)],              # Set 1
                    [e_p(a,c), c, e_p(c,d)],              # Set 2
                    [e_p(a,e), e, e_p(d,e)],              # Set 3
                    [e_p(a,f), f, e_p(d,f)],              # Set 4
                    [v_p(a,b,c), e_p(b,c), v_p(b,c,d)],    # Set 5
                    [v_p(a,b,f), e_p(b,f), v_p(b,d,f)],    # Set 6
                    [v_p(a,c,e), e_p(c,e), v_p(c,d,e)],    # Set 7
                    [v_p(a,e,f), e_p(e,f), v_p(d,e,f)]     # Set 8
                ]
            
            sets_passed = sum(1 for s in sets if any(is_black_before(p, off, thr) for off in s))

            if sets_passed >= min8:
                return True
            
    return False

def check_26_connectivity_8ring(black_offsets):
    """Checks if the black points in the 8-point middle plane ring are 26-connected."""
    if not black_offsets:
        return True

    visited = {black_offsets[0]}
    stack = [black_offsets[0]]

    while stack:
        curr = stack.pop()
        for other in black_offsets:
            if other not in visited:
                if max(abs(curr[0]-other[0]), abs(curr[1]-other[1]), abs(curr[2]-other[2])) <= 1:
                    visited.add(other)
                    stack.append(other)

    return len(visited) == len(black_offsets)


def cond3(p, img, thr):
    """
    'cond 3: every middle plane'
    Checks if p satisfies Condition 3 for all three coordinate planes.
    """
    def is_black_before(offset):
        pt = (p[0]+offset[0], p[1]+offset[1], p[2]+offset[2])
        return img[pt] >= thr - 1

    def is_black_current(offset):
        pt = (p[0]+offset[0], p[1]+offset[1], p[2]+offset[2])
        return img[pt] >= 0

    for i in range(3):
        a, d = OPPOSITE_PAIRS[i]

        mid_s_pairs = [OPPOSITE_PAIRS[j] for j in range(3) if j != i]
        b, e = mid_s_pairs[0]
        c, f = mid_s_pairs[1]

        mid_e_points = [e_p(b, c), e_p(c, e), e_p(e, f), e_p(f, b)]
        mid_s_points = [b, e, c, f]

        # All e-points black before current iteration → this plane is satisfied
        if all(is_black_before(ep) for ep in mid_e_points):
            continue

        # Tunnel exists if all mid-plane s-points are currently black
        if all(is_black_current(sp) for sp in mid_s_points):
            return False

        # Current black points in the middle plane must form a single 26-component
        all_plane_offsets = mid_s_points + mid_e_points
        current_black_offsets = [o for o in all_plane_offsets if is_black_current(o)]

        if not current_black_offsets:
            return False

        if not check_26_connectivity_8ring(current_black_offsets):
            return False

    return True


def is_shape_point(p, img, thr):
    mat = (img >= thr-1)
    if mat[p] == 0: return False

    return cond1(p, img, thr) or cond2_len(p, img, thr, min8 = 7)

def is_simple_point(p, full_mat):
    """
    Checks if a voxel is a (26, 6) simple point using only loops and numpy.
    
    'simple point: (26,6) 
    1. p has at least one black 26-neighbor. 
    2. p has at least one white 6-neighbor. 
    3. The set of black 26-neighbors of p is 26-connected. 
    4. The set of white 6-neighbors of p is 6-connected in the set of white 18-neighbors of p.'
    """
    z, y, x = p
    # Extract the 3x3x3 neighborhood. Center is at (1, 1, 1)
    # Note: 1 = black, 0 = white
    nb = full_mat[z-1:z+2, y-1:y+2, x-1:x+2].astype(np.int8)
    
    # Neighborhood Offset Definitions [3, 4]
    S_OFFS = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
    
    # Generate all 26-neighbors (all points in 3x3x3 except center) [4, 5]
    N26_OFFS = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0: continue
                N26_OFFS.append((dz, dy, dx))
                
    # 18-neighbors (s-points + e-points) [4, 5]
    N18_OFFS = [o for o in N26_OFFS if abs(o[0]) + abs(o[1]) + abs(o[2]) <= 2]

    # --- 1. Black 26-neighbor check ---
    black_26_coords = []
    for dz, dy, dx in N26_OFFS:
        if nb[1+dz, 1+dy, 1+dx] == 1:
            black_26_coords.append((1+dz, 1+dy, 1+dx))
    if not black_26_coords:
        return False

    # --- 2. White 6-neighbor check ---
    white_6_coords = []
    for dz, dy, dx in S_OFFS:
        if nb[1+dz, 1+dy, 1+dx] == 0:
            white_6_coords.append((1+dz, 1+dy, 1+dx))
    if not white_6_coords:
        return False

    # --- 3. Black 26-connectivity check ---
    # We must ensure all black 26-neighbors form a single component using 26-adjacency
    visited = np.zeros((3, 3, 3), dtype=bool)
    stack = [black_26_coords[0]]
    visited[black_26_coords[0]] = True
    reachable_black_count = 0
    
    while stack:
        cz, cy, cx = stack.pop()
        reachable_black_count += 1
        # Check all 26 possible neighbors for each visited black point
        for dz, dy, dx in N26_OFFS:
            nz, ny, nx = cz + dz, cy + dy, cx + dx
            # Must stay within 3x3x3 and not be the center point p
            if 0 <= nz < 3 and 0 <= ny < 3 and 0 <= nx < 3 and not (nz == 1 and ny == 1 and nx == 1):
                if nb[nz, ny, nx] == 1 and not visited[nz, ny, nx]:
                    visited[nz, ny, nx] = True
                    stack.append((nz, ny, nx))
                    
    if reachable_black_count != len(black_26_coords):
        return False

    # --- 4. White 6-connectivity in White 18-neighbors ---
    # We must ensure all white 6-neighbors are connected to each other 
    # through a path of white 18-neighbors using only 6-connectivity steps.
    
    # Territory: set of white points that are at least 18-adjacent to p
    white_18_mask = np.zeros((3, 3, 3), dtype=bool)
    for dz, dy, dx in N18_OFFS:
        if nb[1+dz, 1+dy, 1+dx] == 0:
            white_18_mask[1+dz, 1+dy, 1+dx] = True
            
    visited_w = np.zeros((3, 3, 3), dtype=bool)
    stack_w = [white_6_coords[0]]
    visited_w[white_6_coords[0]] = True
    reachable_white_6_count = 0
    
    # Set of s-point coordinates in the 3x3x3 grid for easy counting
    s_point_set = set((1+dz, 1+dy, 1+dx) for dz, dy, dx in S_OFFS)
    
    while stack_w:
        cz, cy, cx = stack_w.pop()
        if (cz, cy, cx) in s_point_set:
            reachable_white_6_count += 1
            
        # Move only in 6 directions (6-connectivity)
        for dz, dy, dx in S_OFFS:
            nz, ny, nx = cz + dz, cy + dy, cx + dx
            if 0 <= nz < 3 and 0 <= ny < 3 and 0 <= nx < 3:
                # Path must stay within white 18-neighbors
                if white_18_mask[nz, ny, nx] and not visited_w[nz, ny, nx]:
                    visited_w[nz, ny, nx] = True
                    stack_w.append((nz, ny, nx))
                    
    if reachable_white_6_count != len(white_6_coords):
        return False

    return True

def is_thick(axis_idx, p, img_before):
    """
    Definition 5: thick(a, d, p).
    Returns True if, before the final thinning pass, a and f1(d,p) are white,
    d is black, and no other opposite pair of s-points is entirely white.
    img_before is a boolean snapshot of the image taken before the final-thinning scan.
    """
    pair = OPPOSITE_PAIRS[axis_idx]
    if pair[0] in _THICK_D_CANDIDATES:
        d, a = pair[0], pair[1]
    elif pair[1] in _THICK_D_CANDIDATES:
        d, a = pair[1], pair[0]
    else:
        return False

    def bef(off):
        pt = (p[0]+off[0], p[1]+off[1], p[2]+off[2])
        if any(coord < 0 or coord >= sz for coord, sz in zip(pt, img_before.shape)):
            return False
        return bool(img_before[pt])

    # a and f1(d, p) must be white before the iteration
    if bef(a) or bef(f1_p(d)):
        return False
    # d must be black before the iteration
    if not bef(d):
        return False
    # For each of the other two opposite pairs, at least one member must be black
    for i in range(3):
        if i == axis_idx:
            continue
        b, e = OPPOSITE_PAIRS[i]
        if not bef(b) and not bef(e):
            return False
    return True


def satisfies_cond3_ft(axis_idx, p, img):
    """
    Condition 3 for final thinning: current black points of middle plane M(a,d,p)
    form a single 26-component and contain no tunnel.
    A tunnel exists iff all four s-points in the plane are currently black.
    Uses the CURRENT (in-progress) state of img.
    """
    b_e, c_f = [OPPOSITE_PAIRS[i] for i in range(3) if i != axis_idx]
    b, e = b_e
    c, f = c_f

    s_points = [b, e, c, f]
    e_points = [e_p(b, c), e_p(c, e), e_p(e, f), e_p(f, b)]

    def now(off):
        pt = (p[0]+off[0], p[1]+off[1], p[2]+off[2])
        if any(coord < 0 or coord >= sz for coord, sz in zip(pt, img.shape)):
            return False
        return img[pt] >= 0

    # Tunnel check: all 4 s-points currently black => topology cannot be preserved
    if all(now(s) for s in s_points):
        return False

    current_black = [off for off in s_points + e_points if now(off)]
    return check_26_connectivity_8ring(current_black)


def final_thinning(img):
    """
    Section 3.2: Final-thinning single-scan procedure.
    Executed once after primary thinning.  Every black point (marked or unmarked)
    is deleted if it is a simple (26,6) point and satisfies Condition 4, 5, or 6
    (Definition 6).  thick() is evaluated on the snapshot taken before this scan;
    cond3 connectivity is evaluated on the current (evolving) image.
    """
    img_before = (img >= 0)   # boolean snapshot before this pass

    deleted = 0
    for p_idx in np.argwhere(img >= 0):
        p = tuple(p_idx)

        # Simplicity check on the current image
        if not is_simple_point(p, img >= 0):
            continue

        t = [is_thick(i, p, img_before) for i in range(3)]
        num_thick = sum(t)

        erodable = False

        if num_thick == 1:      # Condition 4
            axis = t.index(True)
            if all(satisfies_cond3_ft(i, p, img) for i in range(3) if i != axis):
                erodable = True

        elif num_thick == 2:    # Condition 5
            axis_thin = t.index(False)
            if satisfies_cond3_ft(axis_thin, p, img):
                erodable = True

        elif num_thick == 3:    # Condition 6
            erodable = True

        if erodable:
            img[p] = -(MAXINT + 1)
            deleted += 1

    print(f'final thinning deleted: {deleted}')
    return img


def thin_3d_saha(voxels, run_final_thinning=True):
    img = np.where(voxels > 0, 0, -MAXINT-1)
    img = np.pad(img, 2, constant_values=-MAXINT-1)
    
    i = 0
    changed = True
    while changed:
        i += 1
        changed = False
        thr = -MAXINT + i
        old_img = (img >= thr-1) 
        print('old_img', old_img.sum())
        
        numso = 0
        # 1: s-open points
        for p_idx in np.argwhere(img == 0):
            p = tuple(p_idx)
            if is_s_open(p, old_img):
                numso += 1
                if is_shape_point(p, img, thr):
                    img[p] = i
                elif is_simple_point(p, img >= 0):
                    img[p] = thr
                    changed = True

        print('iteration', i)
        print('num s-open', numso)
        print('marked', (img > 0).sum())
        print('marked this it', (img == i).sum())
        print('unmarked', (img == 0).sum())
        print('deleted this it', (img == thr).sum())

        # 2: e-open points
        for p_idx in np.argwhere(img == 0):
            p = tuple(p_idx)
            if is_e_open(p, old_img) and cond3(p, img, thr) and is_simple_point(p, img >= 0): 
                img[p] = thr
                changed = True

        # 3: v-open points
        for p_idx in np.argwhere(img == 0):
            p = tuple(p_idx)
            if is_v_open(p, old_img) and is_simple_point(p, img >= 0):
                img[p] = thr
                changed = True

    if run_final_thinning:
        img = final_thinning(img)

    return img[2:-2, 2:-2, 2:-2] >= 0, img[2:-2, 2:-2, 2:-2], i

# PARALLEL
import threading

def subfield_worker(img, old_img, thr, l, scan_type, z_range, iteration_num):
    """
    Worker function for a single thread to process a chunk of Z-slices.
    Strictly follows parallel erosion logic for sub-field l [Source 111, 112].
    """
    z_start, z_end = z_range
    for z in range(z_start, z_end):
        for y in range(2, img.shape[1] - 2):
            for x in range(2, img.shape[2] - 2):
                # Check if voxel (z, y, x) belongs to current sub-field l [Source 112]
                if (4 * (z % 2) + 2 * (y % 2) + (x % 2)) != l:
                    continue
                
                # Process only unmarked black voxels (value 0) [Source 82]
                if img[z, y, x] != 0:
                    continue
                
                p = (z, y, x)
                # Apply scan logic based on the current scan phase [Source 95, 97]
                if scan_type == 's': # s-open scan
                    if is_s_open(p, old_img):
                        if is_shape_point(p, img, thr):
                            img[p] = iteration_num # Mark shape point [Source 82]
                        elif is_simple_point(p, img >= 0):
                            img[p] = thr # Delete point [Source 81, 82]

                elif scan_type == 'e': # e-open scan
                    if is_e_open(p, old_img) and cond3(p, img, thr) and \
                       is_simple_point(p, img >= 0):
                        img[p] = thr

                elif scan_type == 'v': # v-open scan
                    if is_v_open(p, old_img) and is_simple_point(p, img >= 0):
                        img[p] = thr

def threaded_subfield_scan(img, old_img, thr, l, scan_type, iteration_num, num_threads=4):
    """Coordinates multiple threads to process sub-field l [Source 112]."""
    depth = img.shape[0]
    # Divide Z-slices into chunks for threads
    chunk_size = (depth - 4) // num_threads
    threads = []
    
    for t in range(num_threads):
        z_start = 2 + (t * chunk_size)
        z_end = (z_start + chunk_size) if t < num_threads - 1 else depth - 2
        
        thread = threading.Thread(
            target=subfield_worker, 
            args=(img, old_img, thr, l, scan_type, (z_start, z_end), iteration_num)
        )
        threads.append(thread)
        thread.start()
        
    for thread in threads:
        thread.join()

def thin_3d_parallel_threaded(voxels, num_threads=8, run_final_thinning=True):
    # Initialize image: 0 for bone, large negative for background [Source 81]
    img = np.where(voxels > 0, 0, -MAXINT-1)
    img = np.pad(img, 2, constant_values=-MAXINT-1)
    
    i = 0
    while True:
        i += 1
        thr = -MAXINT + i
        # snapshot of image BEFORE the iteration for shape/openness checks [Source 81, 114]
        old_img = (img >= thr-1)
        prev_sum = np.sum(img == thr)
        
        # Section 3.1: Each iteration consists of three scans [Source 83, 95]
        # Each scan requires 8 sub-field cycles for parallel safety [Source 112]
        for scan_type in ['s', 'e', 'v']:
            for l in range(8):
                threaded_subfield_scan(img, old_img, thr, l, scan_type, i, num_threads)
        
        # Exit if no points were deleted in this iteration [Source 95]
        if np.sum(img == thr) == prev_sum:
            break
            
    if run_final_thinning:
        img = final_thinning(img) # Single sequential scan [Source 100]
        
    return img[2:-2, 2:-2, 2:-2] >= 0, img[2:-2, 2:-2, 2:-2], i

def calculate_bone_quantification(expanded, skeleton_cla, voxel_size=1.0):
    """
    Calculates bone microarchitecture parameters based on topological classification.

    Parameters:
    -----------
    expanded : ndarray (int8)
        Reconstructed volume: 0=background, 1=plate, 2=rod.
    skeleton_cla : ndarray (int)
        Classified skeleton: 0=bg, 1=surface (plate), 2=arc (rod).
    voxel_size : float
        Physical dimension of one voxel.
    """
    # --- 1. Basic Volume Counts (Voxel Units) ---
    tv_vox = expanded.size
    p_vox = np.sum(expanded == 1)
    r_vox = np.sum(expanded == 2)
    total_bone_vox = p_vox + r_vox

    # Skeleton counts for Thinning Ratio
    p_ske_vox = np.sum(skeleton_cla == 1)
    r_ske_vox = np.sum(skeleton_cla == 2)
    total_ske_vox = p_ske_vox + r_ske_vox

    # --- 2. BV/TV Calculations (Individual and Overall) ---
    p_bv_tv = p_vox / tv_vox
    r_bv_tv = r_vox / tv_vox
    overall_bv_tv = total_bone_vox / tv_vox

    # --- 3. Global Parameters (Porosity and Pore Size) ---
    porosity = 1.0 - overall_bv_tv

    # Global Thinning Ratio: Total Bone Volume / Total Skeleton Volume
    global_th = (total_bone_vox / total_ske_vox * voxel_size) if total_ske_vox > 0 else 0

    # Global Tb.N: (Overall BV/TV) / Global Tb.Th
    global_n = (overall_bv_tv / global_th) if global_th > 0 else 0

    # Global Pore Size (Tb.Sp): (1 / Global Tb.N) - Global Tb.Th
    global_pore_size = (1 / global_n - global_th) if global_n > 0 else 0

    # --- 4. Independent Parameters (Rods and Plates) ---
    p_th = (p_vox / p_ske_vox * voxel_size) if p_ske_vox > 0 else 0
    r_th = (r_vox / r_ske_vox * voxel_size) if r_ske_vox > 0 else 0

    p_n = (p_bv_tv / p_th) if p_th > 0 else 0
    r_n = (r_bv_tv / r_th) if r_th > 0 else 0

    return {
        "TV": tv_vox * (voxel_size**3),
        "BV/TV": overall_bv_tv,
        "porosity": porosity,
        "pore_size": global_pore_size,
        "pBV/TV": p_bv_tv,
        "pTb.Th": p_th,
        "pTb.N": p_n,
        "pBV/BV": p_vox / total_bone_vox if total_bone_vox > 0 else 0,
        "rBV/TV": r_bv_tv,
        "rTb.Th": r_th,
        "rTb.N": r_n,
        "rBV/BV": r_vox / total_bone_vox if total_bone_vox > 0 else 0,
    }


voxeld = np.load('s01_voxel.npy')
print(voxeld.shape)
rng = np.random.default_rng()

for i in range(34, 2000):
    run_start = time.time()
    savename = f'rand{i}'
    print(f'\n--- Starting run: {savename} ---')

    r1 = rng.integers(low=80, high=1400)
    r2 = rng.integers(low=350, high=1100)
    r3 = rng.integers(low=350, high=1100)
    z_range = [int(r1), int(r1+100)]
    y_range = [int(r2), int(r2+100)]
    x_range = [int(r3), int(r3+100)]
    print(f'random submatrix {z_range[0]} {z_range[1]} {y_range[0]} {y_range[1]} {x_range[0]} {x_range[1]}')
    
    sub = voxeld[z_range[0]:z_range[1], y_range[0]:y_range[1], x_range[0]:x_range[1]]

    labeled, n = label(sub, structure=np.ones((3,3,3)))
    if n > 0:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        sub = (labeled == counts.argmax())

    print('input extracted')

    # ske,img,fi = thin_3d_saha(sub)
    ske,img,fi = thin_3d_parallel_threaded(sub)
    print(f"elapsed thin: {time.time() - run_start:.2f} seconds")
    print(f'true cnt: {ske.sum()}')
    np.save('ske_'+savename+'.npy', ske)
    # ske = np.load('s01_80180div2_ske.npy')
    # scout('../scad/img8.11', render_3d_mat_color(img, fi))
    cla = classify_mat(ske)
    np.save('classified_ske_'+savename+'.npy', ske)
    expanded = expand_labels(img, cla, fi)
    np.save('classified_'+savename+'.npy', ske)
    # scout('../scad/orig_'+savename, render_3d_mat(sub))
    # scout('../scad/img_'+savename, color_class(cla))
    # scout('../scad/expimg_'+savename, color_class(expanded))

    quant = calculate_bone_quantification(expanded, cla)
    quant['range'] = {
        'z': z_range,
        'y': y_range,
        'x': x_range
    }

    with open('quant_'+savename+'.json', 'w') as json_file:
        json.dump(quant, json_file, indent=4) # indent for human readability

    print("\n--- Bone Quantification ---")
    print(f"TV: {quant['TV']:.0f} | BV/TV: {quant['BV/TV']} | "
          f"Porosity: {quant['porosity']:.4f} | Pore size: {quant['pore_size']:.4f}")
    print(f"  Plates | pBV/TV: {quant['pBV/TV']:.4f} | pTb.Th: {quant['pTb.Th']:.4f} | "
          f"pTb.N: {quant['pTb.N']:.4f} | pBV/BV: {quant['pBV/BV']:.4f}")
    print(f"  Rods   | rBV/TV: {quant['rBV/TV']:.4f} | rTb.Th: {quant['rTb.Th']:.4f} | "
          f"rTb.N: {quant['rTb.N']:.4f} | rBV/BV: {quant['rBV/BV']:.4f}")

    print(f"Run {savename} elapsed: {time.time() - run_start:.2f} seconds")

