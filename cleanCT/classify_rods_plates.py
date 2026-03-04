import time
import subprocess
import numpy as np
from scipy.ndimage import label
from collections import deque
from solid import *

MAXINT = 1000000

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

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

def render_volume_labels(volume_labels, cube_size=0.1):
    """Render the full classified volume: rod=blue, plate=red."""
    positions = np.argwhere(volume_labels > 0)
    if len(positions) == 0:
        return cube(0)
    cells = None
    for z, y, x in positions:
        lbl = volume_labels[z, y, x]
        if lbl == 1:
            c = color([0.4, 0.5, 0.8])(cube(cube_size))   # blue - rod
        else:
            c = color([0.8, 0.4, 0.4])(cube(cube_size))   # red - plate
        c = translate([x * cube_size, y * cube_size, z * cube_size])(c)
        cells = c if cells is None else cells + c
    return cells

def render_skeleton_labels(skel_labels, cube_size=0.1):
    """Render only skeleton voxels: rod=blue, plate=red."""
    return render_volume_labels(skel_labels, cube_size)

# ---------------------------------------------------------------------------
# Saha skeletonization (identical to thin_fin.py)
# ---------------------------------------------------------------------------

S_PTS = [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]

OPPOSITE_PAIRS = [
    ((-1,0,0), (1,0,0)),
    ((0,-1,0), (0,1,0)),
    ((0,0,-1), (0,0,1)),
]

_THICK_D_CANDIDATES = {(-1, 0, 0), (0, -1, 0), (0, 0, -1)}

def is_s_open(s, mat):
    z, y, x = s
    for sp in S_PTS:
        dz, dy, dx = sp
        if mat[(z+dz, y+dy, x+dx)] == 0:
            return True

def surface(a_off, s):
    z, y, x = s
    dz, dy, dx = a_off
    positions = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if dz != 0:
                positions.append((dz, i, j))
            elif dy != 0:
                positions.append((i, dy, j))
            else:
                positions.append((i, j, dx))
    return positions

def check_6closed(white_em_offsets):
    PN, PS, PE, PW = (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
    PNE, PSE, PSW, PNW = (0, 1, 1), (0, -1, 1), (0, -1, -1), (0, 1, -1)
    f1S, f2SE, f2SW = (0, -2, 0), (0, -2, 1), (0, -2, -1)
    f1W, f2WN, f2WS = (0, 0, -2), (0, 1, -2), (0, -1, -2)

    valid_circuits = [
        [PN, PNE, PE, PSE, PS, PSW, PW, PNW],
        [PN, PNE, PE, PSE, f2SE, f1S, f2SW, PSW, PW, PNW],
        [PN, PNE, PE, PSE, PS, PSW, f2WS, f1W, f2WN, PNW],
        [PN, PNE, PE, PSE, f2SE, f1S, f2SW, PSW, f2WS, f1W, f2WN, PNW]
    ]

    white_set = set(white_em_offsets)
    for circuit in valid_circuits:
        if all(offset in white_set for offset in circuit):
            return True
    return False

def cond1(p, img, thr):
    PT, PN, PE = (-1, 0, 0), (0, 1, 0), (0, 0, 1)
    PS, PB, PW = (0, -1, 0), (1, 0, 0), (0, 0, -1)

    OPPOSITE_PAIRS_LOCAL = [
        ((-1,0,0), (1,0,0)),
        ((0,1,0), (0,-1,0)),
        ((0,0,1), (0,0,-1))
    ]

    def is_white(offset):
        nb_pt = (p[0]+offset[0], p[1]+offset[1], p[2]+offset[2])
        if any(c < 0 or c >= img.shape[i] for i, c in enumerate(nb_pt)):
            return True
        return img[nb_pt] < thr

    for a_off, d_off in OPPOSITE_PAIRS_LOCAL:
        surf_a = surface(a_off, p)
        surf_d = surface(d_off, p)
        if not (any(not is_white(off) for off in surf_a) and
                any(not is_white(off) for off in surf_d)):
            continue

        others = [o for o in [PT, PB, PN, PS, PE, PW] if o != a_off and o != d_off]
        em_offsets = others.copy()
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                if others[i] != tuple(-np.array(others[j])):
                    em_offsets.append(tuple(np.array(others[i]) + np.array(others[j])))

        targets = [o for o in others if o in [PS, PB, PW]]
        for x in targets:
            em_offsets.append(tuple(2 * np.array(x)))
            for y in others:
                if x != tuple(-np.array(y)) and x != y:
                    em_offsets.append(tuple(2 * np.array(x) + np.array(y)))
        if len(targets) == 2:
            em_offsets.append(tuple(2 * np.array(targets[0]) + 2 * np.array(targets[1])))

        white_em = [o for o in set(em_offsets) if is_white(o)]
        if len(white_em) < 4:
            continue

        essential_pts = [o for o in white_em if o in [PT, PN, PE]]
        if len(essential_pts) >= 2:
            if check_6closed(white_em):
                return True

    return False

def e_p(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def v_p(a, b, c):
    return (a[0]+b[0]+c[0], a[1]+b[1]+c[1], a[2]+b[2]+c[2])

def f1_p(a):
    return (2*a[0], 2*a[1], 2*a[2])

def is_e_open(p, mat):
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
    if is_s_open(p, mat) or is_e_open(p, mat):
        return False

    def is_black(off):
        return mat[(p[0]+off[0], p[1]+off[1], p[2]+off[2])]

    for a_off in OPPOSITE_PAIRS[0]:
        for b_off in OPPOSITE_PAIRS[1]:
            for c_off in OPPOSITE_PAIRS[2]:
                if (not is_black(v_p(a_off, b_off, c_off)) and
                        is_black(f1_p(a_off)) and
                        is_black(f1_p(b_off)) and
                        is_black(f1_p(c_off))):
                    return True
    return False

def cond2_len(p, img, thr, min8=7):
    def is_black_before(p, off, thr):
        pt = (p[0]+off[0], p[1]+off[1], p[2]+off[2])
        if 0 <= pt[0] < img.shape[0] and 0 <= pt[1] < img.shape[1] and 0 <= pt[2] < img.shape[2]:
            return img[pt] >= thr
        return False

    D_CANDIDATES = {(-1, 0, 0), (0, -1, 0), (0, 0, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)}

    for i in range(3):
        pair = OPPOSITE_PAIRS[i]
        if pair[0] in D_CANDIDATES:
            d, a = pair[0], pair[1]
        elif pair[1] in D_CANDIDATES:
            d, a = pair[1], pair[0]
        else:
            continue

        pair_be = OPPOSITE_PAIRS[(i + 1) % 3]
        pair_cf = OPPOSITE_PAIRS[(i + 2) % 3]
        b, e = pair_be
        c, f = pair_cf

        if not is_black_before(p, a, thr) and (not is_black_before(p, d, thr) or not is_black_before(p, f1_p(d), thr)):
            sets = [
                [e_p(a,b), b, e_p(b,d)],
                [e_p(a,c), c, e_p(c,d)],
                [e_p(a,e), e, e_p(d,e)],
                [e_p(a,f), f, e_p(d,f)],
                [v_p(a,b,c), e_p(b,c), v_p(b,c,d)],
                [v_p(a,b,f), e_p(b,f), v_p(b,d,f)],
                [v_p(a,c,e), e_p(c,e), v_p(c,d,e)],
                [v_p(a,e,f), e_p(e,f), v_p(d,e,f)]
            ]
            sets_passed = sum(1 for s in sets if any(is_black_before(p, off, thr) for off in s))
            if sets_passed >= min8:
                return True

    return False

def check_26_connectivity_8ring(black_offsets):
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

        if all(is_black_before(ep) for ep in mid_e_points):
            continue
        if all(is_black_current(sp) for sp in mid_s_points):
            return False

        all_plane_offsets = mid_s_points + mid_e_points
        current_black_offsets = [o for o in all_plane_offsets if is_black_current(o)]
        if not current_black_offsets:
            return False
        if not check_26_connectivity_8ring(current_black_offsets):
            return False

    return True

def is_shape_point(p, img, thr):
    mat = (img >= thr-1)
    if mat[p] == 0:
        return False
    return cond1(p, img, thr) or cond2_len(p, img, thr)

def is_simple_point(p, full_mat):
    z, y, x = p
    nb = full_mat[z-1:z+2, y-1:y+2, x-1:x+2].astype(np.int8)

    S_OFFS = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
    N26_OFFS = [(dz, dy, dx) for dz in [-1,0,1] for dy in [-1,0,1] for dx in [-1,0,1]
                if not (dz==0 and dy==0 and dx==0)]
    N18_OFFS = [o for o in N26_OFFS if abs(o[0])+abs(o[1])+abs(o[2]) <= 2]

    black_26_coords = [(1+dz, 1+dy, 1+dx) for dz,dy,dx in N26_OFFS if nb[1+dz,1+dy,1+dx]==1]
    if not black_26_coords:
        return False

    white_6_coords = [(1+dz, 1+dy, 1+dx) for dz,dy,dx in S_OFFS if nb[1+dz,1+dy,1+dx]==0]
    if not white_6_coords:
        return False

    visited = np.zeros((3,3,3), dtype=bool)
    stack = [black_26_coords[0]]
    visited[black_26_coords[0]] = True
    reachable_black_count = 0
    while stack:
        cz, cy, cx = stack.pop()
        reachable_black_count += 1
        for dz, dy, dx in N26_OFFS:
            nz, ny, nx = cz+dz, cy+dy, cx+dx
            if 0 <= nz < 3 and 0 <= ny < 3 and 0 <= nx < 3 and not (nz==1 and ny==1 and nx==1):
                if nb[nz,ny,nx]==1 and not visited[nz,ny,nx]:
                    visited[nz,ny,nx] = True
                    stack.append((nz,ny,nx))
    if reachable_black_count != len(black_26_coords):
        return False

    white_18_mask = np.zeros((3,3,3), dtype=bool)
    for dz, dy, dx in N18_OFFS:
        if nb[1+dz,1+dy,1+dx] == 0:
            white_18_mask[1+dz,1+dy,1+dx] = True

    visited_w = np.zeros((3,3,3), dtype=bool)
    stack_w = [white_6_coords[0]]
    visited_w[white_6_coords[0]] = True
    reachable_white_6_count = 0
    s_point_set = set((1+dz,1+dy,1+dx) for dz,dy,dx in S_OFFS)
    while stack_w:
        cz, cy, cx = stack_w.pop()
        if (cz,cy,cx) in s_point_set:
            reachable_white_6_count += 1
        for dz, dy, dx in S_OFFS:
            nz, ny, nx = cz+dz, cy+dy, cx+dx
            if 0 <= nz < 3 and 0 <= ny < 3 and 0 <= nx < 3:
                if white_18_mask[nz,ny,nx] and not visited_w[nz,ny,nx]:
                    visited_w[nz,ny,nx] = True
                    stack_w.append((nz,ny,nx))
    if reachable_white_6_count != len(white_6_coords):
        return False

    return True

def is_thick(axis_idx, p, img_before):
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

    if bef(a) or bef(f1_p(d)):
        return False
    if not bef(d):
        return False
    for i in range(3):
        if i == axis_idx:
            continue
        b, e = OPPOSITE_PAIRS[i]
        if not bef(b) and not bef(e):
            return False
    return True

def satisfies_cond3_ft(axis_idx, p, img):
    b_e, c_f = [OPPOSITE_PAIRS[i] for i in range(3) if i != axis_idx]
    b, e = b_e
    c, f = c_f
    s_points = [b, e, c, f]
    e_points = [e_p(b,c), e_p(c,e), e_p(e,f), e_p(f,b)]

    def now(off):
        pt = (p[0]+off[0], p[1]+off[1], p[2]+off[2])
        if any(coord < 0 or coord >= sz for coord, sz in zip(pt, img.shape)):
            return False
        return img[pt] >= 0

    if all(now(s) for s in s_points):
        return False
    current_black = [off for off in s_points + e_points if now(off)]
    return check_26_connectivity_8ring(current_black)

def final_thinning(img):
    img_before = (img >= 0)
    deleted = 0
    for p_idx in np.argwhere(img >= 0):
        p = tuple(p_idx)
        if not is_simple_point(p, img >= 0):
            continue
        t = [is_thick(i, p, img_before) for i in range(3)]
        num_thick = sum(t)
        erodable = False
        if num_thick == 1:
            axis = t.index(True)
            if all(satisfies_cond3_ft(i, p, img) for i in range(3) if i != axis):
                erodable = True
        elif num_thick == 2:
            axis_thin = t.index(False)
            if satisfies_cond3_ft(axis_thin, p, img):
                erodable = True
        elif num_thick == 3:
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

        for p_idx in np.argwhere(img == 0):
            p = tuple(p_idx)
            if is_e_open(p, old_img) and cond3(p, img, thr) and is_simple_point(p, img >= 0):
                img[p] = thr
                changed = True

        for p_idx in np.argwhere(img == 0):
            p = tuple(p_idx)
            if is_v_open(p, old_img) and is_simple_point(p, img >= 0):
                img[p] = thr
                changed = True

    if run_final_thinning:
        img = final_thinning(img)

    return img[2:-2, 2:-2, 2:-2] >= 0, img[2:-2, 2:-2, 2:-2], i

# ---------------------------------------------------------------------------
# PCA-based classification helpers
# ---------------------------------------------------------------------------

_NEIGHBOR_OFFSETS = [
    (dz, dy, dx)
    for dz in [-1, 0, 1]
    for dy in [-1, 0, 1]
    for dx in [-1, 0, 1]
    if not (dz == 0 and dy == 0 and dx == 0)
]

def _pca_classify(neighbor_offsets):
    """
    Given a list of (dz, dy, dx) offsets for occupied neighbors, return
    'rod' or 'plate' via SVD.  s[1]/s[0] > 0.5 => coplanar => plate.
    """
    if len(neighbor_offsets) < 2:
        return 'rod'
    pts = np.array(neighbor_offsets, dtype=float)
    pts -= pts.mean(axis=0)
    if pts.shape[0] < 2:
        return 'rod'
    _, s, _ = np.linalg.svd(pts, full_matrices=False)
    if s[0] > 0 and s[1] / s[0] > 0.5:
        return 'plate'
    return 'rod'

def _skeleton_neighbor_offsets(ske, p):
    """Return offsets of skeleton voxels in the 26-neighborhood of p."""
    z, y, x = p
    offsets = []
    for dz, dy, dx in _NEIGHBOR_OFFSETS:
        nz, ny, nx = z + dz, y + dy, x + dx
        if 0 <= nz < ske.shape[0] and 0 <= ny < ske.shape[1] and 0 <= nx < ske.shape[2]:
            if ske[nz, ny, nx]:
                offsets.append((dz, dy, dx))
    return offsets

def _volume_neighbor_offsets(volume, p):
    """Return offsets of original-volume voxels in the 26-neighborhood of p."""
    z, y, x = p
    offsets = []
    for dz, dy, dx in _NEIGHBOR_OFFSETS:
        nz, ny, nx = z + dz, y + dy, x + dx
        if 0 <= nz < volume.shape[0] and 0 <= ny < volume.shape[1] and 0 <= nx < volume.shape[2]:
            if volume[nz, ny, nx]:
                offsets.append((dz, dy, dx))
    return offsets

# ---------------------------------------------------------------------------
# Step 1 – classify every skeleton voxel
# ---------------------------------------------------------------------------

ROD   = 1
PLATE = 2

def classify_skeleton(ske, original_volume):
    """
    Classify each skeleton voxel as ROD (1) or PLATE (2).

    Strategy:
      • If the voxel has >= 3 skeleton neighbours → PCA on skeleton neighbours.
      • Else if the voxel has >= 3 original-volume neighbours → PCA on those.
      • Otherwise mark as uncertain and resolve iteratively from labelled neighbours.

    Returns an int8 array of the same shape as ske:
      0 = not a skeleton voxel
      1 = rod
      2 = plate
    """
    skel_labels = np.zeros(ske.shape, dtype=np.int8)
    uncertain = []

    for p_idx in np.argwhere(ske):
        p = tuple(p_idx)

        ske_offs = _skeleton_neighbor_offsets(ske, p)
        if len(ske_offs) >= 3:
            kind = _pca_classify(ske_offs)
            skel_labels[p] = ROD if kind == 'rod' else PLATE
            continue

        vol_offs = _volume_neighbor_offsets(original_volume, p)
        if len(vol_offs) >= 3:
            kind = _pca_classify(vol_offs)
            skel_labels[p] = ROD if kind == 'rod' else PLATE
            continue

        # Truly isolated – resolve later
        uncertain.append(p)

    # Iterative label propagation for isolated voxels
    for _ in range(len(uncertain) + 1):
        if not uncertain:
            break
        still_uncertain = []
        for p in uncertain:
            z, y, x = p
            neighbor_labels = []
            for dz, dy, dx in _NEIGHBOR_OFFSETS:
                nz, ny, nx = z + dz, y + dy, x + dx
                if 0 <= nz < ske.shape[0] and 0 <= ny < ske.shape[1] and 0 <= nx < ske.shape[2]:
                    lbl = skel_labels[nz, ny, nx]
                    if lbl > 0:
                        neighbor_labels.append(lbl)
            if neighbor_labels:
                # Majority vote; ties go to plate (conservative)
                rod_count   = neighbor_labels.count(ROD)
                plate_count = neighbor_labels.count(PLATE)
                skel_labels[p] = PLATE if plate_count >= rod_count else ROD
            else:
                still_uncertain.append(p)
        uncertain = still_uncertain

    # Any voxel still unresolved (completely isolated structure) → rod by default
    for p in uncertain:
        skel_labels[p] = ROD

    rod_count   = int((skel_labels == ROD).sum())
    plate_count = int((skel_labels == PLATE).sum())
    print(f'skeleton classification: {rod_count} rod voxels, {plate_count} plate voxels')
    return skel_labels

# ---------------------------------------------------------------------------
# Step 2 – spread labels from skeleton back to the full original volume (BFS)
# ---------------------------------------------------------------------------

def spread_labels_to_volume(ske, skel_labels, original_volume):
    """
    Multi-source BFS: start simultaneously from every skeleton voxel.
    Each non-skeleton voxel in the original volume is assigned the label
    of the nearest skeleton voxel (26-connected distance).

    Returns an int8 array of the same shape as original_volume:
      0 = background (not in original volume)
      1 = rod
      2 = plate
    """
    volume_labels = np.zeros(original_volume.shape, dtype=np.int8)
    queue = deque()

    # Seed queue with all skeleton voxels
    for p_idx in np.argwhere(ske):
        p = tuple(p_idx)
        volume_labels[p] = skel_labels[p]
        queue.append(p)

    # BFS over 26-connected neighbours within the original volume
    while queue:
        z, y, x = queue.popleft()
        lbl = volume_labels[z, y, x]
        for dz, dy, dx in _NEIGHBOR_OFFSETS:
            nz, ny, nx = z + dz, y + dy, x + dx
            if (0 <= nz < original_volume.shape[0] and
                    0 <= ny < original_volume.shape[1] and
                    0 <= nx < original_volume.shape[2]):
                if original_volume[nz, ny, nx] and volume_labels[nz, ny, nx] == 0:
                    volume_labels[nz, ny, nx] = lbl
                    queue.append((nz, ny, nx))

    rod_count   = int((volume_labels == ROD).sum())
    plate_count = int((volume_labels == PLATE).sum())
    print(f'volume classification: {rod_count} rod voxels, {plate_count} plate voxels')
    return volume_labels

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

start = time.time()

voxeld = np.load('s01_voxel.npy')
sub = voxeld[80:180, 750:850, 850:950]
sub = sub[::2, ::2, ::2]

# Keep only the largest connected component
labeled, n = label(sub, structure=np.ones((3, 3, 3)))
if n > 0:
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    sub = (labeled == counts.argmax())

print('input extracted')

# ---- Skeletonization ----
ske, img, fi = thin_3d_saha(sub)
print(f'skeleton voxel count: {ske.sum()}')
np.save('s01_ske.npy', ske)

# ---- Classify skeleton voxels ----
skel_labels = classify_skeleton(ske, sub)
np.save('s01_skel_labels.npy', skel_labels)

# ---- Spread labels back to the full volume ----
volume_labels = spread_labels_to_volume(ske, skel_labels, sub)
np.save('s01_volume_labels.npy', volume_labels)

# ---- Render ----
scout('../scad/rp_skeleton', render_skeleton_labels(skel_labels))
scout('../scad/rp_volume',   render_volume_labels(volume_labels))

print(f"elapsed: {time.time() - start:.2f} seconds")
