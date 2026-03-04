import time
import subprocess
import numpy as np
from scipy.ndimage import *

from solid import *

def scout(filename, obj, to_stl=False):
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


#  Neighbor offsets 
S_OFFSETS = np.array([
    [ 0,  0,  1], [ 0,  0, -1],
    [ 0,  1,  0], [ 0, -1,  0],
    [ 1,  0,  0], [-1,  0,  0],
], dtype=np.int8)

E_OFFSETS = np.array([
    [ 0,  1,  1], [ 0,  1, -1], [ 0, -1,  1], [ 0, -1, -1],
    [ 1,  0,  1], [ 1,  0, -1], [-1,  0,  1], [-1,  0, -1],
    [ 1,  1,  0], [ 1, -1,  0], [-1,  1,  0], [-1, -1,  0],
], dtype=np.int8)

V_OFFSETS = np.array([
    [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
    [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1],
], dtype=np.int8)

ALL26_OFFSETS = np.vstack([S_OFFSETS, E_OFFSETS, V_OFFSETS])


def offsets_to_flat_idx(offsets):
    o = np.asarray(offsets, dtype=int)
    if o.ndim == 1:
        o = o[None]
    return ((o[:, 0] + 1) * 9 + (o[:, 1] + 1) * 3 + (o[:, 2] + 1)).astype(np.int8)


S_IDX   = offsets_to_flat_idx(S_OFFSETS)
E_IDX   = offsets_to_flat_idx(E_OFFSETS)
V_IDX   = offsets_to_flat_idx(V_OFFSETS)
N26_IDX = offsets_to_flat_idx(ALL26_OFFSETS)

# Opposite s-pairs as tuples
S_OPPOSITE_PAIRS = [
    (tuple(sa.tolist()), tuple((-sa).tolist()))
    for sa in S_OFFSETS.astype(int)
    if tuple(sa.tolist()) < tuple((-sa).tolist())
]


#  Precomputed lookup tables built at module load 

def build_e_parent_table():
    s = S_OFFSETS.astype(int)
    rows = []
    for ei, e in enumerate(E_OFFSETS.astype(int)):
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if np.array_equal(s[i] + s[j], e):
                    f1a = s[i] * 2
                    f1b = s[j] * 2
                    fi_a = int(offsets_to_flat_idx(f1a)[0]) if np.abs(f1a).max() <= 1 else -1
                    fi_b = int(offsets_to_flat_idx(f1b)[0]) if np.abs(f1b).max() <= 1 else -1
                    si_a = int(S_IDX[i])
                    si_b = int(S_IDX[j])
                    rows.append((ei, si_a, si_b, fi_a, fi_b))
    return rows


def build_v_parent_table():
    s = S_OFFSETS.astype(int)
    rows = []
    for vi, v in enumerate(V_OFFSETS.astype(int)):
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                for k in range(j + 1, len(s)):
                    if np.array_equal(s[i] + s[j] + s[k], v):
                        f1a = s[i] * 2
                        f1b = s[j] * 2
                        f1c = s[k] * 2
                        fi_a = int(offsets_to_flat_idx(f1a)[0]) if np.abs(f1a).max() <= 1 else -1
                        fi_b = int(offsets_to_flat_idx(f1b)[0]) if np.abs(f1b).max() <= 1 else -1
                        fi_c = int(offsets_to_flat_idx(f1c)[0]) if np.abs(f1c).max() <= 1 else -1
                        rows.append((vi, fi_a, fi_b, fi_c))
    return rows


def build_middle_plane_table():
    s = S_OFFSETS.astype(int)
    seen = set()
    rows = []
    for sa in s:
        a = tuple(sa.tolist())
        d = tuple((-sa).tolist())
        key = tuple(sorted([a, d]))
        if key in seen:
            continue
        seen.add(key)
        in_plane_s = [tuple(sc.tolist()) for sc in s if tuple(sc.tolist()) != a and tuple(sc.tolist()) != d]
        in_plane_e = [
            tuple(ec.tolist()) for ec in E_OFFSETS.astype(int)
            if int(np.dot(ec, sa)) == 0
        ]
        s_idx = [int(offsets_to_flat_idx(np.array(o))[0]) for o in in_plane_s]
        e_idx = [int(offsets_to_flat_idx(np.array(o))[0]) for o in in_plane_e]
        rows.append((np.array(s_idx), np.array(e_idx)))
    return rows


def build_surface_table():
    from itertools import combinations
    s_all = [tuple(o.tolist()) for o in S_OFFSETS.astype(int)]
    table = {}
    for a in s_all:
        opp = tuple((-np.array(a)).tolist())
        non_opp = [sc for sc in s_all if sc != a and sc != opp]
        pts = [a]
        for sc in non_opp:
            pts.append(tuple((np.array(a) + np.array(sc)).tolist()))
        for sc1, sc2 in combinations(non_opp, 2):
            pts.append(tuple((np.array(a) + np.array(sc1) + np.array(sc2)).tolist()))
        table[a] = pts
    return table


def build_ext_middle_plane_table():
    from itertools import combinations
    s_all = [tuple(o.tolist()) for o in S_OFFSETS.astype(int)]
    table = {}
    for a, d in S_OPPOSITE_PAIRS:
        for a2, d2 in [(a, d), (d, a)]:
            non_opp = [sc for sc in s_all if sc != a2 and sc != d2]
            plane = set(non_opp)
            for sc1, sc2 in combinations(non_opp, 2):
                plane.add(tuple((np.array(sc1) + np.array(sc2)).tolist()))
            for sc in non_opp:
                f1 = tuple((np.array(sc) * 2).tolist())
                plane.add(f1)
                for sc2 in non_opp:
                    if sc2 == sc:
                        continue
                    f2 = tuple((np.array(f1) + np.array(sc2)).tolist())
                    plane.add(f2)
                    f3 = tuple((np.array(f2) + np.array(sc)).tolist())
                    plane.add(f3)
            table[(a2, d2)] = list(plane)
    return table


E_PARENT_TABLE      = build_e_parent_table()
V_PARENT_TABLE      = build_v_parent_table()
MIDDLE_PLANE_TABLE  = build_middle_plane_table()
SURFACE_TABLE       = build_surface_table()
EXT_MIDDLE_PLANE_TABLE = build_ext_middle_plane_table()

# Precomputed condition-2 sets for each (a, d) bottom-direction pair
BOTTOM_DIRS = {(1, 0, 0), (0, 1, 0), (0, 0, 1)}

def build_condition2_table():
    s_all = [tuple(o.tolist()) for o in S_OFFSETS.astype(int)]
    table = {}
    for a, d in S_OPPOSITE_PAIRS:
        for a2, d2 in [(a, d), (d, a)]:
            if d2 not in BOTTOM_DIRS:
                continue
            rem = [s for s in s_all if s != a2 and s != d2]
            b, c, e, f = rem
            def ep(p, q): return tuple((np.array(p) + np.array(q)).tolist())
            def vp(p, q, r): return tuple((np.array(p) + np.array(q) + np.array(r)).tolist())
            sets8 = [
                (ep(a2,b), b, ep(b,d2)),
                (ep(a2,c), c, ep(c,d2)),
                (ep(a2,e), e, ep(d2,e)),
                (ep(a2,f), f, ep(d2,f)),
                (vp(a2,b,c), ep(b,c), vp(b,c,d2)),
                (vp(a2,b,f), ep(b,f), vp(b,d2,f)),
                (vp(a2,c,e), ep(c,e), vp(c,d2,e)),
                (vp(a2,e,f), ep(e,f), vp(d2,e,f)),
            ]
            table[(a2, d2)] = sets8
    return table

CONDITION2_TABLE = build_condition2_table()


#  Patch extraction 

MAXINT = 10 ** 7

def pad_vol(vol, pad=2):
    return np.pad(vol, pad, mode='constant', constant_values=-MAXINT)


def extract_patches(vol_padded, coords):
    patches = np.zeros((len(coords), 3, 3, 3), dtype=vol_padded.dtype)
    for di in range(3):
        for dj in range(3):
            for dk in range(3):
                patches[:, di, dj, dk] = vol_padded[
                    coords[:, 0] + di,
                    coords[:, 1] + dj,
                    coords[:, 2] + dk,
                ]
    return patches.reshape(len(coords), 27)


def get_patch(vol_padded, z, y, x):
    return vol_padded[z:z+3, y:y+3, x:x+3].ravel()


#  Simple point check (Section 2, four conditions) 

STRUCT26 = np.ones((3, 3, 3), dtype=bool)
STRUCT6  = np.zeros((3, 3, 3), dtype=bool)
STRUCT6[1, 1, 0] = STRUCT6[1, 1, 2] = True
STRUCT6[1, 0, 1] = STRUCT6[1, 2, 1] = True
STRUCT6[0, 1, 1] = STRUCT6[2, 1, 1] = True
STRUCT6[1, 1, 1] = True


def is_26_connected(black26_grid):
    _, n = label(black26_grid, structure=STRUCT26)
    return n <= 1


def is_6_connected_in_18(white6_grid, white18_grid):
    visited = np.zeros((3, 3, 3), dtype=bool)
    starts = np.argwhere(white6_grid)
    if len(starts) == 0:
        return True
    stack = [tuple(starts[0])]
    visited[tuple(starts[0])] = True
    while stack:
        cur = np.array(stack.pop())
        for nb_idx in np.argwhere(STRUCT6):
            nb = cur + nb_idx - 1
            if np.any(nb < 0) or np.any(nb > 2):
                continue
            t = tuple(nb)
            if visited[t] or not white6_grid[t]:
                continue
            mid = tuple(((cur + nb) // 2).tolist())
            if mid == (1, 1, 1) or white18_grid[mid]:
                visited[t] = True
                stack.append(t)
    return bool(visited[white6_grid].all())


def is_simple_point(patch27, thr):
    black26 = np.zeros(27, dtype=bool)
    black26[N26_IDX] = patch27[N26_IDX] > -MAXINT
    if not black26.any():
        return False

    white6  = patch27[S_IDX] <= -MAXINT
    white18 = np.concatenate([patch27[S_IDX] <= -MAXINT, patch27[E_IDX] <= -MAXINT])

    if not white6.any():
        return False

    black26_grid = black26.reshape(3, 3, 3)
    if not is_26_connected(black26_grid):
        return False

    white6_grid  = np.zeros((3, 3, 3), dtype=bool)
    white18_grid = np.zeros((3, 3, 3), dtype=bool)
    white6_grid.ravel()[S_IDX[white6]]   = True
    s_w18 = patch27[S_IDX] <= -MAXINT
    e_w18 = patch27[E_IDX] <= -MAXINT
    white18_grid.ravel()[S_IDX[s_w18]]  = True
    white18_grid.ravel()[E_IDX[e_w18]]  = True

    if not is_6_connected_in_18(white6_grid, white18_grid):
        return False

    return True


#  Open-layer labeling (vectorized over all black voxels) 

def label_open_layers(patches_before, thr):
    n = len(patches_before)
    black_before = patches_before >= thr

    s_before = black_before[:, S_IDX]
    e_before = black_before[:, E_IDX]
    v_before = black_before[:, V_IDX]

    is_s_open = ~s_before.all(axis=1)

    is_e_open = np.zeros(n, dtype=bool)
    for (ei, si_a, si_b, fi_a, fi_b) in E_PARENT_TABLE:
        e_white = ~e_before[:, ei]
        f1a_black = black_before[:, fi_a] if fi_a >= 0 else np.ones(n, dtype=bool)
        f1b_black = black_before[:, fi_b] if fi_b >= 0 else np.ones(n, dtype=bool)
        is_e_open |= (~is_s_open & e_white & black_before[:, si_a] & black_before[:, si_b]
                      & f1a_black & f1b_black)

    is_v_open = np.zeros(n, dtype=bool)
    for (vi, fi_a, fi_b, fi_c) in V_PARENT_TABLE:
        v_white = ~v_before[:, vi]
        f1a_black = black_before[:, fi_a] if fi_a >= 0 else np.ones(n, dtype=bool)
        f1b_black = black_before[:, fi_b] if fi_b >= 0 else np.ones(n, dtype=bool)
        f1c_black = black_before[:, fi_c] if fi_c >= 0 else np.ones(n, dtype=bool)
        is_v_open |= (~is_s_open & ~is_e_open & v_white & f1a_black & f1b_black & f1c_black)

    return is_s_open, is_e_open, is_v_open


#  Condition 3 (2D topology in coordinate planes)

def check_condition3(patch27):
    currently_black = patch27 > -MAXINT
    for s_plane_idx, e_plane_idx in MIDDLE_PLANE_TABLE:
        if currently_black[e_plane_idx].all():
            continue
        if currently_black[s_plane_idx].all():
            return False
    return True


#  Condition 1: arc-like shape

def has_6_closed_path_encircling(white_pts_set, a, d):
    s_all = [tuple(o.tolist()) for o in S_OFFSETS.astype(int)]
    non_opp = [sc for sc in s_all if sc != a and sc != d]
    if len(white_pts_set) < 3:
        return False
    visited = set()
    start = next(iter(white_pts_set))
    stack = [start]
    visited.add(start)
    while stack:
        cur = stack.pop()
        for nb in white_pts_set:
            if nb not in visited:
                if abs(cur[0]-nb[0]) + abs(cur[1]-nb[1]) + abs(cur[2]-nb[2]) == 1:
                    visited.add(nb)
                    stack.append(nb)
    return len(visited) >= 3 and len([sc for sc in non_opp if sc in visited]) >= 2


#  Shape-point detection (Conditions 1 and 2) ─

def vol_sample_before(vol_padded, coord, off, thr):
    oc = np.array(coord) + 1 + np.array(off)
    sh = np.array(vol_padded.shape)
    if np.any(oc < 0) or np.any(oc >= sh):
        return True
    return bool(vol_padded[oc[0], oc[1], oc[2]] >= thr)


def is_shape_point(vol_padded, coord, thr):
    def before(off): return vol_sample_before(vol_padded, coord, off, thr)

    for a, d in S_OPPOSITE_PAIRS:
        em_pts = EXT_MIDDLE_PLANE_TABLE[(a, d)]
        white_em = {o for o in em_pts if not before(o)}
        if has_6_closed_path_encircling(white_em, a, d):
            if any(before(o) for o in SURFACE_TABLE[a]) and any(before(o) for o in SURFACE_TABLE[d]):
                return True

    for (a2, d2), sets8 in CONDITION2_TABLE.items():
        a_white = not before(a2)
        d_or_f1d_white = not before(d2) or not before(tuple((np.array(d2) * 2).tolist()))
        if a_white and d_or_f1d_white:
            if all(any(before(o) for o in s) for s in sets8):
                return True

    return False


#  thick(a, d, p) – Definition 5 

def thick(patch27_before, a_off, d_off):
    def b(off):
        idx = int(offsets_to_flat_idx(np.array(off))[0])
        if 0 <= idx < 27:
            return bool(patch27_before[idx])
        return True

    if b(a_off):
        return False
    f1d = tuple((np.array(d_off) * 2).tolist())
    if b(f1d):
        return False

    s_all = [tuple(o.tolist()) for o in S_OFFSETS.astype(int)]
    others = [s for s in s_all if s != tuple(a_off) and s != tuple(d_off)]
    others_idx = offsets_to_flat_idx(np.array(others))
    return bool(patch27_before[others_idx].all())


def is_erodable(patch27, thr):
    if not is_simple_point(patch27, thr):
        return False
    patch_before = (patch27 >= thr).astype(np.int32)
    for d in BOTTOM_DIRS:
        a = tuple((-np.array(d)).tolist())
        if thick(patch_before, a, d):
            return True
    return False


#  Primary thinning 

def primary_thinning(vol):
    work = np.where(vol > 0, 0, -MAXINT).astype(np.int32)
    changed = True
    iteration = 0

    while changed:
        iteration += 1
        thr = -MAXINT + iteration
        changed = False

        coords = np.argwhere(work >= 0)
        if len(coords) == 0:
            break

        wp2 = pad_vol(work, pad=2)
        patches_before = extract_patches(wp2, coords + 2)
        is_s_open, is_e_open, is_v_open = label_open_layers(patches_before, thr)

        unmarked = work[coords[:, 0], coords[:, 1], coords[:, 2]] == 0
        s_coords = coords[is_s_open & unmarked]
        e_coords = coords[is_e_open & unmarked]
        v_coords = coords[is_v_open & unmarked]

        wp1 = pad_vol(work, pad=1)
        for z, y, x in s_coords:
            patch = get_patch(wp1, z, y, x)
            if is_shape_point(wp1, (z, y, x), thr):
                work[z, y, x] = iteration
            elif is_simple_point(patch, thr):
                work[z, y, x] = thr
                changed = True

        wp1 = pad_vol(work, pad=1)
        for z, y, x in e_coords:
            patch = get_patch(wp1, z, y, x)
            if is_simple_point(patch, thr) and check_condition3(patch):
                work[z, y, x] = thr
                changed = True

        wp1 = pad_vol(work, pad=1)
        for z, y, x in v_coords:
            patch = get_patch(wp1, z, y, x)
            if is_simple_point(patch, thr):
                work[z, y, x] = thr
                changed = True

    return work


# Final thinning 

def final_thinning(work):
    wp1 = pad_vol(work, pad=1)
    coords = np.argwhere(work >= 0)
    to_delete = [
        (z, y, x) for z, y, x in coords
        if is_erodable(get_patch(wp1, z, y, x), thr=0)
    ]
    for z, y, x in to_delete:
        work[z, y, x] = -MAXINT
    return work


def skeletonize_3d(binary_vol):
    work = primary_thinning(binary_vol)
    # work = final_thinning(work)
    return work >= 0


def largest_connected_component(mat):
    labeled, n = label(mat, structure=STRUCT26)
    if n == 0:
        return np.zeros_like(mat, dtype=bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # exclude background
    largest_label = counts.argmax()
    return labeled == largest_label


start = time.time()

voxeld = np.load('s01_voxel.npy')
# sub = voxeld[550:650, 700:800, 800:900]
sub = voxeld[80:180, 750:850, 850:950]
sub = largest_connected_component(sub[::2, ::2, ::2])
print(sub.shape)
print(f'true cnt: {sub.sum()}')

skeleton = skeletonize_3d(sub)
print(f'skeleton cnt: {skeleton.sum()}')
np.save('s01_8_skeleton.npy', skeleton);
print('saved')

end = time.time()
print(f"Elapsed: {end - start:.2f} seconds")

scout('../scad/ske8.2', render_3d_mat(skeleton));

end = time.time()
print(f"Elapsed: {end - start:.2f} seconds")
