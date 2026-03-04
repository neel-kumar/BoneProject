import time
import subprocess
import numpy as np
from scipy.ndimage import label
from solid import *

MAXINT = 1000000

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

def has_opposite_white_neighbors(white_keys):
    opposites = [('N', 'S'), ('E', 'W'), ('T', 'B')]
    for p1, p2 in opposites:
        if p1 in white_keys and p2 in white_keys:
            return True
    return False

def get_middle_plane(p, pair_name, mat):
    z, y, x = p
    if pair_name in ['T', 'B']: return mat[z, y-1:y+2, x-1:x+2]
    if pair_name in ['N', 'S']: return mat[z-1:z+2, y, x-1:x+2]
    return mat[z-1:z+2, y-1:y+2, x] # E, W

def get_ext_middle_plane(p, pair_name, mat):
    z, y, x = p
    if pair_name in ['T', 'B']: return mat[z, y-2:y+2, x-2:x+2]
    if pair_name in ['N', 'S']: return mat[z-2:z+2, y, x-2:x+2]
    return mat[z-2:z+2, y-2:y+2, x] # E, W

def is_simple(p, current_mat):
    # Standard (26, 6) simplicity check
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

def is_e_open(p, mat):
    if is_s_open(p, mat):
        return False

    s = get_s_points(p, mat)
    white_keys = [k for k, v in s.items() if v == 0]
    
    if len(white_keys) == 2 and not has_opposite_white_neighbors(white_keys):
        return True
    return False

def is_v_open(p, mat):
    if is_e_open(p, mat) or is_e_open(p, mat):
        return False

    s = get_s_points(p, mat)
    white_keys = [k for k, v in s.items() if v == 0]
    
    if len(white_keys) == 3 and not has_opposite_white_neighbors(white_keys):
        return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# Direction / offset tables
# ─────────────────────────────────────────────────────────────────────────────

# Each named s-point direction → (dz, dy, dx) offset from p
DIR_OFFSET = {
    'T': (-1, 0, 0), 'B': ( 1, 0, 0),
    'N': ( 0,-1, 0), 'S': ( 0, 1, 0),
    'W': ( 0, 0,-1), 'E': ( 0, 0, 1),
}

# Opposite direction of each named direction
OPPOSITE_DIR = {'T':'B','B':'T', 'N':'S','S':'N', 'E':'W','W':'E'}

# The three unordered opposite-pairs of s-points
ALL_OPP_PAIRS = [('T','B'), ('N','S'), ('E','W')]

# Paper's {P_E, P_S, P_W} — the "positive" half of each axis.
# Condition 2 only applies when d ∈ BOTTOM_DIRS.
BOTTOM_DIRS = {'E', 'S', 'B'}


# ─────────────────────────────────────────────────────────────────────────────
# Neighbourhood arithmetic helpers
# (mat is a binary padded volume; p = (z, y, x) in padded coordinates)
# ─────────────────────────────────────────────────────────────────────────────

def _off(d):
    """Return the (dz,dy,dx) offset for direction string d."""
    return DIR_OFFSET[d]

def _at(p, d, mat):
    """
    Paper: value of the neighbour of p in direction d.
    Returns mat[p + off(d)].  1 = black, 0 = white.
    """
    z,y,x = p
    dz,dy,dx = _off(d)
    return mat[z+dz, y+dy, x+dx]

def _at2(p, d1, d2, mat):
    """
    Paper: e(a,b,p) — e-point neighbour at offset off(d1)+off(d2).
    d1 and d2 must be non-opposite directions.
    """
    z,y,x = p
    dz = DIR_OFFSET[d1][0] + DIR_OFFSET[d2][0]
    dy = DIR_OFFSET[d1][1] + DIR_OFFSET[d2][1]
    dx = DIR_OFFSET[d1][2] + DIR_OFFSET[d2][2]
    return mat[z+dz, y+dy, x+dx]

def _at3(p, d1, d2, d3, mat):
    """
    Paper: v(a,b,c,p) — v-point neighbour at offset off(d1)+off(d2)+off(d3).
    d1, d2, d3 must be mutually non-opposite directions.
    """
    z,y,x = p
    dz = DIR_OFFSET[d1][0] + DIR_OFFSET[d2][0] + DIR_OFFSET[d3][0]
    dy = DIR_OFFSET[d1][1] + DIR_OFFSET[d2][1] + DIR_OFFSET[d3][1]
    dx = DIR_OFFSET[d1][2] + DIR_OFFSET[d2][2] + DIR_OFFSET[d3][2]
    return mat[z+dz, y+dy, x+dx]

def _at_f1(p, d, mat):
    """
    Paper: f1(a,p) — point one step beyond direction d (i.e. at 2*off(d)).
    This lies outside N(p).
    """
    z,y,x = p
    dz,dy,dx = _off(d)
    return mat[z+2*dz, y+2*dy, x+2*dx]

def _other_dirs(a, d):
    """
    Given one opposite pair (a, d), return (b, c, e, f) where
    (b,e) and (c,f) are the other two opposite pairs.
    Paper notation: (a,d), (b,e), (c,f) are all three unordered opposite pairs.
    """
    remaining = [pair for pair in ALL_OPP_PAIRS if a not in pair]
    b, e = remaining[0]
    c, f = remaining[1]
    return b, c, e, f


# ─────────────────────────────────────────────────────────────────────────────
# Extended middle plane — EM(a, d, p)
# Paper Section 2:
#   EM(a,d,p) = M(a,d,p)
#     ∪ {f1(x,p)   | x ∈ {b,e,c,f},   x ∈ BOTTOM_DIRS}              [C3]
#     ∪ {f2(x,y,p) | x,y ∈ {b,e,c,f}, non-opp, x ∈ BOTTOM_DIRS}     [C4]
#     ∪ {f3(x,y,p) | x,y ∈ {b,e,c,f}, non-opp, x,y ∈ BOTTOM_DIRS}   [C5]
#
# f2(x,y,p) = f1(x) + off(y)   (paper: 6-adjacent to f1(x,p) and e(x,y,p))
# f3(x,y,p) = f1(x) + f1(y)    (paper: 6-adjacent to f2(x,y,p) and f2(y,x,p))
# ─────────────────────────────────────────────────────────────────────────────

def _get_ext_middle_plane_pts(a, d):
    """
    Return all (dz,dy,dx) offsets that belong to EM(a,d,p).
    These are relative to p and may extend beyond the ±1 neighbourhood.
    Called once per (a,d) pair; results are cached in EMP_OFFSETS.
    """
    b, c, e, f = _other_dirs(a, d)
    non_ad = [b, c, e, f]

    # M(a,d,p): the four non-(a,d) s-points and all non-opposite e-pairs among them
    pts = set()
    for x in non_ad:
        pts.add(_off(x))                            # C1: s-point
        for y in non_ad:
            if y != x and OPPOSITE_DIR[x] != y:    # x,y non-opposite
                o = tuple(DIR_OFFSET[x][i]+DIR_OFFSET[y][i] for i in range(3))
                pts.add(o)                          # C2: e-point e(x,y)

    # EM extensions
    for x in non_ad:
        ox = _off(x)
        if x in BOTTOM_DIRS:
            # C3: f1(x) = 2*off(x)
            pts.add(tuple(2*ox[i] for i in range(3)))

        for y in non_ad:
            if y == x or OPPOSITE_DIR[x] == y:
                continue                            # must be non-opposite
            oy = _off(y)
            if x in BOTTOM_DIRS:
                # C4: f2(x,y) = 2*off(x) + off(y)
                pts.add(tuple(2*ox[i]+oy[i] for i in range(3)))
            if x in BOTTOM_DIRS and y in BOTTOM_DIRS:
                # C5: f3(x,y) = 2*off(x) + 2*off(y)
                pts.add(tuple(2*ox[i]+2*oy[i] for i in range(3)))

    return frozenset(pts)


# Precompute EM offsets for all 6 directions (both orientations of 3 pairs)
EMP_OFFSETS = {
    (a, d): _get_ext_middle_plane_pts(a, d)
    for a, d in [(a,d) for a,d in ALL_OPP_PAIRS] + [(d,a) for a,d in ALL_OPP_PAIRS]
}


# ─────────────────────────────────────────────────────────────────────────────
# surface(a, p)
# Paper Section 2:
#   surface(a,p) = {a} ∪ {e(a,x,p) | x ∈ {b,e,c,f}} ∪ {v(a,x,y,p) | x,y non-opp}
# Returns a list of (dz,dy,dx) offsets — the 9-point face on side a.
# ─────────────────────────────────────────────────────────────────────────────

def _get_surface_pts(a):
    """Return all (dz,dy,dx) offsets belonging to surface(a,p)."""
    d = OPPOSITE_DIR[a]
    b, c, e, f = _other_dirs(a, d)
    non_ad = [b, c, e, f]

    oa = _off(a)
    pts = {oa}                                      # {a} itself
    for x in non_ad:
        ox = _off(x)
        # e(a,x): a + x
        pts.add(tuple(oa[i]+ox[i] for i in range(3)))
        for y in non_ad:
            if y != x and OPPOSITE_DIR[x] != y:    # non-opposite
                oy = _off(y)
                # v(a,x,y): a + x + y  (avoid duplicate by only forward pairs)
                pts.add(tuple(oa[i]+ox[i]+oy[i] for i in range(3)))

    return frozenset(pts)


# Precompute surface offsets for each of the 6 directions
SURFACE_OFFSETS = {d: _get_surface_pts(d) for d in DIR_OFFSET}
print('SURFACE OFFSETS:', SURFACE_OFFSETS)


# ─────────────────────────────────────────────────────────────────────────────
# 6-connected component flood-fill (within a set of offsets, 6-adjacency)
# Used to find the 6-closed path in EM for Condition 1.
# ─────────────────────────────────────────────────────────────────────────────

def _6connected_component(pts_set):
    """BFS over 6-adjacency within pts_set. Returns the component of the first point."""
    if not pts_set:
        return set()
    visited = set()
    start = next(iter(pts_set))
    stack = [start]
    visited.add(start)
    while stack:
        cur = stack.pop()
        for nb in pts_set:
            if nb not in visited:
                if sum(abs(cur[i]-nb[i]) for i in range(3)) == 1:
                    visited.add(nb)
                    stack.append(nb)
    return visited


# ─────────────────────────────────────────────────────────────────────────────
# Condition 1  —  arc-like shape
# Paper p.1942:
#   p satisfies Condition 1 if there exist opposite s-points a, d ∈ N(p) such that:
#     (i)  EM(a,d,p) contains a 6-closed path of white points encircling p
#     (ii) surface(a,p) contains ≥ 1 black point before the iteration
#     (iii)surface(d,p) contains ≥ 1 black point before the iteration
#
# "Encircles p": the 6-connected white component in EM visits ≥ 2 of the
# four non-(a,d) s-points  (paper: "contains two points of EM ∩ {PT,PN,PE}").
# ─────────────────────────────────────────────────────────────────────────────

def condition_1(p, mat):
    z, y, x = p

    for a, d in ALL_OPP_PAIRS:
        for a_dir, d_dir in [(a, d), (d, a)]:
            b, c, e, f = _other_dirs(a_dir, d_dir)
            non_ad_offsets = {_off(b), _off(c), _off(e), _off(f)}

            # Collect white points in EM(a_dir, d_dir, p)
            white_em = frozenset(
                off for off in EMP_OFFSETS[(a_dir, d_dir)]
                if not mat[z+off[0], y+off[1], x+off[2]]
            )

            # (i) Find 6-connected white component; check encirclement
            component = _6connected_component(white_em)
            if len(component) < 3:
                continue
            # "Encircles p" = path touches ≥ 2 of the 4 non-(a,d) s-points
            encircling = sum(1 for o in non_ad_offsets if o in component)
            if encircling < 2:
                continue

            # (ii) surface(a) has ≥ 1 black point
            if not any(mat[z+o[0], y+o[1], x+o[2]] for o in SURFACE_OFFSETS[a_dir]):
                continue

            # (iii) surface(d) has ≥ 1 black point
            if not any(mat[z+o[0], y+o[1], x+o[2]] for o in SURFACE_OFFSETS[d_dir]):
                continue

            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Condition 2 — surface-like shape
# Paper p.1943:
#   p satisfies Condition 2 if there exists an opposite pair (a, d) with
#   d ∈ BOTTOM_DIRS such that:
#     (i)  a is white
#     (ii) d or f1(d,p) is white
#     (iii)each of the 8 sets below contains ≥ 1 black point:
#            {e(a,b), b, e(b,d)},  {e(a,c), c, e(c,d)},
#            {e(a,e), e, e(d,e)},  {e(a,f), f, e(d,f)},
#            {v(a,b,c), e(b,c), v(b,c,d)},  {v(a,b,f), e(b,f), v(b,d,f)},
#            {v(a,c,e), e(c,e), v(c,d,e)},  {v(a,e,f), e(e,f), v(d,e,f)}
# ─────────────────────────────────────────────────────────────────────────────

def condition_2(p, mat):
    z, y, x = p

    for a, d in ALL_OPP_PAIRS:
        for a_dir, d_dir in [(a, d), (d, a)]:
            if d_dir not in BOTTOM_DIRS:
                continue

            # (i) a must be white
            if _at(p, a_dir, mat):
                continue

            # (ii) d or f1(d) must be white
            if _at(p, d_dir, mat) and _at_f1(p, d_dir, mat):
                continue

            b, c, e, f = _other_dirs(a_dir, d_dir)

            # (iii) each of the 8 sets must contain ≥ 1 black point
            # 4 edge-triplets
            sets8 = [
                (_at2(p,a_dir,b,mat), _at(p,b,mat), _at2(p,b,d_dir,mat)),
                (_at2(p,a_dir,c,mat), _at(p,c,mat), _at2(p,c,d_dir,mat)),
                (_at2(p,a_dir,e,mat), _at(p,e,mat), _at2(p,d_dir,e,mat)),
                (_at2(p,a_dir,f,mat), _at(p,f,mat), _at2(p,d_dir,f,mat)),
                # 4 corner-triplets
                (_at3(p,a_dir,b,c,mat), _at2(p,b,c,mat), _at3(p,b,c,d_dir,mat)),
                (_at3(p,a_dir,b,f,mat), _at2(p,b,f,mat), _at3(p,b,d_dir,f,mat)),
                (_at3(p,a_dir,c,e,mat), _at2(p,c,e,mat), _at3(p,c,d_dir,e,mat)),
                (_at3(p,a_dir,e,f,mat), _at2(p,e,f,mat), _at3(p,d_dir,e,f,mat)),
            ]

            if all(any(s) for s in sets8):
                return True

    return False

# def condition_1(p, mat):
#     pass
#
# def condition_2(p, mat):
#     pass

def is_shape_point(p, mat):
    # Definition 4: A point p is a shape point if it satisfies Condition 1 or Condition 2.
    return condition_1(p, mat) or condition_2(p, mat)

def condition_3(p, current_mat, old_mat):
    # Refined drill prevention
    for pair in [('T','B'), ('N','S'), ('W','E')]:
        m_plane = get_middle_plane(p, pair[0], current_mat).copy()
        m_plane[1, 1] = 0
        _, n_comp = label(m_plane, structure=np.ones((3,3)))
        if n_comp != 1: return False
        s_in_m = [m_plane[0,1], m_plane[2,1], m_plane[1,0], m_plane[1,2]]
        if all(s_in_m): return False
    return True

# def thick(p, old_mat):
#     s = get_s_points(p, old_mat)
#     # no pair of opposite s-neighbors are both 0
#     if (s['N'] == 0 and s['S'] == 0): return 0
#     if (s['E'] == 0 and s['W'] == 0): return 0
#     if (s['T'] == 0 and s['B'] == 0): return 0
#     return 1

def thick(p, mat, direction_of_white_d):
    # 'p has no two opposite s-points b, e such that both b and e are white'
    s = get_s_points(p, mat)
    
    if direction_of_white_d in ['T', 'B']:
        if (s['N'] == 0 and s['S'] == 0) or (s['E'] == 0 and s['W'] == 0):
            return 0
    elif direction_of_white_d in ['N', 'S']:
        if (s['T'] == 0 and s['B'] == 0) or (s['E'] == 0 and s['W'] == 0):
            return 0
    elif direction_of_white_d in ['E', 'W']:
        if (s['T'] == 0 and s['B'] == 0) or (s['N'] == 0 and s['S'] == 0):
            return 0
            
    return 1

def is_topo_safe(plane_3x3):
    n8 = plane_3x3.copy()
    n8[1, 1] = 0
    if np.sum(n8) == 0: return False
    _, n_comp = label(n8, structure=np.ones((3,3)))
    if n_comp != 1: return False
    
    # no tunnel: tot all 4 s-neighbors in the 3x3 are black
    s_points_2d = [plane_3x3[0,1], plane_3x3[2,1], plane_3x3[1,0], plane_3x3[1,2]]
    if all(s_points_2d): return False
    
    return True

def get_other_axes(d):
    # d is a direction string like 'T', 'N', 'E'
    # returns the other two axis labels (as pairs) for the middle plane checks
    all_axes = [('T','B'), ('N','S'), ('E','W')]
    return [pair for pair in all_axes if d not in pair]

def axis_to_dir(axis_pair):
    # takes one of the axis pairs and returns the canonical direction for thick()
    # thick() takes the "direction of white d", so return the BOTTOM_DIRS member
    return next(d for d in axis_pair if d in BOTTOM_DIRS)

def get_white_axes(s):
    # s is the dict from get_s_points — returns axis pairs where at least one direction in the pair has a white (0) s-neighbor
    all_axes = [('T','B'), ('N','S'), ('E','W')]
    return [pair for pair in all_axes if s[pair[0]] == 0 or s[pair[1]] == 0]

def get_remaining_axis(white_axes):
    # white_axes is a list of axis pairs — returns the one axis pair not in it
    all_axes = [('T','B'), ('N','S'), ('E','W')]
    return next(pair for pair in all_axes if pair not in white_axes)

def get_middle_plane_by_axis(p, axis_pair, mat):
    # axis_pair is e.g. ('T','B') — delegates to get_middle_plane using
    # the canonical direction name for that axis
    return get_middle_plane(p, axis_pair[0], mat)

# def condition_4(p, mat):
#     s = get_s_points(p, mat)
#     white_dirs = [k for k, v in s.items() if v == 0]
#
#     for d in white_dirs:
#         # If the point is 'thick' relative to this white neighbor
#         if thick(p, mat, d):
#             # Check if the other two middle planes are topologically safe
#             if d == 'N' or d == 'S':
#                 planes = [get_middle_plane(p, ax, mat) for ax in ['E','W','T','B']]
#                 if all(is_topo_safe(pl) for pl in planes):
#                     return True
#             elif d == 'E' or d == 'W':
#                 planes = [get_middle_plane(p, ax, mat) for ax in ['N','S','T','B']]
#                 if all(is_topo_safe(pl) for pl in planes):
#                     return True
#             else: # T, B
#                 planes = [get_middle_plane(p, ax, mat) for ax in ['E','W','N','S']]
#                 if all(is_topo_safe(pl) for pl in planes):
#                     return True
#     return False
#
# def condition_5(p, mat):
#     s = get_s_points(p, mat)
#     # Identify the two axes that contain white s-neighbors
#     white_axes = get_white_axes(s) 
#
#     # Check if p is thick on BOTH white axes
#     if all(thick(p, mat, axis_to_dir(ax)) for ax in white_axes):
#         # Check if the third (remaining) axis is topo safe
#         remaining_ax = get_remaining_axis(white_axes)
#         if is_topo_safe(get_middle_plane(p, remaining_ax, mat)):
#             return True
#     return False

def condition_4(p, mat):
    all_axes = [('T','B'), ('N','S'), ('E','W')]
    for a_dir, d_dir in [(a,d) for a,d in all_axes] + [(d,a) for a,d in all_axes]:
        if d_dir not in BOTTOM_DIRS:
            continue
        if not thick(p, mat, d_dir):
            continue
        other_planes = [get_middle_plane(p, ax[0], mat) for ax in all_axes if d_dir not in ax]
        if all(is_topo_safe(pl) for pl in other_planes):
            return True
    return False


def condition_5(p, mat):
    all_axes = [('T','B'), ('N','S'), ('E','W')]
    for i, (a1, d1) in enumerate(all_axes):
        if d1 not in BOTTOM_DIRS:
            continue
        if not thick(p, mat, d1):
            continue
        for a2, d2 in all_axes[i+1:]:
            if d2 not in BOTTOM_DIRS:
                continue
            if not thick(p, mat, d2):
                continue
            remaining = [ax for ax in all_axes if ax != (a1,d1) and ax != (a2,d2)][0]
            if is_topo_safe(get_middle_plane(p, remaining[0], mat)):
                return True
    return False

def condition_6(p, mat):
    return (thick(p, mat, 'T') and thick(p, mat, 'N') and thick(p, mat, 'E'))

def def6(p, img):
    return is_simple(p, img >= 0) and ( condition_4(p, img) or condition_5(p, img) or condition_6(p, img) )

def condition_1(p, img):


def thin_3d_saha(voxels, final_thinning=True):
    img = np.where(voxels > 0, 0, -MAXINT)
    img = np.pad(img, 2, constant_values=-MAXINT)
    its = 1
    while True:
        changed = False
        thr = -MAXINT + its
        old_img = (img >= thr) 
        
        # 1st Scan: s-open
        for p_idx in np.argwhere(img == 0):
            p = tuple(p_idx)
            # Use iteration-start state for classification
            if is_s_open(p, old_img):
                if is_shape_point(p, old_img):
                    img[p] = its # Mark
                elif is_simple(p, img >= 0):
                    img[p] = thr # Delete
                    changed = True

        # # 2nd Scan: e-open
        # for p_idx in np.argwhere(img == 0):
        #     p = tuple(p_idx)
        #     # Strict Condition 3 for all potentially non-shape e-points
        #     if is_e_open(p, old_img):
        #         assert(not is_shape_point(p, old_img) and img[p] != its)
        #         if is_simple(p, img >= 0) and condition_3(p, img >= 0, old_img): 
        #             img[p] = thr
        #             changed = True

        # # 3rd Scan: v-open
        # for p_idx in np.argwhere(img == 0):
        #     p = tuple(p_idx)
        #     if is_v_open(p, old_img) and is_simple(p, img >= 0):
        #         img[p] = thr
        #         changed = True

        if not changed:
            break
        # if its > 10:
        #     break
        its += 1

    if final_thinning:
        # One pass to handle slanted surfaces
        for p_idx in np.argwhere(img >= 0):
            p = tuple(p_idx)
            if def6(p, img):
                img[p] = -MAXINT + its
            
    return img[2:-2, 2:-2, 2:-2] >= 0, img, its


def render_3d_mat_type(mat, cube_size=0.1):
    positions = np.argwhere(mat)
    if len(positions) == 0:
        return cube(0)
    
    # Pad to safely check neighbors
    padded = np.pad(mat, 1, constant_values=0)
    cells = None
    
    for z, y, x in positions:
        p_pad = (z + 1, y + 1, x + 1)
        
        # Classification logic
        if is_v_open(p_pad, padded) and is_simple(p_pad, padded):
            # V-points: Red
            c = color([1, 0, 0])(cube(cube_size))
        elif is_v_open(p_pad, padded):
            # less red
            c = color([0.8, 0.4, 0.4])(cube(cube_size))
        elif is_e_open(p_pad, padded):
            # E-points: Green
            c = color([0.5, 0.7, 0.5])(cube(cube_size))
        elif is_s_open(p_pad, padded):
            # S-points: Blue
            c = color([0.4, 0.5, 0.8])(cube(cube_size))
        else:
            # Interior: Gray
            c = color([0.5, 0.5, 0.5])(cube(cube_size))
            
        c = translate([x * cube_size, y * cube_size, z * cube_size])(c)
        cells = c if cells is None else cells + c
        
    return cells

start = time.time()
voxeld = np.load('s01_voxel.npy')
sub = voxeld[80:180, 750:850, 850:950]
sub = sub[::2, ::2, ::2]
sub = largest_connected_component(sub)
print('input extracted')
ske,img,its = thin_3d_saha(sub, final_thinning=False)
print('iterations', its)
print('marked', (img > 0).sum())
print('unmarked', (img == 0).sum())
print(f'true cnt: {ske.sum()}')
scout('../scad/img8.10', render_3d_mat(ske))
print(f"elapsed: {time.time() - start:.2f} seconds")
