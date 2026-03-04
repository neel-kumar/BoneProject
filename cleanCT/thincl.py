"""
3D Shape Preserving Parallel Thinning Algorithm
Based on: Saha, Chaudhuri, Dutta Majumder (1997)
"A New Shape Preserving Parallel Thinning Algorithm for 3D Digital Images"
Pattern Recognition, Vol. 30, No. 12, pp. 1939-1955

Implementation follows user's notes exactly for each section.
"""

import numpy as np
from itertools import product as iterproduct

MAXINT = 2**30

# =============================================================================
# Neighborhood definitions
# =============================================================================

# "two image: check old for s/e/v-open, and if shape point; check new/current to check for simple point"

# 6-adjacent offsets (s-points): face neighbors
ADJ6 = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
], dtype=np.int32)

# 18-adjacent offsets: face + edge neighbors
ADJ18 = np.array([d for d in iterproduct([-1, 0, 1], repeat=3)
                   if 0 < abs(d[0]) + abs(d[1]) + abs(d[2]) <= 2], dtype=np.int32)

# 26-adjacent offsets: face + edge + vertex neighbors
ADJ26 = np.array([d for d in iterproduct([-1, 0, 1], repeat=3)
                   if d != (0, 0, 0)], dtype=np.int32)


def _neighbors(img, p, offsets):
    """Get neighbor coordinates and values."""
    coords = p + offsets
    valid = np.all((coords >= 0) & (coords < np.array(img.shape)), axis=1)
    return coords[valid], valid


def _get(img, p):
    """Get image value at point p (tuple or array)."""
    if isinstance(p, np.ndarray):
        p = tuple(p)
    if all(0 <= p[i] < img.shape[i] for i in range(3)):
        return img[p]
    return -MAXINT  # out of bounds treated as white


def _set(img, p, val):
    """Set image value at point p."""
    if isinstance(p, np.ndarray):
        p = tuple(p)
    img[p] = val


# =============================================================================
# Named s-points of N(p) - the 6 face neighbors
# Using paper's convention: N, S, E, W, T, B
# =============================================================================

# Map names to offsets relative to p
# Convention from paper Fig 1:
# T = top (+z or +y depending on convention), B = bottom
# N = north, S = south, E = east, W = west
# We use: x=E/W, y=N/S, z=T/B
S_POINTS = {
    'N':  np.array([0, 1, 0]),
    'S':  np.array([0, -1, 0]),
    'E':  np.array([1, 0, 0]),
    'W':  np.array([-1, 0, 0]),
    'T':  np.array([0, 0, 1]),
    'B':  np.array([0, 0, -1]),
}

# Opposite pairs: (a, d)
OPPOSITE_PAIRS = [
    ('N', 'S'), ('E', 'W'), ('T', 'B'),
]

# All 3 pairs of opposite s-points as (a,d), (b,e), (c,f) permutations
def _all_opposite_triple_assignments():
    """Generate all assignments of 3 opposite pairs to (a,d),(b,e),(c,f)."""
    from itertools import permutations
    pairs = [('N', 'S'), ('E', 'W'), ('T', 'B')]
    result = []
    for perm in permutations(pairs):
        # Also consider flipping each pair
        for flip in iterproduct([False, True], repeat=3):
            assignment = []
            for i, (x, y) in enumerate(perm):
                if flip[i]:
                    assignment.append((y, x))
                else:
                    assignment.append((x, y))
            result.append(tuple(assignment))
    return result

ALL_TRIPLE_ASSIGNMENTS = _all_opposite_triple_assignments()


def _spoint(name):
    """Get offset for named s-point."""
    return S_POINTS[name]


def _are_opposite(name1, name2):
    """Check if two s-point names are opposite."""
    for a, b in OPPOSITE_PAIRS:
        if (name1 == a and name2 == b) or (name1 == b and name2 == a):
            return True
    return False


def _are_non_opposite(name1, name2):
    """Check if two s-point names are non-opposite (and distinct)."""
    return name1 != name2 and not _are_opposite(name1, name2)


# =============================================================================
# Functions e, v, f1, f2, f3 from paper
# "e - 2 letter on diagram" "v - 3 letter on diagram" "js define the funcs"
# =============================================================================

def e_func(a_name, b_name, p):
    """e(a,b,p) = q where q in N*(p) and 6-adjacent to both a and b.
    "e - 2 letter on diagram" - e.g. e(N,T,p) = P_TN"""
    # q is the e-point: offset = offset_a + offset_b (since they're non-opposite s-points)
    return p + S_POINTS[a_name] + S_POINTS[b_name]


def v_func(a_name, b_name, c_name, p):
    """v(a,b,c,p) = q where q in N*(p) and 6-adjacent to e(a,b,p), e(b,c,p), e(c,a,p).
    "v - 3 letter on diagram" - e.g. v(N,T,E,p) = P_TNE"""
    return p + S_POINTS[a_name] + S_POINTS[b_name] + S_POINTS[c_name]


def f1_func(a_name, p):
    """f1(a,p) = q where q not in N(p) and 6-adjacent to a.
    "whats f1: point that is 6-adj to a and NOT in N(p)" - so it's 2 steps in a's direction"""
    # "whats f1: point that is 6-adj to d and 26-adj to p" - wait, not 26-adj actually
    # f1(a,p) = q|q not in N(p) and 6-adjacent to a
    # This means q = p + 2*offset_a
    return p + 2 * S_POINTS[a_name]


def f2_func(a_name, b_name, p):
    """f2(a,b,p) = q where q not in N(p) and 6-adjacent to f1(a,p) and e(a,b,p).
    From paper example: p=(5,5,5), a=W=(4,5,5), b=S=(5,4,5) -> f2=(3,4,5)"""
    return p + 2 * S_POINTS[a_name] + S_POINTS[b_name]


def f3_func(a_name, b_name, p):
    """f3(a,b,p) = q where q not in N(p) and 6-adjacent to f2(a,b,p) and f2(b,a,p).
    From paper example: p=(5,5,5), a=W=(4,5,5), b=S=(5,4,5) -> f3=(3,3,5)"""
    return p + 2 * S_POINTS[a_name] + 2 * S_POINTS[b_name]


# =============================================================================
# Middle plane and extended middle plane
# =============================================================================

def middle_plane(a_name, d_name, p):
    """M(a,d,p) = {x|C1} U {e(x,y,p)|C2}
    where C1: x in {b,e,c,f}, C2: x,y in {b,e,c,f} and non-opposite.
    The 4 s-points other than a,d plus 4 e-points formed by non-opposite pairs among them."""
    all_snames = set(S_POINTS.keys())
    others = list(all_snames - {a_name, d_name})
    pts = []
    # C1: the 4 s-points not a or d
    for name in others:
        pts.append(p + S_POINTS[name])
    # C2: pairs of non-opposite among others
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            if _are_non_opposite(others[i], others[j]):
                pts.append(e_func(others[i], others[j], p))
    return pts


def extended_middle_plane(a_name, d_name, p):
    """EM(a,d,p) = M(a,d,p) U {f1(x,p)|C3} U {f2(x,y,p)|C4} U {f3(x,y,p)|C5}
    C3: x in {b,e,c,f} and x in {PB, PS, PW}
    C4: x,y in {b,e,c,f}, non-opposite, x in {PB, PS, PW}
    C5: x,y in {b,e,c,f}, non-opposite, x,y in {PB, PS, PW}"""
    all_snames = set(S_POINTS.keys())
    others = list(all_snames - {a_name, d_name})
    lower = {'B', 'S', 'W'}  # {PB, PS, PW}

    pts = list(middle_plane(a_name, d_name, p))

    # C3
    for name in others:
        if name in lower:
            pts.append(f1_func(name, p))
    # C4
    for i in range(len(others)):
        for j in range(len(others)):
            if i != j and _are_non_opposite(others[i], others[j]):
                if others[i] in lower:
                    pts.append(f2_func(others[i], others[j], p))
    # C5
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            if _are_non_opposite(others[i], others[j]):
                if others[i] in lower and others[j] in lower:
                    pts.append(f3_func(others[i], others[j], p))
    return pts


def surface_func(a_name, p):
    """surface(a,p) = {a} U {e(a,x,p)|C1} U {v(a,x,y,p)|C2}
    where C1: x in other 4 s-points, C2: x,y non-opposite among other 4."""
    all_snames = set(S_POINTS.keys())
    # Find opposite of a
    opp_a = None
    for n1, n2 in OPPOSITE_PAIRS:
        if a_name == n1:
            opp_a = n2
        elif a_name == n2:
            opp_a = n1
    others = list(all_snames - {a_name, opp_a})

    pts = [p + S_POINTS[a_name]]
    # e(a, x, p) for x in others
    for name in others:
        pts.append(e_func(a_name, name, p))
    # v(a, x, y, p) for x, y non-opposite in others
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            if _are_non_opposite(others[i], others[j]):
                pts.append(v_func(a_name, others[i], others[j], p))
    return pts


# =============================================================================
# "two image:" - black/white checks on old vs current image
# "before: black >= thr" "cur: white < 0, black >= 0"
# "thr = -maxint + i (i >= 1) - delete" "black, marked = i"
# =============================================================================

def is_black_before(img, pt, thr):
    """A point is black before the iteration if its value >= thr.
    "before: black >= thr" """
    val = _get(img, pt)
    return val >= thr


def is_white_before(img, pt, thr):
    """A point is white before the iteration if its value < thr."""
    return not is_black_before(img, pt, thr)


def is_black_current(img, pt):
    """A point is currently black if its value >= 0.
    "cur: white < 0, black >= 0" """
    val = _get(img, pt)
    return val >= 0


def is_white_current(img, pt):
    """A point is currently white if its value < 0."""
    return not is_black_current(img, pt)


def is_marked(img, pt):
    """A point is marked if it has non-zero positive value.
    "black, marked = i" """
    val = _get(img, pt)
    return val > 0


def is_unmarked_black(img, pt):
    """A point with zero value is an unmarked black point."""
    return _get(img, pt) == 0


# =============================================================================
# Simple point detection (26,6)
# "simple point: (26,6)
#  1. p has at least one black 26-neighbor.
#  2. p has at least one white 6-neighbor.
#  3. The set of black 26-neighbors of p is 26-connected.
#  4. The set of white 6-neighbors of p is 6-connected in the set of white 18-neighbors of p."
# "check new/current to check for simple point"
# =============================================================================

def _flood_fill_local(points_set, adj_offsets, seed):
    """Flood fill in a local set of points using given adjacency."""
    visited = set()
    stack = [tuple(seed)]
    while stack:
        curr = stack.pop()
        if curr in visited:
            continue
        visited.add(curr)
        curr_arr = np.array(curr)
        for off in adj_offsets:
            nb = tuple(curr_arr + off)
            if nb in points_set and nb not in visited:
                stack.append(nb)
    return visited


def is_simple_point(img, p):
    """Check if p is a (26,6) simple point on the CURRENT image.
    "check new/current to check for simple point"
    "simple point: (26,6)
     1. p has at least one black 26-neighbor.
     2. p has at least one white 6-neighbor.
     3. The set of black 26-neighbors of p is 26-connected.
     4. The set of white 6-neighbors of p is 6-connected in the set of white 18-neighbors of p." """

    p = np.array(p)

    # Collect neighbors
    black_26 = []
    for off in ADJ26:
        nb = p + off
        if all(0 <= nb[i] < img.shape[i] for i in range(3)):
            if is_black_current(img, nb):
                black_26.append(tuple(nb))

    white_6 = []
    for off in ADJ6:
        nb = p + off
        if all(0 <= nb[i] < img.shape[i] for i in range(3)):
            if is_white_current(img, nb):
                white_6.append(tuple(nb))

    white_18 = []
    for off in ADJ18:
        nb = p + off
        if all(0 <= nb[i] < img.shape[i] for i in range(3)):
            if is_white_current(img, nb):
                white_18.append(tuple(nb))

    # "1. p has at least one black 26-neighbor."
    if len(black_26) == 0:
        return False

    # "2. p has at least one white 6-neighbor."
    if len(white_6) == 0:
        return False

    # "3. The set of black 26-neighbors of p is 26-connected."
    black_26_set = set(black_26)
    visited = _flood_fill_local(black_26_set, ADJ26, black_26[0])
    if len(visited) != len(black_26_set):
        return False

    # "4. The set of white 6-neighbors of p is 6-connected in the set of white 18-neighbors of p."
    white_18_set = set(white_18)
    # We need white 6-neighbors to be 6-connected within the white 18-neighbor set
    white_6_set = set(white_6)
    if len(white_6_set) == 0:
        return False
    visited = _flood_fill_local(white_18_set, ADJ6, white_6[0])
    # All white 6-neighbors must be reached
    if not white_6_set.issubset(visited):
        return False

    return True


# =============================================================================
# Open points (outer-layer)
# "check old for s/e/v-open"
# =============================================================================

def is_s_open(img, p, thr):
    """Definition 1: s-open point if at least one s-point of N(p) is white before the iteration.
    "check old for s/e/v-open" """
    p = np.array(p)
    for name, off in S_POINTS.items():
        nb = p + off
        if is_white_before(img, nb, thr):
            return True
    return False


def is_e_open(img, p, thr):
    """Definition 2: e-open point if p is NOT s-open and an e-point e(a,b,p) is white
    while f1(a,p), f1(b,p) are black before the iteration."""
    p = np.array(p)
    if is_s_open(img, p, thr):
        return False
    all_names = list(S_POINTS.keys())
    for i in range(len(all_names)):
        for j in range(i + 1, len(all_names)):
            a, b = all_names[i], all_names[j]
            if _are_non_opposite(a, b):
                e_pt = e_func(a, b, p)
                if is_white_before(img, e_pt, thr):
                    f1a = f1_func(a, p)
                    f1b = f1_func(b, p)
                    if is_black_before(img, f1a, thr) and is_black_before(img, f1b, thr):
                        return True
    return False


def is_v_open(img, p, thr):
    """Definition 3: v-open point if p is neither s-open nor e-open and a v-point v(a,b,c,p)
    is white while f1(a,p), f1(b,p), f1(c,p) are black before the iteration."""
    p = np.array(p)
    if is_s_open(img, p, thr) or is_e_open(img, p, thr):
        return False
    all_names = list(S_POINTS.keys())
    for i in range(len(all_names)):
        for j in range(i + 1, len(all_names)):
            for k in range(j + 1, len(all_names)):
                a, b, c = all_names[i], all_names[j], all_names[k]
                if _are_non_opposite(a, b) and _are_non_opposite(b, c) and _are_non_opposite(a, c):
                    v_pt = v_func(a, b, c, p)
                    if is_white_before(img, v_pt, thr):
                        f1a = f1_func(a, p)
                        f1b = f1_func(b, p)
                        f1c = f1_func(c, p)
                        if (is_black_before(img, f1a, thr) and
                                is_black_before(img, f1b, thr) and
                                is_black_before(img, f1c, thr)):
                            return True
    return False


# =============================================================================
# Shape point detection
# "shape point: black, cond 1 OR cond 2"
# "check old for ... if shape point"
# =============================================================================

def _check_condition1(img, p, thr):
    """Condition 1: "cond 1: some plane with 6-closed path;
    6-closed in the middle, Fig 4 - you can just check every combo;
    surface right on top and bottom have at least 1 black"

    There exist two opposite s-points a,d such that EM(a,d,p) contains a 6-closed path
    of white points encircling p and each of surface(a,p) and surface(d,p) contains
    at least one black point before the iteration."""
    p = np.array(p)

    for a_name, d_name in OPPOSITE_PAIRS:
        # Also check the reverse (d,a)
        for an, dn in [(a_name, d_name), (d_name, a_name)]:
            # Check surfaces have at least one black point before iteration
            # "surface right on top and bottom have at least 1 black"
            surf_a = surface_func(an, p)
            surf_d = surface_func(dn, p)

            has_black_a = any(is_black_before(img, pt, thr) for pt in surf_a)
            has_black_d = any(is_black_before(img, pt, thr) for pt in surf_d)

            if not (has_black_a and has_black_d):
                continue

            # Check if EM(a,d,p) contains a 6-closed path of white points encircling p
            # "6-closed in the middle, Fig 4 - you can just check every combo"
            # A 6-closed path encircles p if it contains two points of EM(a,d,p) ∩ {PT, PN, PE}
            em_pts = extended_middle_plane(an, dn, p)

            # Build set of white points in EM
            white_em = set()
            for pt in em_pts:
                if is_white_before(img, pt, thr):
                    white_em.add(tuple(pt))

            if len(white_em) < 2:
                continue

            # Check for 6-closed path of white points encircling p
            # "you can just check every combo"
            # A path encircles p if it contains points from two of {PT, PN, PE} direction sets
            # We check if the white points in EM form a connected component (via 6-adj)
            # that contains points from at least 2 of the "upper" directions
            # The encircling condition: contains two points of EM ∩ {pT, pN, pE}
            target_pts = set()
            for tname in ['T', 'N', 'E']:
                tp = tuple(p + S_POINTS[tname])
                if tp in white_em:
                    target_pts.add(tp)

            if len(target_pts) < 2:
                # Also check if any connected component of white_em has a cycle
                # that passes through 2 target directions
                # Try all pairs of target-adjacent white points
                continue

            # Check if at least 2 target points are in the same 6-connected component
            # and that component has a cycle (closed path)
            target_list = list(target_pts)
            # Check if any two target points are 6-connected in white_em
            for ti in range(len(target_list)):
                visited = _flood_fill_local(white_em, ADJ6, target_list[ti])
                count_targets_in_component = sum(1 for t in target_list if t in visited)
                if count_targets_in_component >= 2:
                    # Now check if this component has a cycle (needed for "closed path")
                    # A connected component of size >= the minimum for a cycle
                    # In a 6-adj grid, a cycle needs at least 4 points
                    component = visited
                    if len(component) >= 4:
                        # Check for cycle: if |edges| >= |vertices| then there's a cycle
                        edges = 0
                        for cp in component:
                            cp_arr = np.array(cp)
                            for off in ADJ6:
                                nb = tuple(cp_arr + off)
                                if nb in component and nb > cp:  # count each edge once
                                    edges += 1
                        if edges >= len(component):
                            return True
                    break  # only need to check one flood fill per target
    return False


def _check_condition2(img, p, thr):
    """Condition 2: "cond 2: pair of opposite s-points (a,d);
    a is white; d or f1(d, p) is white;
    sets contain at least one black point - figure out how to define sets;
    check all manually"

    There exists a pair (a,d) such that d in {PB, PS, PW}, a is white,
    d or f1(d,p) is white and each of the 8 sets contains at least one black point
    before the iteration."""
    p = np.array(p)
    lower = {'B', 'S', 'W'}

    for pair in OPPOSITE_PAIRS:
        for a_name, d_name in [pair, (pair[1], pair[0])]:
            if d_name not in lower:
                continue

            # "a is white"
            a_pt = p + S_POINTS[a_name]
            if not is_white_before(img, a_pt, thr):
                continue

            # "d or f1(d, p) is white"
            d_pt = p + S_POINTS[d_name]
            f1d = f1_func(d_name, p)
            if not (is_white_before(img, d_pt, thr) or is_white_before(img, f1d, thr)):
                continue

            # Find b,c,e,f: the other two pairs of opposite s-points
            others = [pr for pr in OPPOSITE_PAIRS if pr != pair]
            # others[0] = (b_name, e_name), others[1] = (c_name, f_name)
            # But we need to try all assignments
            for (b_name, e_name), (c_name, f_name) in [
                (others[0], others[1]),
                (others[1], others[0]),
                ((others[0][1], others[0][0]), others[1]),
                (others[0], (others[1][1], others[1][0])),
                ((others[0][1], others[0][0]), (others[1][1], others[1][0])),
                ((others[1][1], others[1][0]), others[0]),
                (others[1], (others[0][1], others[0][0])),
                ((others[1][1], others[1][0]), (others[0][1], others[0][0])),
            ]:
                # "sets contain at least one black point - figure out how to define sets"
                # "check all manually"
                # The 8 sets from the paper:
                # {e(a,b,p), b, e(b,d,p)}
                # {e(a,c,p), c, e(c,d,p)}
                # {e(a,e,p), e, e(d,e,p)}
                # {e(a,f,p), f, e(d,f,p)}
                # {v(a,b,c,p), e(b,c,p), v(b,c,d,p)}
                # {v(a,b,f,p), e(b,f,p), v(b,d,f,p)}
                # {v(a,c,e,p), e(c,e,p), v(c,d,e,p)}
                # {v(a,e,f,p), e(e,f,p), v(d,e,f,p)}

                sets = [
                    [e_func(a_name, b_name, p), p + S_POINTS[b_name], e_func(b_name, d_name, p)],
                    [e_func(a_name, c_name, p), p + S_POINTS[c_name], e_func(c_name, d_name, p)],
                    [e_func(a_name, e_name, p), p + S_POINTS[e_name], e_func(e_name, d_name, p)],
                    [e_func(a_name, f_name, p), p + S_POINTS[f_name], e_func(f_name, d_name, p)],
                    [v_func(a_name, b_name, c_name, p), e_func(b_name, c_name, p), v_func(b_name, c_name, d_name, p)],
                    [v_func(a_name, b_name, f_name, p), e_func(b_name, f_name, p), v_func(b_name, d_name, f_name, p)],
                    [v_func(a_name, c_name, e_name, p), e_func(c_name, e_name, p), v_func(c_name, d_name, e_name, p)],
                    [v_func(a_name, e_name, f_name, p), e_func(e_name, f_name, p), v_func(d_name, e_name, f_name, p)],
                ]

                all_have_black = True
                for s in sets:
                    has_black = False
                    for pt in s:
                        if is_black_before(img, pt, thr):
                            has_black = True
                            break
                    if not has_black:
                        all_have_black = False
                        break

                if all_have_black:
                    return True

    return False


def is_shape_point(img, p, thr):
    """Definition 4: A black point is a shape point if it satisfies Condition 1 or 2.
    "shape point: black, cond 1 OR cond 2"
    "check old for ... if shape point" """
    if not is_black_before(img, p, thr):
        return False
    return _check_condition1(img, p, thr) or _check_condition2(img, p, thr)


# =============================================================================
# Condition 3 - 2D topology preservation in middle planes
# "cond 3: every middle plane - all e-points black before current it"
# =============================================================================

def _middle_plane_has_tunnel(img, a_name, d_name, p):
    """The black points of M(a,d,p) generate a tunnel iff all s-points of N(p)
    belonging to M(a,d,p) are currently black."""
    p = np.array(p)
    all_snames = set(S_POINTS.keys())
    mp_spoints = list(all_snames - {a_name, d_name})
    return all(is_black_current(img, p + S_POINTS[name]) for name in mp_spoints)


def _check_condition3(img, p, thr):
    """Condition 3: "cond 3: every middle plane - all e-points black before current it"
    OR the current black points of M(a,d,p) generate single 26-component without tunnel.

    During an iteration p satisfies Condition 3 if each middle plane M(a,d,p) holds that:
    either all e-points in M(a,d,p) are black before the iteration
    or the current black points of M(a,d,p) generate single 26-component without any tunnel."""
    p = np.array(p)

    for a_name, d_name in OPPOSITE_PAIRS:
        # Get middle plane
        mp_pts = middle_plane(a_name, d_name, p)

        # Get e-points in middle plane (the ones formed by e_func)
        all_snames = set(S_POINTS.keys())
        others = list(all_snames - {a_name, d_name})
        e_pts_in_mp = []
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                if _are_non_opposite(others[i], others[j]):
                    e_pts_in_mp.append(e_func(others[i], others[j], p))

        # "all e-points black before current it"
        all_e_black = all(is_black_before(img, pt, thr) for pt in e_pts_in_mp)
        if all_e_black:
            continue

        # Otherwise check: current black points of M form single 26-component without tunnel
        black_in_mp = set()
        for pt in mp_pts:
            if is_black_current(img, pt):
                black_in_mp.add(tuple(pt))

        if len(black_in_mp) == 0:
            continue

        # Check single 26-component
        visited = _flood_fill_local(black_in_mp, ADJ26, next(iter(black_in_mp)))
        if len(visited) != len(black_in_mp):
            return False

        # Check no tunnel
        if _middle_plane_has_tunnel(img, a_name, d_name, p):
            return False

    return True


# =============================================================================
# thick function and Conditions 4, 5, 6
# =============================================================================

def thick(img, a_name, d_name, p, thr):
    """Definition 5: thick(a,d,p) = 1 if a, f1(d,p) are white while b,c,d are black
    (meaning p has at least three black non-opposite s-points, i.e. no two opposite
    s-points are both white before the iteration)."""
    p = np.array(p)
    a_pt = p + S_POINTS[a_name]
    f1d = f1_func(d_name, p)

    if not (is_white_before(img, a_pt, thr) and is_white_before(img, f1d, thr)):
        return False

    # "b,c,d are black" means no two opposite s-points are both white
    # Actually: the remaining 3 non-opposite-to-a s-points (b,c,d) must be black
    # d is the opposite of a, b and c are from the other two pairs
    # But the paper says "b,c,d are black" meaning d and two others from other pairs
    # This means: p has at least 3 black non-opposite s-points
    # "no two opposite s-points b,e such that both b and e are white"
    for pair in OPPOSITE_PAIRS:
        n1, n2 = pair
        if is_white_before(img, p + S_POINTS[n1], thr) and is_white_before(img, p + S_POINTS[n2], thr):
            return False

    return True


def _check_condition4(img, p, thr):
    """Condition 4: thick(a,d,p) where d in {PB,PS,PW} is true and current black points
    of each of M(b,e,p) and M(c,f,p) generate single 26-component without tunnel."""
    p = np.array(p)
    lower = {'B', 'S', 'W'}

    for pair in OPPOSITE_PAIRS:
        for a_name, d_name in [pair, (pair[1], pair[0])]:
            if d_name not in lower:
                continue
            if not thick(img, a_name, d_name, p, thr):
                continue

            others = [pr for pr in OPPOSITE_PAIRS if pr != pair]
            all_ok = True
            for other_pair in others:
                b_name, e_name = other_pair
                mp_pts = middle_plane(b_name, e_name, p)
                black_in_mp = set(tuple(pt) for pt in mp_pts if is_black_current(img, pt))

                if len(black_in_mp) == 0:
                    continue

                visited = _flood_fill_local(black_in_mp, ADJ26, next(iter(black_in_mp)))
                if len(visited) != len(black_in_mp):
                    all_ok = False
                    break
                if _middle_plane_has_tunnel(img, b_name, e_name, p):
                    all_ok = False
                    break

            if all_ok:
                return True
    return False


def _check_condition5(img, p, thr):
    """Condition 5: thick(a,d,p), thick(b,e,p) where d,e in {PB,PS,PW} are true
    and current black points of M(c,f,p) generate single 26-component without tunnel."""
    p = np.array(p)
    lower = {'B', 'S', 'W'}

    pairs = list(OPPOSITE_PAIRS)
    for i in range(len(pairs)):
        for j in range(len(pairs)):
            if i == j:
                continue
            pair_ad = pairs[i]
            pair_be = pairs[j]

            for a_name, d_name in [pair_ad, (pair_ad[1], pair_ad[0])]:
                if d_name not in lower:
                    continue
                for b_name, e_name in [pair_be, (pair_be[1], pair_be[0])]:
                    if e_name not in lower:
                        continue

                    if not thick(img, a_name, d_name, p, thr):
                        continue
                    if not thick(img, b_name, e_name, p, thr):
                        continue

                    # Find remaining pair
                    remaining = [pr for pr in pairs if pr != pair_ad and pr != pair_be]
                    if len(remaining) != 1:
                        continue
                    c_name, f_name = remaining[0]

                    mp_pts = middle_plane(c_name, f_name, p)
                    black_in_mp = set(tuple(pt) for pt in mp_pts if is_black_current(img, pt))

                    if len(black_in_mp) == 0:
                        return True  # vacuously true

                    visited = _flood_fill_local(black_in_mp, ADJ26, next(iter(black_in_mp)))
                    if len(visited) == len(black_in_mp):
                        if not _middle_plane_has_tunnel(img, c_name, f_name, p):
                            return True
    return False


def _check_condition6(img, p, thr):
    """Condition 6: thick(a,d,p), thick(b,e,p), thick(c,f,p) where d,e,f in {PB,PS,PW}."""
    p = np.array(p)
    lower = {'B', 'S', 'W'}

    pairs = list(OPPOSITE_PAIRS)
    for a_name, d_name in [pairs[0], (pairs[0][1], pairs[0][0])]:
        if d_name not in lower:
            continue
        for b_name, e_name in [pairs[1], (pairs[1][1], pairs[1][0])]:
            if e_name not in lower:
                continue
            for c_name, f_name in [pairs[2], (pairs[2][1], pairs[2][0])]:
                if f_name not in lower:
                    continue
                if (thick(img, a_name, d_name, p, thr) and
                        thick(img, b_name, e_name, p, thr) and
                        thick(img, c_name, f_name, p, thr)):
                    return True
    return False


def is_erodable(img, p, thr):
    """Definition 6: A black point is erodable if it is a simple point and satisfies
    any of Conditions 4, 5, or 6."""
    if not is_simple_point(img, p):
        return False
    return _check_condition4(img, p, thr) or _check_condition5(img, p, thr) or _check_condition6(img, p, thr)


# =============================================================================
# Sub-fields for parallel implementation
# =============================================================================

def get_subfield(l, shape):
    """Get points belonging to subfield O_l.
    O_l = {(2i+f, 2j+g, 2k+h) | f,g,h in {0,1} and 4f+2g+h = l}"""
    f = (l >> 2) & 1
    g = (l >> 1) & 1
    h = l & 1
    xs = np.arange(f, shape[0], 2)
    ys = np.arange(g, shape[1], 2)
    zs = np.arange(h, shape[2], 2)
    return np.array([(x, y, z) for x in xs for y in ys for z in zs], dtype=np.int32)


# =============================================================================
# Main thinning algorithm
# =============================================================================

def thin_3d(binary_image, verbose=True):
    """Main 3D parallel thinning algorithm.

    "two image: check old for s/e/v-open, and if shape point;
     check new/current to check for simple point"

    "thr = -maxint + i (i >= 1) - delete"
    "black, marked = i"
    "before: black >= thr"
    "cur: white < 0, black >= 0"

    Parameters
    ----------
    binary_image : ndarray of bool or int
        3D binary image where True/1 = object (black), False/0 = background (white).

    Returns
    -------
    result : ndarray of bool
        Thinned 3D binary image.
    """
    # Initialize image values
    # "every white point is assigned -maxint and each black point is assigned 0"
    img = np.full(binary_image.shape, -MAXINT, dtype=np.int64)
    img[binary_image > 0] = 0

    if verbose:
        print(f"Image shape: {img.shape}")
        print(f"Black points: {np.sum(img >= 0)}")

    # === PRIMARY THINNING ===
    # "primary-thinning is an iterative procedure and iterations are continued
    #  as long as any point is deleted in the last iteration"
    iteration = 1
    while True:
        # "thr = -maxint + i (i >= 1)"
        thr = -MAXINT + iteration
        deleted_count = 0

        if verbose:
            print(f"\n--- Primary-thinning iteration {iteration} ---")
            print(f"  thr = {thr}")

        # Pre-label all open points and shape points on the image BEFORE this iteration
        # "check old for s/e/v-open, and if shape point"
        black_points = list(zip(*np.where(img >= 0)))
        # Only consider unmarked black points that are currently black
        unmarked_black = [p for p in black_points if img[p] == 0]

        if verbose:
            print(f"  Unmarked black points: {len(unmarked_black)}")

        # Pre-compute open and shape labels
        s_open_set = set()
        e_open_set = set()
        v_open_set = set()
        shape_set = set()

        for p in unmarked_black:
            if is_s_open(img, p, thr):
                s_open_set.add(p)
            elif is_e_open(img, p, thr):
                e_open_set.add(p)
            elif is_v_open(img, p, thr):
                v_open_set.add(p)

        for p in s_open_set:
            if is_shape_point(img, p, thr):
                shape_set.add(p)

        if verbose:
            print(f"  s-open: {len(s_open_set)}, e-open: {len(e_open_set)}, "
                  f"v-open: {len(v_open_set)}, shape: {len(shape_set)}")

        # --- SCAN 1: s-open points ---
        # "During the first scan the set of unmarked s-open points is used for erosion.
        #  An unmarked s-open point is marked if it is a shape point.
        #  When it is not a shape point, it is deleted if it is a simple point,
        #  otherwise it is left unmarked."
        for l in range(8):
            subfield = get_subfield(l, img.shape)
            for pt_arr in subfield:
                p = tuple(pt_arr)
                if p not in s_open_set:
                    continue
                if img[p] != 0:  # must be unmarked
                    continue
                if p in shape_set:
                    # "marked = i" - mark with iteration number
                    img[p] = iteration
                else:
                    # "check new/current to check for simple point"
                    if is_simple_point(img, p):
                        # "delete" - assign thr value
                        img[p] = thr
                        deleted_count += 1

        # --- SCAN 2: e-open points ---
        # "During the second scan the set of unmarked e-open points is used for erosion.
        #  An e-open point can never be a shape point and hence is never marked.
        #  An unmarked e-open point is deleted if it is a simple point and satisfies Condition 3."
        for l in range(8):
            subfield = get_subfield(l, img.shape)
            for pt_arr in subfield:
                p = tuple(pt_arr)
                if p not in e_open_set:
                    continue
                if img[p] != 0:
                    continue
                # "check new/current to check for simple point"
                if is_simple_point(img, p) and _check_condition3(img, p, thr):
                    img[p] = thr
                    deleted_count += 1

        # --- SCAN 3: v-open points ---
        # "During the third scan the set of unmarked v-open points is used for erosion.
        #  An unmarked v-open point is deleted if it is a simple point."
        for l in range(8):
            subfield = get_subfield(l, img.shape)
            for pt_arr in subfield:
                p = tuple(pt_arr)
                if p not in v_open_set:
                    continue
                if img[p] != 0:
                    continue
                if is_simple_point(img, p):
                    img[p] = thr
                    deleted_count += 1

        if verbose:
            print(f"  Deleted in this iteration: {deleted_count}")

        if deleted_count == 0:
            break
        iteration += 1

    # === FINAL THINNING ===
    # "final-thinning is a single iteration procedure and the iteration consists of single scan.
    #  During this scan a black point p (irrespective of whether p is marked or unmarked)
    #  is deleted if it is an erodable point."
    if verbose:
        print("\n--- Final thinning ---")

    thr_final = -MAXINT + iteration + 1
    final_deleted = 0
    black_points = list(zip(*np.where(img >= 0)))

    for l in range(8):
        subfield = get_subfield(l, img.shape)
        for pt_arr in subfield:
            p = tuple(pt_arr)
            if img[p] < 0:
                continue
            if is_erodable(img, p, thr_final):
                img[p] = thr_final
                final_deleted += 1

    if verbose:
        print(f"  Final thinning deleted: {final_deleted}")
        print(f"  Remaining black points: {np.sum(img >= 0)}")

    return img >= 0


# =============================================================================
# Test
# =============================================================================

import time
from solid import *
from scipy.ndimage import label

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

start = time.time()
voxeld = np.load('s01_voxel.npy')
sub = voxeld[80:180, 750:850, 850:950]
sub = sub[::2, ::2, ::2]
sub = largest_connected_component(sub)
print('input extracted')
print(f'true cnt before: {sub.sum()}')
ske = thin_3d(sub, verbose=True)
print(f'true cnt: {ske.sum()}')
scout('../scad/new8.1', render_3d_mat(ske))
print(f"elapsed: {time.time() - start:.2f} seconds")

# if __name__ == '__main__':
#     # Simple test: thin a small cube
#     print("Testing 3D parallel thinning algorithm...")
#
#     # Create a small solid cube
#     size = 11
#     img = np.zeros((size, size, size), dtype=bool)
#     img[2:9, 2:9, 2:9] = True
#
#     print(f"\nOriginal object: {size}x{size}x{size} cube, black points: {np.sum(img)}")
#
#     result = thin_3d(img, verbose=True)
#
#     print(f"\nSkeleton points: {np.sum(result)}")
#     skeleton_coords = np.array(np.where(result)).T
#     if len(skeleton_coords) > 0:
#         print(f"Skeleton coordinate range:")
#         print(f"  x: [{skeleton_coords[:, 0].min()}, {skeleton_coords[:, 0].max()}]")
#         print(f"  y: [{skeleton_coords[:, 1].min()}, {skeleton_coords[:, 1].max()}]")
#         print(f"  z: [{skeleton_coords[:, 2].min()}, {skeleton_coords[:, 2].max()}]")
#
#     # Test with an L-shape
#     print("\n\n=== Testing L-shape ===")
#     img2 = np.zeros((15, 15, 7), dtype=bool)
#     img2[2:13, 2:6, 1:6] = True   # vertical bar
#     img2[2:6, 2:13, 1:6] = True   # horizontal bar
#
#     print(f"L-shape black points: {np.sum(img2)}")
#     result2 = thin_3d(img2, verbose=True)
#     print(f"Skeleton points: {np.sum(result2)}")
