import pytest

pytestmark = [pytest.mark.remeshing, pytest.mark.processing, pytest.mark.vtk]
pytest.importorskip("vtk", reason="VTK is required for this test module")

"""Nested replacement: replacing the nested triangles on a short edge with one element.

The fixtures are the patch from the owner's description, placed in larger meshes:

      w4
       |
      w3
       |
      w2
       |
      w1
     /   \\
    u-----v

(u,v,w1) is an element; (u,v,w2) ... (u,v,wk) are triangles of edges, not elements.
"""
import itertools
import math

import numpy as np
import torch

import torch_fea_mesh.processing.isotropic_triangle_surface_remesher as r
from test.helper.curved_patch_example import repairable_curved_patch
from test.remeshing.test_isotropic_triangle_surface_remesher import attempt, context

U, V = 0, 1
TARGET = 2.0  # (u,v) of length 0.4 is short, and the host edges stay clear of the length limits.
LIMIT = 0.1 * TARGET


def patch(depth=3, uv=0.4, height=1.1, sag=0.0):
    """u, v, then a chain of depth inner vertices and one outer vertex."""
    points = [[-uv / 2, 0, 0], [uv / 2, 0, 0]]
    for i in range(depth + 1):
        points.append([0, height * (i + 1) / (depth + 1), sag if i < depth else 0.0])
    faces = [(U, V, 2)]
    for i in range(depth):
        a, b = 2 + i, 3 + i
        faces += [(U, a, b), (V, b, a)]
    return np.array(points, float), faces, len(points) - 1


def ring_around(points, faces, loop, n=16, radii=(2.5, 3.5, 5.0)):
    """Add flat rings around a counter-clockwise loop of vertices; the outer ring stays free."""
    points = [list(p) for p in points]
    faces = list(faces)
    center = np.mean([points[i][:2] for i in loop], axis=0)
    angle = [float(np.arctan2(*(np.array(points[i][:2]) - center)[::-1]) % (2 * math.pi)) for i in loop]
    for k in range(1, len(angle)):
        while angle[k] < angle[k - 1]:
            angle[k] += 2 * math.pi
    angle.append(angle[0] + 2 * math.pi)
    step = 2 * math.pi / n
    first = int(math.ceil(angle[0] / step))
    base = len(points)
    for radius in radii:
        for j in range(n):
            a = (first + j) * step
            points.append([center[0] + radius * math.cos(a), center[1] + radius * math.sin(a), 0.0])

    def ring(k, j):
        return base + k * n + (j % n)

    i = j = 0
    while i < len(loop) or j < n:
        if i < len(loop) and (j == n or angle[i + 1] <= (first + j + 1) * step):
            faces.append((loop[i], ring(0, j), loop[(i + 1) % len(loop)]))
            i += 1
        else:
            faces.append((loop[i % len(loop)], ring(0, j), ring(0, j + 1)))
            j += 1
    for k in range(len(radii) - 1):
        for j in range(n):
            faces += [
                (ring(k, j), ring(k + 1, j), ring(k + 1, j + 1)),
                (ring(k, j), ring(k + 1, j + 1), ring(k, j + 1)),
            ]
    return np.array(points, float), np.array(faces)


def plate(depth=3, n=16, **kw):
    points, faces, outer = patch(depth, **kw)
    v, f = ring_around(points, faces, [U, V, outer], n=n)
    return v, f, dict(uv=(U, V), inner=list(range(2, outer)), outer=outer, patch_faces=2 * depth + 1)


def closed_sphere(depth=3, both_poles=False):
    """Map the plate onto a hemisphere and close it with a mirror copy."""
    v, f, meta = plate(depth)
    v = v.copy()
    v[:, :2] -= v[[U, V, meta['outer']], :2].mean(axis=0)
    rout = np.linalg.norm(v[:, :2], axis=1).max()
    radius = rout * 2 / math.pi
    rho = np.linalg.norm(v[:, :2], axis=1)
    phi = rho / rout * math.pi / 2
    scale = np.where(rho > 0, np.sin(phi) / np.maximum(rho, 1e-300), 0.0)
    top = np.c_[radius * scale * v[:, 0], radius * scale * v[:, 1], radius * np.cos(phi)]
    keys, points = {}, []

    def vertex(p):
        key = tuple(float(x) + 0.0 for x in np.round(p, 9))
        if key not in keys:
            keys[key] = len(points)
            points.append(list(p))
        return keys[key]

    a = [vertex(p) for p in top]
    b = [vertex(p * np.array([1, 1, -1])) for p in top]
    faces = [tuple(a[i] for i in t) for t in f.tolist()]
    if both_poles:
        faces += [(b[t[0]], b[t[2]], b[t[1]]) for t in f.tolist()]
    else:
        faces += [(b[t[0]], b[t[2]], b[t[1]]) for t in f.tolist()[meta['patch_faces'] :]]
        faces.append((b[U], b[meta['outer']], b[V]))
    used = sorted({i for t in faces for i in t})
    remap = {old: new for new, old in enumerate(used)}
    meta = dict(meta, uv=tuple(sorted(remap[a[i]] for i in (U, V))), inner=[remap[a[i]] for i in meta['inner']])
    if both_poles:
        meta['south_uv'] = tuple(sorted(remap[b[i]] for i in (U, V)))
    return np.array(points, float)[used], np.array([[remap[i] for i in t] for t in faces]), meta


def icosphere(level=1):
    t = (1 + 5**0.5) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(level):
        cache, nf = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                p = v[a] + v[b]
                v.append(p / np.linalg.norm(p))
                cache[k] = len(v) - 1
            return cache[k]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    return np.array(v), np.array(f)


def mesh_of(v, f):
    return r.TriangleMesh(torch.tensor(np.asarray(v), dtype=torch.float64), torch.tensor(np.asarray(f)))


def make(v, f, target=TARGET, **kw):
    return context(mesh_of(v, f), length=target, **kw)


def replace_with_reason(ctx, edge=(U, V), iteration=9):
    m, t, f, b, ref, p, tol = ctx
    return r._replace_nested_triangles(m, t, edge, f, b, p, tol, iteration_index=iteration, repair_failure='test')


def replace(ctx, edge=(U, V), iteration=9):
    return replace_with_reason(ctx, edge, iteration)[0]


def late(ctx, edge=(U, V), iteration=9):
    return attempt(ctx, tuple(sorted(edge)), iteration, tuple(sorted(edge)) in ctx[1]['edge_to_faces'] and tuple(sorted(edge)) in ctx[1]['boundary_edges'])


def faces_of(ctx):
    return np.asarray(ctx[1]['faces'])


def stats(ctx):
    t = ctx[1]
    used = {int(v) for face in t['faces'] for v in face}
    return dict(
        faces=len(t['faces']),
        euler=len(used) - len(t['edge_to_faces']) + len(t['faces']),
        free=sum(len(x) == 1 for x in t['edge_to_faces'].values()),
        high=sum(len(x) > 2 for x in t['edge_to_faces'].values()),
        components=r._component_count(ctx[0]),
    )


def volume(ctx):
    p = ctx[1]['vertex_positions'][faces_of(ctx)]
    return abs(float(np.einsum('ij,ij->i', p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6))


def distance_to_mesh(ctx, point):
    return r._closest_point(point, r._build_cell_locator(ctx[0]))[1]


def segment_hits_triangle(p, q, tri):
    d, e1, e2 = q - p, tri[1] - tri[0], tri[2] - tri[0]
    h = np.cross(d, e2)
    det = e1 @ h
    if abs(det) < 1e-12:
        return False
    s = p - tri[0]
    a = (s @ h) / det
    qv = np.cross(s, e1)
    b = (d @ qv) / det
    t = (e2 @ qv) / det
    return a >= -1e-10 and b >= -1e-10 and a + b <= 1 + 1e-10 and 1e-10 < t < 1 - 1e-10


def crossings(ctx, sheet):
    x, faces = ctx[1]['vertex_positions'], faces_of(ctx)
    total = 0
    for face in faces.tolist():
        if set(face) & set(sheet):
            continue
        for a, b in ((0, 1), (1, 2), (2, 0)):
            total += segment_hits_triangle(x[sheet[a]], x[sheet[b]], x[face])
    return total


# What gets replaced


@pytest.mark.parametrize('depth', [2, 3, 6, 12, 30, 127])
def test_flat_chain_of_any_supported_depth(depth):
    v, f, m = plate(depth)
    ctx = make(v, f)
    assert late(ctx) == (True, 'ordinary_collapsed')
    s = stats(ctx)
    assert s['faces'] == len(f) - (2 * depth + 1) - 1  # the nest plus the other element on (u,v)
    assert (s['euler'], s['high'], s['components']) == (1, 0, 1)


def test_replacement_leaves_the_outer_element_and_nothing_else_changes():
    v, f, m = plate()
    ctx = make(v, f)
    before = {tuple(sorted(t)) for t in f.tolist()[m['patch_faces'] :]}
    assert replace(ctx) == 1
    after = {tuple(sorted(t)) for t in faces_of(ctx).tolist()}
    assert after == before | {(U, V, m['outer'])}
    assert np.array_equal(ctx[1]['vertex_positions'][:len(v)], v)
    assert all(not ctx[1]['vertex_to_faces'][w] for w in m['inner'])
    r._check_adjacency_consistency(ctx[0], ctx[1])


def test_new_element_keeps_the_orientation_of_the_element_it_replaces():
    v, f, m = plate()
    ctx = make(v, f)
    assert replace(ctx) == 1
    key = ctx[1]['face_keys'][tuple(sorted((U, V, m['outer'])))]
    assert r._edge_direction(ctx[1]['faces'][key], (U, V)) == r._edge_direction(f[0], (U, V))


def test_flat_host_keeps_its_area_exactly():
    v, f, m = plate()
    ctx = make(v, f)
    assert late(ctx)[0]
    x, g = ctx[1]['vertex_positions'], faces_of(ctx)
    area = lambda p, q: float(0.5 * np.linalg.norm(np.cross(p[q[:, 1]] - p[q[:, 0]], p[q[:, 2]] - p[q[:, 0]]), axis=1).sum())
    assert area(x, g) == pytest.approx(area(v, f), abs=1e-9)


@pytest.mark.parametrize(
    'name,build,expected',
    [
        ('split edge', 'broken', 82),
        ('split fan face', 'refined', 2562),
        ('inner triangle', 'inner_triangle', 82),
    ],
)
def test_nests_that_are_not_exact_chains_are_resolved(name, build, expected):
    if build == 'broken':
        points, faces, outer = patch()
        points = np.vstack([points, (points[U] + points[3]) / 2])
        p = len(points) - 1
        faces = [t for t in faces if set(t) not in ({U, 2, 3}, {U, 3, 4})] + [(U, 2, p), (p, 2, 3), (U, p, 4), (p, 3, 4)]
        v, f = ring_around(points, faces, [U, V, outer])
        ctx = make(v, f)
    elif build == 'refined':
        points, faces, outer = patch()
        a, b, c = faces[1]
        p = len(points)
        points = np.vstack([points, points[[a, b, c]].mean(axis=0)])
        faces = faces[:1] + faces[2:] + [(a, b, p), (b, c, p), (c, a, p)]
        v, f = ring_around(points, faces, [U, V, outer], n=512)
        ctx = make(v, f)
    else:
        u, v_, w, a, b, c = 0, 1, 2, 3, 4, 5
        points = [[-1, 0, 0], [1, 0, 0], [0, 1.8, 0], [0, 0.3, 0], [0.3, 0.8, 0], [-0.3, 0.8, 0]]
        faces = [(u, v_, a), (v_, b, a), (v_, w, b), (w, c, b), (w, u, c), (u, a, c), (a, b, c)]
        v, f = ring_around(points, faces, [u, v_, w])
        ctx = make(v, f, target=4.0)
    assert late(ctx) == (True, 'ordinary_collapsed')
    assert stats(ctx)['faces'] == expected and stats(ctx)['euler'] == 1


def test_nests_on_both_sides_of_uv():
    points, faces, top = patch(2)
    base = len(points)
    extra = [[0, -p[1], 0] for p in points[2:]]
    pts = np.vstack([points, extra])
    w = lambda i: base + i
    faces = list(faces) + [(V, U, w(0)), (U, w(1), w(0)), (V, w(0), w(1)), (U, w(2), w(1)), (V, w(1), w(2))]
    v, f = ring_around(pts, faces, [U, w(2), V, top])
    ctx = make(v, f)
    assert replace(ctx) == 2
    assert stats(ctx)['faces'] == len(f) - 8
    assert stats(ctx)['high'] == 0


def test_uv_on_free_boundary():
    points, faces, outer = patch()
    pts = [list(p) for p in points]
    n, radii = 8, (2.5, 3.5, 5.0)
    base = len(pts)
    for radius in radii:
        pts += [[radius * math.cos(j * math.pi / n), radius * math.sin(j * math.pi / n), 0.0] for j in range(n + 1)]
    ring = lambda k, j: base + k * (n + 1) + j
    out_faces = list(faces)
    loop = [V, outer, U]
    angle = [0.0, math.pi / 2, math.pi]
    i = j = 0
    while i < 2 or j < n:
        if i < 2 and (j == n or angle[i + 1] <= (j + 1) * math.pi / n):
            out_faces.append((loop[i], ring(0, j), loop[i + 1]))
            i += 1
        else:
            out_faces.append((loop[i], ring(0, j), ring(0, j + 1)))
            j += 1
    for k in range(len(radii) - 1):
        for j in range(n):
            out_faces += [(ring(k, j), ring(k + 1, j), ring(k + 1, j + 1)), (ring(k, j), ring(k + 1, j + 1), ring(k, j + 1))]
    ctx = make(np.array(pts), np.array(out_faces))
    assert late(ctx)[0]
    s = stats(ctx)
    assert s['faces'] == len(out_faces) - 7 and s['high'] == 0 and s['components'] == 1


# Surfaces that must never be erased or flattened


def test_closed_surface_keeps_everything_but_the_nest():
    v, f, m = closed_sphere()
    ctx = make(v, f)
    assert late(ctx, m['uv'])[0]
    s = stats(ctx)
    assert s['faces'] == len(f) - 8 and s['euler'] == 2 and s['free'] == 0 and s['components'] == 1


def test_two_nests_on_a_closed_surface_are_independent():
    v, f, m = closed_sphere(both_poles=True)
    ctx = make(v, f)
    assert late(ctx, m['uv'])[0] and late(ctx, m['south_uv'])[0]
    s = stats(ctx)
    assert s['faces'] == len(f) - 16 and s['euler'] == 2 and s['free'] == 0


@pytest.mark.parametrize('level', [1, 2, 3])
def test_ordinary_closed_mesh_has_nothing_to_replace(level):
    v, f = icosphere(level)
    ctx = make(v * 4, f)
    for edge in sorted(ctx[1]['edge_to_faces']):
        assert replace(ctx, edge) == 0
    assert stats(ctx)['faces'] == len(f)


@pytest.mark.parametrize('radius,target', [(1.9, 2.0), (0.95, 1.0), (0.3, 1.0), (5.0, 2.0)])
def test_full_run_on_a_sphere_never_erases_it(radius, target):
    v, f = icosphere(3)
    out = r.isotropic_triangle_surface_remesh(mesh_of(v * radius, f), target, True, max_iter=10)
    assert len(out.element) >= 4


@pytest.mark.parametrize('reverse', [False, True])
def test_small_closed_solid_is_not_flattened(reverse):
    v = np.array([[-.6, 0, 0], [.6, 0, 0], [0, 1.2, 0], [0, .4, .15], [0, .4, 0.]])
    f = np.array([(0, 1, 3), (1, 2, 3), (2, 0, 3), (1, 0, 4), (2, 1, 4), (0, 2, 4)])
    ctx = make(v, f[::-1].copy() if reverse else f)
    before = volume(ctx)
    assert before > 0
    late(ctx)
    assert stats(ctx)['faces'] == 6
    assert volume(ctx) == pytest.approx(before)


def test_closed_solid_with_a_refined_flat_base_is_not_flattened():
    v = np.array([[-.6, 0, 0], [.6, 0, 0], [0, 1.2, 0], [0, .4, .15], [0, .4, 0.]])
    f = [(0, 1, 3), (1, 2, 3), (2, 0, 3), (1, 0, 4), (2, 1, 4), (0, 2, 4)]
    a, b, c = f[4]
    v2 = np.vstack([v, v[[a, b, c]].mean(axis=0)])
    f2 = f[:4] + f[5:] + [(a, b, 5), (b, c, 5), (c, a, 5)]
    ctx = make(v2, f2)
    before = volume(ctx)
    assert replace(ctx) == 0
    assert volume(ctx) == pytest.approx(before)


def test_nest_on_an_edge_with_more_than_two_elements_is_refused():
    # Such input is out of policy, but until an input check rejects it, it must not be changed here.
    v, f, m = plate()
    v2 = np.vstack([v, [[0.0, -0.3, 1.0]]])
    f2 = np.vstack([f, [[U, V, len(v)]]])
    ctx = make(v2, f2)
    assert len(ctx[1]['edge_to_faces'][(U, V)]) == 3
    assert replace(ctx) == 0
    assert stats(ctx)['faces'] == len(f2)


def test_tetrahedron_is_left_alone():
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    f = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    ctx = make(v, f, target=8.0)
    for edge in sorted(ctx[1]['edge_to_faces']):
        assert replace(ctx, edge) == 0
    assert stats(ctx)['faces'] == 4


def test_patch_with_nothing_around_it_keeps_the_outer_element():
    points, faces, outer = patch()
    ctx = make(points, faces)
    assert late(ctx) == (True, 'nested_triangles_replaced')
    assert {tuple(sorted(t)) for t in faces_of(ctx).tolist()} == {(U, V, outer)}


# The shape limit


@pytest.mark.parametrize('sag,expected', [(0.0, 1), (0.1, 1), (0.19, 1), (0.21, 0), (0.3, 0), (1.0, 0)])
def test_limit_on_inner_vertices(sag, expected):
    v, f, m = plate(sag=sag)
    ctx = make(v, f)
    assert replace(ctx) == expected
    assert stats(ctx)['faces'] == (len(f) - m['patch_faces'] + 1 if expected else len(f))


def test_limit_is_one_tenth_of_the_target():
    v, f, m = plate(sag=0.19)
    assert replace(make(v, f, target=2.0)) == 1  # limit 0.2
    assert replace(make(v, f, target=1.8)) == 0  # limit 0.18


def test_limit_checks_the_whole_nest_not_each_intermediate_triangle():
    points, faces, outer = patch()
    points[2:5, 2] = [0.19, -0.19, 0.19]
    v, f = ring_around(points, faces, [U, V, outer])
    assert r._distance_to_triangle(v[2], v[U], v[V], v[3]) > LIMIT
    assert replace(make(v, f)) == 1


@pytest.mark.parametrize('seed', range(20))
def test_replacement_stays_within_the_limit_in_both_directions(seed):
    rng = np.random.default_rng(seed)
    v, f, m = patch(5, uv=1.2, height=1.2)
    corners = v[[U, V, m]]
    for j in range(2, m):
        v[j] = rng.dirichlet([1.0, 1.0, 1.0]) @ corners
        v[j, 2] = rng.uniform(-0.199, 0.199)
    ctx = make(v, f)
    assert replace(ctx) == 1
    samples = rng.dirichlet([1.0, 1.0, 1.0], size=300)
    old = np.einsum('ij,kjl->kil', samples, v[np.array(f)]).reshape(-1, 3)
    new = np.einsum('ij,kjl->kil', samples, ctx[1]['vertex_positions'][faces_of(ctx)]).reshape(-1, 3)
    old_locator = r._build_cell_locator(r._new_mesh(v, np.array(f)))
    new_locator = r._build_cell_locator(ctx[0])
    assert max(r._closest_point(p, new_locator)[1] for p in old) <= LIMIT + 1e-10
    assert max(r._closest_point(p, old_locator)[1] for p in new) <= LIMIT + 1e-10


def test_smaller_nest_cannot_hide_a_vertex_over_the_limit():
    v = np.array([[-.6, 0, 0], [.6, 0, 0], [0, 1.2, 0], [0, .5, .15], [0, .25, .25], [.18, .6, .1]])
    f = [(0, 1, 4), (1, 3, 4), (3, 0, 4), (1, 2, 5), (2, 3, 5), (3, 1, 5), (2, 0, 3)]
    v2, f2 = ring_around(v, f, [0, 1, 2], n=128)
    ctx = make(v2, f2)
    assert replace(ctx, (0, 1)) == 0 and replace(ctx, (1, 2)) == 0
    assert stats(ctx)['faces'] == len(f2)
    assert distance_to_mesh(ctx, v[4]) == pytest.approx(0.0, abs=1e-12)


def test_face_budget_fails_closed():
    v, f, m = plate(depth=128, sag=0.15)
    v = v.copy()
    high = 2 + int(128 * 0.8)
    v[high, 2] = 0.28
    ctx = make(v, f)
    assert replace(ctx) == 0
    assert stats(ctx)['faces'] == len(f)
    assert distance_to_mesh(ctx, v[high]) == pytest.approx(0.0, abs=1e-12)


def test_flat_chain_over_the_face_budget_is_left_alone():
    v, f, m = plate(depth=128)
    ctx = make(v, f)
    assert replace(ctx) == 0
    assert stats(ctx)['faces'] == len(f)


@pytest.mark.xfail(strict=True, reason='Known limit: the shape limit applies to each edit, not across edits.')
def test_known_limit_vertex_hidden_through_another_edge():
    v = [[-.6, 0, 0], [.6, 0, 0], [0, 1.2, 0], [0, .45, .15], [.22, .55, .15]]
    f = [(0, 1, 3), (0, 3, 2), (2, 3, 4), (2, 4, 1)]
    chain = []
    for i in range(10):
        t = (i + 1) / 11
        p = (1 - t) * (np.array(v[3]) + v[4]) / 2 + t * np.array(v[1])
        p[2] += 0.13 if i == 0 else 0.05
        chain.append(len(v))
        v.append(p.tolist())
    chain.append(1)
    f.append((4, 3, chain[0]))
    for a, b in zip(chain, chain[1:]):
        f.extend([(4, a, b), (3, b, a)])
    v2, f2 = ring_around(v, f, [0, 1, 2], n=128)
    ctx = make(v2, f2)
    for edge in ((0, 1), (3, 4), (0, 1)):
        late(ctx, edge)
    assert distance_to_mesh(ctx, np.array(v[5])) <= LIMIT


# Nearby surfaces


def nest_under_sheet(sheet):
    v, f, m = patch(3, uv=0.6, sag=0.15)
    v, f = ring_around(v, f, [U, V, m], radii=(2.0, 3.0, 4.0))
    start = len(v)
    return np.vstack([v, sheet]), np.vstack([f, [[start, start + 1, start + 2]]]), [start, start + 1, start + 2]


def test_replacement_does_not_pass_through_a_nearby_sheet():
    v, f, sheet = nest_under_sheet([[-.04, .4, -.04], [.04, .4, .04], [.04, .45, -.04]])
    ctx = make(v, f)
    assert crossings(ctx, sheet) == 0
    assert replace(ctx) == 0
    assert stats(ctx)['faces'] == len(f)
    late(ctx)
    assert crossings(ctx, sheet) == 0


def test_sheet_clear_of_the_new_element_does_not_block_it():
    v, f, sheet = nest_under_sheet([[-.04, .4, .5], [.04, .4, .5], [.04, .45, .6]])
    assert replace(make(v, f)) == 1


# Reasons for a refused replacement


def small_closed_solid():
    v = np.array([[-.6, 0, 0], [.6, 0, 0], [0, 1.2, 0], [0, .4, .15], [0, .4, 0.]])
    return make(v, [(0, 1, 3), (1, 2, 3), (2, 0, 3), (1, 0, 4), (2, 1, 4), (0, 2, 4)])


def nest_on_an_edge_with_three_elements():
    v, f, _ = plate()
    return make(np.vstack([v, [[0.0, -0.3, 1.0]]]), np.vstack([f, [[U, V, len(v)]]]))


def plate_with_feature_point():
    v, f, m = plate()
    return make(v, f, points=[m['inner'][1]])


SHEET_ACROSS_THE_NEW_ELEMENT = [[-.04, .4, -.04], [.04, .4, .04], [.04, .45, -.04]]


@pytest.mark.parametrize(
    'build,reason',
    [
        (lambda: make(*plate(sag=0.3)[:2]), 'nest_over_limit'),
        (lambda: make(*plate(depth=128)[:2]), 'nest_over_budget'),
        (small_closed_solid, 'other_side_within_limit'),
        (lambda: make(*nest_under_sheet(SHEET_ACROSS_THE_NEW_ELEMENT)[:2]), 'element_crosses_surface'),
        (nest_on_an_edge_with_three_elements, 'edge_has_more_than_two_faces'),
        (plate_with_feature_point, 'replacement_modified_feature_point'),
    ],
)
def test_refused_replacement_gives_its_reason(build, reason):
    assert replace_with_reason(build()) == (0, reason)


def test_blocked_edge_with_repair_off_gets_no_repair_or_replacement():
    m, t, f, b, ref, p, tol = make(*plate()[:2])
    faces = len(m.element)
    assert r._attempt_collapse(
        m, t, (U, V), f, b, p, tol, boundary_only=False, iteration_index=9, allow_repair=False
    ) == (False, 'link_obstruction')
    assert len(m.element) == faces


def test_blocked_edge_names_the_repair_failure_and_the_refusal():
    assert late(make(*plate(sag=0.3)[:2])) == (False, 'neighbor_length, nest_over_limit')


# Features and frozen boundaries


def test_feature_point_blocks_replacement_and_stays_attached():
    v, f, m = plate()
    point = m['inner'][1]
    ctx = make(v, f, points=[point])
    assert replace(ctx) == 0
    assert stats(ctx)['faces'] == len(f)
    assert ctx[1]['vertex_to_faces'][point]


def test_feature_edge_blocks_replacement():
    v, f, m = plate()
    ctx = make(v, f, curves=[[m['inner'][0], m['inner'][1]]])
    assert replace(ctx) == 0
    assert stats(ctx)['faces'] == len(f)


def test_feature_point_on_u_keeps_u_in_place():
    v, f, m = plate()
    ctx = make(v, f, target=4.0, points=[U])
    assert late(ctx)[0]
    assert np.array_equal(ctx[1]['vertex_positions'][U], v[U])


def test_frozen_boundary_is_kept():
    v, f, m = plate()
    ctx = make(v, f, boundary=False)
    free_before = {e for e, x in ctx[1]['edge_to_faces'].items() if len(x) == 1}
    assert replace(ctx) == 1
    assert {e for e, x in ctx[1]['edge_to_faces'].items() if len(x) == 1} == free_before


# Numbering, scale and rotation


@pytest.mark.parametrize('seed', range(5))
def test_numbering_and_face_order_do_not_matter(seed):
    v, f, m = plate()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(v))
    v2 = np.empty_like(v)
    v2[perm] = v
    f2 = perm[f][rng.permutation(len(f))]
    f2 = np.array([np.roll(t, int(rng.integers(3))) for t in f2])
    uv = tuple(sorted((int(perm[U]), int(perm[V]))))
    ctx = make(v2, f2)
    assert late(ctx, uv)[0]
    assert stats(ctx)['faces'] == len(f) - 8


@pytest.mark.parametrize('factor', [1e-3, 1.0, 1e3])
def test_scale_does_not_matter(factor):
    v, f, m = plate()
    ctx = make(v * factor, f, target=TARGET * factor)
    assert late(ctx)[0]
    assert stats(ctx)['faces'] == len(f) - 8


def test_rotation_does_not_matter():
    v, f, m = plate()
    q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    q = q * np.sign(np.linalg.det(q))
    ctx = make(v @ q.T, f)
    assert late(ctx)[0]
    assert stats(ctx)['faces'] == len(f) - 8


# Scheduling and reporting


def test_replacement_waits_for_the_late_half(monkeypatch):
    v, f, m = plate()
    monkeypatch.setattr(r, '_try_repair_collapse_patch', lambda *a, **k: (None, 'forced'))
    ctx = make(v, f)
    assert late(ctx, iteration=0) == (False, 'deferred_after_repair_failure')
    assert late(ctx, iteration=4) == (False, 'deferred_after_repair_failure')
    assert ctx[1]['revision'] == 0
    assert late(ctx, iteration=5) == (True, 'ordinary_collapsed')
    record = ctx[1]['replacements']
    assert len(record) == 1
    assert record[0]['iteration'] == 6 and record[0]['removed_faces'] == m['patch_faces']
    assert record[0]['added_faces'] == 1 and record[0]['repair_failure'] == 'forced'
    assert record[0]['outer_triangle'] == [U, V, m['outer']]


def test_successful_repair_never_reaches_replacement(monkeypatch):
    source, meta = repairable_curved_patch()
    monkeypatch.setattr(r, '_replace_nested_triangles', lambda *a, **k: pytest.fail('replacement after a successful repair'))
    ctx = context(source, length=float(meta['target']))
    assert attempt(ctx, tuple(sorted(meta['edge'])), 9, False) == (True, 'repaired_collapsed')


def test_failed_collapse_after_a_replacement_still_reports_a_change():
    points, faces, outer = patch()
    ctx = make(points, faces)
    changed, reason = late(ctx)
    assert changed and reason == 'nested_triangles_replaced'
    assert ctx[1]['counts']['nested_replacement'] == 1


def test_full_run_records_replacements_in_the_report():
    v, f, m = plate()
    out = r.isotropic_triangle_surface_remesh(mesh_of(v, f), 0.8, False, max_iter=10)
    report = out.mesh_data['remeshing_report']
    records = report['nested_replacements']
    assert records, 'this fixture must reach the late fallback through the public API'
    assert report['operations']['nested_replacement'] == len(records)
    for record in records:
        assert record['iteration'] >= 6 and record['repair_failure']
        assert record['removed_faces'] >= 3 and record['added_faces'] == 1
        assert len(record['outer_triangle']) == 3
    g = out.element.numpy()
    edges = {tuple(sorted(e)) for t in g.tolist() for e in itertools.combinations(t, 2)}
    assert len(np.unique(g)) - len(edges) + len(g) == 1
    assert len(np.unique(g)) == len(out.node)


def test_no_replacement_with_zero_iterations():
    v, f, m = plate()
    out = r.isotropic_triangle_surface_remesh(mesh_of(v, f), TARGET, False, max_iter=0)
    assert not out.mesh_data['remeshing_report']['nested_replacements']


# The original bug: deletion emptied closed meshes


@pytest.mark.parametrize('seed', range(10))
def test_closed_sphere_with_noise_is_never_erased(seed):
    v, f = icosphere(3)
    v = v * 0.95 + np.random.default_rng(seed).normal(0, 1e-9, v.shape)
    out = r.isotropic_triangle_surface_remesh(mesh_of(v, f), 1.0, True, max_iter=10)
    assert len(out.element) > 0


@pytest.mark.parametrize('builder', [plate, closed_sphere])
def test_full_remesh_with_a_patch_stays_valid(builder):
    v, f, m = builder()
    out = r.isotropic_triangle_surface_remesh(mesh_of(v, f), 1.0, True, max_iter=10)
    g = out.element.numpy()
    counts = {}
    for t in g.tolist():
        for e in itertools.combinations(sorted(t), 2):
            counts[e] = counts.get(e, 0) + 1
    assert len(g) > 0 and max(counts.values()) <= 2
