import pytest

pytestmark = [pytest.mark.remeshing, pytest.mark.processing, pytest.mark.vtk]
pytest.importorskip("vtk", reason="VTK is required for this test module")

"""Interior smoothing: the plain average by default, or coverage then tangential smoothing."""
import inspect

import numpy as np
import torch

import torch_fea_mesh.processing.isotropic_triangle_surface_remesher as r
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh
from test.remeshing.test_isotropic_triangle_surface_remesher import context

RING = [[3, 0, 0], [0.5, 1, 0.1], [-1, 0.5, -0.1], [-1, -1, 0.2], [1, -2, 0]]
GENTLE_APEX = [0.2, 0.1, 0.3]


def mesh_of(points, faces):
    return TriangleMesh(torch.tensor(np.array(points, dtype=float), dtype=torch.float64), torch.tensor(faces))


def fan(apex, ring):
    """A vertex with a ring of neighbors; only the vertex is interior."""
    n = len(ring)
    return mesh_of([apex] + ring, [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)])


def small_sphere():
    """A subdivided octahedron of radius 1."""
    points = [np.array(p, float) for p in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    for _ in range(3):
        middle, finer = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in middle:
                p = points[a] + points[b]
                points.append(p / np.linalg.norm(p))
                middle[key] = len(points) - 1
            return middle[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            finer += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = finer
    return mesh_of(points, faces)


def relax(source, flag, project=False, use_coverage=False):
    # At length 2 the short limit is 1.4, so these fans have short and long edges.
    m, t, f, b, reference, p, tolerances = context(source, length=2.0, project=project)
    p['flag_use_coverage_then_tangential'] = flag
    r._relax_vertex(m, t, f, b, reference, p, tolerances, boundary_only=False, use_coverage=use_coverage)
    return t['vertex_positions'][0].copy()


def test_flag_is_the_last_argument_and_off_by_default():
    last = list(inspect.signature(r.isotropic_triangle_surface_remesh).parameters.values())[-1]
    assert last.name == 'flag_use_coverage_then_tangential'
    assert last.default is False and last.kind is inspect.Parameter.KEYWORD_ONLY


def test_flag_must_be_bool():
    with pytest.raises(TypeError, match='flag_use_coverage_then_tangential'):
        r.isotropic_triangle_surface_remesh(small_sphere(), 1.0, flag_use_coverage_then_tangential=1)


def test_default_moves_a_vertex_to_the_plain_average():
    assert np.allclose(relax(fan(GENTLE_APEX, RING), False), np.array(RING).mean(axis=0))


def test_tangential_step_removes_the_normal_part_of_the_move():
    # Uneven faces, so an unweighted normal gives a different answer.
    apex, points = np.array(GENTLE_APEX), np.array(RING)
    normal = np.sum([np.cross(points[i] - apex, points[(i + 1) % 5] - apex) / 2 for i in range(5)], axis=0)
    normal /= np.linalg.norm(normal)
    average = points.mean(axis=0)
    assert np.allclose(relax(fan(GENTLE_APEX, RING), True), average - normal * ((average - apex) @ normal))


def test_tangential_step_leaves_a_vertex_with_no_tangent_plane():
    # These four area-weighted normals cancel exactly. Projection is on, as by default: with it
    # off, validation refuses the bad position and hides a missing check.
    apex = [0, 0, 1]
    moved = relax(fan(apex, [[-1, 0, 0], [-0.5, 1, 0], [1, 0, 0], [0.5, 1, 0]]), True, project=True)
    assert np.array_equal(moved, np.array(apex, dtype=float))


def test_tangential_step_leaves_a_vertex_on_a_sharp_corner():
    apex = [0.2, 0.1, 3.0]
    assert np.array_equal(relax(fan(apex, RING), True), np.array(apex))


def test_coverage_samples_follow_area_and_each_vertex_moves_to_the_centre_of_its_own():
    # The second triangle has nine times the area of the first.
    source = mesh_of([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-3, 0, 0], [0, -3, 0], [3, -3, 0]], [(0, 1, 2), (3, 4, 5)])
    m, t, f, b, reference, p, tolerances = context(source, length=2.0)
    points = r.coverage_samples(reference, p)
    assert 0.07 < np.mean(points[:, 1] >= 0) < 0.13
    centers = r.coverage_smoothing_positions(m, t, reference, p)
    x = t['vertex_positions']
    owner = np.argmin(np.linalg.norm(points[:, None] - x[None], axis=2), axis=1)
    for vertex, center in centers.items():
        assert np.allclose(center, points[owner == vertex].mean(axis=0))


def test_coverage_step_moves_a_vertex_to_the_centre_of_its_samples():
    source = fan(GENTLE_APEX, RING)
    m, t, f, b, reference, p, tolerances = context(source, length=2.0, project=False)
    expected = r.coverage_smoothing_positions(m, t, reference, p)[0]
    assert np.allclose(relax(source, True, use_coverage=True), expected)


def smoothing_per_iteration(monkeypatch, max_iter, **option):
    calls = []
    original = r._relax_vertex

    def spy(*a, **k):
        if not k.get('boundary_only'):
            calls.append('coverage' if k.get('use_coverage') else 'tangential')
        return original(*a, **k)

    monkeypatch.setattr(r, '_relax_vertex', spy)
    r.isotropic_triangle_surface_remesh(small_sphere(), 0.5, max_iter=max_iter, **option)
    return calls


@pytest.mark.parametrize('max_iter, expected', [
    (1, ['coverage']),
    (2, ['coverage', 'tangential']),
    (3, ['coverage', 'tangential', 'tangential']),
    (5, ['coverage', 'coverage', 'coverage', 'tangential', 'tangential']),
])
def test_first_iteration_always_uses_coverage(monkeypatch, max_iter, expected):
    calls = smoothing_per_iteration(monkeypatch, max_iter, flag_use_coverage_then_tangential=True)
    assert calls == expected


def test_coverage_runs_except_in_the_last_two_iterations(monkeypatch):
    calls = []
    original = r._relax_vertex

    def spy(*a, **k):
        if not k.get('boundary_only'):
            calls.append(k.get('use_coverage'))
        return original(*a, **k)

    monkeypatch.setattr(r, '_relax_vertex', spy)
    r.isotropic_triangle_surface_remesh(small_sphere(), 0.5, max_iter=4, flag_use_coverage_then_tangential=True)
    assert calls == [True, True, False, False]
    calls.clear()
    r.isotropic_triangle_surface_remesh(small_sphere(), 0.5, max_iter=1, flag_use_coverage_then_tangential=True)
    assert calls == [True]
    calls.clear()
    r.isotropic_triangle_surface_remesh(small_sphere(), 0.5, max_iter=2)
    assert calls == [False, False]


def test_small_sphere_shrinks_by_default_and_keeps_its_size_with_the_flag():
    # Radius 1 at target 1: the plain average leaves about a tenth of the area.
    area = lambda output: output.mesh_data['remeshing_report']['failures']['area_ratio']
    source = small_sphere()
    assert area(r.isotropic_triangle_surface_remesh(source, 1.0, max_iter=30)) < 0.2
    assert area(r.isotropic_triangle_surface_remesh(source, 1.0, max_iter=30, flag_use_coverage_then_tangential=True)) > 0.7
