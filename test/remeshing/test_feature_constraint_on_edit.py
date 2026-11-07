import pytest

pytestmark = [pytest.mark.remeshing, pytest.mark.processing, pytest.mark.vtk]
pytest.importorskip("vtk", reason="VTK is required for this test module")

"""The feature check on an edit looks only at what the edit touches."""
import numpy as np
import torch

import torch_fea_mesh.processing.isotropic_triangle_surface_remesher as r
from test.remeshing.test_isotropic_triangle_surface_remesher import context


def square():
    node = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=torch.float64)
    return r.TriangleMesh(node, torch.tensor([[0, 1, 2], [0, 2, 3]]))


def closed_hexagon():
    """A fan around node 0 over a hexagon, closed below by triangles with chords (1,3), (3,5), (5,1)."""
    ring = [[np.cos(k * np.pi / 3), np.sin(k * np.pi / 3), 0] for k in range(6)]
    node = torch.tensor([[0, 0, 1], *ring], dtype=torch.float64)
    fan = [[0, 1 + k, 1 + (k + 1) % 6] for k in range(6)]
    below = [[1, 3, 2], [3, 5, 4], [5, 1, 6], [1, 5, 3]]
    return r.TriangleMesh(node, torch.tensor(fan + below))


def check(ctx, kind='collapse', **edit):
    mesh, topology, features, *_ = ctx
    return r._check_feature_constraint(mesh, topology, features, r._proposal(topology, kind, **edit))


def test_edit_that_moves_or_retires_a_protected_point_is_refused():
    ctx = context(square(), points=[0])
    here, there = np.array([0.0, 0, 0]), np.array([0.5, 0, 0])
    assert check(ctx, 'move', positions={0: there}) == (False, 'modified_feature_point')
    assert check(ctx, 'move', positions={0: here}) == (True, 'ok')
    assert check(ctx, 'move', positions={1: there}) == (True, 'ok')
    assert check(ctx, retired={0}) == (False, 'modified_feature_point')
    assert check(ctx, retired={1}) == (True, 'ok')


def test_edit_that_removes_every_face_of_a_protected_edge_is_refused():
    ctx = context(square(), curves=[[0, 2]])
    assert check(ctx, remove=[0]) == (True, 'ok')
    assert check(ctx, remove=[0, 1]) == (False, 'lost_feature_edge')
    assert check(ctx, remove=[0, 1], add=[(0, 2, 1)]) == (True, 'ok')


def test_edit_check_does_not_visit_every_protected_item():
    class NoVisit(dict):
        def items(self):
            raise AssertionError('visited every protected point')

    class NoVisitSet(set):
        def __iter__(self):
            raise AssertionError('visited every protected edge')

    mesh, topology, features, *_ = context(square(), curves=[[0, 2]])
    features = dict(
        features, positions=NoVisit(features['positions']), edges=NoVisitSet(features['edges'])
    )
    edit = r._proposal(topology, 'collapse', remove=[0, 1], retired={1})
    assert r._check_feature_constraint(mesh, topology, features, edit) == (False, 'lost_feature_edge')



def grid():
    """A flat 4 x 4 grid of nodes, node i + 4 * j at (i, j). Nodes 5, 6, 9 and 10 are inside."""
    node = torch.tensor([[i, j, 0] for j in range(4) for i in range(4)], dtype=torch.float64)
    face = []
    for j in range(3):
        for i in range(3):
            a = i + 4 * j
            face += [[a, a + 1, a + 5], [a, a + 5, a + 4]]
    return r.TriangleMesh(node, torch.tensor(face))


def collapse_placements(ctx, edge):
    _, topology, features, boundary, *_ = ctx
    return [
        (edit['survivor'], edit['positions'][edit['survivor']].tolist())
        for edit in r._propose_edge_collapse(topology, edge, features, boundary)
    ]


@pytest.mark.parametrize(
    'edge, option, expected',
    [
        ((5, 6), {}, [(5, [1.5, 1, 0]), (6, [2, 1, 0]), (5, [1, 1, 0])]),
        ((5, 6), dict(points=[6]), [(6, [2, 1, 0])]),
        ((5, 6), dict(points=[5, 6]), []),
        ((1, 5), {}, [(1, [1, 0, 0])]),  # an interior edge keeps its end point on the boundary
        ((1, 5), dict(points=[5]), []),
        ((0, 1), {}, [(0, [0.5, 0, 0]), (1, [1, 0, 0]), (0, [0, 0, 0])]),
        ((0, 1), dict(boundary=False), []),  # an edge of a frozen boundary
        ((5, 9), dict(boundary=False), [(5, [1, 1.5, 0]), (9, [1, 2, 0]), (5, [1, 1, 0])]),
        ((1, 5), dict(boundary=False), [(1, [1, 0, 0])]),
    ],
)
def test_collapse_proposal_keeps_protected_end_points(edge, option, expected):
    assert collapse_placements(context(grid(), **option), edge) == expected


@pytest.mark.parametrize('flag_process_boundary', [True, False])
def test_collapse_proposal_does_not_visit_every_protected_point(flag_process_boundary):
    class LookUpOnly:
        # Not a set subclass: set() and | copy one of those without calling __iter__.
        def __init__(self, items):
            self.items = frozenset(items)

        def __contains__(self, item):
            return item in self.items

        def __iter__(self):
            raise AssertionError('visited every protected point')

    ctx = context(grid(), boundary=flag_process_boundary, points=[0, 6, 15])
    mesh, topology, features, boundary, *rest = ctx
    no_visit = (
        mesh,
        topology,
        dict(features, points=LookUpOnly(features['points'])),
        dict(boundary, frozen=dict(boundary['frozen'], points=LookUpOnly(boundary['frozen']['points']))),
        *rest,
    )
    for edge in [(5, 6), (6, 10), (1, 5), (0, 1), (5, 9)]:
        assert collapse_placements(no_visit, edge) == collapse_placements(ctx, edge)

def test_final_check_finds_a_moved_or_lost_point():
    mesh, topology, features, *_ = context(square(), points=[0])
    assert r._check_feature_constraint(mesh, topology, features) == (True, 'ok')
    assert r._check_feature_constraint(mesh, topology, features, old_to_new_vertex={}) == (
        False,
        'lost_feature_point',
    )
    topology['vertex_positions'][0] += 1
    assert r._check_feature_constraint(mesh, topology, features) == (False, 'modified_feature_point')


def test_patch_refuses_edits_while_a_protected_edge_is_absent():
    mesh, topology, features, boundary, _, parameters, _ = context(closed_hexagon(), curves=[[3, 5]])
    patch, _ = r._extract_repair_patch(mesh, topology, (0, 2), features, boundary, parameters, ring_count=1)
    local_mesh = r._copy_source_mesh(patch['mesh'])
    local_topology = r._initialize_topology(local_mesh)
    local_features = patch['features']
    assert len(local_features['absent_edges']) == 1
    edge = next(iter(local_features['absent_edges']))
    other = next(v for v in range(len(local_mesh.node)) if v not in edge)

    def result(**edit):
        return r._check_feature_constraint(
            local_mesh, local_topology, local_features, r._proposal(local_topology, 'collapse', **edit)
        )

    assert result(remove=[0]) == (False, 'lost_feature_edge')
    assert result(remove=[0], add=[(*edge, other)]) == (True, 'ok')
    assert r._check_feature_constraint(local_mesh, local_topology, local_features) == (False, 'lost_feature_edge')
