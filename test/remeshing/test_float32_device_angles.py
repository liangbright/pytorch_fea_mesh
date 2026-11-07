import pytest

pytestmark = [pytest.mark.remeshing, pytest.mark.processing, pytest.mark.vtk]
pytest.importorskip("vtk", reason="VTK is required for this test module")

"""Float32 sources, the device of the output, and what validating an edit computes."""
import numpy as np
import torch

import torch_fea_mesh.processing.isotropic_triangle_surface_remesher as r
from test.remeshing.test_isotropic_triangle_surface_remesher import context, grid
from test.remeshing.test_nested_replacement import U, V, make, plate

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device')


def small_loop_tube(seed):
    """A float32 tube far from the origin, whose boundary loops are about 3 target lengths around."""
    generator = np.random.default_rng(seed)
    count, rows = 16, 9
    angle = 2 * np.pi * np.arange(count) / count
    node = np.array(
        [[0.5 * np.cos(a), 0.5 * np.sin(a), 2 * j / (rows - 1)] for j in range(rows) for a in angle]
    )
    node = node + 100 + 1e-6 * generator.standard_normal(node.shape)
    element = []
    for j in range(rows - 1):
        for i in range(count):
            a, b = j * count + i, j * count + (i + 1) % count
            element += [(a, b, b + count), (a, b + count, a + count)]
    order = generator.permutation(len(node))
    return r.TriangleMesh(
        torch.tensor(node[np.argsort(order)], dtype=torch.float32), torch.tensor(order[element])
    )


def test_float32_run_finishes_and_reports_what_it_writes():
    for seed in range(4):
        output = r.isotropic_triangle_surface_remesh(small_loop_tube(seed), 1.0, True, max_iter=10)
        report = output.mesh_data['remeshing_report']
        output.build_edge()
        length = output.node.double()[output.edge]
        length = (length[:, 0] - length[:, 1]).norm(dim=1)
        assert report['failures']['short_edges'] == int(
            (length <= report['parameters']['min_edge_length']).sum()
        )
        assert report['failures']['long_edges'] == int(
            (length > report['parameters']['max_edge_length']).sum()
        )


def test_float32_source_rounds_new_positions_everywhere_and_float64_does_not():
    point = np.array([1 / 3, 2 / 3, 0.1])
    rounded = point.astype(np.float32).astype(float)
    source = grid(dtype=torch.float64)
    _, topology, *_ = context(source)
    edit = r._proposal(topology, 'move', positions={0: point}, new_points=[point])
    assert np.array_equal(edit['positions'][0], point) and np.array_equal(edit['new_points'][0], point)
    topology['round_positions'] = True
    edit = r._proposal(topology, 'move', positions={0: point}, new_points=[point])
    assert np.array_equal(edit['positions'][0], rounded) and np.array_equal(edit['new_points'][0], rounded)
    # A repair patch works on its own topology, which must round too.
    m, t, f, b, _, p, tolerances = make(*plate()[:2])
    t['round_positions'] = True
    patch, _ = r._extract_repair_patch(m, t, (U, V), f, b, p, ring_count=1)
    captured = []
    initialize = r._initialize_topology
    r._initialize_topology = lambda *a, **k: captured.append(k) or initialize(*a, **k)
    try:
        r._repair_patch_geometry(patch, p, tolerances)
    finally:
        r._initialize_topology = initialize
    assert captured == [{'round_positions': True}]
    public = []
    r._initialize_topology = lambda *a, **k: public.append(k) or initialize(*a, **k)
    try:
        for dtype in (torch.float32, torch.float64):
            r.isotropic_triangle_surface_remesh(grid(dtype=dtype), 0.6, max_iter=0)
    finally:
        r._initialize_topology = initialize
    assert [k.get('round_positions') for k in public if k] == [True, False]


def test_validating_an_edit_computes_no_angles(monkeypatch):
    _, topology, *_, tolerances = context(grid())
    edit = r._proposal(topology, 'move', positions={5: topology['vertex_positions'][5] + 0.01})
    assert r._geometry(np.zeros((1, 3, 3)) + np.eye(3))[2].shape == (1, 3)

    def no_angles(*a, **k):
        raise AssertionError('computed corner angles')

    monkeypatch.setattr(r.TriangleMesh, 'cal_element_corner_angle', staticmethod(no_angles))
    assert r._validate_edit_geometry(topology, edit, tolerances) == (True, 'ok')
    assert r._geometry(np.zeros((1, 3, 3)) + np.eye(3), with_angles=False)[2] is None


@needs_cuda
def test_no_tensor_follows_the_default_device():
    for node_device in ('cpu', 'cuda'):
        source = grid()
        source.node, source.element = source.node.to(node_device), source.element.to(node_device)
        with torch.device('cuda'):
            output = r.isotropic_triangle_surface_remesh(source, 0.6, True, max_iter=2)
            empty = r._new_mesh([], [])
        assert output.node.device == source.node.device
        assert output.element.device == source.node.device
        assert empty.node.device.type == 'cpu' and empty.element.device.type == 'cpu'


@needs_cuda
def test_output_element_goes_to_the_node_device():
    source = grid()
    source.node = source.node.to('cuda')
    assert source.element.device.type == 'cpu'
    output = r.isotropic_triangle_surface_remesh(source, 0.6, max_iter=1)
    assert output.element.device == source.node.device
