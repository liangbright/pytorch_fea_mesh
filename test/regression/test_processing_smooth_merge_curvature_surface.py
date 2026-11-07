"""Tests for the fixes in mesh_processing.py, polygon_mesh_processing.py and polyhedron_mesh_processing.py."""
import inspect
import itertools
import math

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.processing, pytest.mark.vtk]
vtk = pytest.importorskip('vtk')

from torch_fea_mesh.mesh.hexahedron_mesh import HexahedronMesh
from torch_fea_mesh.mesh.mesh import Mesh
from torch_fea_mesh.mesh.polygon_mesh import PolygonMesh
from torch_fea_mesh.mesh.polyhedron_mesh import PolyhedronMesh
from torch_fea_mesh.mesh.polyline_mesh import PolylineMesh
from torch_fea_mesh.mesh.quad_mesh import QuadMesh
from torch_fea_mesh.mesh.quad_triangle_mesh import QuadTriangleMesh
from torch_fea_mesh.mesh.tet10_mesh import Tet10Mesh
from torch_fea_mesh.mesh.tetrahedron_mesh import TetrahedronMesh
from torch_fea_mesh.mesh.tri6_mesh import Tri6Mesh
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh
from torch_fea_mesh.processing import mesh_processing as mp
from torch_fea_mesh.processing import polygon_mesh_processing as pmp
from torch_fea_mesh.processing import polyhedron_mesh_processing as vp

F = torch.float64
CUDA = 'cuda' if torch.cuda.is_available() else None
needs_cuda = pytest.mark.skipif(CUDA is None, reason='CUDA device unavailable')
TRI = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
SQUARE = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]


def T(value):
    return torch.tensor(value, dtype=F)


def grid_node(n):
    xs = torch.linspace(0, 1, n, dtype=F)
    X, Y = torch.meshgrid(xs, xs, indexing='ij')
    return torch.stack([X.flatten(), Y.flatten(), 0.1 * torch.sin(3 * X.flatten())], 1)


def grid_tri(n=4):
    element = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = a + n
            element += [[a, b, a + 1], [a + 1, b, b + 1]]
    return TriangleMesh(grid_node(n), element)


def grid_quad(n=4):
    element = [[i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1] for i in range(n - 1) for j in range(n - 1)]
    return QuadMesh(grid_node(n), element)


def tri6():
    node = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [.5, 0, 0], [.5, .5, 0], [0, .5, 0], [1, 1, 0], [1, .5, 0], [.5, 1, 0]]
    return Tri6Mesh(T(node), [[0, 1, 2, 3, 4, 5], [1, 6, 2, 7, 8, 4]])


def quad_tri():
    return QuadTriangleMesh(T(SQUARE + [[2, 0, 0]]), [[0, 1, 2, 3], [1, 4, 2]])


def hex_block(nx=2, ny=1, nz=1):
    X, Y, Z = torch.meshgrid(*[torch.arange(k + 1, dtype=F) for k in (nx, ny, nz)], indexing='ij')
    node = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], 1)
    idx = np.arange((nx + 1) * (ny + 1) * (nz + 1)).reshape(nx + 1, ny + 1, nz + 1)
    corner = [idx[:-1, :-1, :-1], idx[1:, :-1, :-1], idx[1:, 1:, :-1], idx[:-1, 1:, :-1],
              idx[:-1, :-1, 1:], idx[1:, :-1, 1:], idx[1:, 1:, 1:], idx[:-1, 1:, 1:]]
    return HexahedronMesh(node, torch.tensor(np.stack(corner, -1).reshape(-1, 8)))


def tet():
    return TetrahedronMesh(T([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]), [[0, 1, 2, 3], [1, 2, 3, 4]])


def tet10():
    mesh = Tet10Mesh()
    mesh.create_from_tet4(tet())
    return mesh


def sphere(R=2.0, res=40):
    source = vtk.vtkSphereSource()
    source.SetRadius(R)
    source.SetThetaResolution(res)
    source.SetPhiResolution(res)
    source.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    source.Update()
    mesh = TriangleMesh()
    mesh.read_mesh_vtk(source.GetOutput(), F)
    return mesh


def on_cuda(mesh):
    mesh.node = mesh.node.to(CUDA)
    mesh.element = mesh.element.to(CUDA)
    return mesh


def fill_geometry_cache(mesh):
    mesh.update_edge_length()
    if isinstance(mesh, PolygonMesh):
        mesh.update_element_area_and_normal()
        mesh.update_node_normal()
    else:
        mesh.update_element_volume()


def test_coordinate_writes_clear_the_cache_and_keep_the_connectivity():
    def assert_cleared(mesh):
        assert mesh.edge_length is None
        if isinstance(mesh, PolygonMesh):
            assert mesh.element_area is None and mesh.element_normal is None and mesh.node_normal is None
        else:
            assert mesh.element_volume is None
    for make in (grid_tri, grid_quad, tri6, quad_tri, tet, tet10, hex_block):
        mesh = make()
        fill_geometry_cache(mesh)
        mp.simple_smoother_for_mesh(mesh, 0.5, None, 1)
        assert_cleared(mesh)
        link, edge = mesh.node_to_node_adj_link, mesh.edge
        assert link is not None and edge is not None
        fill_geometry_cache(mesh)
        mp.simple_smoother_for_mesh(mesh, 0.5, None, 1)
        assert_cleared(mesh)
        assert mesh.node_to_node_adj_link is link and mesh.edge is edge
    mesh = grid_quad()
    fill_geometry_cache(mesh)
    pmp.simple_smoother_for_quad_mesh(mesh, 0.5, None, 1)
    assert_cleared(mesh)
    assert mesh.node_to_element_adj_table is not None

    mesh = grid_tri()
    mesh.node[:, 2] += 0.3
    fill_geometry_cache(mesh)
    mesh.build_node_to_node_adj_link()
    pmp.smooth_and_project(mesh, grid_tri(6), 0.5, torch.ones(16), 1, 1, smooth_first=False)
    assert_cleared(mesh)
    assert mesh.node_to_node_adj_link is not None


def test_clear_geometry_derived_data_switch_on_every_mesh_class():
    connectivity = ('quad_element', 'quad_element_idx', 'tri_element', 'tri_element_idx')
    user_data = ('mesh_type', 'name', 'node', 'element', 'element_type', 'mapping_from_node_name_to_index',
                 'node_set', 'element_set', 'node_data', 'element_data', 'mesh_data')
    for make in (grid_tri, grid_quad, tri6, quad_tri, tet, tet10, hex_block, lambda: PolygonMesh(T(TRI), [[0, 1, 2]]),
                 lambda: PolyhedronMesh(T(TRI + [[0, 0, 1]]), [[0, 1, 2, 3]]), lambda: PolylineMesh(T(TRI), [[0, 1], [1, 2]])):
        mesh = make()
        adjacency = [k for k in vars(Mesh(None, None, None, mesh.mesh_type)) if k not in user_data and k != 'edge_length']
        adjacency += [k for k in connectivity if hasattr(mesh, k)]
        cached = [k for k, v in vars(mesh).items() if k not in adjacency and k not in user_data]
        assert 'edge_length' in cached
        def stale(mesh):
            for name in adjacency + cached:
                setattr(mesh, name, 'stale')
            return mesh
        mesh = stale(make())
        mesh.clear_geometry_derived_data(clear_adjacency_data=False)
        assert [k for k in cached if getattr(mesh, k) is not None] == []
        assert [k for k in adjacency if getattr(mesh, k) != 'stale'] == []
        for call in (lambda m: m.clear_geometry_derived_data(), lambda m: m.clear_geometry_derived_data(clear_adjacency_data=True)):
            mesh = stale(make())
            call(mesh)
            assert [k for k in adjacency + cached if getattr(mesh, k) == 'stale'] == []
    # QuadTriangleMesh keeps its classification, and a later update still works
    mesh = quad_tri()
    mesh.clear_geometry_derived_data(clear_adjacency_data=False)
    assert mesh.quad_element_idx == [0] and mesh.tri_element_idx == [1]
    mesh.update_node_normal()


def test_quad_smoother_with_no_elements_is_a_no_op():
    for node in (T([[0, 0, 0], [1, 0, 0]]), torch.empty((0, 3), dtype=F)):
        mesh = QuadMesh(node.clone(), torch.empty((0, 4), dtype=torch.int64))
        pmp.simple_smoother_for_quad_mesh(mesh, 0.5, None, 1)
        assert torch.equal(mesh.node, node)


def test_remove_unused_node_keeps_first_use_order_and_plain_integers():
    # rectangular: the result is a tensor on the same device; the flag has no effect on connectivity tables
    for flag in (True, False):
        mesh = TriangleMesh(T(TRI + [[1, 1, 0], [5, 5, 5]]), [[3, 0, 1], [1, 0, 2]])
        mesh.build_edge()
        kept = mp.remove_unused_node(mesh, return_node_idx_list=True, clear_adj_info=flag)
        assert kept.tolist() == [3, 0, 1, 2] and kept.dtype == torch.int64
        assert mesh.element.tolist() == [[0, 1, 2], [2, 1, 3]] and mesh.edge is None
        assert torch.equal(mesh.node, T([[1, 1, 0]] + TRI))
    # ragged: entries are plain ints, so a sub mesh works afterwards
    mesh = PolygonMesh(T(SQUARE + [[2, 0, 0], [2, 1, 0], [9, 9, 9]]), [[3, 0, 1, 2], [1, 4, 5], [2, 1, 5]])
    kept = mp.remove_unused_node(mesh, return_node_idx_list=True)
    assert kept.tolist() == [3, 0, 1, 2, 4, 5]
    assert mesh.element == [[0, 1, 2, 3], [2, 4, 5], [3, 2, 5]]
    assert all(type(idx) is int for elm in mesh.element for idx in elm)
    assert [list(map(int, e)) for e in mesh.get_sub_mesh([1, 2]).element] == [[0, 2, 3], [1, 0, 3]]
    with pytest.raises(ValueError):
        mp.remove_unused_node(PolygonMesh(T(TRI), [[0, 1, -1], [0, 1, 2, 2]]))
    # no elements: every node goes
    mesh = TriangleMesh(T(TRI), torch.empty((0, 3), dtype=torch.int64))
    mp.remove_unused_node(mesh)
    assert mesh.node.shape == (0, 3)


def test_merge_mesh_warns_about_either_input(capsys):
    class NotAMesh:
        def __init__(self, mesh):
            self.__dict__.update(mesh.__dict__)
            self.make_element_copy = mesh.make_element_copy
    mp.merge_mesh(grid_tri(), None, NotAMesh(grid_tri()), None, 0)
    assert 'warning' in capsys.readouterr().out
    mp.merge_mesh(grid_tri(), None, grid_tri(), None, 0)
    assert 'warning' not in capsys.readouterr().out


def test_merge_mesh_gives_integer_indices_and_keeps_dtype_and_class():
    a = PolygonMesh(T(SQUARE), [[0, 1, 2, 3]])
    b = TriangleMesh(T([[1, 0, 0], [1, 1, 0], [2, 0, 0]]), [[0, 2, 1]])
    merged = mp.merge_mesh(a, [1, 2], b, [0, 1], 1e-6)
    assert merged.element == [[0, 1, 2, 3], [1, 4, 2]]
    assert all(type(idx) is int for elm in merged.element for idx in elm)
    merged.build_node_to_element_adj_table()

    a32 = TriangleMesh(torch.tensor(TRI, dtype=torch.float32), [[0, 1, 2]])
    assert mp.merge_mesh(a32, None, a32, None, 0).node.dtype == torch.float32

    tri2 = TriangleMesh(T([[1, 0, 0], [1, 1, 0], [0, 1, 0]]), [[0, 1, 2]])
    for first, second, expected in [(grid_tri(), grid_tri(), TriangleMesh), (grid_quad(), grid_quad(), QuadMesh),
                                    (grid_tri(), grid_quad(), PolygonMesh), (quad_tri(), quad_tri(), QuadTriangleMesh),
                                    (hex_block(), hex_block(), HexahedronMesh), (hex_block(), tet(), PolyhedronMesh)]:
        assert type(mp.merge_mesh(first, None, second, None, 0)) is expected
    base = Mesh(T(TRI), [[0, 1, 2]], None, 'polygon_tri3')
    assert type(mp.merge_mesh(base, None, grid_tri(), None, 0)) is Mesh
    assert mp.merge_mesh(base, None, grid_quad(), None, 0).mesh_type == 'polygon'
    merged = pmp.merge_mesh_on_boundary([TriangleMesh(T(TRI), [[0, 1, 2]]), tri2], 1e-6)
    assert type(merged) is TriangleMesh and len(merged.node) == 4
    merged.update_node_normal()
    with pytest.raises(NotImplementedError):
        pmp.merge_mesh_on_boundary([tet(), tet()], 1e-6)

    with pytest.raises(ValueError, match='dimension'):
        mp.merge_mesh(hex_block(), None, grid_quad(), None, 0)


def test_merge_mesh_gives_one_label_per_element(tmp_path):
    def loaded(mesh, name):
        mesh.save_as_vtk(str(tmp_path / name))
        output = PolyhedronMesh()
        output.load_from_vtk(str(tmp_path / name), 'float64')
        return output
    # labels are kept in order and the merged mesh saves and reloads with both shapes
    merged = mp.merge_mesh(loaded(hex_block(1, 1, 1), 'h.vtk'), None, loaded(tet(), 't.vtk'), None, 0)
    assert list(merged.element_type) == ['HEX8', None, None]
    assert [len(e) for e in merged.get_sub_mesh([0, 1]).element] == [8, 4]
    merged.save_as_vtk(str(tmp_path / 'm.vtk'))
    again = PolyhedronMesh()
    again.load_from_vtk(str(tmp_path / 'm.vtk'), 'float64')
    assert [len(e) for e in again.element] == [8, 4, 4]
    # classes that name a shape get labels only if a save needs them
    merged = mp.merge_mesh(tet(), None, hex_block(1, 1, 1), None, 0)
    assert list(merged.element_type) == ['TET4', 'TET4', 'HEX8']
    assert mp.merge_mesh(grid_tri(), None, grid_quad(), None, 0).element_type is None
    assert mp.merge_mesh(hex_block(), None, hex_block(), None, 0).element_type is None
    # a polygon mesh never mixes linear and quadratic elements
    for linear in (grid_tri(2), grid_quad()):
        with pytest.raises(ValueError, match='never mixes linear and quadratic'):
            mp.merge_mesh(tri6(), None, linear, None, 0)
    assert type(mp.merge_mesh(tri6(), None, tri6(), None, 0)) is Tri6Mesh
    # a label from one side only: the other side still gets a slot
    labelled = grid_tri(2)
    labelled.element_type = np.array(['S3', 'S3'])
    assert list(mp.merge_mesh(labelled, None, quad_tri(), None, 0).element_type) == ['S3', 'S3', None, None]


def test_find_nearest_node_gives_one_result_per_query():
    mesh = TriangleMesh(T(TRI), [[0, 1, 2]])
    assert mp.find_nearest_node(mesh, T([[5, 5, 5], [1, 0, 0]]), distance_threshold=0.5) == [None, 1]
    assert mp.find_nearest_node(mesh, T([5, 5, 5]), distance_threshold=0.5) is None
    assert mp.find_nearest_node(mesh, T([1, 0, 0]), distance_threshold=0.5) == 1
    assert mp.find_nearest_node(mesh, np.array([[0.1, 0, 0]])) == [0]
    empty = TriangleMesh(torch.empty((0, 3), dtype=F), torch.empty((0, 3), dtype=torch.int64))
    assert mp.find_nearest_node(empty, T([[0, 0, 0]])) == [None]


@needs_cuda
def test_cuda_smoothers_projection_and_devices():
    # each result matches the CPU result and stays on the input's device
    field = mp.simple_smoother(T([[0.], [2.], [4.]]).to(CUDA), torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]]), .5, None, False)
    assert field.device.type == 'cuda' and field.cpu().tolist() == [[1.], [2.], [3.]]
    mesh = on_cuda(grid_tri())
    mp.simple_smoother_for_mesh(mesh, 0.5, None, 2)
    cpu = grid_tri()
    mp.simple_smoother_for_mesh(cpu, 0.5, None, 2)
    assert mesh.node.device.type == 'cuda' and torch.allclose(mesh.node.cpu(), cpu.node, rtol=1e-12)

    mesh = on_cuda(grid_quad())
    pmp.simple_smoother_for_quad_mesh(mesh, 0.5, None, 2)
    cpu = grid_quad()
    pmp.simple_smoother_for_quad_mesh(cpu, 0.5, None, 2)
    assert torch.allclose(mesh.node.cpu(), cpu.node, rtol=1e-12)

    mesh = on_cuda(grid_tri())
    pmp.simple_smoother_for_mesh_node_normal(mesh, 0.5, None, 2)
    assert mesh.node_normal.device.type == 'cuda'
    mesh = on_cuda(grid_tri())
    pmp.simple_smoother_for_mesh_element_normal(mesh, 0.5, None, 2)
    assert mesh.element_normal.device.type == 'cuda'

    for smooth_first in (True, False):
        for mask in (None, torch.ones(16, device=CUDA)):
            move = on_cuda(grid_tri())
            move.node[:, 2] += 0.3
            pmp.smooth_and_project(move, on_cuda(grid_tri(6)), 0.5, mask, 1, 2, smooth_first=smooth_first)
            assert move.node.device.type == 'cuda'

    a, b = on_cuda(grid_tri(4)), on_cuda(grid_tri(5))
    b.node[:, 2] += 0.1
    assert pmp.project_point_to_mesh(a.node, b)[0].device.type == 'cuda'
    assert pmp.project_point_to_mesh(a.node.cpu().numpy(), b)[0].device.type == 'cuda'
    assert pmp.project_point_to_mesh(a.node.cpu(), b)[0].device.type == 'cpu'
    assert pmp.cal_point_to_mesh_distance(a.node, b).device.type == 'cuda'
    assert pmp.cal_point_to_mesh_distance(a.node.cpu().numpy(), b).device.type == 'cuda'
    distance = pmp.cal_mesh_to_mesh_distance(a, b, 'mean')
    c, d = grid_tri(4), grid_tri(5)
    d.node[:, 2] += 0.1
    assert distance.device.type == 'cuda' and float(distance) == pytest.approx(float(pmp.cal_mesh_to_mesh_distance(c, d, 'mean')), rel=1e-12)

    assert mp.find_nearest_node(on_cuda(grid_tri()), np.array([[0., 0, 0]])) == [0]
    assert mp.find_nearest_node(on_cuda(grid_tri()), T([[0, 0, 0]])) == [0]

    for merged in (mp.merge_mesh(on_cuda(grid_tri()), None, on_cuda(grid_tri()), None, 0),
                   mp.merge_mesh(on_cuda(grid_tri()), [3], on_cuda(grid_tri()), [0], 10.0)):
        assert merged.node.device.type == 'cuda' and merged.element.device.type == 'cuda'
        merged.update_node_normal()
    merged = mp.merge_mesh(on_cuda(TriangleMesh(torch.tensor(TRI, dtype=torch.float32), [[0, 1, 2]])), None,
                           on_cuda(TriangleMesh(torch.tensor(TRI, dtype=torch.float32), [[0, 1, 2]])), None, 0)
    assert merged.node.dtype == torch.float32

    surface = vp.extract_surface_mesh(on_cuda(hex_block()))
    assert surface.node.device.type == 'cuda' and surface.element.device.type == 'cuda'
    surface.update_node_normal()


def test_projection_of_no_points_and_onto_no_elements():
    mesh = grid_tri()
    point, element = pmp.project_point_to_mesh(torch.empty((0, 3), dtype=F), mesh)
    assert point.shape == (0, 3) and element == []
    assert pmp.cal_point_to_mesh_distance(torch.empty((0, 3), dtype=F), mesh).shape == (0,)
    no_element = TriangleMesh(T(TRI), torch.empty((0, 3), dtype=torch.int64))
    with pytest.raises(ValueError, match='no elements'):
        pmp.project_point_to_mesh(T([[0, 0, 1]]), no_element)
    assert pmp.project_point_to_mesh(np.array([[0.25, 0.25, 1]]), TriangleMesh(T(TRI), [[0, 1, 2]]))[0].tolist() == [[0.25, 0.25, 0]]


def test_normal_smoothers_leave_a_held_tensor_alone_and_give_clear_errors():
    mesh = grid_tri()
    mesh.update_node_normal()
    held, before = mesh.node_normal, mesh.node_normal.clone()
    pmp.simple_smoother_for_mesh_node_normal(mesh, 0.5, None, 1, update_node_normal=False)
    assert torch.equal(held, before) and held is not mesh.node_normal
    assert not torch.equal(mesh.node_normal, before)
    assert torch.allclose(mesh.node_normal.norm(dim=1), torch.ones(16, dtype=F))

    mesh = grid_tri()
    mesh.update_element_normal()
    held, before = mesh.element_normal, mesh.element_normal.clone()
    pmp.simple_smoother_for_mesh_element_normal(mesh, 0.5, None, 1, update_element_normal=False)
    assert torch.equal(held, before) and not torch.equal(mesh.element_normal, before)

    plain = PolygonMesh(T(TRI), [[0, 1, 2]])
    with pytest.raises(TypeError, match='plain PolygonMesh'):
        pmp.simple_smoother_for_mesh_node_normal(plain, 0.5, None, 1)
    with pytest.raises(TypeError, match='plain PolygonMesh'):
        pmp.simple_smoother_for_mesh_element_normal(plain, 0.5, None, 1)
    with pytest.raises(ValueError, match='node_normal is None'):
        pmp.simple_smoother_for_mesh_node_normal(grid_tri(), 0.5, None, 1, update_node_normal=False)
    with pytest.raises(ValueError, match='element_normal is None'):
        pmp.simple_smoother_for_mesh_element_normal(grid_tri(), 0.5, None, 1, update_element_normal=False)
    for make in (grid_tri, grid_quad, tri6, quad_tri):
        pmp.simple_smoother_for_mesh_node_normal(make(), 0.5, None, 1)
        pmp.simple_smoother_for_mesh_element_normal(make(), 0.5, None, 1)


def test_smooth_and_project_takes_every_mask_type_simple_smoother_takes():
    def run(mask, smooth_first=True):
        move = grid_tri()
        move.node[:, 2] += 0.3
        pmp.smooth_and_project(move, grid_tri(6), 0.5, mask, 1, 1, smooth_first=smooth_first)
        return move.node
    reference = run(torch.ones(16))
    for mask in (None, [1.] * 16, np.ones(16), 1, 1.0, torch.ones(16, 1)):
        assert torch.equal(run(mask), reference)
        assert torch.equal(run(mask, False), run(torch.ones(16), False))
    # nodes with mask 0 stay
    mask = np.zeros(16)
    mask[5] = 1
    before = grid_tri().node
    before[:, 2] += 0.3
    moved = run(mask, False)
    assert torch.equal(moved[np.arange(16) != 5], before[np.arange(16) != 5]) and not torch.equal(moved[5], before[5])
    with pytest.raises(ValueError):
        run('bad mask')


def test_subdivide_uses_update_with_a_capital():
    assert '.Update()' in inspect.getsource(pmp.subdivide) and '.update()' not in inspect.getsource(pmp.subdivide)
    assert len(pmp.subdivide(grid_tri(), 1, 'loop').element) == 72


def test_trace_boundary_direction_and_end_node(capsys):
    node = T(SQUARE)
    direction = set()
    default_direction = set()

    def signed_area(n, curve):
        p = n[curve][:, :2]
        q = p.roll(-1, 0)
        return float((p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]).sum())
    for perm in itertools.permutations(range(4)):
        new_node = node[[perm.index(i) for i in range(4)]]
        mesh = TriangleMesh(new_node, [[perm[0], perm[1], perm[2]], [perm[0], perm[2], perm[3]]])
        curve = pmp.trace_mesh_boundary_curve(mesh, 0, follow_element_orientation=True)
        direction.add(signed_area(new_node, curve) > 0)
        default_direction.add(signed_area(new_node, pmp.trace_mesh_boundary_curve(mesh, 0)) > 0)
        loops = pmp.find_mesh_boundary_curve(mesh, follow_element_orientation=True)
        assert signed_area(new_node, loops[0]) > 0
    assert direction == {True} and default_direction == {True, False}
    # the default keeps the old direction, which comes from the first boundary edge found
    assert pmp.trace_mesh_boundary_curve(TriangleMesh(node, [[0, 1, 2], [0, 2, 3]]), 0) == [0, 1, 2, 3]
    assert pmp.trace_mesh_boundary_curve(TriangleMesh(node, [[0, 3, 2], [0, 2, 1]]), 0) == [0, 1, 2, 3]
    assert pmp.trace_mesh_boundary_curve(TriangleMesh(node, [[0, 3, 2], [0, 2, 1]]), 0, follow_element_orientation=True) == [0, 3, 2, 1]
    assert pmp.trace_mesh_boundary_curve(TriangleMesh(node, [[0, 1, 2], [0, 2, 3]]), 0, next_node_idx=3, follow_element_orientation=True) == [0, 3, 2, 1]
    with pytest.raises(NotImplementedError):
        pmp.trace_mesh_boundary_curve(tri6(), 0, follow_element_orientation=True)
    # an end node on another loop is reported
    B = 8
    theta = torch.arange(B, dtype=F) * 2 * math.pi / B
    ring = torch.cat([r * torch.stack([torch.cos(theta), torch.sin(theta), 0 * theta], 1) for r in (1, 2)])
    element = []
    for k in range(B):
        a, b = k, (k + 1) % B
        element += [[a, a + B, b + B], [a, b + B, b]]
    annulus = TriangleMesh(ring, element)
    capsys.readouterr()
    pmp.trace_mesh_boundary_curve(annulus, 0, end_node_idx=2)
    assert 'warning' not in capsys.readouterr().out
    assert 2 + B not in pmp.trace_mesh_boundary_curve(annulus, 0, end_node_idx=2 + B)
    assert 'warning' in capsys.readouterr().out


def test_curvature_has_every_mode_and_works_on_quads():
    mesh = sphere()
    expected = {'mean': 0.5, 'gaussian': 0.25, 'Gauss_Curvature': 0.25, 'max': 0.5, 'Maximum_Curvature': 0.5, 'min': 0.5}
    for name, value in expected.items():
        assert float(pmp.cal_curvature(mesh, name).median()) == pytest.approx(value, rel=0.06)
    with pytest.raises(ValueError):
        pmp.cal_curvature(mesh, 'unknown')
    # a latitude-longitude sphere of quads, away from its open ends
    nt, np_ = 48, 24
    theta = torch.arange(nt, dtype=F) * 2 * math.pi / nt
    phi = (torch.arange(np_, dtype=F) + 1) * math.pi / (np_ + 1)
    node = torch.stack([2 * torch.outer(torch.sin(phi), torch.cos(theta)), 2 * torch.outer(torch.sin(phi), torch.sin(theta)),
                        2 * torch.outer(torch.cos(phi), torch.ones(nt, dtype=F))], -1).reshape(-1, 3)
    element = [[i * nt + j, (i + 1) * nt + j, (i + 1) * nt + (j + 1) % nt, i * nt + (j + 1) % nt] for i in range(np_ - 1) for j in range(nt)]
    quad = QuadMesh(node, element)
    band = (torch.arange(np_ // 2 - 2, np_ // 2 + 2)[:, None] * nt + torch.arange(nt)[None]).flatten()
    for name, value in (('mean', 0.5), ('gaussian', 0.25)):
        curvature = pmp.cal_curvature(quad, name)
        assert curvature.shape == (len(node),)
        assert float(curvature[band].median()) == pytest.approx(value, rel=0.01)
    # no cells: one zero per node
    assert pmp.cal_curvature(TriangleMesh(T(TRI), torch.empty((0, 3), dtype=torch.int64))).tolist() == [0, 0, 0]


def test_clip_with_two_outputs_keeps_only_used_nodes():
    mesh = grid_tri(9)
    mesh.node_data['h'] = mesh.node[:, :1].clone()
    plane = ([.51, .5, 0], [1, 0, 0])
    one = pmp.clip_mesh_by_plane(mesh, *plane)
    kept, other = pmp.clip_mesh_by_plane(mesh, *plane, return_clipped_output=True)
    assert torch.equal(kept.node, one.node) and [list(map(int, e)) for e in kept.element] == [list(map(int, e)) for e in one.element]
    assert len(other.node) == len({int(i) for e in other.element for i in e}) < len(mesh.node)
    one = pmp.clip_mesh_by_node_attribute(mesh, .51, 'h')
    kept, other = pmp.clip_mesh_by_node_attribute(mesh, .51, 'h', return_clipped_output=True)
    assert torch.equal(kept.node, one.node) and len(kept.node_data['h']) == len(kept.node)
    assert len(other.node) == len({int(i) for e in other.element for i in e})
    # one side empty
    kept, other = pmp.clip_mesh_by_plane(mesh, [5, .5, 0], [1, 0, 0], return_clipped_output=True)
    assert kept.node is None and len(other.element) == len(mesh.element)
    kept, other = pmp.clip_mesh_by_node_attribute(mesh, 5., 'h', return_clipped_output=True)
    assert other.node is None and len(kept.element) == len(mesh.element)


def test_functions_for_linear_elements_reject_quadratic_ones():
    mesh = tri6()
    for call in (lambda: pmp.cut_mesh_by_curve(mesh, TRI, [.1, .1, 0]),
                 lambda: pmp.clip_mesh_by_plane(mesh, [.5, 0, 0], [1, 0, 0]),
                 lambda: pmp.clip_mesh_by_node_attribute(mesh, 0., 'h'),
                 lambda: pmp.convert_polygon_mesh_to_triangle_mesh(mesh),
                 lambda: pmp.cal_curvature(mesh),
                 lambda: pmp.fill_hole(mesh, 1.),
                 lambda: pmp.subdivide(mesh, 1)):
        with pytest.raises(ValueError, match='linear elements'):
            call()
    # cut still works on linear elements, and gives the cap of a sphere
    surface = sphere(1.0)
    theta = torch.arange(24, dtype=F) * 2 * math.pi / 24
    r = math.sin(math.pi / 3)
    curve = torch.stack([r * torch.cos(theta), r * torch.sin(theta), 0.5 + 0 * theta], 1)
    for straight_cut in (False, True):
        cap = pmp.cut_mesh_by_curve(surface, curve, [0, 0, 1], straight_cut=straight_cut)
        triangle = pmp.convert_polygon_mesh_to_triangle_mesh(cap)
        area = float(TriangleMesh.cal_element_area_and_normal(triangle.node, triangle.element)[0].sum())
        assert area == pytest.approx(math.pi, rel=0.02)


def test_empty_vtk_outputs_are_empty_meshes():
    mesh = TriangleMesh(T(TRI), torch.empty((0, 3), dtype=torch.int64))
    for output in (pmp.fill_hole(mesh, 1.), pmp.subdivide(mesh, 1)):
        assert type(output) is PolygonMesh and output.node is None and output.element is None


def test_surface_node_mapping_and_mixed_polyhedron_mesh():
    for mesh, kind in ((hex_block(), QuadMesh), (tet(), TriangleMesh), (tet10(), Tri6Mesh)):
        surface, node_idx_list = vp.extract_surface_mesh(mesh, return_node_idx_list=True)
        assert type(surface) is kind
        assert torch.equal(surface.node, mesh.node[node_idx_list])
        assert torch.equal(mesh.node[node_idx_list[surface.element]], mesh.node[torch.tensor(vp.extract_surface_element(mesh))])
        assert type(vp.extract_surface_mesh(mesh)) is kind
    # a plain PolyhedronMesh: all tets, all hexes, and a mix, with faces pointing outward
    block = hex_block()
    tets = tet()
    assert sorted(map(tuple, vp.extract_surface_element(PolyhedronMesh(block.node, block.element)))) == sorted(map(tuple, vp.extract_surface_element(block)))
    assert sorted(map(tuple, vp.extract_surface_element(PolyhedronMesh(tets.node, tets.element)))) == sorted(map(tuple, vp.extract_surface_element(tets)))
    one = hex_block(1, 1, 1)
    node = torch.cat([one.node, T([[.5, .5, 2]])])
    # the tet stands on the hex's top face without sharing a whole face with it
    mixed = PolyhedronMesh(node, [one.element[0].tolist(), [4, 5, 6, 8]], np.array(['HEX8', 'TET4']))
    surface, node_idx_list = vp.extract_surface_mesh(mixed, return_node_idx_list=True)
    assert type(surface) is PolygonMesh and sorted(len(e) for e in surface.element) == [3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
    assert len(surface.node) == 9 and torch.equal(surface.node, node[torch.as_tensor(node_idx_list)])
    with pytest.raises(NotImplementedError, match='6 nodes'):
        vp.extract_surface_element(PolyhedronMesh(T(SQUARE + [[0, 0, 1], [1, 0, 1]]), [[0, 1, 2, 3, 4, 5]]))
    # no elements
    empty = TetrahedronMesh(T(TRI), torch.empty((0, 4), dtype=torch.int64))
    assert vp.extract_surface_mesh(empty).element is None
