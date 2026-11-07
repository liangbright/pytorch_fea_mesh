"""Fixes for the open bugs in Mesh_classes_bug_fix_2026-11-6_new_bugs_table.md. Each test names its bug."""
import inspect
import json

import numpy as np
import pytest
import torch

from torch_fea_mesh.mesh.hexahedron_mesh import HexahedronMesh
from torch_fea_mesh.mesh.mesh import Mesh
from torch_fea_mesh.mesh.polygon_mesh import PolygonMesh
from torch_fea_mesh.mesh.polyhedron_mesh import PolyhedronMesh
from torch_fea_mesh.mesh.polyline_mesh import PolylineMesh
from torch_fea_mesh.mesh.quad_mesh import QuadMesh
from torch_fea_mesh.mesh.quad_triangle_mesh import QuadTriangleMesh
from torch_fea_mesh.mesh.tetrahedron_mesh import TetrahedronMesh
from torch_fea_mesh.mesh.tri6_mesh import Tri6Mesh
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh

F32, F64 = torch.float32, torch.float64
SQUARE = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=F64)
MIXED_NODE = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], dtype=F64)
MIXED = [[0, 1, 2, 3], [1, 4, 2]]
TET_NODE = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=F64)
HEX_NODE = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=F64)
TRI6_NODE = torch.tensor([[0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=F64)
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')


def normal_length(normal):
    return torch.linalg.vector_norm(normal).item()


#%% bug-2
@needs_cuda
def test_bug2_list_element_goes_on_the_device_of_the_node():
    node = SQUARE.cuda()
    triangle = TriangleMesh(node, [[0, 1, 2], [0, 2, 3]])
    quad = QuadMesh(node, [[0, 1, 2, 3]])
    tri6 = Tri6Mesh(TRI6_NODE.cuda(), [[0, 1, 2, 3, 4, 5]])
    from_tri3 = Tri6Mesh()
    from_tri3.create_from_tri3(triangle)
    copied = TriangleMesh()
    copied.copy(node, [[0, 1, 2]])
    for mesh in (triangle, TriangleMesh(node, np.array([[0, 1, 2]])), copied, triangle.subdivide_to_tri_1to4(),
                 triangle.subdivide_to_tri_1to3(), triangle.subdivide_to_quad(), quad.subdivide_to_quad(),
                 quad.quad_to_tri(), tri6.subdivide_to_tri3(), from_tri3):
        assert mesh.element.is_cuda
    triangle.subdivide_to_tri_1to4().update_node_normal()
    triangle.subdivide_to_tri_1to4().sample_point(5)
    tri6.subdivide_to_tri3().update_node_normal()
    from_tri3.update_node_normal()
    # a mixed mesh stays a list, and its tables must follow the node
    mixed = QuadTriangleMesh(MIXED_NODE.cuda(), MIXED)
    expected = QuadTriangleMesh(MIXED_NODE, MIXED)
    mixed.update_node_normal()
    expected.update_node_normal()
    assert torch.equal(mixed.node_normal.cpu(), expected.node_normal)
    assert mixed.quad_element.is_cuda and mixed.tri_element.is_cuda


#%% bug-6, bug-7, bug-8
def test_bug6_get_sub_mesh_rejects_a_mask():
    rectangular = TriangleMesh(SQUARE, [[0, 1, 2], [0, 2, 3]])
    ragged = PolygonMesh(MIXED_NODE, MIXED)
    for mesh in (rectangular, ragged):
        for mask in ([True, False], torch.tensor([True, False]), np.array([True, False]), [np.True_, np.False_],
                     [torch.tensor(True), torch.tensor(False)]):
            with pytest.raises(ValueError, match='boolean mask'):
                mesh.get_sub_mesh(mask)


def test_bug7_get_sub_mesh_takes_any_form_of_index_on_a_labeled_mesh():
    mesh = PolyhedronMesh(TET_NODE, [[0, 1, 2, 3], [1, 2, 3, 4]], element_type=['C3D4', 'TET4'])
    for index in ([1], (1,), range(1, 2), torch.tensor([1]), np.array([1]), [np.int64(1)], [torch.tensor(1)]):
        sub_mesh, node_idx_list = mesh.get_sub_mesh(index, return_node_idx_list=True)
        assert sub_mesh.element.tolist() == [[0, 1, 2, 3]], index
        assert sub_mesh.element_type.tolist() == ['TET4'], index
        assert list(node_idx_list) == [1, 2, 3, 4], index
    assert mesh.get_sub_mesh(torch.tensor([0, 1])).element_type.tolist() == ['C3D4', 'TET4']
    assert mesh.get_sub_mesh(torch.empty(0, dtype=torch.int64)).element is None
    with pytest.raises(ValueError, match='1D'):
        mesh.get_sub_mesh(torch.tensor([[0, 1]]))
    with pytest.raises(TypeError):
        mesh.get_sub_mesh([0.5])
    if torch.cuda.is_available():
        index = torch.tensor([1]).cuda()
        assert mesh.get_sub_mesh(index).element_type.tolist() == ['TET4']
        on_cuda = PolyhedronMesh(TET_NODE.cuda(), torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]).cuda(), element_type=['C3D4', 'TET4'])
        assert on_cuda.get_sub_mesh(index).element_type.tolist() == ['TET4']


def test_bug8_docstring_says_what_get_sub_mesh_drops():
    for name in ('node_data', 'element_data', 'node_set', 'element_set', 'mesh_data', 'name', 'boolean mask'):
        assert name in Mesh.get_sub_mesh.__doc__
    assert 'node_data' in inspect.getdoc(TriangleMesh.get_sub_mesh)


#%% bug-10, bug-11
def test_bug10_element_with_a_repeated_node_has_no_self_link():
    for mesh, adj in ((PolylineMesh(TET_NODE[:3], [[0, 1, 2, 0]]), 'node'), (HexahedronMesh(HEX_NODE, [[0, 0, 2, 3, 4, 5, 6, 7]]), 'node'),
                      (PolylineMesh(TET_NODE[:2], [[0, 1, 0]]), 'edge'), (HexahedronMesh(HEX_NODE, [[0, 1, 2, 3, 0, 1, 2, 3]]), 'face')):
        mesh.build_element_to_element_adj_link(adj)
        assert mesh.element_to_element_adj_link[adj].shape == (0, 2), adj
    neighbours = TriangleMesh(SQUARE, [[0, 1, 2], [0, 2, 3]])
    neighbours.build_element_to_element_adj_link('node')
    assert neighbours.element_to_element_adj_link['node'].tolist() == [[0, 1], [1, 0]]


def test_bug11_polyline_edge_adjacency_table():
    mesh = PolylineMesh(TET_NODE, [[0, 1, 2], [1, 2, 3], [3, 4]])
    assert mesh.element_to_element_adj_table['edge'] is None
    mesh.build_element_to_element_adj_table('edge')
    assert mesh.element_to_element_adj_table['edge'] == [[1], [0], []]


#%% bug-12
def test_bug12_copy_converts_like_the_constructor():
    ragged = np.empty(2, dtype=object)
    ragged[0], ragged[1] = [0, 1, 2, 3], [1, 4, 2]
    triangle = [[0, 1, 2]]
    for node, element, node_dtype in ((np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), triangle, F32),      # integer numpy node
                                      (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32), triangle, F32),
                                      (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64), np.array(triangle, dtype=np.int32), F64),
                                      (SQUARE.tolist(), torch.tensor(triangle, dtype=torch.int32), F64),  # int32 element
                                      (MIXED_NODE.numpy(), ragged, F64),                                  # ragged object array
                                      (MIXED_NODE, MIXED, F64),
                                      (torch.tensor([[0, 0, 0]]), [[0]], torch.int64),                    # integer tensor node
                                      (torch.empty(0), [], torch.get_default_dtype()), ([], [], F64)):
        built = PolygonMesh(node, element)
        copied = PolygonMesh(SQUARE, [[0, 1, 2]])
        copied.copy(node, element)
        assert built.node.dtype == node_dtype and copied.node.dtype == node_dtype
        assert torch.equal(copied.node, built.node)
        if torch.is_tensor(built.element):
            assert built.element.dtype == torch.int64
        assert type(copied.element) is type(built.element)
        if torch.is_tensor(built.element):
            assert copied.element.dtype == built.element.dtype and torch.equal(copied.element, built.element)
        else:
            assert copied.element == built.element
    # the same errors, and a failed copy keeps the mesh
    mesh = PolygonMesh(SQUARE, [[0, 1, 2]])
    for node, element, message in (({1: 2}, [[0, 1, 2]], 'type of node'), (SQUARE, {1: 2}, 'type of element'),
                                   (SQUARE[0], [[0, 1, 2]], 'node.shape'), (SQUARE, [0, 1, 2], 'element.shape')):
        for call in (lambda: PolygonMesh(node, element), lambda: mesh.copy(node, element)):
            with pytest.raises(ValueError, match=message):
                call()
    assert mesh.element.tolist() == [[0, 1, 2]]


def test_bug12_copy_is_independent_and_keeps_dtype_and_detach():
    node = SQUARE.clone().requires_grad_()
    element = torch.tensor([[0, 1, 2]])
    mesh = PolygonMesh()
    mesh.copy(node, element)
    assert mesh.node is not node and not mesh.node.requires_grad
    assert not np.shares_memory(mesh.node.numpy(), node.detach().numpy())
    assert not np.shares_memory(mesh.element.numpy(), element.numpy())
    mesh.copy(node, element, detach=False)
    assert mesh.node.requires_grad
    mesh.copy(node, element, dtype=F32)
    assert mesh.node.dtype == F32
    rows = [[0, 1, 2, 3], [1, 4, 2]]
    mesh.copy(MIXED_NODE, rows)
    assert mesh.element == rows and mesh.element is not rows and mesh.element[0] is not rows[0]
    # dtype comes first, so a large integer isn't rounded through float32
    large = np.array([[2**24+1, 0, 0]])
    mesh.copy(large, [[0]], dtype=F64)
    assert mesh.node[0, 0].item() == 2**24+1 and mesh.node.dtype == F64
    mesh.copy(large.tolist(), [[0]], dtype=F32)
    assert mesh.node.dtype == F32


#%% bug-13, bug-14
def test_bug13_torch_file_keeps_the_name(tmp_path):
    path = str(tmp_path / 'mesh.pt')
    mesh = TriangleMesh(SQUARE, [[0, 1, 2]])
    mesh.name = 'abc'
    mesh.save_as_torch(path)
    restored = TriangleMesh()
    restored.load_from_torch(path)
    assert restored.name == 'abc'
    # a file from before this fix has no name
    torch.save({'node': SQUARE, 'element': torch.tensor([[0, 1, 2]])}, path)
    restored.load_from_torch(path)
    assert restored.name == ''


def test_bug14_saved_mesh_type_is_used_only_if_it_says_more(tmp_path):
    path = str(tmp_path / 'mesh.pt')
    tri6 = Tri6Mesh(TRI6_NODE, [[0, 1, 2, 3, 4, 5]])
    tri6.save_as_torch(path, save_adj_info=False)
    plain = PolygonMesh()
    plain.load_from_torch(path)
    assert plain.mesh_type == 'polygon_tri6'
    plain.build_edge()
    tri6.build_edge()
    assert torch.equal(plain.edge, tri6.edge)
    # the real files are saved as polyhedron and loaded in a HexahedronMesh, which must still save a hex
    PolyhedronMesh(HEX_NODE, [list(range(8))]).save_as_torch(path)
    hexahedron = HexahedronMesh()
    hexahedron.load_from_torch(path)
    assert hexahedron.mesh_type == 'polyhedron_hex8'
    hexahedron.save_as_vtk(str(tmp_path / 'hex.vtk'))
    HexahedronMesh(HEX_NODE, [list(range(8))]).save_as_torch(path)
    solid = PolyhedronMesh()
    solid.load_from_torch(path)
    assert solid.mesh_type == 'polyhedron_hex8'
    solid.save_as_vtk(str(tmp_path / 'solid.vtk'))
    for owner, expected in ((TriangleMesh, 'polygon_tri3'), (PolylineMesh, 'polyline'), (QuadTriangleMesh, 'polygon_quad4_tri3')):
        TriangleMesh(SQUARE, [[0, 1, 2]]).save_as_torch(path)
        if owner is PolylineMesh:
            TetrahedronMesh(TET_NODE, [[0, 1, 2, 3]]).save_as_torch(path)
        mesh = owner()
        mesh.load_from_torch(path)
        assert mesh.mesh_type == expected
    torch.save({'node': SQUARE, 'element': torch.tensor([[0, 1, 2]]), 'mesh_type': None}, path)
    legacy = PolygonMesh()
    legacy.load_from_torch(path)
    assert legacy.mesh_type == 'polygon'


#%% bug-15
def test_bug15_docstrings_give_the_small_element_limit():
    node, element = lambda side: SQUARE*side, torch.tensor([[0, 1, 2]])
    quad = torch.tensor([[0, 1, 2, 3]])
    tri6 = torch.tensor([[0, 1, 2, 3, 4, 5]])
    assert 'will not work for very small triangles' in TriangleMesh.cal_element_area_and_normal.__doc__
    assert 'will not work for very small quads' in QuadMesh.cal_element_area_and_normal.__doc__
    assert 'will not work for very small quads' in QuadMesh.cal_element_normal.__doc__
    assert 'will not work for very small triangles' in Tri6Mesh.cal_element_area_and_normal.__doc__
    # the numbers in the docstrings
    assert normal_length(TriangleMesh.cal_element_area_and_normal(node(1e-6), element)[1]) == pytest.approx(1, abs=1e-6)
    assert normal_length(TriangleMesh.cal_element_area_and_normal(node(1e-7), element)[1]) == pytest.approx(.01)
    assert normal_length(QuadMesh.cal_element_normal(node(2e-6), quad)) == pytest.approx(1)
    assert normal_length(QuadMesh.cal_element_normal(node(1e-6), quad)) == pytest.approx(.25)
    assert normal_length(QuadMesh.cal_element_normal(node(1e-7), quad)) == pytest.approx(.0025)
    # TRI6_NODE has side 2
    assert normal_length(Tri6Mesh.cal_element_area_and_normal(TRI6_NODE*1e-6, tri6)[1]) == pytest.approx(1)
    assert normal_length(Tri6Mesh.cal_element_area_and_normal(TRI6_NODE*.5e-6, tri6)[1]) == pytest.approx(.25)
    assert normal_length(Tri6Mesh.cal_element_area_and_normal(TRI6_NODE*.5e-7, tri6)[1]) == pytest.approx(.0025)
    assert normal_length(TriangleMesh.cal_element_area_and_normal(node(1e-7), element, eps=1e-30)[1]) == pytest.approx(1)
    assert normal_length(QuadMesh.cal_element_normal(node(1e-7), quad, eps=1e-30)) == pytest.approx(1)


#%% bug-16
def test_bug16_quad_area_keeps_float32_digits_at_a_large_offset():
    # integer coordinates, so that float32 stores them exactly and only the sums round
    torch.manual_seed(0)
    element = torch.tensor([[0, 1, 2, 3]])
    corner = torch.tensor([[0, 0, 0], [8, 0, 0], [8, 8, 0], [0, 8, 0]], dtype=F64)
    worst = 0
    for _ in range(200):
        x = (corner+torch.randint(-3, 4, (4, 3), dtype=F64)+2.0**23)
        area32 = QuadMesh.cal_element_area_and_normal(x.to(F32), element)[0].item()
        area64 = QuadMesh.cal_element_area_and_normal(x, element)[0].item()
        worst = max(worst, abs(area32-area64)/area64)
    assert worst < 1e-5


#%% bug-17, bug-18, bug-19
def test_bug17_load_from_vtk_can_return_the_vtk_mesh(tmp_path):
    path = str(tmp_path / 'mesh.vtk')
    QuadTriangleMesh(MIXED_NODE, MIXED).save_as_vtk(path)
    mesh = QuadTriangleMesh()
    assert mesh.load_from_vtk(path, F64) is None
    assert mesh.load_from_vtk(path, F64, return_mesh_vtk=True).GetNumberOfPoints() == 5
    assert mesh.element == MIXED and mesh.quad_element.tolist() == [MIXED[0]]


def test_bug18_static_element_normal():
    normal = QuadTriangleMesh.cal_element_normal(MIXED_NODE, MIXED)
    assert torch.equal(normal, QuadTriangleMesh.cal_element_area_and_normal(MIXED_NODE, MIXED)[1])
    assert normal.tolist() == [[0, 0, 1], [0, 0, 1]]


def test_bug19_third_argument_of_corner_angle_is_return_cos():
    quad, tri = torch.tensor([MIXED[0]]), torch.tensor([MIXED[1]])
    for return_cos in (False, True):
        angle_quad, angle_tri = QuadTriangleMesh.cal_element_corner_angle(MIXED_NODE, MIXED, return_cos)
        assert torch.equal(angle_quad, QuadMesh.cal_element_corner_angle(MIXED_NODE, quad, return_cos))
        assert torch.equal(angle_tri, TriangleMesh.cal_element_corner_angle(MIXED_NODE, tri, return_cos))
    assert angle_quad.abs().max().item() < 1e-12 # cos of 90 degrees
    QuadTriangleMesh.cal_element_corner_angle(MIXED_NODE, MIXED, eps=1e-9)


#%% bug-20
def test_bug20_element_rows_may_be_tensors_or_numpy_arrays():
    tensor_rows = [torch.tensor(row) for row in MIXED]
    numpy_rows = [np.array(row) for row in MIXED]
    expected = QuadTriangleMesh(MIXED_NODE, MIXED)
    expected.update_element_area_and_normal()
    for rows in (tensor_rows, numpy_rows, tuple(tensor_rows), [MIXED[0], tensor_rows[1]]):
        mesh = QuadTriangleMesh(MIXED_NODE, rows)
        assert mesh.element == MIXED
        mesh.update_element_area_and_normal()
        assert torch.equal(mesh.element_area, expected.element_area)
        copied = QuadTriangleMesh()
        copied.copy(MIXED_NODE, rows)
        assert copied.element == MIXED and copied.quad_element.tolist() == [MIXED[0]]
        assert PolygonMesh(MIXED_NODE, rows).element == MIXED
    # rows of one length give a tensor, on every class
    for cls, rows in ((TriangleMesh, [[0, 1, 2], [0, 2, 3]]), (QuadMesh, [[0, 1, 2, 3]])):
        for form in ([torch.tensor(row) for row in rows], list(torch.tensor(rows)), [np.array(row) for row in rows]):
            mesh = cls(SQUARE, form)
            assert torch.is_tensor(mesh.element) and mesh.element.tolist() == rows
            mesh.update_element_area_and_normal()
    # assigned by hand
    mesh = QuadTriangleMesh(MIXED_NODE, MIXED)
    mesh.element = tensor_rows
    mesh.classify_element()
    assert mesh.quad_element.tolist() == [MIXED[0]] and mesh.tri_element.tolist() == [MIXED[1]]
    mesh = TriangleMesh(SQUARE, [[0, 1, 2]])
    mesh.element = [torch.tensor([0, 1, 2]), torch.tensor([0, 2, 3])]
    assert mesh.subdivide_to_tri_1to3().element.shape == (6, 3)
    # the wrong width is still rejected, and plain rows are not copied
    with pytest.raises(ValueError, match='must have 3 nodes'):
        TriangleMesh(SQUARE, [torch.tensor([0, 1, 2, 3])])
    plain = [[0, 1, 2, 3], [1, 4, 2]]
    assert PolygonMesh(MIXED_NODE, plain).element is plain
    if torch.cuda.is_available():
        mesh = TriangleMesh(SQUARE.cuda(), [torch.tensor([0, 1, 2]).cuda()])
        assert mesh.element.is_cuda


#%% read_mesh_vtk_slow
def test_read_mesh_vtk_rejects_other_vtk_classes():
    vtk = pytest.importorskip('vtk')
    assert not hasattr(Mesh, 'read_mesh_vtk_slow') and not hasattr(QuadTriangleMesh, 'read_mesh_vtk_slow')
    grid = vtk.vtkImageData()
    grid.SetDimensions(2, 2, 2)
    mesh = PolyhedronMesh(TET_NODE, [[0, 1, 2, 3]])
    with pytest.raises(ValueError, match='vtkImageData'):
        mesh.read_mesh_vtk(grid, F64)
    assert mesh.element.tolist() == [[0, 1, 2, 3]]


#%% bug-21
def test_bug21_failed_load_keeps_the_mesh(tmp_path):
    pentagon = PolygonMesh(torch.rand(5, 3, dtype=F64), [[0, 1, 2, 3, 4]])
    vtk_file, vtp_file, torch_file = (str(tmp_path / name) for name in ('p.vtk', 'p.vtp', 'p.pt'))
    pentagon.save_as_vtk(vtk_file)
    pentagon.save_as_vtp(vtp_file)
    pentagon.save_as_torch(torch_file)
    mesh_vtk = PolygonMesh().load_from_vtk(vtk_file, F64, return_mesh_vtk=True)
    mesh = QuadTriangleMesh(MIXED_NODE, MIXED)
    mesh.update_element_area_and_normal()
    mesh.name = 'keep'
    area = mesh.element_area
    calls = [lambda: mesh.load_from_vtk(vtk_file, F64), lambda: mesh.load_from_vtp(vtp_file, F64),
             lambda: mesh.load_from_torch(torch_file), lambda: mesh.copy(pentagon.node, pentagon.element),
             lambda: mesh.read_mesh_vtk(mesh_vtk, F64)]
    for call in calls:
        with pytest.raises(ValueError):
            call()
        assert mesh.element == MIXED and mesh.name == 'keep'
        assert mesh.quad_element.tolist() == [MIXED[0]] and mesh.tri_element.tolist() == [MIXED[1]]
        assert mesh.element_area is area and torch.equal(mesh.node, MIXED_NODE)
    with pytest.raises(ValueError, match='3 or 4 nodes'):
        QuadTriangleMesh(pentagon.node, pentagon.element)


#%% bug-22
def test_bug22_mdk_json_takes_tensor_and_numpy_sets(tmp_path):
    path = str(tmp_path / 'mesh.json')
    mesh = PolygonMesh(SQUARE, [[0, 1, 2], [0, 2, 3]])
    mesh.node_set = {'tensor': torch.tensor([0, 1]), 'numpy': np.array([2, 3]), 'list': [0], 'numbers': [np.int64(1)], 'tuple': (2,)}
    mesh.element_set = {'tensor': torch.tensor([1]), 'numpy': np.array([0])}
    mesh.save_as_mdk_json(path)
    restored = PolygonMesh()
    restored.load_from_mdk_json(path, F64)
    assert restored.node_set == {'tensor': [0, 1], 'numpy': [2, 3], 'list': [0], 'numbers': [1], 'tuple': [2]}
    assert restored.element_set == {'tensor': [1], 'numpy': [0]}
    # a failure leaves the old file whole
    before = open(path).read()
    mesh.name = object()
    with pytest.raises(TypeError):
        mesh.save_as_mdk_json(path)
    assert open(path).read() == before
    json.loads(before)


#%% bug-23
def test_bug23_curve_keeps_its_gradient_and_is_not_shared():
    curve = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=F64, requires_grad=True)
    for form in (curve, [curve, curve], list(curve), torch.stack([curve, curve])):
        mesh = PolylineMesh.from_curve(form)
        assert mesh.node.requires_grad
        curve.grad = None
        mesh.node.sum().backward()
        assert curve.grad is not None
    plain = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=F64)
    mesh = PolylineMesh.from_curve(plain)
    mesh.node[0, 0] = 5
    assert plain[0, 0].item() == 0 and not mesh.node.requires_grad
    with pytest.raises(ValueError, match='same number of coordinates'):
        PolylineMesh.from_curve([plain, plain[:, :2]])


@needs_cuda
def test_bug23_curve_stays_on_its_device():
    curve = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=F64).cuda()
    other = np.array([[5, 0, 0], [6, 0, 0]], dtype=np.float32)
    for form, n_node in ((curve, 3), ([curve, curve], 6), (list(curve), 3), ([curve, other], 5), ([other, curve], 5),
                         ([other.tolist(), curve], 5), (torch.stack([curve, curve]), 6)):
        mesh = PolylineMesh.from_curve(form)
        assert mesh.node.is_cuda and mesh.node.dtype == F64 and mesh.node.shape == (n_node, 3)
    assert PolylineMesh.from_curve([other, curve]).element == [[0, 1], [2, 3], [3, 4]]


#%% bug-24
def test_bug24_boundary_edge_does_not_search_a_list(monkeypatch):
    class NoSearch(list):
        def __contains__(self, item):
            raise AssertionError('searching a list makes this quadratic')
    mesh = PolylineMesh(TET_NODE, [[0, 1], [1, 2], [3, 4]])
    found = PolylineMesh.find_boundary_node(mesh)
    monkeypatch.setattr(PolylineMesh, 'find_boundary_node', lambda self: NoSearch(found))
    assert mesh.find_boundary_edge() == [0, 1, 2]
    boundary_node, boundary_edge = mesh.find_boundary_node_and_edge()
    assert list(boundary_node) == [0, 2, 3, 4] and boundary_edge == [0, 1, 2]
