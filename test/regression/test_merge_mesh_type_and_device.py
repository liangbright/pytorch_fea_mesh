"""merge_mesh: the mesh_type and the device of the output.

mesh_type: load_from_torch can give a plain PolygonMesh or PolyhedronMesh a mesh_type that says more,
such as polygon_tri6 or polyhedron_hex8. The node order and the VTK cell type depend on it, so a merged
mesh that lost it would get wrong edges or fail to save. When the two inputs have different mesh_types,
the result is the same as for the classes they came from.

device: the two inputs can be on different devices, and the output is on the device of meshA.node.
"""
import numpy as np
import pytest
import torch

from torch_fea_mesh.mesh.hexahedron_mesh import HexahedronMesh
from torch_fea_mesh.mesh.polygon_mesh import PolygonMesh
from torch_fea_mesh.mesh.polyhedron_mesh import PolyhedronMesh
from torch_fea_mesh.mesh.quad_mesh import QuadMesh
from torch_fea_mesh.mesh.quad_triangle_mesh import QuadTriangleMesh
from torch_fea_mesh.mesh.tet10_mesh import Tet10Mesh
from torch_fea_mesh.mesh.tetrahedron_mesh import TetrahedronMesh
from torch_fea_mesh.mesh.tri6_mesh import Tri6Mesh
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh
from torch_fea_mesh.processing.mesh_processing import merge_mesh
from torch_fea_mesh.processing.polygon_mesh_processing import merge_mesh_on_boundary

pytestmark = pytest.mark.processing

F32, F64 = torch.float32, torch.float64

GRID_NODE = [[0, 0, 0], [1, 0, 0], [3, 0, 0], [0, 1, 0], [1, 1, 0], [3, 1, 0]]
HEXAGON_NODE = [[0, 0, 0], [2, 0, 0], [3, 1, 0], [2, 2, 0], [0, 2, 0], [-1, 1, 0]]
HEXAGON_ELEMENT = [list(range(6))]
# Three right triangles with legs of 2, each with an area of 2. Corners first, then midpoints.
# The second shares an edge with the first, and the third shares an edge with the second.
TRI6_FIRST = [[0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
TRI6_SECOND = [[2, 0, 0], [2, 2, 0], [0, 2, 0], [2, 1, 0], [1, 2, 0], [1, 1, 0]]
TRI6_THIRD = [[2, 0, 0], [4, 0, 0], [2, 2, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0]]
TRI6_ELEMENT = [[0, 1, 2, 3, 4, 5]]
HEX_NODE = [[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)]
HEX_ELEMENT = [[0, 1, 3, 2, 4, 5, 7, 6]]
TET_NODE = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
TET_ELEMENT = [[0, 1, 2, 3]]
TET10_NODE = TET_NODE + [[.5, 0, 0], [.5, .5, 0], [0, .5, 0], [0, 0, .5], [.5, 0, .5], [0, .5, .5]]
TET10_ELEMENT = [list(range(10))]

# class: its mesh_type, node and element
POLYGON_DATA = {
    TriangleMesh: ('polygon_tri3', GRID_NODE, [[0, 1, 4], [0, 4, 3]]),
    QuadMesh: ('polygon_quad4', GRID_NODE, [[0, 1, 4, 3], [1, 2, 5, 4]]),
    QuadTriangleMesh: ('polygon_quad4_tri3', GRID_NODE, [[0, 1, 4, 3], [1, 2, 5]]),
    Tri6Mesh: ('polygon_tri6', TRI6_FIRST, TRI6_ELEMENT),
}
POLYHEDRON_DATA = {
    HexahedronMesh: ('polyhedron_hex8', HEX_NODE, HEX_ELEMENT),
    TetrahedronMesh: ('polyhedron_tet4', TET_NODE, TET_ELEMENT),
    Tet10Mesh: ('polyhedron_tet10', TET10_NODE, TET10_ELEMENT),
}
CLASS_DATA = {**POLYGON_DATA, **POLYHEDRON_DATA}

# Two triangles that share an edge: node 1 and 2 of A are node 0 and 2 of B.
NODE_A = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
NODE_B = [[1, 0, 0], [1, 1, 0], [0, 1, 0]]
TRIANGLE_ELEMENT = [[0, 1, 2]]

devices = pytest.mark.parametrize('device_a, device_b', [('cuda', 'cpu'), ('cpu', 'cuda')])
# how the two inputs are merged: no node merging, and merging the two shared nodes
ways = pytest.mark.parametrize('idx_a, idx_b, threshold, n_node, element_b', [
    (None, None, 0, 6, [3, 4, 5]),
    ([1, 2], [0, 2], 1e-6, 4, [1, 3, 2]),
])


def make(cls, node, element, device='cpu'):
    return cls(torch.tensor(node, dtype=F64, device=device), element)


def triangle_mesh(node, device, dtype=F64, requires_grad=False):
    node = torch.tensor(node, dtype=dtype, device=device, requires_grad=requires_grad)
    return TriangleMesh(node, torch.tensor(TRIANGLE_ELEMENT, device=device))


@pytest.fixture
def plain(tmp_path):
    """Return a plain mesh loaded from a file saved by a subclass, as load_from_torch sets its mesh_type."""
    counter = iter(range(100))

    def load(plain_class, cls, node, element):
        path = str(tmp_path / f'mesh{next(counter)}.pt')
        make(cls, node, element).save_as_torch(path)
        mesh = plain_class()
        mesh.load_from_torch(path)
        assert mesh.mesh_type == cls().mesh_type
        return mesh
    return load


def edge_set(mesh):
    mesh.build_edge()
    return {tuple(sorted(edge)) for edge in torch.as_tensor(mesh.edge).tolist()}


def tri6_edge_set(element):
    edges = set()
    for c0, c1, c2, m01, m12, m20 in element:
        edges |= {tuple(sorted(edge)) for edge in [(c0, m01), (m01, c1), (c1, m12), (m12, c2), (c2, m20), (m20, c0)]}
    return edges


def element_list(mesh):
    return torch.as_tensor(mesh.element).tolist()


# --------------------------------------------------------------------- Two plain meshes, same mesh_type
@pytest.mark.parametrize('cls', list(POLYGON_DATA))
def test_two_plain_polygons_keep_mesh_type(cls, plain):
    mesh_type, node, element = POLYGON_DATA[cls]
    merged = merge_mesh(plain(PolygonMesh, cls, node, element), None, plain(PolygonMesh, cls, node, element), None, 0)
    assert type(merged) is PolygonMesh
    assert merged.mesh_type == mesh_type
    assert merged.element_type is None
    assert len(merged.element) == 2 * len(element)


@pytest.mark.parametrize('cls', list(POLYHEDRON_DATA))
def test_two_plain_polyhedrons_keep_mesh_type(cls, plain):
    mesh_type, node, element = POLYHEDRON_DATA[cls]
    merged = merge_mesh(plain(PolyhedronMesh, cls, node, element), None,
                        plain(PolyhedronMesh, cls, node, element), None, 0)
    assert type(merged) is PolyhedronMesh
    assert merged.mesh_type == mesh_type
    assert merged.element_type is None


def test_tri6_edges_of_two_plain_polygons(plain):
    merged = merge_mesh(plain(PolygonMesh, Tri6Mesh, TRI6_FIRST, TRI6_ELEMENT), None,
                        plain(PolygonMesh, Tri6Mesh, TRI6_SECOND, TRI6_ELEMENT), None, 0)
    assert element_list(merged) == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
    assert edge_set(merged) == tri6_edge_set(element_list(merged))


def test_tri6_edges_when_nodes_are_merged(plain):
    merged = merge_mesh(plain(PolygonMesh, Tri6Mesh, TRI6_FIRST, TRI6_ELEMENT), [1, 2, 4],
                        plain(PolygonMesh, Tri6Mesh, TRI6_SECOND, TRI6_ELEMENT), [0, 2, 5], 1e-6)
    assert merged.mesh_type == 'polygon_tri6'
    assert len(merged.node) == 9
    assert element_list(merged) == [[0, 1, 2, 3, 4, 5], [1, 6, 2, 7, 8, 4]]
    assert edge_set(merged) == tri6_edge_set(element_list(merged))
    assert len(edge_set(merged)) == 10


def test_merge_on_boundary_keeps_tri6_through_each_step(plain):
    merged = merge_mesh_on_boundary([plain(PolygonMesh, Tri6Mesh, TRI6_FIRST, TRI6_ELEMENT),
                                     plain(PolygonMesh, Tri6Mesh, TRI6_SECOND, TRI6_ELEMENT),
                                     plain(PolygonMesh, Tri6Mesh, TRI6_THIRD, TRI6_ELEMENT)], 1e-6)
    assert merged.mesh_type == 'polygon_tri6'
    assert len(merged.element) == 3
    assert len(merged.node) == 12


@pytest.mark.vtk
def test_two_plain_tri6_save_as_quadratic_cells(plain, tmp_path):
    pytest.importorskip('vtk')
    merged = merge_mesh(plain(PolygonMesh, Tri6Mesh, TRI6_FIRST, TRI6_ELEMENT), None,
                        plain(PolygonMesh, Tri6Mesh, TRI6_SECOND, TRI6_ELEMENT), None, 0)
    path = str(tmp_path / 'merged.vtk')
    merged.save_as_vtk(path)
    loaded = Tri6Mesh()  # it refuses any cell other than VTK_QUADRATIC_TRIANGLE
    loaded.load_from_vtk(path, 'float64')
    assert element_list(loaded) == element_list(merged)
    area, _ = Tri6Mesh.cal_element_area_and_normal(loaded.node, loaded.element)
    assert area.sum().item() == pytest.approx(4.0)


@pytest.mark.vtk
@pytest.mark.parametrize('cls', list(POLYHEDRON_DATA))
def test_two_plain_polyhedrons_save_and_load(cls, plain, tmp_path):
    pytest.importorskip('vtk')
    _, node, element = POLYHEDRON_DATA[cls]
    merged = merge_mesh(plain(PolyhedronMesh, cls, node, element), None,
                        plain(PolyhedronMesh, cls, node, element), None, 0)
    path = str(tmp_path / 'merged.vtk')
    merged.save_as_vtk(path)
    loaded = cls()
    loaded.load_from_vtk(path, 'float64')
    assert element_list(loaded) == element_list(merged)
    assert torch.allclose(loaded.node, merged.node)


# ------------------------------------------------------------- A plain mesh and a class, same mesh_type
@pytest.mark.parametrize('plain_first', [True, False])
def test_plain_tri6_and_tri6_mesh(plain, plain_first):
    one = plain(PolygonMesh, Tri6Mesh, TRI6_FIRST, TRI6_ELEMENT)
    two = make(Tri6Mesh, TRI6_SECOND, TRI6_ELEMENT)
    merged = merge_mesh(one, None, two, None, 0) if plain_first else merge_mesh(two, None, one, None, 0)
    assert type(merged) is PolygonMesh
    assert merged.mesh_type == 'polygon_tri6'
    assert edge_set(merged) == tri6_edge_set(element_list(merged))


@pytest.mark.vtk
@pytest.mark.parametrize('plain_first', [True, False])
def test_plain_hex8_and_hexahedron_mesh(plain, plain_first, tmp_path):
    pytest.importorskip('vtk')
    one = plain(PolyhedronMesh, HexahedronMesh, HEX_NODE, HEX_ELEMENT)
    two = make(HexahedronMesh, HEX_NODE, HEX_ELEMENT)
    merged = merge_mesh(one, None, two, None, 0) if plain_first else merge_mesh(two, None, one, None, 0)
    assert type(merged) is PolyhedronMesh
    assert merged.mesh_type == 'polyhedron_hex8'
    path = str(tmp_path / 'merged.vtk')
    merged.save_as_vtk(path)
    loaded = HexahedronMesh()
    loaded.load_from_vtk(path, 'float64')
    assert element_list(loaded) == element_list(merged)


def test_labelled_and_unlabelled_plain_tri6(plain):
    labelled = plain(PolygonMesh, Tri6Mesh, TRI6_FIRST, TRI6_ELEMENT)
    labelled.element_type = np.array(['TRI6'])
    merged = merge_mesh(labelled, None, plain(PolygonMesh, Tri6Mesh, TRI6_SECOND, TRI6_ELEMENT), None, 0)
    assert merged.mesh_type == 'polygon_tri6'
    assert list(merged.element_type) == ['TRI6', 'TRI6']
    assert edge_set(merged) == tri6_edge_set(element_list(merged))


# ---------------------------------------------------------------- Two plain meshes, different mesh_types
@pytest.mark.parametrize('tri6_first', [True, False])
def test_plain_tri6_and_plain_linear_polygon_raise(plain, tri6_first):
    tri6 = plain(PolygonMesh, Tri6Mesh, TRI6_FIRST, TRI6_ELEMENT)
    linear = make(PolygonMesh, GRID_NODE, [[0, 1, 4]])
    with pytest.raises(ValueError, match='never mixes'):
        merge_mesh(tri6, None, linear, None, 0) if tri6_first else merge_mesh(linear, None, tri6, None, 0)


@pytest.mark.parametrize('cls', [TriangleMesh, QuadMesh, QuadTriangleMesh])
def test_plain_tri6_and_plain_linear_type_raise(plain, cls):
    _, node, element = POLYGON_DATA[cls]
    with pytest.raises(ValueError, match='never mixes'):
        merge_mesh(plain(PolygonMesh, Tri6Mesh, TRI6_FIRST, TRI6_ELEMENT), None,
                   plain(PolygonMesh, cls, node, element), None, 0)


def test_plain_tri3_and_plain_quad4_give_polygon(plain):
    _, node_a, element_a = POLYGON_DATA[TriangleMesh]
    _, node_b, element_b = POLYGON_DATA[QuadMesh]
    merged = merge_mesh(plain(PolygonMesh, TriangleMesh, node_a, element_a), None,
                        plain(PolygonMesh, QuadMesh, node_b, element_b), None, 0)
    assert merged.mesh_type == 'polygon'
    assert merged.element_type is None


@pytest.mark.vtk
def test_plain_hex8_and_plain_tet4_get_labels(plain, tmp_path):
    pytest.importorskip('vtk')
    merged = merge_mesh(plain(PolyhedronMesh, HexahedronMesh, HEX_NODE, HEX_ELEMENT), None,
                        plain(PolyhedronMesh, TetrahedronMesh, TET_NODE, TET_ELEMENT), None, 0)
    assert merged.mesh_type == 'polyhedron'
    assert list(merged.element_type) == ['HEX8', 'TET4']
    path = str(tmp_path / 'merged.vtk')
    merged.save_as_vtk(path)
    loaded = PolyhedronMesh()
    loaded.load_from_vtk(path, 'float64')
    assert [len(element) for element in loaded.element] == [8, 4]


# ------------------------------------------------------------------- mesh_type: what stays the same
def test_two_truly_plain_polygons_stay_polygon():
    merged = merge_mesh(make(PolygonMesh, HEXAGON_NODE, HEXAGON_ELEMENT), None,
                        make(PolygonMesh, HEXAGON_NODE, HEXAGON_ELEMENT), None, 0)
    assert merged.mesh_type == 'polygon'
    assert merged.element_type is None
    assert (0, 1) in edge_set(merged) and (6, 7) in edge_set(merged)


def test_two_labelled_plain_polyhedrons_stay_polyhedron():
    one = PolyhedronMesh(torch.tensor(HEX_NODE, dtype=F64), HEX_ELEMENT, np.array(['C3D8']))
    two = PolyhedronMesh(torch.tensor(HEX_NODE, dtype=F64), HEX_ELEMENT, np.array(['C3D8']))
    merged = merge_mesh(one, None, two, None, 0)
    assert merged.mesh_type == 'polyhedron'
    assert list(merged.element_type) == ['C3D8', 'C3D8']


@pytest.mark.parametrize('cls', list(CLASS_DATA))
def test_two_of_a_class_keep_class_and_mesh_type(cls):
    mesh_type, node, element = CLASS_DATA[cls]
    merged = merge_mesh(make(cls, node, element), None, make(cls, node, element), None, 0)
    assert type(merged) is cls
    assert merged.mesh_type == mesh_type
    assert merged.element_type is None


def test_different_classes_give_the_closest_class():
    _, node_a, element_a = POLYGON_DATA[TriangleMesh]
    _, node_b, element_b = POLYGON_DATA[QuadMesh]
    polygon = merge_mesh(make(TriangleMesh, node_a, element_a), None, make(QuadMesh, node_b, element_b), None, 0)
    assert type(polygon) is PolygonMesh and polygon.mesh_type == 'polygon'
    polyhedron = merge_mesh(make(HexahedronMesh, HEX_NODE, HEX_ELEMENT), None,
                            make(TetrahedronMesh, TET_NODE, TET_ELEMENT), None, 0)
    assert type(polyhedron) is PolyhedronMesh and polyhedron.mesh_type == 'polyhedron'
    assert list(polyhedron.element_type) == ['HEX8', 'TET4']


def test_tri6_mesh_and_triangle_mesh_still_raise():
    _, node, element = POLYGON_DATA[TriangleMesh]
    with pytest.raises(ValueError, match='never mixes'):
        merge_mesh(make(Tri6Mesh, TRI6_FIRST, TRI6_ELEMENT), None, make(TriangleMesh, node, element), None, 0)


# ------------------------------------------------------------------------ Inputs on different devices
@pytest.mark.gpu
@devices
@ways
def test_output_is_on_the_device_of_mesh_a(device_a, device_b, idx_a, idx_b, threshold, n_node, element_b):
    merged = merge_mesh(triangle_mesh(NODE_A, device_a), idx_a, triangle_mesh(NODE_B, device_b), idx_b, threshold)
    assert type(merged) is TriangleMesh
    assert merged.node.device.type == device_a
    assert merged.element.device.type == device_a
    assert merged.node.shape == (n_node, 3)
    assert merged.element.tolist() == [[0, 1, 2], element_b]


@pytest.mark.gpu
@devices
@ways
def test_same_result_as_with_both_on_the_cpu(device_a, device_b, idx_a, idx_b, threshold, n_node, element_b):
    expected = merge_mesh(triangle_mesh(NODE_A, 'cpu'), idx_a, triangle_mesh(NODE_B, 'cpu'), idx_b, threshold)
    merged = merge_mesh(triangle_mesh(NODE_A, device_a), idx_a, triangle_mesh(NODE_B, device_b), idx_b, threshold)
    assert torch.equal(merged.node.cpu(), expected.node)
    assert torch.equal(merged.element.cpu(), expected.element)


@pytest.mark.gpu
@devices
@ways
def test_inputs_stay_on_their_devices(device_a, device_b, idx_a, idx_b, threshold, n_node, element_b):
    mesh_a = triangle_mesh(NODE_A, device_a)
    mesh_b = triangle_mesh(NODE_B, device_b)
    merge_mesh(mesh_a, idx_a, mesh_b, idx_b, threshold)
    assert mesh_a.node.device.type == device_a and mesh_a.element.device.type == device_a
    assert mesh_b.node.device.type == device_b and mesh_b.element.device.type == device_b


@pytest.mark.gpu
@devices
@ways
def test_gradient_reaches_both_inputs(device_a, device_b, idx_a, idx_b, threshold, n_node, element_b):
    mesh_a = triangle_mesh(NODE_A, device_a, requires_grad=True)
    mesh_b = triangle_mesh(NODE_B, device_b, requires_grad=True)
    merge_mesh(mesh_a, idx_a, mesh_b, idx_b, threshold).node.sum().backward()
    assert mesh_a.node.grad.device.type == device_a and mesh_a.node.grad.sum().item() == 9
    assert mesh_b.node.grad.device.type == device_b
    # each input node that stays in the output adds 3 to the sum, and a merged node of meshB is not in the output
    assert mesh_b.node.grad.sum().item() == 3 * (n_node - 3)


@pytest.mark.gpu
@devices
@ways
def test_dtype_is_kept_across_devices(device_a, device_b, idx_a, idx_b, threshold, n_node, element_b):
    merged = merge_mesh(triangle_mesh(NODE_A, device_a, F32), idx_a, triangle_mesh(NODE_B, device_b, F32), idx_b, threshold)
    assert merged.node.dtype == F32
    assert merged.node.device.type == device_a


@pytest.mark.gpu
@pytest.mark.parametrize('device_b', ['cuda', 'cpu'])
def test_node_of_mesh_a_decides_the_device_and_not_its_element(device_b):
    mesh_a = TriangleMesh(torch.tensor(NODE_A, dtype=F64, device='cuda'), torch.tensor(TRIANGLE_ELEMENT, device='cpu'))
    merged = merge_mesh(mesh_a, None, triangle_mesh(NODE_B, device_b), None, 0)
    assert merged.node.is_cuda and merged.element.is_cuda


@pytest.mark.gpu
@devices
@pytest.mark.parametrize('cls, node, element', [(cls, data[1], data[2]) for cls, data in CLASS_DATA.items()]
                         + [(PolygonMesh, HEXAGON_NODE, HEXAGON_ELEMENT)])
def test_each_class_across_devices(device_a, device_b, cls, node, element):
    merged = merge_mesh(make(cls, node, element, device_a), None, make(cls, node, element, device_b), None, 0)
    assert type(merged) is cls
    assert merged.node.device.type == device_a
    if torch.is_tensor(merged.element):
        assert merged.element.device.type == device_a
    assert len(merged.element) == 2 * len(element)


@pytest.mark.gpu
@devices
def test_merge_on_boundary_across_devices(device_a, device_b):
    merged = merge_mesh_on_boundary([triangle_mesh(NODE_A, device_a), triangle_mesh(NODE_B, device_b)], 1e-6)
    assert merged.node.device.type == device_a
    assert len(merged.node) == 4


@pytest.mark.gpu
def test_two_nodes_of_mesh_b_on_one_node_still_raise_across_devices():
    node_b = [[1, 0, 0], [1, 0, 0], [0, 1, 0]]
    with pytest.raises(ValueError, match='same node'):
        merge_mesh(triangle_mesh(NODE_A, 'cuda'), [1], triangle_mesh(node_b, 'cpu'), [0, 1], 1e-6)
