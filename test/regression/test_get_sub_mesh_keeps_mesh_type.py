"""get_sub_mesh of a plain PolygonMesh or PolyhedronMesh keeps a mesh_type that says more.

load_from_torch can give a plain mesh a mesh_type such as polygon_tri6 or polyhedron_hex8. The node
order and the VTK cell type depend on it, so a sub-mesh that lost it would get wrong edges or fail to save.
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

pytestmark = pytest.mark.geometry

F64 = torch.float64
GRID_NODE = [[0, 0, 0], [1, 0, 0], [3, 0, 0], [0, 1, 0], [1, 1, 0], [3, 1, 0]]
# Two right triangles with legs of 2, so each has an area of 2. Corners first, then midpoints.
TRI6_NODE = [[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 1, 0], [1, 2, 0]]
TRI6_ELEMENT = [[0, 1, 2, 4, 5, 6], [1, 3, 2, 7, 8, 5]]
HEX_NODE = [[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1, 2)]
HEX_ELEMENT = [[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]]
TET_NODE = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
TET_ELEMENT = [[0, 1, 2, 3], [1, 2, 3, 4]]


def tet10_mesh_data():
    # Corners first, then the midpoints of edges 01, 12, 20, 03, 13 and 23. The two elements share no node.
    corner = torch.tensor(TET_NODE[:4], dtype=F64)
    one = torch.cat([corner, (corner[[0, 1, 2, 0, 1, 2]] + corner[[1, 2, 0, 3, 3, 3]]) / 2], dim=0)
    return torch.cat([one, one + torch.tensor([5., 0, 0], dtype=F64)], dim=0).tolist(), [list(range(10)), list(range(10, 20))]


POLYGON_DATA = {
    TriangleMesh: ('polygon_tri3', GRID_NODE, [[0, 1, 4], [0, 4, 3]]),
    QuadMesh: ('polygon_quad4', GRID_NODE, [[0, 1, 4, 3], [1, 2, 5, 4]]),
    QuadTriangleMesh: ('polygon_quad4_tri3', GRID_NODE, [[0, 1, 4, 3], [1, 2, 5]]),
    Tri6Mesh: ('polygon_tri6', TRI6_NODE, TRI6_ELEMENT),
}
POLYHEDRON_DATA = {
    HexahedronMesh: ('polyhedron_hex8', HEX_NODE, HEX_ELEMENT),
    TetrahedronMesh: ('polyhedron_tet4', TET_NODE, TET_ELEMENT),
    Tet10Mesh: ('polyhedron_tet10', *tet10_mesh_data()),
}
polygon_classes = pytest.mark.parametrize('cls', list(POLYGON_DATA))
polyhedron_classes = pytest.mark.parametrize('cls', list(POLYHEDRON_DATA))


def make(cls, data):
    _, node, element = data[cls]
    return cls(torch.tensor(node, dtype=F64), element)


def plain_mesh_from_file(plain_class, cls, data, tmp_path):
    path = str(tmp_path / 'mesh.pt')
    make(cls, data).save_as_torch(path)
    mesh = plain_class()
    mesh.load_from_torch(path)
    assert mesh.mesh_type == data[cls][0]
    return mesh


def edge_set(mesh):
    mesh.build_edge()
    return {tuple(sorted(edge)) for edge in torch.as_tensor(mesh.edge).tolist()}


# ---------------------------------------------------------------------------------- Plain polygon
@polygon_classes
def test_plain_polygon_sub_mesh_keeps_mesh_type(cls, tmp_path):
    mesh = plain_mesh_from_file(PolygonMesh, cls, POLYGON_DATA, tmp_path)
    sub_mesh = mesh.get_sub_mesh([0])
    assert type(sub_mesh) is PolygonMesh
    assert sub_mesh.mesh_type == POLYGON_DATA[cls][0]


@polygon_classes
def test_plain_polygon_sub_mesh_keeps_mesh_type_with_node_idx_list(cls, tmp_path):
    mesh = plain_mesh_from_file(PolygonMesh, cls, POLYGON_DATA, tmp_path)
    sub_mesh, node_idx_list = mesh.get_sub_mesh([1], return_node_idx_list=True)
    assert sub_mesh.mesh_type == POLYGON_DATA[cls][0]
    assert torch.equal(sub_mesh.node, mesh.node[torch.as_tensor(node_idx_list)])


def test_plain_polygon_sub_mesh_with_no_element_keeps_mesh_type(tmp_path):
    mesh = plain_mesh_from_file(PolygonMesh, Tri6Mesh, POLYGON_DATA, tmp_path)
    assert mesh.get_sub_mesh([]).mesh_type == 'polygon_tri6'


def test_tri6_edges_of_plain_polygon_sub_mesh(tmp_path):
    mesh = plain_mesh_from_file(PolygonMesh, Tri6Mesh, POLYGON_DATA, tmp_path)
    sub_mesh = mesh.get_sub_mesh([1])
    c0, c1, c2, m01, m12, m20 = torch.as_tensor(sub_mesh.element)[0].tolist()
    expected = {tuple(sorted(edge)) for edge in [(c0, m01), (m01, c1), (c1, m12), (m12, c2), (c2, m20), (m20, c0)]}
    assert edge_set(sub_mesh) == expected
    assert edge_set(sub_mesh) == edge_set(make(Tri6Mesh, POLYGON_DATA).get_sub_mesh([1]))


def test_tri6_label_on_plain_polygon_sub_mesh(tmp_path):
    mesh = plain_mesh_from_file(PolygonMesh, Tri6Mesh, POLYGON_DATA, tmp_path)
    mesh.element_type = np.array(['TRI6', 'TRI6'])
    sub_mesh = mesh.get_sub_mesh([0])
    assert sub_mesh.element_type.tolist() == ['TRI6']
    assert len(edge_set(sub_mesh)) == 6


@pytest.mark.vtk
def test_tri6_plain_polygon_sub_mesh_saves_as_quadratic_cells(tmp_path):
    pytest.importorskip('vtk')
    mesh = plain_mesh_from_file(PolygonMesh, Tri6Mesh, POLYGON_DATA, tmp_path)
    path = str(tmp_path / 'sub_mesh.vtk')
    mesh.get_sub_mesh([1]).save_as_vtk(path)
    loaded = Tri6Mesh()  # it refuses any cell other than VTK_QUADRATIC_TRIANGLE
    loaded.load_from_vtk(path, 'float64')
    assert torch.as_tensor(loaded.element).shape == (1, 6)
    area, _ = Tri6Mesh.cal_element_area_and_normal(loaded.node, loaded.element)
    assert area.sum().item() == pytest.approx(2.0)


def test_plain_polygon_stays_polygon():
    node = [[0, 0, 0], [2, 0, 0], [3, 1, 0], [2, 2, 0], [0, 2, 0], [-1, 1, 0], [9, 9, 0], [8, 9, 0], [8, 8, 0]]
    mesh = PolygonMesh(torch.tensor(node, dtype=F64), [[0, 1, 2, 3, 4, 5], [6, 7, 8]])
    sub_mesh = mesh.get_sub_mesh([0])
    assert sub_mesh.mesh_type == 'polygon'
    assert edge_set(sub_mesh) == {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)}


@polygon_classes
def test_polygon_subclass_sub_mesh_keeps_class_and_mesh_type(cls):
    sub_mesh = make(cls, POLYGON_DATA).get_sub_mesh([0])
    assert type(sub_mesh) is cls
    assert sub_mesh.mesh_type == POLYGON_DATA[cls][0]


# ------------------------------------------------------------------------------- Plain polyhedron
@polyhedron_classes
def test_plain_polyhedron_sub_mesh_keeps_mesh_type(cls, tmp_path):
    mesh = plain_mesh_from_file(PolyhedronMesh, cls, POLYHEDRON_DATA, tmp_path)
    sub_mesh = mesh.get_sub_mesh([0])
    assert type(sub_mesh) is PolyhedronMesh
    assert sub_mesh.mesh_type == POLYHEDRON_DATA[cls][0]


@polyhedron_classes
def test_plain_polyhedron_sub_mesh_keeps_mesh_type_with_node_idx_list(cls, tmp_path):
    mesh = plain_mesh_from_file(PolyhedronMesh, cls, POLYHEDRON_DATA, tmp_path)
    sub_mesh, node_idx_list = mesh.get_sub_mesh([1], return_node_idx_list=True)
    assert sub_mesh.mesh_type == POLYHEDRON_DATA[cls][0]
    assert torch.equal(sub_mesh.node, mesh.node[torch.as_tensor(node_idx_list)])


def test_plain_polyhedron_sub_mesh_with_no_element_keeps_mesh_type(tmp_path):
    mesh = plain_mesh_from_file(PolyhedronMesh, HexahedronMesh, POLYHEDRON_DATA, tmp_path)
    assert mesh.get_sub_mesh([]).mesh_type == 'polyhedron_hex8'


@pytest.mark.vtk
@polyhedron_classes
def test_plain_polyhedron_sub_mesh_saves_and_loads(cls, tmp_path):
    pytest.importorskip('vtk')
    mesh = plain_mesh_from_file(PolyhedronMesh, cls, POLYHEDRON_DATA, tmp_path)
    sub_mesh = mesh.get_sub_mesh([1])
    path = str(tmp_path / 'sub_mesh.vtk')
    sub_mesh.save_as_vtk(path)
    loaded = cls()
    loaded.load_from_vtk(path, 'float64')
    assert torch.equal(torch.as_tensor(loaded.element), torch.as_tensor(sub_mesh.element))
    assert torch.allclose(loaded.node, sub_mesh.node)


def test_labelled_plain_polyhedron_stays_polyhedron():
    _, node, element = POLYHEDRON_DATA[HexahedronMesh]
    mesh = PolyhedronMesh(torch.tensor(node, dtype=F64), element, np.array(['C3D8', 'C3D8']))
    sub_mesh = mesh.get_sub_mesh([0])
    assert sub_mesh.mesh_type == 'polyhedron'
    assert sub_mesh.element_type.tolist() == ['C3D8']


@polyhedron_classes
def test_polyhedron_subclass_sub_mesh_keeps_class_and_mesh_type(cls):
    sub_mesh = make(cls, POLYHEDRON_DATA).get_sub_mesh([0])
    assert type(sub_mesh) is cls
    assert sub_mesh.mesh_type == POLYHEDRON_DATA[cls][0]
