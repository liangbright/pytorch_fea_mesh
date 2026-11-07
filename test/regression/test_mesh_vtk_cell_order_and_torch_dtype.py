"""Reading VTK cells into a polygon mesh, get_sub_mesh with a list element_type, and the
dtype input of load_from_torch.

A polygon mesh never mixes linear and quadratic elements. Each one has its own node order, so
a VTK cell of the other kind would be read with wrong edges or a wrong area and no error.
"""
import numpy as np
import pytest
import torch

vtk = pytest.importorskip('vtk')

from torch_fea_mesh.mesh.mesh import Mesh
from torch_fea_mesh.mesh.polygon_mesh import PolygonMesh
from torch_fea_mesh.mesh.polyhedron_mesh import PolyhedronMesh
from torch_fea_mesh.mesh.quad_triangle_mesh import QuadTriangleMesh
from torch_fea_mesh.mesh.tet10_mesh import Tet10Mesh
from torch_fea_mesh.mesh.tri6_mesh import Tri6Mesh
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh

pytestmark = [pytest.mark.geometry, pytest.mark.io, pytest.mark.vtk]

F32 = torch.float32
F64 = torch.float64
# A right triangle with legs of 2, so its area is 2. Corners first, then midpoints: tri6 order.
TRI6_NODE = [[0., 0, 0], [2, 0, 0], [0, 2, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
TRI6_ORDER = [0, 1, 2, 3, 4, 5]
BOUNDARY_ORDER = [0, 3, 1, 4, 2, 5]


def grid(cells, n_node=None, node=None):
    """Return a vtkUnstructuredGrid with cells [(vtk cell type, node ids), ...]."""
    if node is None:
        node = [[float(k), float(k % 3), 0.] for k in range(n_node)]
    points = vtk.vtkPoints()
    for x in node:
        points.InsertNextPoint(*x)
    data = vtk.vtkUnstructuredGrid()
    data.SetPoints(points)
    for cell_type, ids in cells:
        id_list = vtk.vtkIdList()
        for k in ids:
            id_list.InsertNextId(k)
        data.InsertNextCell(cell_type, id_list)
    return data


def write(data, path):
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetInputData(data)
    writer.SetFileName(str(path))
    assert writer.Write() == 1
    return str(path)


# ------------------------------------------------------------------- A polygon mesh refuses another node order
def test_tri6_mesh_refuses_a_file_with_a_6_node_polygon(tmp_path):
    path = write(grid([(vtk.VTK_POLYGON, BOUNDARY_ORDER)], node=TRI6_NODE), tmp_path / 'a.vtk')
    with pytest.raises(ValueError, match='never mixes linear and quadratic'):
        Tri6Mesh().load_from_vtk(path, F64)


def test_polygon_mesh_refuses_a_quadratic_triangle():
    data = grid([(vtk.VTK_QUADRATIC_TRIANGLE, TRI6_ORDER)], node=TRI6_NODE)
    with pytest.raises(ValueError, match='never mixes linear and quadratic'):
        PolygonMesh().read_mesh_vtk(data, F64)


@pytest.mark.parametrize('cell_type, n', [('VTK_QUADRATIC_QUAD', 8), ('VTK_BIQUADRATIC_QUAD', 9),
                                          ('VTK_QUADRATIC_LINEAR_QUAD', 6), ('VTK_BIQUADRATIC_TRIANGLE', 7)])
def test_polygon_mesh_refuses_other_cells_that_list_corners_first(cell_type, n):
    data = grid([(getattr(vtk, cell_type), list(range(n)))], n_node=n)
    with pytest.raises(ValueError, match='never mixes linear and quadratic'):
        PolygonMesh().read_mesh_vtk(data, F64)


def test_one_quadratic_cell_among_linear_cells_is_refused():
    data = grid([(vtk.VTK_TRIANGLE, [0, 1, 2]), (vtk.VTK_QUADRATIC_TRIANGLE, TRI6_ORDER)], node=TRI6_NODE)
    with pytest.raises(ValueError, match='never mixes linear and quadratic'):
        PolygonMesh().read_mesh_vtk(data, F64)


@pytest.mark.parametrize('cell_type', ['VTK_QUADRATIC_LINEAR_QUAD', 'VTK_POLYGON'])
def test_tri6_mesh_refuses_other_6_node_cells(cell_type):
    # both have the 6 nodes that the width check of Tri6Mesh looks for
    data = grid([(getattr(vtk, cell_type), TRI6_ORDER)], node=TRI6_NODE)
    with pytest.raises(ValueError, match='never mixes linear and quadratic'):
        Tri6Mesh().read_mesh_vtk(data, F64)


def test_tri6_mesh_type_refuses_a_linear_triangle():
    mesh = Mesh(None, None, None, 'polygon_tri6')
    with pytest.raises(ValueError, match='never mixes linear and quadratic'):
        mesh.read_mesh_vtk(grid([(vtk.VTK_TRIANGLE, [0, 1, 2])], n_node=3), F64)


def test_quad8_mesh_type_refuses_an_8_node_polygon():
    mesh = Mesh(None, None, None, 'polygon_quad8')
    with pytest.raises(ValueError, match='never mixes linear and quadratic'):
        mesh.read_mesh_vtk(grid([(vtk.VTK_POLYGON, list(range(8)))], n_node=8), F64)


def test_a_refused_read_keeps_the_mesh():
    mesh = PolygonMesh(torch.tensor(TRI6_NODE[:3], dtype=F64), [[0, 1, 2]])
    with pytest.raises(ValueError):
        mesh.read_mesh_vtk(grid([(vtk.VTK_QUADRATIC_TRIANGLE, TRI6_ORDER)], node=TRI6_NODE), F64)
    assert mesh.node.shape == (3, 3)
    assert torch.as_tensor(mesh.element).tolist() == [[0, 1, 2]]


# ----------------------------------------------------------------- Reads that still work
def test_tri6_mesh_still_reads_its_own_file_with_the_right_area(tmp_path):
    source = Tri6Mesh(torch.tensor(TRI6_NODE, dtype=F64), [TRI6_ORDER])
    source.save_as_vtk(str(tmp_path / 'a.vtk'))
    mesh = Tri6Mesh()
    mesh.load_from_vtk(str(tmp_path / 'a.vtk'), F64)
    mesh.update_element_area_and_normal()
    assert mesh.element.tolist() == [TRI6_ORDER]
    assert mesh.element_area.item() == pytest.approx(2.0)


def test_polygon_mesh_still_reads_its_own_hexagon(tmp_path):
    source = PolygonMesh(torch.tensor(TRI6_NODE, dtype=F64), [BOUNDARY_ORDER])
    source.save_as_vtk(str(tmp_path / 'a.vtk'))
    source.save_as_vtp(str(tmp_path / 'a.vtp'))
    for name in ('a.vtk', 'a.vtp'):
        mesh = PolygonMesh()
        mesh.load(str(tmp_path / name), F64)
        assert torch.as_tensor(mesh.element).tolist() == [BOUNDARY_ORDER]


@pytest.mark.parametrize('cell_type, ids', [('VTK_TRIANGLE', [0, 1, 2]), ('VTK_QUAD', [0, 1, 2, 3]),
                                            ('VTK_PIXEL', [0, 1, 2, 3]), ('VTK_POLYGON', [0, 1, 2, 3, 4]),
                                            ('VTK_TRIANGLE_STRIP', [0, 1, 2, 3])])
def test_polygon_mesh_still_reads_every_linear_cell(cell_type, ids):
    mesh = PolygonMesh()
    mesh.read_mesh_vtk(grid([(getattr(vtk, cell_type), ids)], n_node=6), F64)
    assert len(mesh.element) >= 1


def test_quadratic_mesh_types_still_read_their_own_cell():
    tri6 = Mesh(None, None, None, 'polygon_tri6')
    tri6.read_mesh_vtk(grid([(vtk.VTK_QUADRATIC_TRIANGLE, TRI6_ORDER)], node=TRI6_NODE), F64)
    assert tri6.element.tolist() == [TRI6_ORDER]
    quad8 = Mesh(None, None, None, 'polygon_quad8')
    quad8.read_mesh_vtk(grid([(vtk.VTK_QUADRATIC_QUAD, list(range(8)))], n_node=8), F64)
    assert quad8.element.shape == (1, 8)


def test_a_dropped_cell_does_not_cause_a_refusal():
    # A quadratic edge is 1D and a hex is 3D, so a polygon mesh drops them before the check.
    data = grid([(vtk.VTK_QUADRATIC_EDGE, [0, 1, 2]), (vtk.VTK_HEXAHEDRON, list(range(8))),
                 (vtk.VTK_TRIANGLE, [0, 1, 2])], n_node=8)
    mesh = PolygonMesh()
    mesh.read_mesh_vtk(data, F64)
    assert torch.as_tensor(mesh.element).tolist() == [[0, 1, 2]]


def test_solid_and_polyline_meshes_are_not_checked():
    tet10 = Tet10Mesh()
    tet10.read_mesh_vtk(grid([(vtk.VTK_QUADRATIC_TETRA, list(range(10)))], n_node=10), F64)
    assert tet10.element.shape == (1, 10)
    polyline = Mesh(None, None, None, 'polyline')
    polyline.read_mesh_vtk(grid([(vtk.VTK_QUADRATIC_EDGE, [0, 1, 2])], n_node=3), F64)
    assert torch.as_tensor(polyline.element).tolist() == [[0, 1, 2]]


# ------------------------------------------------------------------------------- get_sub_mesh with a list element_type
HEX_NODE = [[float(k), float(k % 3), float(k % 2)] for k in range(12)]
LABELS = ['C3D4', 'C3D8', 'C3D4']
SOLID = [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11], [1, 2, 3, 4]]


@pytest.mark.parametrize('index', [[1], [2, 0], np.array([0, 1]), torch.tensor([1, 2])])
def test_get_sub_mesh_with_a_list_element_type(index):
    mesh = PolyhedronMesh(torch.tensor(HEX_NODE), SOLID)
    mesh.element_type = list(LABELS)
    sub = mesh.get_sub_mesh(index)
    assert np.asarray(sub.element_type).tolist() == [LABELS[k] for k in np.asarray(index).tolist()]
    assert len(sub.element) == len(index)


def test_get_sub_mesh_with_a_list_on_a_plain_mesh_and_a_polygon_mesh():
    node = torch.tensor(TRI6_NODE)
    plain = Mesh(node, [[0, 1, 2], [0, 1, 3, 4]], None, 'polygon')
    plain.element_type = ['S3', 'S4']
    assert np.asarray(plain.get_sub_mesh([1]).element_type).tolist() == ['S4']
    polygon = PolygonMesh(node, [[0, 1, 2], [0, 1, 3, 4]])
    polygon.element_type = ['S3', 'S4']
    assert np.asarray(polygon.get_sub_mesh([0]).element_type).tolist() == ['S3']


def test_get_sub_mesh_with_a_list_restored_by_load_from_torch(tmp_path):
    mesh = PolyhedronMesh(torch.tensor(HEX_NODE), SOLID)
    mesh.element_type = list(LABELS)
    mesh.save_as_torch(str(tmp_path / 'a.pt'))
    restored = PolyhedronMesh()
    restored.load_from_torch(str(tmp_path / 'a.pt'))
    assert isinstance(restored.element_type, list)
    assert np.asarray(restored.get_sub_mesh([1]).element_type).tolist() == ['C3D8']
    assert restored.element_type == LABELS


def test_get_sub_mesh_with_a_list_with_a_missing_label_and_no_index():
    mesh = PolyhedronMesh(torch.tensor(HEX_NODE), SOLID)
    mesh.element_type = ['C3D4', None, 'C3D4']
    assert np.asarray(mesh.get_sub_mesh([1]).element_type).tolist() == [None]
    assert len(mesh.get_sub_mesh([]).element_type) == 0


def test_get_sub_mesh_with_a_numpy_element_type_still_works():
    mesh = PolyhedronMesh(torch.tensor(HEX_NODE), SOLID, np.array(LABELS))
    assert mesh.get_sub_mesh([1]).element_type.tolist() == ['C3D8']


# ---------------------------------------------------------------- The dtype of load_from_torch
def saved_mesh(path, dtype=F32):
    mesh = TriangleMesh(torch.tensor(TRI6_NODE[:3], dtype=dtype), [[0, 1, 2]])
    mesh.node_data['stress'] = torch.ones((3, 2), dtype=dtype)
    mesh.node_data['label'] = torch.tensor([[1], [2], [3]], dtype=torch.int64)
    mesh.element_data['area'] = torch.ones((1, 1), dtype=dtype)
    mesh.element_data['flag'] = torch.tensor([[True]])
    mesh.save_as_torch(str(path))
    return str(path)


@pytest.mark.parametrize('dtype', [F64, 'float64'])
def test_load_from_torch_converts_node_and_float_fields(tmp_path, dtype):
    mesh = TriangleMesh()
    mesh.load_from_torch(saved_mesh(tmp_path / 'a.pt'), dtype)
    assert mesh.node.dtype == F64
    assert mesh.node_data['stress'].dtype == F64
    assert mesh.element_data['area'].dtype == F64


def test_load_from_torch_converts_down_too(tmp_path):
    mesh = TriangleMesh()
    mesh.load_from_torch(saved_mesh(tmp_path / 'a.pt', F64), 'float32')
    assert mesh.node.dtype == F32
    assert mesh.node_data['stress'].dtype == F32


def test_load_from_torch_keeps_integer_and_bool_data_and_element(tmp_path):
    mesh = TriangleMesh()
    mesh.load_from_torch(saved_mesh(tmp_path / 'a.pt'), F64)
    assert mesh.node_data['label'].dtype == torch.int64
    assert mesh.element_data['flag'].dtype == torch.bool
    assert mesh.element.dtype == torch.int64


@pytest.mark.parametrize('args', [(), (None,)])
def test_load_from_torch_without_dtype_keeps_the_saved_types(tmp_path, args):
    for saved in (F32, F64):
        mesh = TriangleMesh()
        mesh.load_from_torch(saved_mesh(tmp_path / 'a.pt', saved), *args)
        assert mesh.node.dtype == saved
        assert mesh.node_data['stress'].dtype == saved


def test_load_passes_dtype_to_load_from_torch(tmp_path, capsys):
    path = saved_mesh(tmp_path / 'a.pt')
    mesh = TriangleMesh()
    mesh.load(path, 'float64')
    assert mesh.node.dtype == F64
    assert 'ignored' not in capsys.readouterr().out
    mesh = TriangleMesh()
    mesh.load(path, None)
    assert mesh.node.dtype == F32


def test_load_from_torch_refuses_a_bad_dtype_and_keeps_the_mesh(tmp_path):
    path = saved_mesh(tmp_path / 'a.pt')
    mesh = TriangleMesh(torch.tensor(TRI6_NODE[:3], dtype=F64), [[0, 1, 2]])
    with pytest.raises(ValueError, match='unsupported dtype'):
        mesh.load_from_torch(path, 'float16x')
    assert mesh.node.dtype == F64


def test_load_from_torch_keeps_node_a_leaf_that_needs_grad(tmp_path):
    node = torch.tensor(TRI6_NODE[:3], dtype=F32, requires_grad=True)
    TriangleMesh(node, [[0, 1, 2]]).save_as_torch(str(tmp_path / 'a.pt'))
    mesh = TriangleMesh()
    mesh.load_from_torch(str(tmp_path / 'a.pt'), F64)
    assert mesh.node.dtype == F64
    assert mesh.node.requires_grad and mesh.node.is_leaf


@pytest.mark.parametrize('use_load', [False, True])
def test_quad_triangle_mesh_takes_dtype(tmp_path, use_load):
    path = str(tmp_path / 'a.pt')
    QuadTriangleMesh(torch.tensor(TRI6_NODE, dtype=F32), [[0, 1, 2], [0, 1, 3, 4]]).save_as_torch(path)
    mesh = QuadTriangleMesh()
    if use_load:
        mesh.load(path, 'float64')
    else:
        mesh.load_from_torch(path, 'float64')
    assert mesh.node.dtype == F64
    plain = QuadTriangleMesh()
    plain.load_from_torch(path)
    assert plain.node.dtype == F32
