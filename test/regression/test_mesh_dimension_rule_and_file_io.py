"""One element dimension per mesh, and the VTK and torch file handling in mesh.py.

A geometry mesh (polyhedron, polygon, polyline) does not have mixed dimension elements. A
polyhedron mesh may mix hex8 and tet4, but will never have a 2D quad. All the save and load
functions in Mesh follow this, except save_as_torch and load_from_torch, which store and
restore a mesh exactly as it is.

Keep this file small: add a test only for a behaviour that nothing here catches yet.
"""
import ast
from pathlib import Path
import struct
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

vtk = pytest.importorskip('vtk')
from vtk.util.numpy_support import numpy_to_vtk

from torch_fea_mesh.io.check_legacy_vtk_file import find_cut_in_legacy_vtk_file
import torch_fea_mesh.mesh.mesh as mesh_module
from torch_fea_mesh.mesh.hexahedron_mesh import HexahedronMesh
from torch_fea_mesh.mesh.mesh import Mesh
from torch_fea_mesh.mesh.polygon_mesh import PolygonMesh
from torch_fea_mesh.mesh.polyhedron_mesh import PolyhedronMesh
from torch_fea_mesh.mesh.polyline_mesh import PolylineMesh
from torch_fea_mesh.mesh.quad_mesh import QuadMesh
from torch_fea_mesh.mesh.tet10_mesh import Tet10Mesh
from torch_fea_mesh.mesh.tetrahedron_mesh import TetrahedronMesh
from torch_fea_mesh.mesh.tri6_mesh import Tri6Mesh
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh

pytestmark = [pytest.mark.geometry, pytest.mark.io, pytest.mark.vtk]

F64 = torch.float64
TRI = [[0., 0, 0], [1, 0, 0], [0, 1, 0]]
TET = [[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
HEX = [[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
STRIP = [[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 2, 0]]
READERS = ['read_mesh_vtk']
CONVERTERS = ['convert_to_vtk', 'convert_to_vtk_slow']


def T(x):
    return torch.tensor(x, dtype=F64)


def E(k):
    return torch.empty((0, k), dtype=torch.int64)


def node_list(n):
    """n distinct points."""
    return [[float(i), float(i * i % 7), float(i % 3)] for i in range(n)]


def cell_array(cells):
    array = vtk.vtkCellArray()
    for cell in cells:
        array.InsertNextCell(len(cell), list(cell))
    return array


def points(coords):
    result = vtk.vtkPoints()
    result.SetDataTypeToDouble()
    for p in coords:
        result.InsertNextPoint(*p)
    return result


def add_tag(data, tags):
    array = numpy_to_vtk(np.asarray(tags, dtype=np.float64).reshape(-1, 1), deep=True)
    array.SetName('tag')
    data.GetCellData().AddArray(array)


def polydata(coords, verts=(), lines=(), polys=(), strips=(), tags=None):
    data = vtk.vtkPolyData()
    data.SetPoints(points(coords))
    data.SetVerts(cell_array(verts))
    data.SetLines(cell_array(lines))
    data.SetPolys(cell_array(polys))
    data.SetStrips(cell_array(strips))
    if tags is not None:
        add_tag(data, tags)
    return data


def unstructured(coords, cells, tags=None):
    data = vtk.vtkUnstructuredGrid()
    data.SetPoints(points(coords))
    for cell_type, ids in cells:
        data.InsertNextCell(cell_type, len(ids), list(ids))
    if tags is not None:
        add_tag(data, tags)
    return data


def write_legacy(data, path):
    writer = vtk.vtkPolyDataWriter() if data.IsA('vtkPolyData') else vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(data)
    assert writer.Write() == 1


def write_vtp(data, path):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(data)
    assert writer.Write() == 1


def elements(mesh):
    """element as a list of lists, whichever form it has."""
    return mesh.element.tolist() if torch.is_tensor(mesh.element) else [list(e) for e in mesh.element]


def labels(mesh):
    return None if mesh.element_type is None else list(mesh.element_type)


def saved_cell_types(mesh, path):
    mesh.save_as_vtk(str(path))
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    return [grid.GetCellType(i) for i in range(grid.GetNumberOfCells())]


# ---------------------------------------------------------------- Labels and VTK cell types
def test_label_gives_vtk_cell():
    mesh = PolyhedronMesh(T(node_list(10)), [list(range(10))], element_type=['C3D10M'])
    output = mesh.convert_to_vtk_slow()
    assert output.GetNumberOfCells() == 1
    assert output.GetCellType(0) == vtk.VTK_QUADRATIC_TETRA


def test_polygon_mesh_does_not_mix_linear_and_quadratic_elements(tmp_path):
    # A polygon lists its nodes around its boundary, and a quadratic element lists its corners first.
    mesh = PolygonMesh(T(node_list(8)), [list(range(8))])
    mesh.element_type = np.array(['S8R'])
    path = tmp_path / 'mesh.vtk'
    for action in (mesh.convert_to_vtk, mesh.convert_to_vtk_slow, mesh.build_edge, lambda: mesh.save_as_vtk(str(path))):
        with pytest.raises(ValueError, match='never mixes linear and quadratic'):
            action()
    assert not path.exists()
    for mesh_type in ('polygon', 'polygon_tri6'):
        with pytest.raises(ValueError, match='never mixes linear and quadratic'):
            Mesh(T(node_list(6)), [list(range(6)), [0, 1, 2]], ['TRI6', 'TRI3'], mesh_type)
    tri6 = Tri6Mesh(T(node_list(6)), [list(range(6))])
    tri6.element_type = np.array(['TRI6'])
    assert tri6.convert_to_vtk().GetCellType(0) == vtk.VTK_QUADRATIC_TRIANGLE
    tri6.build_edge()


def test_label_that_contradicts_node_count_is_rejected():
    mesh = PolyhedronMesh(T(HEX), [list(range(8))], element_type=['C3D4'])
    with pytest.raises(ValueError, match='needs 4 nodes'):
        mesh.convert_to_vtk_slow()


def test_wrong_number_of_labels_is_rejected():
    mesh = PolyhedronMesh(T(TET + [[1., 1, 1]]), [[0, 1, 2, 3], [1, 2, 3, 4]], element_type=['C3D4'])
    with pytest.raises(ValueError, match='len\\(element_type\\)'):
        mesh.convert_to_vtk_slow()


# ---------------------------------------------------------------- Empty arrays and links
def test_constructor_accepts_empty_node_and_element():
    mesh = PolylineMesh([], [])
    assert mesh.node.shape == (0, 3)
    assert mesh.element.shape == (0, 2)


def test_copy_gives_two_dimensional_empty_arrays():
    mesh = PolylineMesh()
    mesh.copy(T(node_list(10)), [])
    assert mesh.element.shape == (0, 2)
    mesh.copy([], [])
    assert mesh.node.shape == (0, 3)
    assert mesh.element.shape == (0, 2)


@pytest.mark.parametrize('adj', ['node', 'edge'])
def test_element_link_of_single_triangle(adj):
    mesh = TriangleMesh(T(TRI), [[0, 1, 2]])
    mesh.build_element_to_element_adj_link(adj)
    link = mesh.element_to_element_adj_link[adj]
    assert link.shape == (0, 2)
    assert link[:, 0].shape == (0,)
    mesh.build_element_to_element_adj_table(adj)
    assert mesh.element_to_element_adj_table[adj] == [[]]


def test_face_link_of_single_tetrahedron():
    mesh = TetrahedronMesh(T(TET), [[0, 1, 2, 3]])
    mesh.build_element_to_element_adj_link('face')
    assert mesh.element_to_element_adj_link['face'].shape == (0, 2)


def test_node_link_without_edges():
    mesh = Mesh(T(TRI), E(3), None, 'polygon')
    mesh.edge = E(2)
    mesh.build_node_to_node_adj_link()
    assert mesh.node_to_node_adj_link.shape == (0, 2)


# ---------------------------------------------------------------- Non-numeric arrays
def with_string_arrays(data):
    for attributes, n in ((data.GetPointData(), data.GetNumberOfPoints()),
                          (data.GetCellData(), data.GetNumberOfCells())):
        text = vtk.vtkStringArray()
        text.SetName('label')
        for i in range(n):
            text.InsertNextValue('v' + str(i))
        attributes.AddArray(text)
        value = numpy_to_vtk(np.arange(n, dtype=np.float64).reshape(-1, 1) + 0.5, deep=True)
        value.SetName('value')
        attributes.AddArray(value)
    return data


@pytest.mark.parametrize('reader', READERS)
def test_string_array_is_skipped(reader, capsys):
    mesh = TriangleMesh()
    getattr(mesh, reader)(with_string_arrays(polydata(TRI, polys=[[0, 1, 2]])), F64)
    assert list(mesh.node_data) == ['value']
    assert list(mesh.element_data) == ['value']
    assert mesh.node_data['value'].flatten().tolist() == [0.5, 1.5, 2.5]
    assert mesh.element_data['value'].flatten().tolist() == [0.5]
    assert 'skipped non-numeric array label' in capsys.readouterr().out


# ---------------------------------------------------------------- Triangle strips
def test_strip_mixed_with_quad_stays_ragged():
    data = polydata(STRIP, polys=[[0, 1, 3, 2]], strips=[[1, 3, 2, 4]], tags=[1, 2])
    mesh = PolygonMesh()
    mesh.read_mesh_vtk(data, F64)
    assert elements(mesh) == [[0, 1, 3, 2], [1, 3, 2], [2, 3, 4]]
    assert mesh.element_data['tag'].flatten().tolist() == [1, 2, 2]


# ---------------------------------------------------------------- Readers keep one dimension
def test_quad_is_not_kept_and_not_saved_as_a_tet(tmp_path, capsys):
    # The rule's own example: a polyhedron mesh never has a 2D quad.
    coords = TET + [[0., 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2]]
    path = tmp_path / 'tet_and_quad.vtk'
    write_legacy(unstructured(coords, [(vtk.VTK_TETRA, [0, 1, 2, 3]), (vtk.VTK_QUAD, [4, 5, 6, 7])], tags=[1, 2]), path)
    mesh = PolyhedronMesh()
    mesh.load(str(path), F64)
    assert elements(mesh) == [[0, 1, 2, 3]]
    assert mesh.element_type is None
    assert mesh.element_data['tag'].flatten().tolist() == [1]
    assert 'dropped 1 cells' in capsys.readouterr().out
    assert saved_cell_types(mesh, tmp_path / 'out.vtk') == [vtk.VTK_TETRA]


@pytest.mark.parametrize('cls, extension, expected', [(PolylineMesh, '.vtk', [[0, 1]]), (PolygonMesh, '.vtp', [[0, 1, 2], [0, 1, 3, 2]])],
                         ids=['polyline', 'polygon'])
def test_mixed_polydata_file_keeps_one_dimension(cls, extension, expected, tmp_path):
    path = tmp_path / ('mixed' + extension)
    data = polydata(STRIP, verts=[[0]], lines=[[0, 1]], polys=[[0, 1, 2], [0, 1, 3, 2]])
    write_legacy(data, path) if extension == '.vtk' else write_vtp(data, path)
    mesh = cls()
    mesh.load(str(path), F64)
    assert elements(mesh) == expected


def test_cell_with_too_few_nodes_is_dropped(capsys):
    mesh = PolylineMesh()
    mesh.read_mesh_vtk(polydata(STRIP, lines=[[0], [0, 1]], tags=[1, 2]), F64)
    assert elements(mesh) == [[0, 1]]
    assert mesh.element_data['tag'].shape == (1, 1)
    assert 'dropped 1 cells' in capsys.readouterr().out


def test_polydata_has_nothing_a_polyhedron_mesh_can_hold(capsys):
    data = polydata(node_list(6), verts=[[0]], lines=[[0, 1]], polys=[[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]], tags=[1, 2, 3, 4, 5])
    mesh = PolyhedronMesh()
    mesh.read_mesh_vtk(data, F64)
    assert mesh.element.shape == (0, 4)
    assert mesh.element_type is None
    assert mesh.element_data['tag'].shape == (0, 1)
    assert 'dropped 5 cells' in capsys.readouterr().out


# ---------------------------------------------------------------- Readers keep the types of solids
def test_solid_cell_type_survives_saving_again(tmp_path):
    # The hex forces labels, because the 'polyhedron' mesh_type has no 8-node cell type.
    data = unstructured(node_list(15), [(vtk.VTK_HEXAHEDRON, list(range(8))), (vtk.VTK_QUADRATIC_WEDGE, list(range(15)))])
    mesh = PolyhedronMesh()
    mesh.read_mesh_vtk(data, F64)
    assert labels(mesh) == ['HEX8', 'WEDGE15']
    assert saved_cell_types(mesh, tmp_path / 'out.vtk') == [vtk.VTK_HEXAHEDRON, vtk.VTK_QUADRATIC_WEDGE]


def test_labels_follow_the_kept_cells():
    coords = HEX + [[2., 0, 0], [3, 0, 0], [2, 1, 0], [2, 0, 1]]
    data = unstructured(coords, [(vtk.VTK_QUAD, [0, 1, 2, 3]), (vtk.VTK_HEXAHEDRON, list(range(8))), (vtk.VTK_VERTEX, [0]),
                                 (vtk.VTK_TETRA, [8, 9, 10, 11])], tags=[1, 2, 3, 4])
    mesh = PolyhedronMesh()
    mesh.read_mesh_vtk(data, F64)
    assert labels(mesh) == ['HEX8', 'TET4']
    assert mesh.element_data['tag'].flatten().tolist() == [2, 4]


def test_voxel_and_pixel_get_the_node_order_of_a_hex_and_a_quad(tmp_path):
    # A voxel lists its corners in grid order, which as a hex has crossed faces and no volume.
    coords = [[i, j, k] for k in (0., 4.) for j in (0., 3.) for i in (0., 2.)]
    grid_order, hex_order = list(range(8)), [0, 1, 3, 2, 4, 5, 7, 6]
    voxel = unstructured(coords, [(vtk.VTK_VOXEL, grid_order)])
    pixel = unstructured([[i, j, 0.] for j in (0., 2.) for i in (0., 3.)], [(vtk.VTK_PIXEL, [0, 1, 2, 3])])
    for reader in READERS:
        mixed = unstructured(coords, [(vtk.VTK_VOXEL, grid_order), (vtk.VTK_TETRA, [0, 1, 2, 4]), (vtk.VTK_VOXEL, grid_order)], tags=[1, 2, 3])
        mesh = PolyhedronMesh()
        getattr(mesh, reader)(mixed, F64)
        assert elements(mesh) == [hex_order, [0, 1, 2, 4], hex_order]
        assert labels(mesh) == ['HEX8', 'TET4', 'HEX8']
        assert mesh.element_data['tag'].flatten().tolist() == [1, 2, 3]
        assert [mixed.GetCell(2).GetPointId(k) for k in range(8)] == grid_order
        hex_mesh = HexahedronMesh()
        getattr(hex_mesh, reader)(voxel, F64)
        hex_mesh.update_element_volume()
        assert hex_mesh.element_volume.flatten().tolist() == pytest.approx([24.])
        quad_mesh = QuadMesh()
        getattr(quad_mesh, reader)(pixel, F64)
        quad_mesh.update_element_area_and_normal()
        assert quad_mesh.element_area.flatten().tolist() == pytest.approx([6.])
    # from a file, and what a save writes is a real hex
    write_legacy(voxel, tmp_path / 'voxel.vtk')
    for mesh_class in (HexahedronMesh, PolyhedronMesh):
        mesh = mesh_class()
        mesh.load(str(tmp_path / 'voxel.vtk'), 'float64')
        assert elements(mesh) == [hex_order]
        assert saved_cell_types(mesh, tmp_path / 'saved.vtk') == [vtk.VTK_HEXAHEDRON]
        again = mesh_class()
        again.load(str(tmp_path / 'saved.vtk'), 'float64')
        assert elements(again) == [hex_order]
    # a cell with the wrong number of nodes for its type is left alone
    mesh = PolyhedronMesh()
    mesh.read_mesh_vtk(unstructured(coords, [(vtk.VTK_VOXEL, [0, 1, 2, 3]), (vtk.VTK_TETRA, [0, 1, 2, 4])]), F64)
    assert elements(mesh) == [[0, 1, 2, 3], [0, 1, 2, 4]]


def test_solid_with_no_label_stays_unlabeled(tmp_path):
    # A hexagonal prism has no label. It stays, and saving it raises as it always did, not silently.
    mesh = PolyhedronMesh()
    mesh.read_mesh_vtk(unstructured(node_list(12), [(vtk.VTK_HEXAGONAL_PRISM, list(range(12))), (vtk.VTK_HEXAHEDRON, list(range(8)))]), F64)
    assert labels(mesh) == [None, 'HEX8']
    with pytest.raises(ValueError, match='not supported'):
        mesh.save_as_vtk(str(tmp_path / 'out.vtk'))


def test_read_replaces_labels_the_mesh_had():
    mesh = PolyhedronMesh(T(TET), [[0, 1, 2, 3]], element_type=['USER'])
    mesh.read_mesh_vtk(unstructured(TET, [(vtk.VTK_TETRA, [0, 1, 2, 3])]), F64)
    assert mesh.element_type is None


# ---------------------------------------------------------------- Saves reject another dimension
@pytest.mark.parametrize('label, mesh_type', [('S4R', 'polyhedron'), ('B31', 'polygon_tri3')])
def test_label_of_another_dimension_is_rejected_when_built(label, mesh_type):
    with pytest.raises(ValueError, match='holds only'):
        Mesh(T(TET), [[0, 1, 2, 3]], [label], mesh_type)


@pytest.mark.parametrize('convert', CONVERTERS)
def test_label_of_another_dimension_is_rejected_when_converted(convert):
    mesh = PolylineMesh(T(node_list(4)), [[0, 1, 2, 3]])
    mesh.element_type = np.array(['C3D4'])
    with pytest.raises(ValueError, match='holds only'):
        getattr(mesh, convert)()


def test_label_of_another_dimension_is_rejected_by_the_text_writers(monkeypatch, tmp_path):
    # save_as_vtk uses the text writers on VTK 9.0, and they go by node count.
    monkeypatch.setattr(Mesh, 'get_vtk_version', staticmethod(lambda: (9, 0)), raising=False)
    mesh = PolylineMesh(T(node_list(2)), [[0, 1]])
    mesh.element_type = np.array(['S3'])
    path = tmp_path / 'out.vtk'
    with pytest.raises(ValueError, match='holds only'):
        mesh.save_as_vtk(str(path))
    assert not path.exists()


def test_element_with_too_few_nodes_is_rejected(tmp_path):
    path = tmp_path / 'out.vtp'
    with pytest.raises(ValueError, match='holds only'):
        PolygonMesh(T(TET), [[0, 1]]).save(str(path))
    assert not path.exists()


def test_one_small_element_among_ragged_elements_is_rejected(tmp_path):
    path = tmp_path / 'out.vtk'
    with pytest.raises(ValueError, match='holds only'):
        PolyhedronMesh(T(TET), [[0, 1, 2, 3], [0, 1, 2]]).save(str(path))
    assert not path.exists()


# ---------------------------------------------------------------- The torch format is the exception to the rule
def test_torch_format_saves_and_loads_a_mesh_exactly_as_it_is(tmp_path):
    mesh = PolylineMesh(T(TET), [[0]])
    path = str(tmp_path / 'mesh.pt')
    mesh.save_as_torch(path)
    restored = PolylineMesh()
    restored.load_from_torch(path)
    assert elements(restored) == [[0]]


def test_torch_format_loads_a_file_of_another_dimension(tmp_path):
    path = str(tmp_path / 'mesh.pt')
    TetrahedronMesh(T(TET), [[0, 1, 2, 3]]).save_as_torch(path)
    mesh = PolylineMesh()
    mesh.load_from_torch(path)
    assert elements(mesh) == [[0, 1, 2, 3]]


# ---------------------------------------------------------------- VTK version check
def fake_vtk(version):
    major, minor = [int(part) for part in version.split('.')[:2]]

    class Version:
        GetVTKVersion = staticmethod(lambda: version)
        GetVTKMajorVersion = staticmethod(lambda: major)
        GetVTKMinorVersion = staticmethod(lambda: minor)

    class Proxy:
        vtkVersion = Version

        def __getattr__(self, name):
            return getattr(vtk, name)
    return Proxy()


def test_newer_major_version_uses_vtk_writer(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(mesh_module, 'vtk', fake_vtk('11.0.0'))
    monkeypatch.setattr(mesh_module, 'save_polygon_mesh_as_vtk', lambda mesh, filename: calls.append(filename))
    path = tmp_path / 'mesh.vtk'
    TriangleMesh(T(TRI), [[0, 1, 2]]).save_as_vtk(str(path))
    assert calls == []
    assert path.read_text().startswith('# vtk DataFile Version 4.2')


# ---------------------------------------------------------------- Empty files and bad files
BAD_LEGACY = {
    'wrong_dataset': '# vtk DataFile Version 4.2\nvtk output\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS 0 double\n',
    'cut_before_points': '# vtk DataFile Version 4.2\nvtk output\nASCII\nDATASET POLYDATA\n',
}


def test_empty_vtp_round_trip(tmp_path):
    for k, source in enumerate((QuadMesh(), QuadMesh(torch.empty((0, 3), dtype=F64), E(4)))):
        path = str(tmp_path / (str(k) + '.vtp'))
        source.save_as_vtp(path)
        restored = QuadMesh()
        restored.load_from_vtp(path, F64)
        assert restored.node.shape == (0, 3)
        assert restored.node.dtype == F64
        assert restored.element.shape == (0, 4)


def test_return_mesh_vtk_for_empty_file(tmp_path):
    path = str(tmp_path / 'empty.vtk')
    TriangleMesh().save_as_vtk(path)
    output = PolygonMesh().load_from_vtk(path, F64, return_mesh_vtk=True)
    assert output.GetNumberOfPoints() == 0


@pytest.mark.parametrize('name', list(BAD_LEGACY))
def test_bad_vtk_file_is_still_an_error(name, tmp_path):
    path = tmp_path / 'bad.vtk'
    path.write_text(BAD_LEGACY[name])
    mesh = TriangleMesh(T(TRI), [[0, 1, 2]])
    with pytest.raises(ValueError):
        mesh.load_from_vtk(str(path), F64)
    assert elements(mesh) == [[0, 1, 2]]


# ---------------------------------------------------------------- Warnings about loading code
def test_torch_load_docstrings_warn():
    for method in (Mesh.load_from_torch, Mesh.load):
        doc = method.__doc__ or ''
        assert 'WARNING' in doc and 'run code' in doc


# ---------------------------------------------------------------- Failed writes
def test_failed_vtk_and_vtp_write_raises(tmp_path):
    mesh = TriangleMesh(T(TRI), [[0, 1, 2]])
    solid = TetrahedronMesh(T(TET), [[0, 1, 2, 3]])
    missing = tmp_path / 'missing'
    for save, name in ((mesh.save_as_vtk, 'm.vtk'), (mesh.save_as_vtp, 'm.vtp'), (solid.save_as_vtk, 's.vtk')):
        with pytest.raises(RuntimeError, match='cannot write'):
            save(str(missing / name))
        save(str(tmp_path / name))
        assert (tmp_path / name).stat().st_size > 0
    if Path('/dev/full').exists():
        with pytest.raises(RuntimeError, match='OutOfDiskSpace'):
            mesh.save_as_vtk('/dev/full')


# ---------------------------------------------------------------- Element width
def test_class_with_one_element_shape_rejects_another_width(tmp_path):
    node, tri, quad, tet, hex8 = T(HEX), [[0, 1, 2]], [[0, 1, 2, 3]], [[0, 1, 3, 4]], [list(range(8))]
    wrong = ((TriangleMesh, quad), (QuadMesh, tri), (Tri6Mesh, tri), (TetrahedronMesh, hex8),
             (Tet10Mesh, tet), (HexahedronMesh, tet), (TriangleMesh, tri+quad))
    for mesh_class, element in wrong:
        with pytest.raises(ValueError, match='nodes, but found'):
            mesh_class(node, element)
        mesh = mesh_class()
        with pytest.raises(ValueError, match='nodes, but found'):
            mesh.copy(node, element)
        assert mesh.node is None
    TriangleMesh(node, tri).copy(node, tri)
    assert TriangleMesh(node, []).element.shape == (0, 3)
    # a load that fails keeps the mesh, and the generic classes still take these files
    QuadMesh(node, quad).save_as_vtk(str(tmp_path / 'quad.vtk'))
    QuadMesh(node, quad).save_as_vtp(str(tmp_path / 'quad.vtp'))
    HexahedronMesh(node, hex8).save_as_vtk(str(tmp_path / 'hex.vtk'))
    triangle = TriangleMesh(node, tri)
    for name in ('quad.vtk', 'quad.vtp'):
        with pytest.raises(ValueError, match='must have 3 nodes'):
            triangle.load(str(tmp_path / name), 'float64')
    assert triangle.element.shape == (1, 3)
    with pytest.raises(ValueError, match='nodes, but found'):
        TriangleMesh().read_mesh_vtk(Mesh.load_vtk_file(str(tmp_path / 'quad.vtk'), 'polygon'), F64)
    with pytest.raises(ValueError, match='must have 4 nodes'):
        TetrahedronMesh().load(str(tmp_path / 'hex.vtk'), 'float64')
    generic = PolygonMesh()
    generic.load(str(tmp_path / 'quad.vtk'), 'float64')
    assert generic.element.shape == (1, 4)


# ---------------------------------------------------------------- Files that end too early
def write_legacy_as(data, path, version, binary):
    writer = vtk.vtkUnstructuredGridWriter() if data.IsA('vtkUnstructuredGrid') else vtk.vtkPolyDataWriter()
    writer.SetInputData(data)
    writer.SetFileName(str(path))
    writer.SetFileVersion(version)
    writer.SetFileTypeToBinary() if binary else writer.SetFileTypeToASCII()
    assert writer.Write() == 1


def test_file_cut_at_any_byte_raises_or_gives_only_right_data(tmp_path, capsys):
    surface = PolygonMesh(T(HEX[:4]), [[0, 1, 2], [0, 2, 3]])
    solid = PolyhedronMesh(T(HEX), [list(range(8))], element_type=['C3D8'])
    for mesh in (surface, solid):
        n_node, n_element = len(mesh.node), len(mesh.element)
        mesh.node_data['u'] = torch.arange(1., 2*n_node+1, dtype=F64).reshape(n_node, 2)*1.5
        mesh.node_data['v'] = torch.arange(1., n_node+1, dtype=F64).reshape(n_node, 1)*2.5
        mesh.element_data['s'] = torch.arange(1., n_element+1, dtype=F64).reshape(n_element, 1)*0.5
    full, cut = tmp_path / 'full.vtk', tmp_path / 'cut.vtk'
    for mesh in (surface, solid):
        for version, binary in ((42, False), (51, False), (42, True), (51, True)):
            write_legacy_as(mesh.convert_to_vtk(), full, version, binary)
            text = full.read_bytes()
            for size in range(len(text)):
                cut.write_bytes(text[:size])
                loaded = type(mesh)()
                capsys.readouterr()
                try:
                    loaded.load_from_vtk(str(cut), 'float64')
                except ValueError:
                    continue
                if not binary and not text[:size][-1:].isspace():
                    assert 'may be cut' in capsys.readouterr().out, size
                    continue
                # Only a cut between two whole sections gets here, and all it can lose is whole sections.
                assert torch.equal(loaded.node, mesh.node), (version, binary, size)
                assert len(loaded.element) == 0 or torch.equal(torch.as_tensor(loaded.element), mesh.element), size
                for mine, theirs in ((loaded.node_data, mesh.node_data), (loaded.element_data, mesh.element_data)):
                    for name in mine:
                        assert torch.equal(mine[name], theirs[name]), (version, binary, size, name)


def test_cut_check_knows_every_section_that_vtk_writes(tmp_path):
    coords = [[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 2, 0]]
    data = polydata(coords, verts=[[0]], lines=[[0, 1]], polys=[[0, 1, 2]], strips=[[0, 1, 2, 3, 4]])
    attributes = data.GetPointData()
    for setter, components, vtk_array in ((attributes.SetScalars, 2, vtk.vtkFloatArray()), (attributes.SetVectors, 3, vtk.vtkDoubleArray()),
                                          (attributes.SetNormals, 3, vtk.vtkFloatArray()), (attributes.SetTCoords, 2, vtk.vtkFloatArray()),
                                          (attributes.SetTensors, 9, vtk.vtkDoubleArray()), (attributes.SetGlobalIds, 1, vtk.vtkIntArray())):
        vtk_array.SetNumberOfComponents(components)
        vtk_array.SetNumberOfTuples(5)
        vtk_array.SetName('a'+str(components))
        vtk_array.Fill(1)
        setter(vtk_array)
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetNumberOfTuples(4)
    colors.SetName('colors')
    colors.Fill(1)
    data.GetCellData().SetScalars(colors)
    for name, vtk_array in (('short', vtk.vtkShortArray()), ('int', vtk.vtkIntArray())):
        vtk_array.SetNumberOfTuples(4)
        vtk_array.SetName(name)
        vtk_array.Fill(1)
        data.GetCellData().AddArray(vtk_array)
    # The last section of the file. A binary bit array ends the check, and must not be called a cut.
    bit = vtk.vtkBitArray()
    bit.SetNumberOfComponents(3)
    bit.SetNumberOfTuples(5)
    bit.SetName('bit')
    attributes.AddArray(bit)
    data.GetFieldData().AddArray(colors)
    path = tmp_path / 'all.vtk'
    for version, binary in ((42, False), (51, False), (42, True), (51, True)):
        write_legacy_as(data, path, version, binary)
        text = path.read_bytes()
        assert find_cut_in_legacy_vtk_file(str(path)) is None
        # a cut inside each section must be found
        keywords = [b'SCALARS', b'VECTORS', b'NORMALS', b'TEXTURE_COORDINATES', b'TENSORS', b'GLOBAL_IDS', b'COLOR_SCALARS', b'short', b'int']
        for keyword in keywords:
            start = text.index(b'\n'+keyword)+1
            data_start = text.index(b'\n', text.index(b'LOOKUP', start) if keyword == b'SCALARS' else start)+1
            path.write_bytes(text[:data_start+4])
            assert find_cut_in_legacy_vtk_file(str(path)) is not None, (version, binary, keyword)
    # The other ways to end too early. A load would only lose whole sections, so the check must say so.
    head = b'# vtk DataFile Version 4.2\nvtk output\n'
    point = b'POINTS 1 double\n0 0 0\n'
    for ending in (b'ASCII\nDATASET POLYDATA\n'+point+b'POINT_DA',
                   b'ASCII\nDATASET UNSTRUCTURED_GRID\n'+point+b'CELLS 0 0\n', b'BINARY\nDATASET POLYDATA\nPOINTS 1 dou'):
        path.write_bytes(head+ending)
        assert find_cut_in_legacy_vtk_file(str(path)) is not None, ending


LEGACY_POINT = [0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0.]
# The first bytes of each coordinate are white space, which must be read as data.
LEGACY_POINT_LIKE_WHITE_SPACE = [struct.unpack('>f', b'\n\n\n\n')[0], struct.unpack('>f', b'    ')[0],
                                 struct.unpack('>f', b'\r\n\t ')[0]]+LEGACY_POINT[3:]
LEGACY_HEADERS = ('POINTS', 'POLYGONS', 'SCALARS', 'LOOKUP_TABLE', 'VECTORS', 'POINT_DATA', 'FIELD', 'f 2', 'CELL_DATA', 'g 1')


def legacy_polygon_file(binary, pad='', only=None, line_break='\n', ending=b'', with_arrays=True, point=LEGACY_POINT):
    # Built by hand, because VTK's writer never puts anything after the last word of a header.
    # pad goes before the line break of each header line, or of the line that starts with only.
    def header(text):
        return (text+(pad if only is None or text.startswith(only) else '')+line_break).encode()

    def values(code, data):
        if binary:
            return struct.pack('>'+str(len(data))+code, *data)+line_break.encode()
        return (' '.join(str(value) for value in data)+line_break).encode()
    text = ('# vtk DataFile Version 4.2'+line_break+'vtk output'+line_break+('BINARY' if binary else 'ASCII')+line_break
            +'DATASET POLYDATA'+line_break).encode()
    text += header('POINTS 4 float')+values('f', point)
    text += header('POLYGONS 2 8')+values('i', [3, 0, 1, 3, 3, 0, 3, 2])
    if with_arrays:
        text += header('POINT_DATA 4')
        text += header('SCALARS s float 1')+header('LOOKUP_TABLE default')+values('f', [1., 2., 3., 4.])
        text += header('VECTORS v float')+values('f', [float(k) for k in range(12)])
        text += header('FIELD FieldData 1')+header('f 2 4 double')+values('d', [float(k) for k in range(8)])
        text += header('CELL_DATA 2')
        text += header('FIELD FieldData 1')+header('g 1 2 float')+values('f', [5., 6.])
    return text+ending


def loaded_legacy_file(text, path):
    path.write_bytes(text)
    mesh = PolygonMesh()
    mesh.load_from_vtk(str(path), 'float64')
    return mesh


def assert_same_mesh(mesh, expected):
    assert torch.equal(mesh.node, expected.node)
    assert torch.equal(torch.as_tensor(mesh.element), torch.as_tensor(expected.element))
    for mine, theirs in ((mesh.node_data, expected.node_data), (mesh.element_data, expected.element_data)):
        assert sorted(mine) == sorted(theirs)
        for name in mine:
            assert torch.equal(mine[name], theirs[name]), name


FILES_THAT_VTK_READS = {
    **{'space on the '+only+' line': dict(pad=' ', only=only) for only in LEGACY_HEADERS},
    'space on each header line': dict(pad=' '),
    'tab on each header line': dict(pad='\t'),
    'CRLF': dict(line_break='\r\n'),
    'CRLF and space': dict(pad=' ', line_break='\r\n'),
    'blank lines at the end': dict(ending=b'\n\n  \n'),
    **{'last word '+repr(ending): dict(ending=ending) for ending in (b'END', b'END\n', b'END ', b'end\n', b'XYZ', b'POINT_DA\n', b'POINT_DA ')},
    **{'no arrays after '+repr(ending): dict(with_arrays=False, ending=ending) for ending in (
        b'POINT_DATA 4', b'POINT_DATA 4\n', b'CELL_DATA 2\n', b'POINT_DATA 0\n', b'POINT_DATA 4\nCELL_DATA 2\n', b'POINT_DATA 4 \n\n')},
}


@pytest.mark.parametrize('binary', [False, True])
@pytest.mark.parametrize('options', FILES_THAT_VTK_READS.values(), ids=FILES_THAT_VTK_READS.keys())
def test_cut_check_accepts_a_file_that_vtk_reads(tmp_path, binary, options):
    expected = loaded_legacy_file(legacy_polygon_file(binary, with_arrays=options.get('with_arrays', True)), tmp_path/'plain.vtk')
    assert len(expected.element) == 2
    path = tmp_path/'file.vtk'
    path.write_bytes(legacy_polygon_file(binary, **options))
    assert find_cut_in_legacy_vtk_file(str(path)) is None
    assert_same_mesh(loaded_legacy_file(path.read_bytes(), path), expected)


@pytest.mark.parametrize('pad', ['', ' ', ' \t '])
@pytest.mark.parametrize('line_break', ['\n', '\r\n'])
def test_cut_inside_binary_data_is_found_whatever_is_after_the_header(tmp_path, pad, line_break):
    text = legacy_polygon_file(True, pad=pad, line_break=line_break)
    path = tmp_path/'cut.vtk'
    path.write_bytes(text)
    assert find_cut_in_legacy_vtk_file(str(path)) is None
    for header in (b'POINTS', b'POLYGONS', b'LOOKUP_TABLE', b'VECTORS', b'f 2', b'g 1'):
        data = text.index(b'\n', text.index(header))+1
        path.write_bytes(text[:data+1])
        assert 'fewer values' in find_cut_in_legacy_vtk_file(str(path)), header
    # no line break yet, so the data has not started
    path.write_bytes(text[:text.index(b'POINTS 4 float')+len(b'POINTS 4 float')])
    assert 'fewer values' in find_cut_in_legacy_vtk_file(str(path))


def test_binary_data_that_looks_like_white_space_is_data(tmp_path):
    for pad in ('', ' '):
        text = legacy_polygon_file(True, pad=pad, point=LEGACY_POINT_LIKE_WHITE_SPACE)
        mesh = loaded_legacy_file(text, tmp_path/'file.vtk')
        assert torch.equal(mesh.node.flatten().float(), torch.tensor(LEGACY_POINT_LIKE_WHITE_SPACE, dtype=torch.float32))
        (tmp_path/'file.vtk').write_bytes(text[:text.index(b'\n', text.index(b'POINTS'))+2])
        assert find_cut_in_legacy_vtk_file(str(tmp_path/'file.vtk')) is not None


@pytest.mark.parametrize('binary', [False, True])
@pytest.mark.parametrize('ending', [b'POINT_DA', b'point_da', b'CELL_', b'CELL', b'P', b'F', b'FIE', b'SCAL', b'PEDIGREE_', b'METADA', b'DIMENS'])
def test_file_that_ends_inside_a_keyword_is_cut(tmp_path, binary, ending):
    path = tmp_path/'cut.vtk'
    path.write_bytes(legacy_polygon_file(binary, ending=ending))
    assert find_cut_in_legacy_vtk_file(str(path)) == 'it ends inside a keyword'


@pytest.mark.parametrize('binary', [False, True])
@pytest.mark.parametrize('version, keyword', [(51, b'OFFSETS'), (51, b'CONNECTIVITY'), (42, b'LOOKUP_TABLE'), (51, b'LOOKUP_TABLE')])
def test_file_that_ends_inside_a_keyword_a_section_needs_is_cut(tmp_path, binary, version, keyword):
    # A load would give a mesh with no elements, or with no array, and no error.
    data = PolygonMesh(T(HEX[:4]), [[0, 1, 2], [0, 2, 3]]).convert_to_vtk()
    scalar = vtk.vtkFloatArray()
    scalar.SetName('s')
    scalar.SetNumberOfTuples(4)
    scalar.Fill(1)
    data.GetPointData().SetScalars(scalar)
    path = tmp_path/'cut.vtk'
    write_legacy_as(data, path, version, binary)
    text = path.read_bytes()
    assert find_cut_in_legacy_vtk_file(str(path)) is None
    start = text.index(b'\n'+keyword)+1
    for size in range(1, len(keyword)):
        for word in (keyword[:size], keyword[:size].lower()):
            path.write_bytes(text[:start]+word)
            assert find_cut_in_legacy_vtk_file(str(path)) == 'it ends inside a keyword', word
            with pytest.raises(ValueError, match='ends too early'):
                PolygonMesh().load_from_vtk(str(path), 'float64')
    # the whole word, but nothing after it
    path.write_bytes(text[:start]+keyword)
    assert 'not complete' in find_cut_in_legacy_vtk_file(str(path))


@pytest.mark.parametrize('binary', [False, True])
@pytest.mark.parametrize('version, keyword', [(51, b'OFFSETS'), (51, b'CONNECTIVITY'), (42, b'LOOKUP_TABLE')])
@pytest.mark.parametrize('word', [b'XYZ', b'FSETS', b'L\n', b'O\n', b'C '])
def test_other_word_where_a_section_needs_a_keyword_is_not_a_cut(tmp_path, binary, version, keyword, word):
    # Nothing was cut if another word stands there, or if white space comes after a part of the keyword.
    data = PolygonMesh(T(HEX[:4]), [[0, 1, 2], [0, 2, 3]]).convert_to_vtk()
    scalar = vtk.vtkFloatArray()
    scalar.SetName('s')
    scalar.SetNumberOfTuples(4)
    scalar.Fill(1)
    data.GetPointData().SetScalars(scalar)
    path = tmp_path/'file.vtk'
    write_legacy_as(data, path, version, binary)
    text = path.read_bytes()
    path.write_bytes(text[:text.index(b'\n'+keyword)+1]+word)
    assert find_cut_in_legacy_vtk_file(str(path)) is None


@pytest.mark.parametrize('binary', [False, True])
def test_section_with_no_value_needs_no_line_break(tmp_path, binary):
    head = b'# vtk DataFile Version 4.2\nvtk output\n'+(b'BINARY' if binary else b'ASCII')+b'\nDATASET POLYDATA\n'
    path = tmp_path/'file.vtk'
    for ending in (b'POINTS 0 float', b'POINTS 0 float\n', b'POINTS 0 float\nPOLYGONS 0 0', b'POINTS 0 float\nPOLYGONS 0 0 \n'):
        path.write_bytes(head+ending)
        assert find_cut_in_legacy_vtk_file(str(path)) is None, ending


# ---------------------------------------------------------------- Exception handling
def test_no_bare_except():
    package = Path(mesh_module.__file__).resolve().parents[1]
    handlers = [(path.name, node) for path in package.rglob('*.py')
                for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))) if isinstance(node, ast.ExceptHandler)]
    assert handlers
    assert [(name, h.lineno) for name, h in handlers if h.type is None] == []


def test_keyboard_interrupt_during_vtk_import_propagates():
    code = textwrap.dedent("""
        import sys
        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == 'vtk' or name.startswith('vtk.'):
                    raise KeyboardInterrupt()
        sys.meta_path.insert(0, Blocker())
        try:
            import torch_fea_mesh.mesh.mesh as m
        except KeyboardInterrupt:
            print('RESULT interrupted')
        else:
            print('RESULT flag', m._Flag_VTK_IMPORT_)
    """)
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=120,
                            cwd=Path(__file__).resolve().parents[2])
    assert 'RESULT interrupted' in result.stdout


def test_polygon_mesh_processing_vtk_import_handler():
    # mesh.py has its own import of vtk and runs first, so vtk is blocked only after that.
    code = textwrap.dedent("""
        import sys
        import torch_fea_mesh.mesh.mesh
        assert 'torch_fea_mesh.processing.polygon_mesh_processing' not in sys.modules
        del sys.modules['vtk']
        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == 'vtk':
                    raise {error}
        sys.meta_path.insert(0, Blocker())
        try:
            import torch_fea_mesh.processing.polygon_mesh_processing as m
        except KeyboardInterrupt:
            print('RESULT interrupted')
        else:
            print('RESULT imported', hasattr(m, 'vtk'))
    """)

    def run(error):
        result = subprocess.run([sys.executable, '-B', '-c', code.format(error=error)], capture_output=True, text=True,
                                timeout=120, cwd=Path(__file__).resolve().parents[2])
        return result.stdout

    assert 'RESULT interrupted' in run('KeyboardInterrupt()')
    output = run("ModuleNotFoundError('blocked')")
    assert 'RESULT imported False' in output
    assert 'cannot import vtk: blocked' in output


def test_load_error_keeps_its_cause(tmp_path):
    for name, text, loader in (('bad.vtk', 'not vtk\n', 'load_from_vtk'), ('bad.vtp', '<not xml', 'load_from_vtp')):
        path = tmp_path / name
        path.write_text(text)
        with pytest.raises(ValueError, match='cannot convert vtk data to mesh') as caught:
            getattr(TriangleMesh(), loader)(str(path), F64)
        assert caught.value.__cause__ is not None
        assert 'cannot load node' in str(caught.value)


def ragged_object_array():
    array = np.empty(2, dtype=object)
    array[0] = [0, 1, 2]
    array[1] = [0, 1, 2, 3]
    return array


@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize('element', [[torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2, 3])], ragged_object_array()],
                         ids=['list_of_tensors', 'object_array'])
def test_ragged_element_still_falls_back_to_list(element):
    mesh = Mesh(T(node_list(4)), element, None, 'polygon')
    assert isinstance(mesh.element, list)
    assert [len(e) for e in mesh.element] == [3, 4]
    if isinstance(element, list):
        mesh.copy(T(node_list(4)), element)
        assert isinstance(mesh.element, list)
        assert [len(e) for e in mesh.element] == [3, 4]
