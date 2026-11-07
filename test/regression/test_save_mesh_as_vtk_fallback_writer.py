"""The pure-Python VTK writers that Mesh.save_as_vtk uses when VTK is unavailable.

They must behave as the VTK path does. Each test saves through the public save_as_vtk with the
fallback selected, and reads the file back with VTK. Keep this file small: add a test only for a
behaviour nothing here catches. A new input usually needs only a new row in SAME_AS_VTK_PATH.
"""
import os

import numpy as np
import pytest
import torch

vtk = pytest.importorskip('vtk')

from vtk.util.numpy_support import vtk_to_numpy

import torch_fea_mesh.io.save_mesh_as_vtk_file as writer_module
import torch_fea_mesh.mesh.mesh as mesh_module
from torch_fea_mesh.mesh.hexahedron_mesh import HexahedronMesh
from torch_fea_mesh.mesh.polygon_mesh import PolygonMesh
from torch_fea_mesh.mesh.polyhedron_mesh import PolyhedronMesh
from torch_fea_mesh.mesh.polyline_mesh import PolylineMesh
from torch_fea_mesh.mesh.quad_mesh import QuadMesh
from torch_fea_mesh.mesh.quad_triangle_mesh import QuadTriangleMesh
from torch_fea_mesh.mesh.tet10_mesh import Tet10Mesh
from torch_fea_mesh.mesh.tetrahedron_mesh import TetrahedronMesh
from torch_fea_mesh.mesh.tri6_mesh import Tri6Mesh
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh

pytestmark = [pytest.mark.io, pytest.mark.vtk]

F64 = torch.float64
TRIANGLE_NODE = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
TRIANGLE_ELEMENT = [[0, 1, 2], [1, 3, 2]]


def triangle_mesh():
    return TriangleMesh(torch.tensor(TRIANGLE_NODE, dtype=F64), TRIANGLE_ELEMENT)


def polyline_mesh():
    return PolylineMesh(torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=F64), [[0, 1], [1, 2]])


def tet_mesh():
    return TetrahedronMesh(torch.tensor(TRIANGLE_NODE, dtype=F64), [[0, 1, 2, 3]])


@pytest.fixture
def save_fallback(monkeypatch):
    def save(mesh, path):
        with monkeypatch.context() as patch:
            patch.setattr(mesh_module, '_Flag_VTK_IMPORT_', False)
            mesh.save_as_vtk(str(path))
    return save


def load(mesh_class, path):
    mesh = mesh_class()
    mesh.load_from_vtk(str(path), 'float64')
    return mesh


def test_polyline_keeps_element_data(save_fallback, tmp_path):
    mesh = polyline_mesh()
    mesh.node_data['u'] = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=F64)
    mesh.element_data['stress'] = torch.tensor([[10.], [20.]], dtype=F64)
    save_fallback(mesh, tmp_path/'polyline.vtk')
    loaded = load(PolylineMesh, tmp_path/'polyline.vtk')
    assert torch.equal(loaded.node_data['u'], mesh.node_data['u'])
    assert torch.equal(loaded.element_data['stress'], mesh.element_data['stress'])


def test_solid_elements_get_their_cell_type_whatever_the_node_numbering(save_fallback, tmp_path):
    # The first node of an element is not 0: VTK_POLYHEDRON would make the reader read it as a
    # face count, and could crash this process. So the types are checked in the text first.
    expected = {4: vtk.VTK_TETRA, 5: vtk.VTK_PYRAMID, 6: vtk.VTK_WEDGE, 8: vtk.VTK_HEXAHEDRON,
                10: vtk.VTK_QUADRATIC_TETRA, 15: vtk.VTK_QUADRATIC_WEDGE, 20: vtk.VTK_QUADRATIC_HEXAHEDRON}
    element, start = [], 7
    for count in expected:
        element.append(list(range(start, start+count)))
        start += count
    labels = ['TET4', 'PYRA5', 'WEDGE6', 'HEX8', 'TET10', 'WEDGE15', 'HEX20']
    mesh = PolyhedronMesh(torch.rand(start, 3, dtype=F64), element, np.array(labels))
    mesh.node_data['u'] = torch.rand(start, 2, dtype=F64)
    mesh.element_data['s'] = torch.rand(len(element), 1, dtype=F64)
    path = tmp_path/'solid.vtk'
    save_fallback(mesh, path)
    written = path.read_text().split('CELL_TYPES')[1].split('POINT_DATA')[0].split()[1:]
    assert written == [str(code) for code in expected.values()]
    loaded = load(PolyhedronMesh, path)
    assert [[int(i) for i in e] for e in loaded.element] == element
    assert list(loaded.element_type) == labels
    assert torch.equal(loaded.node_data['u'], mesh.node_data['u'])
    assert torch.equal(loaded.element_data['s'], mesh.element_data['s'])


def test_solid_element_of_another_node_count_raises_and_writes_nothing(save_fallback, tmp_path):
    path = tmp_path/'solid.vtk'
    mesh = PolyhedronMesh(torch.rand(17, 3, dtype=F64), [[0, 1, 2, 3], list(range(4, 17))])
    with pytest.raises(ValueError, match='not supported: polyhedron 13'):
        save_fallback(mesh, path)
    assert not path.exists()


def make_node(n_node):
    return torch.tensor([[float(k), float(k % 3), float(k % 2)] for k in range(n_node)], dtype=F64)


def mesh_of(mesh_class, n_node, labels=None, n_element=1, mesh_type=None, element=None, **attributes):
    """Return a function that makes the mesh. Labels and attributes are set after the constructor, which checks them."""
    def make():
        mesh = mesh_class(make_node(n_node), [list(range(n_node))]*n_element if element is None else element)
        if labels is not None:
            mesh.element_type = np.array(labels, dtype=object)
        if mesh_type is not None:
            mesh.mesh_type = mesh_type
        for name, value in attributes.items():
            setattr(mesh, name, value)
        return mesh
    return make


def mesh_with_element(make, value, n_node):
    def make_with_element():
        mesh = make()
        mesh.element = value(n_node)
        return mesh
    return make_with_element


def mesh_with_field(make, where, name, value):
    def make_with_field():
        mesh = make()
        getattr(mesh, where)[name] = value
        return mesh
    return make_with_field


def seen_by_a_vtk_reader(path):
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()

    def arrays(group):
        return {group.GetAbstractArray(k).GetName(): vtk_to_numpy(group.GetAbstractArray(k)).tolist()
                for k in range(group.GetNumberOfArrays())}
    cells = []
    for k in range(data.GetNumberOfCells()):
        cell = data.GetCell(k)
        cells.append((cell.GetCellType(), [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]))
    points = vtk_to_numpy(data.GetPoints().GetData()).tolist() if data.GetNumberOfPoints() > 0 else []
    # repr, because nan != nan
    return repr((data.GetClassName(), points, cells, arrays(data.GetPointData()), arrays(data.GetCellData())))


def behaviour(save, make, path):
    try:
        save(make(), path)
    except Exception as error:
        assert not path.exists()
        return type(error), str(error)
    return seen_by_a_vtk_reader(path)


TRI6_ELEMENT = [[0, 1, 2, 3, 4, 5]]
SAME_AS_VTK_PATH = {
    # each class
    'triangle': mesh_of(TriangleMesh, 3), 'quad': mesh_of(QuadMesh, 4), 'tri6': mesh_of(Tri6Mesh, 6, n_element=2),
    'quad and triangle': mesh_of(QuadTriangleMesh, 5, element=[[0, 1, 2, 3], [1, 2, 4]]),
    'tet': mesh_of(TetrahedronMesh, 4), 'tet10': mesh_of(Tet10Mesh, 10), 'hex': mesh_of(HexahedronMesh, 8),
    'polygon': mesh_of(PolygonMesh, 5), 'polygon of 6': mesh_of(PolygonMesh, 6), 'polygon of 8': mesh_of(PolygonMesh, 8),
    'ragged polygon': mesh_of(PolygonMesh, 7, element=[[0, 1, 2], [1, 2, 3, 4], list(range(7))]),
    'line': mesh_of(PolylineMesh, 2), 'polyline': mesh_of(PolylineMesh, 3),
    'ragged polyline': mesh_of(PolylineMesh, 4, element=[[0, 1], [1, 2, 3]]),
    # mesh_type, with no label
    'polygon_quad8': mesh_of(PolygonMesh, 8, mesh_type='polygon_quad8'),
    'polygon_tri6 of 3': mesh_of(PolygonMesh, 3, mesh_type='polygon_tri6'),
    'polygon_tri6 of 5': mesh_of(PolygonMesh, 5, mesh_type='polygon_tri6'),
    'polygon_tri6 of 8': mesh_of(PolygonMesh, 8, mesh_type='polygon_tri6'),
    'polygon_quad8 of 4': mesh_of(PolygonMesh, 8, mesh_type='polygon_quad8', element=[[0, 1, 2, 3]]),
    'polygon_quad8 of 6': mesh_of(PolygonMesh, 6, mesh_type='polygon_quad8'),
    'polygon_tri6 ragged': mesh_of(PolygonMesh, 6, mesh_type='polygon_tri6', element=[[0, 1, 2, 3, 4, 5], [0, 1, 2]]),
    'polyhedron_hex8': mesh_of(PolyhedronMesh, 8, mesh_type='polyhedron_hex8'),
    'polyhedron_wedge6': mesh_of(PolyhedronMesh, 6, mesh_type='polyhedron_wedge6'),
    'polyhedron_tet10': mesh_of(PolyhedronMesh, 10, mesh_type='polyhedron_tet10'),
    'polyhedron_hex8 of 6': mesh_of(PolyhedronMesh, 6, mesh_type='polyhedron_hex8'),
    'polyhedron_hex20': mesh_of(PolyhedronMesh, 20, mesh_type='polyhedron_hex20'),
    # solids: a label says the shape, and the node count alone does not
    **{'solid of '+str(n): mesh_of(PolyhedronMesh, n) for n in (4, 5, 6, 7, 8, 10, 13, 15, 20)},
    **{'solid of '+str(n)+' labelled '+label: mesh_of(PolyhedronMesh, n, [label]) for n, label in (
        (4, 'TET4'), (4, 'C3D4H'), (5, 'PYRA5'), (6, 'WEDGE6'), (6, 'PENTA6'), (8, 'HEX8'), (8, 'C3D8R'), (8, 'hex8'), (8, ' HEX8 '),
        (10, 'TET10'), (15, 'C3D15'), (20, 'HEX20'), (20, 'C3D20R'),
        (8, 'PYRA5'), (4, 'HEX8'), (6, 'HEX8'), (8, 'MY_ELEMENT'), (4, 'MY_ELEMENT'), (4, 'TRI3'), (4, 'LINE2'))},
    'solid of 8 with a missing label': mesh_of(PolyhedronMesh, 8, [None]),
    'solid of 4 with a missing label': mesh_of(PolyhedronMesh, 4, [None]),
    'solids with mixed labels': mesh_of(PolyhedronMesh, 12, ['HEX8', 'TET4', None], element=[list(range(8)), [0, 1, 2, 3], [4, 5, 6, 7]]),
    'solids with one wrong label': mesh_of(PolyhedronMesh, 12, ['HEX8', 'PYRA5'], element=[list(range(8)), [0, 1, 2, 3]]),
    # polygons and polylines: a label must fit too, and it does not change the cell of a polygon
    **{'polygon of '+str(n)+' labelled '+label: mesh_of(PolygonMesh, n, [label]) for n, label in (
        (3, 'S3'), (4, 'S4R'), (4, 'QUAD4'), (4, 'TRI3'), (3, 'QUAD4'), (5, 'S3'), (5, 'MY_ELEMENT'), (6, 'TRI6'), (8, 'S8'), (4, 'HEX8'))},
    'polygon of 5 with a missing label': mesh_of(PolygonMesh, 5, [None]),
    **{'polyline of '+str(n)+' labelled '+label: mesh_of(PolylineMesh, n, [label]) for n, label in (
        (2, 'LINE2'), (2, 'T3D2'), (3, 'LINE2'), (3, 'MY_ELEMENT'), (3, 'TRI3'))},
    **{mesh_type+' of '+str(n)+' labelled '+label: mesh_of(PolygonMesh, n, [label], mesh_type=mesh_type) for mesh_type, n, label in (
        ('polygon_tri6', 6, 'TRI6'), ('polygon_tri6', 6, 'STRI65'), ('polygon_tri6', 6, 'M3D6'), ('polygon_tri6', 6, 'MY_ELEMENT'),
        ('polygon_tri6', 3, 'TRI6'), ('polygon_tri6', 6, 'TRI3'), ('polygon_tri6', 6, 'S8'),
        ('polygon_quad8', 8, 'QUAD8'), ('polygon_quad8', 8, 'S8'), ('polygon_quad8', 8, 'S4'), ('polygon_quad8', 4, 'QUAD8'))},
    # how labels are stored
    'label in a list': mesh_of(PolyhedronMesh, 8, element_type=['HEX8']),
    'label in a 2D array': mesh_of(PolyhedronMesh, 8, element_type=np.array([['HEX8']])),
    'wrong label in a 2D array': mesh_of(PolyhedronMesh, 8, element_type=np.array([['PYRA5']])),
    'wrong polygon label in a 2D array': mesh_of(PolygonMesh, 4, element_type=np.array([['TRI3']])),
    **{name+', 1 label for 2 elements': mesh_of(mesh_class, n, [None], n_element=2) for name, mesh_class, n in (
        ('solid', PolyhedronMesh, 4), ('polygon', PolygonMesh, 3), ('polyline', PolylineMesh, 2), ('tri6', Tri6Mesh, 6))},
    **{name+', labels with no element': mesh_of(mesh_class, n, [None], element=[]) for name, mesh_class, n in (
        ('solid', PolyhedronMesh, 4), ('polygon', PolygonMesh, 3), ('polyline', PolylineMesh, 2), ('tri6', Tri6Mesh, 6))},
    # no node or no element
    **{'empty '+mesh_class.__name__: mesh_class for mesh_class in (PolylineMesh, PolygonMesh, PolyhedronMesh, Tri6Mesh, TetrahedronMesh)},
    **{'no element in '+mesh_class.__name__: mesh_of(mesh_class, 6, element=[]) for mesh_class in (PolylineMesh, PolygonMesh, PolyhedronMesh, Tri6Mesh)},
    # how elements are stored
    **{name+' '+kind: mesh_with_element(mesh_of(mesh_class, n, mesh_type=mesh_type), value, n)
       for name, mesh_class, n, mesh_type in (('polygon', PolygonMesh, 3, None), ('polyline', PolylineMesh, 3, None),
                                              ('solid', PolyhedronMesh, 4, None), ('tri6', PolygonMesh, 6, 'polygon_tri6'))
       for kind, value in (('tensor', lambda n: torch.arange(n).reshape(1, n)),
                           ('int32 tensor', lambda n: torch.arange(n, dtype=torch.int32).reshape(1, n)),
                           ('numpy', lambda n: np.arange(n).reshape(1, n)),
                           ('list of tensors', lambda n: [torch.arange(n)]),
                           ('list of numpy', lambda n: [np.arange(n)]),
                           ('float tensor', lambda n: torch.arange(n, dtype=F64).reshape(1, n)),
                           ('float list', lambda n: [[k+0.7 for k in range(n)]]),
                           ('1D tensor', lambda n: torch.arange(n)),
                           ('3D tensor', lambda n: torch.arange(n).reshape(1, 1, n)),
                           ('list of 2D rows', lambda n: [[list(range(n))]*n]),
                           ('strings', lambda n: [['a']*n]))},
    # field names, and the order of errors
    **{where+' named '+repr(name): mesh_with_field(mesh_of(TriangleMesh, 3), where, name, torch.ones(count, 1, dtype=F64))
       for where, count in (('node_data', 3), ('element_data', 1)) for name in (None, b'abc', '', 'my stress', '100%', 'σ_vm')},
    'wrong node and wrong label': mesh_of(PolyhedronMesh, 8, ['PYRA5'], node=torch.rand(8, 2, dtype=F64)),
    'wrong label and wrong field': mesh_with_field(mesh_of(PolyhedronMesh, 8, ['PYRA5']), 'node_data', 'u', torch.ones(1, 1, dtype=F64)),
    'no shape and wrong field': mesh_with_field(mesh_of(PolyhedronMesh, 8), 'element_data', 's', torch.ones(5, 1, dtype=F64)),
    'wrong element and wrong field': mesh_with_field(mesh_with_element(mesh_of(PolygonMesh, 3), lambda n: [[[0, 1, 2]]*3], 3),
                                                     'node_data', 'u', torch.ones(1, 1, dtype=F64)),
    'wrong node and wrong field': mesh_with_field(mesh_of(PolygonMesh, 3, node=torch.rand(3, 2, dtype=F64)), 'node_data', 'u', torch.ones(1, 1, dtype=F64)),
}


@pytest.mark.parametrize('make', SAME_AS_VTK_PATH.values(), ids=SAME_AS_VTK_PATH.keys())
def test_fallback_behaves_as_the_vtk_path(save_fallback, tmp_path, make):
    expected = behaviour(lambda mesh, path: mesh.save_as_vtk(str(path)), make, tmp_path/'vtk.vtk')
    assert behaviour(save_fallback, make, tmp_path/'fallback.vtk') == expected


def test_cell_type_numbers_are_those_of_vtk():
    for name, number in writer_module.VTK_CELL_TYPE_BY_NAME.items():
        assert getattr(vtk, name) == number, name
    for name, _ in mesh_module.VTK_CELL_BY_ELEMENT_TYPE.values():
        assert name in writer_module.VTK_CELL_TYPE_BY_NAME, name


@pytest.mark.parametrize('mesh_class, n_node', [(TriangleMesh, 3), (Tri6Mesh, 6), (TetrahedronMesh, 4), (PolylineMesh, 2)])
def test_file_that_cannot_be_written_raises_as_in_the_vtk_path(save_fallback, tmp_path, mesh_class, n_node):
    make = mesh_of(mesh_class, n_node)
    for filename in [tmp_path/'no_folder'/'mesh.vtk', tmp_path]+(['/dev/full'] if os.path.exists('/dev/full') else []):
        with pytest.raises(RuntimeError) as from_vtk:
            make().save_as_vtk(str(filename))
        with pytest.raises(RuntimeError) as from_fallback:
            save_fallback(make(), str(filename))
        assert str(from_fallback.value) == str(from_vtk.value)


def test_a_number_is_not_taken_as_a_file_descriptor(monkeypatch, tmp_path):
    # open(number) would write the mesh to whatever that file descriptor is
    path = tmp_path/'open.txt'
    with open(path, 'w') as file:
        for filename in (file.fileno(), None):
            with pytest.raises(TypeError):
                mesh_of(TriangleMesh, 3)().save_as_vtk(filename)
            with monkeypatch.context() as patch:
                patch.setattr(mesh_module, '_Flag_VTK_IMPORT_', False)
                with pytest.raises(TypeError):
                    mesh_of(TriangleMesh, 3)().save_as_vtk(filename)
    assert path.read_text() == ''


def test_field_name_that_is_not_a_string_raises_as_in_the_vtk_path(save_fallback, tmp_path):
    make = mesh_with_field(mesh_of(TriangleMesh, 3), 'node_data', 5, torch.ones(3, 1, dtype=F64))
    with pytest.raises(TypeError):
        make().save_as_vtk(str(tmp_path/'vtk.vtk'))
    with pytest.raises(TypeError):
        save_fallback(make(), tmp_path/'fallback.vtk')
    assert not (tmp_path/'fallback.vtk').exists()


def test_field_names_are_written_as_the_vtk_writer_writes_them(save_fallback, tmp_path):
    names = ['stress', 'my stress', 'a%20b', '100%', 'tab\tname', 'new\nline', 'quo"te', "apo'st", 'σ_vm', '']

    def make():
        mesh = triangle_mesh()
        for k, name in enumerate(names):
            mesh.node_data[name] = torch.full((4, 1), float(k), dtype=F64)
        mesh.element_data['cell name'] = torch.ones(2, 1, dtype=F64)
        return mesh

    def field_lines(path):
        # VTK writes CELL_DATA first, so the order is not compared.
        return sorted(line for line in path.read_text().split('\n')[5:] if line.endswith(' double'))

    make().save_as_vtk(str(tmp_path/'vtk.vtk'))
    save_fallback(make(), tmp_path/'fallback.vtk')
    assert field_lines(tmp_path/'fallback.vtk') == field_lines(tmp_path/'vtk.vtk')
    loaded = load(TriangleMesh, tmp_path/'fallback.vtk')
    assert sorted(loaded.node_data) == sorted(names[:-1]+['unknown'])
    assert torch.equal(loaded.node_data['my stress'], torch.full((4, 1), 1., dtype=F64))
    assert list(loaded.element_data) == ['cell name']


def test_invalid_shapes_raise_as_in_the_vtk_backend(save_fallback, tmp_path):
    def with_field(make, where, value):
        mesh = make()
        getattr(mesh, where)['x'] = value
        return mesh
    bad = [(triangle_mesh, 'node_data', torch.rand(5, 1)), (triangle_mesh, 'node_data', torch.rand(3, 1)),
           (triangle_mesh, 'node_data', torch.empty(4, 0)), (triangle_mesh, 'node_data', torch.rand(4)),
           (triangle_mesh, 'element_data', torch.rand(3, 1)), (triangle_mesh, 'element_data', torch.rand(2, 3, 3)),
           (polyline_mesh, 'node_data', torch.rand(4, 1)), (polyline_mesh, 'element_data', torch.rand(3, 1)),
           (tet_mesh, 'node_data', torch.rand(5, 1)), (tet_mesh, 'element_data', torch.rand(2, 1))]
    for make, where, value in bad:
        mesh = with_field(make, where, value)
        with pytest.raises(ValueError, match='invalid data shape'):
            mesh.save_as_vtk(str(tmp_path/'vtk.vtk'))
        with pytest.raises(ValueError, match='invalid data shape'):
            save_fallback(mesh, tmp_path/'fallback.vtk')
    for column in (2, 4):
        for make in (triangle_mesh, polyline_mesh, tet_mesh):
            mesh = make()
            mesh.node = torch.rand(len(mesh.node), column, dtype=F64)
            mesh.node_data.clear()
            with pytest.raises(ValueError, match='node must have shape'):
                save_fallback(mesh, tmp_path/'fallback.vtk')
    assert not (tmp_path/'fallback.vtk').exists()


def test_field_input_types_are_those_of_the_vtk_backend(save_fallback, tmp_path):
    fields = {'list': [[1.], [2.], [3.], [4.]], 'numpy': np.arange(8.).reshape(4, 2),
              'int': torch.tensor([[1], [2], [3], [2**40]]), 'bool': torch.tensor([[True], [False], [True], [False]]),
              'float32': torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.float32),
              'bfloat16': torch.tensor([[0.5], [1.5], [2.5], [3.5]], dtype=torch.bfloat16),
              'grad': torch.rand(4, 3, dtype=F64).requires_grad_()}
    mesh = triangle_mesh()
    mesh.node_data.update(fields)
    save_fallback(mesh, tmp_path/'fallback.vtk')
    mesh.save_as_vtk(str(tmp_path/'vtk.vtk'))
    loaded, reference = load(TriangleMesh, tmp_path/'fallback.vtk'), load(TriangleMesh, tmp_path/'vtk.vtk')
    assert sorted(loaded.node_data) == sorted(fields)
    for name in fields:
        assert torch.equal(loaded.node_data[name], reference.node_data[name]), name


def test_numbers_and_text_format_are_exact(save_fallback, tmp_path):
    mesh = TriangleMesh(torch.tensor([[0.1, 1/3, 1e-300], [-0.0, 1e300, 5e-324], [1e22, 2.5, 1e-5]], dtype=F64), [[0, 1, 2]])
    mesh.node_data['u'] = torch.tensor([[float('nan'), 1.7976931348623157e308], [float('inf'), 1e-320], [-float('inf'), 0.1]], dtype=F64)
    mesh.element_data['s'] = torch.tensor([[1., 2.]], dtype=F64)
    path = tmp_path/'exact.vtk'
    save_fallback(mesh, path)
    assert path.read_text() == ('# vtk DataFile Version 4.2\nvtk output\nASCII\nDATASET POLYDATA\n'
                                'POINTS 3 double\n0.1 0.3333333333333333 1e-300\n-0.0 1e+300 5e-324\n1e+22 2.5 1e-05\n'
                                'POLYGONS 1 4\n3 0 1 2\n'
                                'POINT_DATA 3\nFIELD FieldData 1\nu 2 3 double\n'
                                'nan 1.7976931348623157e+308 \ninf 1e-320 \n-inf 0.1 \n'
                                'CELL_DATA 1\nFIELD FieldData 1\ns 2 1 double\n1.0 2.0 \n')
    loaded = load(TriangleMesh, path)
    assert torch.equal(loaded.node, mesh.node)
    assert torch.equal(loaded.element_data['s'], mesh.element_data['s'])
    assert np.array_equal(loaded.node_data['u'].numpy(), mesh.node_data['u'].numpy(), equal_nan=True)


def test_mesh_without_fields_or_elements(save_fallback, tmp_path):
    for k, make in enumerate((polyline_mesh, triangle_mesh, tet_mesh)):
        path = tmp_path/(str(k)+'.vtk')
        save_fallback(make(), path)
        assert 'FIELD' not in path.read_text()
    for k, mesh_class in enumerate((PolylineMesh, TriangleMesh, TetrahedronMesh)):
        path = tmp_path/('empty'+str(k)+'.vtk')
        mesh = mesh_class()
        save_fallback(mesh, path)
        assert len(load(mesh_class, path).element) == 0
        mesh.node, mesh.element = torch.empty(0, dtype=F64), []
        save_fallback(mesh, path)
        assert 'POINTS 0 double' in path.read_text()


def test_cuda_node_and_field(save_fallback, tmp_path):
    device = 'cuda:0'
    mesh = TriangleMesh(torch.tensor(TRIANGLE_NODE, dtype=F64, device=device), torch.tensor(TRIANGLE_ELEMENT, device=device))
    mesh.node_data['u'] = torch.rand(4, 2, dtype=F64, device=device)
    mesh.element_data['s'] = torch.rand(2, 1, dtype=F64, device=device)
    save_fallback(mesh, tmp_path/'cuda.vtk')
    loaded = load(TriangleMesh, tmp_path/'cuda.vtk')
    assert torch.equal(loaded.node, mesh.node.cpu())
    assert torch.equal(loaded.node_data['u'], mesh.node_data['u'].cpu())
    assert torch.equal(loaded.element_data['s'], mesh.element_data['s'].cpu())
