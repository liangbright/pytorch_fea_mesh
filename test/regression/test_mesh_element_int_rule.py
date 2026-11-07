"""self.element is an int64 tensor or a nested list of python int, wherever the class sets it."""
import numpy as np
import pytest
import torch

from torch_fea_mesh.mesh.hexahedron_mesh import HexahedronMesh
from torch_fea_mesh.mesh.polygon_mesh import PolygonMesh
from torch_fea_mesh.mesh.polyline_mesh import PolylineMesh
from torch_fea_mesh.mesh.quad_triangle_mesh import QuadTriangleMesh
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh

F64 = torch.float64
NODE = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], dtype=F64)
SAME = [[0, 1, 2], [0, 2, 3]]
RAGGED = [[0, 1, 2, 3], [1, 4, 2]]


def assert_nested_python_int(element, expected):
    assert type(element) is list and all(type(row) is list for row in element)
    assert all(type(value) is int for row in element for value in row)
    assert element == expected


def assert_int64_tensor(element, expected):
    assert torch.is_tensor(element) and element.dtype == torch.int64 and element.tolist() == expected


def build_and_copy(element):
    built = PolygonMesh(NODE, element)
    copied = PolygonMesh()
    copied.copy(NODE, element)
    return built, copied


def test_float_and_bool_are_rejected_in_every_form():
    bad = [[[0, 1.5, 2]], [[0., 1., 2.]], [[0, 1.5, 2, 3], [1, 4, 2]], [[False, True, 2]], [[False, True, 2, 3], [1, 4, 2]],
           [[np.float64(0), 1, 2]], [[np.True_, 1, 2]], [[torch.tensor(1.5), 1, 2]], [[torch.tensor(True), 1, 2]],
           [torch.tensor([0., 1., 2.])], [torch.tensor([True, False, True])], [np.array([0., 1., 2.])], [np.array([True, False, True])],
           np.array([[0., 1., 2.]]), np.array([[True, False, True]]), torch.tensor([[0., 1., 2.]]), torch.tensor([[True, False, True]]),
           torch.tensor([[0., 1., 2.]], dtype=F64), np.array([[0., 1.], [1., 4., 2.]], dtype=object), [[[0, 1, 2]]],
           torch.tensor([[0, 1, 2]], dtype=torch.complex64), np.array([[0, 1, 2]], dtype=np.complex64)]
    mesh = PolygonMesh(NODE, SAME)
    for element in bad:
        with pytest.raises(ValueError, match='integers'):
            PolygonMesh(NODE, element)
        with pytest.raises(ValueError, match='integers'):
            mesh.copy(NODE, element)
        assert mesh.element.tolist() == SAME


def test_any_integer_type_becomes_int64_or_python_int():
    for dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        for mesh in build_and_copy(torch.tensor(SAME, dtype=dtype)):
            assert_int64_tensor(mesh.element, SAME)
    for dtype in (np.int8, np.uint8, np.int16, np.uint32, np.int64):
        for mesh in build_and_copy(np.array(SAME, dtype=dtype)):
            assert_int64_tensor(mesh.element, SAME)
    for rows in (SAME, tuple(SAME), [tuple(row) for row in SAME], [[np.int32(v) for v in row] for row in SAME],
                 [[torch.tensor(v) for v in row] for row in SAME], [torch.tensor(row, dtype=torch.int32) for row in SAME],
                 [np.array(row, dtype=np.int16) for row in SAME], np.array(SAME, dtype=object)):
        for mesh in build_and_copy(rows):
            assert_int64_tensor(mesh.element, SAME)
    ragged_object = np.empty(2, dtype=object)
    ragged_object[0], ragged_object[1] = np.array(RAGGED[0], dtype=np.int32), RAGGED[1]
    for rows in (RAGGED, tuple(RAGGED), [tuple(row) for row in RAGGED], [[np.int32(v) for v in row] for row in RAGGED],
                 [[torch.tensor(v) for v in row] for row in RAGGED], [torch.tensor(row, dtype=torch.int16) for row in RAGGED],
                 [np.array(row, dtype=np.uint8) for row in RAGGED], ragged_object):
        for mesh in build_and_copy(rows):
            assert_nested_python_int(mesh.element, RAGGED)


def test_valid_nested_list_is_not_copied_by_the_constructor():
    rows = [[0, 1, 2, 3], [1, 4, 2]]
    assert PolygonMesh(NODE, rows).element is rows
    copied = PolygonMesh()
    copied.copy(NODE, rows)
    assert copied.element == rows and copied.element is not rows


def test_empty_input_may_have_any_dtype():
    for element in ([], (), torch.tensor([]), np.array([]), torch.empty((0, 3)), torch.empty((0, 3), dtype=torch.bool)):
        for mesh in build_and_copy(element):
            assert mesh.element.dtype == torch.int64 and mesh.element.shape == (0, 3)


def test_flat_input_is_still_a_shape_error():
    for element in ([0, 1, 2], [torch.tensor(0), torch.tensor(1)], [torch.tensor([[0, 1, 2]])],
                    [0, [1, 2, 3]], [torch.tensor(0), [1, 2, 3]], [torch.tensor([[0, 1, 2]]), [1, 2, 3]], ['ab', [1, 2, 3]]):
        with pytest.raises(ValueError, match='element.shape'):
            PolygonMesh(NODE, element)


def test_load_from_torch_follows_the_rule_and_keeps_the_shape(tmp_path):
    path = str(tmp_path / 'mesh.pt')
    save = lambda element: torch.save({'node': NODE, 'element': element}, path)
    mesh = PolygonMesh(NODE, SAME)
    # a list stays a list, and a file is not held to the width of this class
    for element, expected in (([[np.int64(0), 1]], [[0, 1]]), (RAGGED, RAGGED), ((tuple(RAGGED[0]), RAGGED[1]), RAGGED)):
        save(element)
        mesh.load_from_torch(path)
        assert_nested_python_int(mesh.element, expected)
    ragged_object = np.empty(2, dtype=object)
    ragged_object[0], ragged_object[1] = RAGGED[0], RAGGED[1]
    save(ragged_object)
    mesh.load_from_torch(path)
    assert_nested_python_int(mesh.element, RAGGED)
    for element in (torch.tensor(SAME, dtype=torch.int32), np.array(SAME, dtype=np.int16)):
        save(element)
        mesh.load_from_torch(path)
        assert_int64_tensor(mesh.element, SAME)
    # a bad file raises, and the mesh stays as it was
    mesh = PolygonMesh(NODE, SAME)
    for element in ([[0, 1.5, 2]], [[True, 1, 2]], torch.tensor([[0., 1., 2.]]), np.array([[0., 1., 2.]]), torch.tensor([[True, False, True]])):
        save(element)
        with pytest.raises(ValueError, match='integers'):
            mesh.load_from_torch(path)
        assert mesh.element.tolist() == SAME
    save(None)
    mesh.load_from_torch(path)
    assert mesh.element is None


def test_what_the_class_sets_follows_the_rule(tmp_path):
    path = str(tmp_path / 'mesh.vtk')
    PolygonMesh(NODE, RAGGED).save_as_vtk(path)
    mesh = PolygonMesh()
    mesh.load_from_vtk(path, F64)
    assert_nested_python_int(mesh.element, RAGGED)
    sub_mesh = PolygonMesh(NODE, RAGGED).get_sub_mesh([1, 0])
    assert_nested_python_int(sub_mesh.element, sub_mesh.element)
    TriangleMesh(NODE, SAME).save_as_vtk(path)
    mesh = PolygonMesh()
    mesh.load_from_vtk(path, F64)
    assert_int64_tensor(mesh.element, SAME)
    assert TriangleMesh(NODE, SAME).subdivide_to_tri_1to4().element.dtype == torch.int64
    assert_nested_python_int(PolylineMesh.from_curve(NODE[:3]).element, [[0, 1], [1, 2]])
    assert_nested_python_int(QuadTriangleMesh(NODE, RAGGED).element, RAGGED)
    assert HexahedronMesh(torch.rand(8, 3, dtype=F64), [list(range(8))]).element.dtype == torch.int64
