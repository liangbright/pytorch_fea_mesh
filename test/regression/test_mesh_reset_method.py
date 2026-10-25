import pytest

pytestmark = [pytest.mark.geometry, pytest.mark.vtk]
pytest.importorskip("vtk", reason="VTK is required for this test module")

"""Behavioral tests for full reset, cache reset, and geometry replacement."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import vtk

from torch_fea_mesh.mesh.mesh import Mesh
from torch_fea_mesh.mesh.polygon_mesh import PolygonMesh
from torch_fea_mesh.mesh.polyline_mesh import PolylineMesh
from torch_fea_mesh.mesh.polyhedron_mesh import PolyhedronMesh
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh
from torch_fea_mesh.mesh.quad_mesh import QuadMesh
from torch_fea_mesh.mesh.quad_triangle_mesh import QuadTriangleMesh
from torch_fea_mesh.mesh.tri6_mesh import Tri6Mesh
from torch_fea_mesh.mesh.tetrahedron_mesh import TetrahedronMesh
from torch_fea_mesh.mesh.tet10_mesh import Tet10Mesh
from torch_fea_mesh.mesh.hexahedron_mesh import HexahedronMesh
from torch_fea_mesh.processing.mesh_processing import remove_unused_node


CLASSES = (Mesh, PolygonMesh, PolylineMesh, PolyhedronMesh, TriangleMesh, QuadMesh,
           QuadTriangleMesh, Tri6Mesh, TetrahedronMesh, Tet10Mesh, HexahedronMesh)
DTYPES = (torch.float32, torch.float64)
PRIMARY = ('name', 'mesh_type', 'node', 'element', 'element_type',
           'mapping_from_node_name_to_index', 'node_set', 'element_set',
           'node_data', 'element_data', 'mesh_data')


def empty(cls):
    return cls(None, None, None, 'polygon') if cls is Mesh else cls()


def fixture(cls, dtype):
    square = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    tetrahedron = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    node, element = square, [[0, 1, 2], [0, 2, 3]]
    if cls is PolylineMesh:
        element = [[0, 1], [1, 2], [2, 3]]
    elif cls is QuadMesh:
        element = [[0, 1, 2, 3]]
    elif cls is QuadTriangleMesh:
        node = square + [[2, 0, 0]]
        element = [[0, 1, 2, 3], [1, 4, 2]]
    elif cls is Tri6Mesh:
        node = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [.5, 0, 0], [.5, .5, 0], [0, .5, 0]]
        element = [[0, 1, 2, 3, 4, 5]]
    elif cls in (PolyhedronMesh, TetrahedronMesh, Tet10Mesh):
        node, element = tetrahedron, [[0, 1, 2, 3]]
        if cls is Tet10Mesh:
            result = Tet10Mesh()
            result.create_from_tet4(TetrahedronMesh(torch.tensor(node, dtype=dtype), element))
            return result
    elif cls is HexahedronMesh:
        node = square + [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
        element = [list(range(8))]
    node = torch.tensor(node, dtype=dtype)
    return cls(node, element, None, 'polygon') if cls is Mesh else cls(node, element)


def add_metadata(mesh):
    mesh.name = 'previous mesh'
    mesh.element_type = np.arange(len(mesh.element), dtype=np.int64)
    mesh.node_data = {'old_nodes': torch.arange(len(mesh.node), dtype=mesh.node.dtype).reshape(-1, 1)}
    mesh.element_data = {'old_elements': torch.ones((len(mesh.element), 1), dtype=mesh.node.dtype)}
    mesh.node_set = {'old_node_set': [0]}
    mesh.element_set = {'old_element_set': [0]}
    mesh.mapping_from_node_name_to_index = {'old_landmark': 0}
    mesh.mesh_data = {'solver': 'example', 'parameters': [1., 2.]}


def fill_cache_slot(mesh):
    # Give every library cache an old value, including rarely built adjacency.
    # Primary geometry/metadata are tested separately for preservation.
    for key, value in list(vars(mesh).items()):
        if key not in PRIMARY:
            setattr(mesh, key, {k: ['cached'] for k in value} if isinstance(value, dict)
                    else torch.tensor([123.]))


class MeshResetTest(unittest.TestCase):
    def assert_value_equal(self, actual, expected):
        self.assertIs(type(actual), type(expected))
        if torch.is_tensor(actual):
            torch.testing.assert_close(actual, expected)
        elif isinstance(actual, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(actual, dict):
            self.assertEqual(set(actual), set(expected))
            for key in actual:
                self.assert_value_equal(actual[key], expected[key])
        elif isinstance(actual, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_value_equal(a, b)
        else:
            self.assertEqual(actual, expected)

    def test_clear_matches_fresh_empty_mesh_for_every_class(self):
        for dtype in DTYPES:
            for cls in CLASSES:
                with self.subTest(dtype=dtype, cls=cls.__name__):
                    mesh = fixture(cls, dtype)
                    add_metadata(mesh)
                    fill_cache_slot(mesh)
                    for _ in range(2):
                        mesh.clear()
                        self.assert_value_equal(vars(mesh), vars(empty(cls)))
                    output = mesh.convert_to_vtk()
                    self.assertEqual(output.GetNumberOfPoints(), 0)
                    self.assertEqual(output.GetNumberOfCells(), 0)

    def test_clear_geometry_derived_data_preserves_geometry_and_all_user_data(self):
        for dtype in DTYPES:
            for cls in CLASSES:
                with self.subTest(dtype=dtype, cls=cls.__name__):
                    mesh = fixture(cls, dtype)
                    add_metadata(mesh)
                    mesh.node.requires_grad_(True)
                    primary = {key: getattr(mesh, key) for key in PRIMARY}
                    values = deepcopy(primary)
                    fill_cache_slot(mesh)
                    for _ in range(2):
                        mesh.clear_geometry_derived_data()
                        for key in PRIMARY:
                            self.assertIs(getattr(mesh, key), primary[key])
                            self.assert_value_equal(getattr(mesh, key), values[key])
                        for key, value in vars(empty(cls)).items():
                            if key not in PRIMARY:
                                self.assert_value_equal(getattr(mesh, key), value)
                    mesh.node.sum().backward()
                    torch.testing.assert_close(mesh.node.grad, torch.ones_like(mesh.node))

    def test_clear_adj_info_preserves_geometry_value(self):
        mesh = fixture(TriangleMesh, torch.float64)
        mesh.update_node_normal()
        mesh.update_edge_length()
        node, element, area, normal = mesh.node, mesh.element, mesh.element_area, mesh.node_normal
        lengths = mesh.edge_length
        mesh.clear_adj_info()
        self.assertIsNone(mesh.edge)
        self.assertIs(mesh.node, node)
        self.assertIs(mesh.element, element)
        self.assertIs(mesh.element_area, area)
        self.assertIs(mesh.node_normal, normal)
        self.assertIs(mesh.edge_length, lengths)

    def test_cached_geometry_can_be_recomputed_after_scaling(self):
        for dtype in DTYPES:
            for cls in (TriangleMesh, QuadMesh, QuadTriangleMesh, Tri6Mesh,
                        TetrahedronMesh, Tet10Mesh, HexahedronMesh):
                with self.subTest(dtype=dtype, cls=cls.__name__):
                    mesh = fixture(cls, dtype)
                    mesh.update_edge_length()
                    lengths = mesh.edge_length.clone()
                    surface = hasattr(mesh, 'update_element_area_and_normal')
                    if surface:
                        mesh.update_element_area_and_normal()
                        measure = mesh.element_area.clone()
                    else:
                        mesh.update_element_volume()
                        measure = mesh.element_volume.clone()
                    mesh.node = 2 * mesh.node
                    mesh.clear_geometry_derived_data()
                    mesh.update_edge_length()
                    torch.testing.assert_close(mesh.edge_length, 2 * lengths)
                    if surface:
                        mesh.update_element_area_and_normal()
                        torch.testing.assert_close(mesh.element_area, 4 * measure)
                        mesh.update_node_normal()
                        mesh.update_element_corner_angle()
                    else:
                        mesh.update_element_volume()
                        torch.testing.assert_close(mesh.element_volume, 8 * measure)

    def test_copy_resets_old_state_and_accepts_own_geometry(self):
        for dtype in DTYPES:
            for cls in CLASSES:
                with self.subTest(dtype=dtype, cls=cls.__name__):
                    mesh = fixture(cls, dtype)
                    add_metadata(mesh)
                    fill_cache_slot(mesh)
                    node, element = mesh.node, mesh.element
                    expected = empty(cls)
                    expected.copy(node, element)
                    mesh.copy(mesh.node, mesh.element)
                    self.assert_value_equal(vars(mesh), vars(expected))
                    self.assertIsNot(mesh.node, node)
                    self.assertIsNot(mesh.element, element)

    def test_copy_preserves_autograd_when_requested(self):
        for dtype in DTYPES:
            mesh = fixture(TriangleMesh, dtype)
            mesh.node.requires_grad_(True)
            old_node = mesh.node
            mesh.update_node_normal()
            mesh.copy(mesh.node, mesh.element, detach=False)
            self.assertIsNone(mesh.node_normal)
            mesh.node.sum().backward()
            torch.testing.assert_close(old_node.grad, torch.ones_like(old_node))

    def test_failed_copy_does_not_erase_existing_mesh(self):
        for node, element in ((object(), [[0, 1, 2]]), ([[0., 0., 0.]], object())):
            mesh = fixture(TriangleMesh, torch.float64)
            add_metadata(mesh)
            mesh.update_node_normal()
            before = deepcopy(vars(mesh))
            with self.assertRaises(NotImplementedError):
                mesh.copy(node, element)
            self.assert_value_equal(vars(mesh), before)

    def test_vtk_reader_replaces_all_state_for_every_class(self):
        for dtype in DTYPES:
            for cls in CLASSES:
                for reader in ('read_mesh_vtk', 'read_mesh_vtk_slow'):
                    with self.subTest(dtype=dtype, cls=cls.__name__, reader=reader):
                        source = fixture(cls, dtype)
                        source.node_data['new_field'] = torch.ones((len(source.node), 1), dtype=dtype)
                        vtk_data = source.convert_to_vtk()
                        expected = empty(cls)
                        getattr(expected, reader)(vtk_data, dtype)
                        mesh = fixture(cls, dtype)
                        add_metadata(mesh)
                        fill_cache_slot(mesh)
                        getattr(mesh, reader)(vtk_data, dtype)
                        self.assert_value_equal(vars(mesh), vars(expected))

    def test_torch_load_resets_missing_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'minimal.pt')
            for dtype in DTYPES:
                for cls in CLASSES:
                    with self.subTest(dtype=dtype, cls=cls.__name__):
                        source = fixture(cls, dtype)
                        torch.save({'node': source.node, 'element': source.element}, path)
                        expected = empty(cls)
                        expected.load_from_torch(path)
                        mesh = fixture(cls, dtype)
                        add_metadata(mesh)
                        fill_cache_slot(mesh)
                        mesh.load_from_torch(path)
                        self.assert_value_equal(vars(mesh), vars(expected))

    def test_torch_load_restores_saved_adjacency_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'saved.pt')
            for cls in (TriangleMesh, QuadTriangleMesh):
                with self.subTest(cls=cls.__name__):
                    source = fixture(cls, torch.float64)
                    add_metadata(source)
                    source.build_node_to_node_adj_table()
                    source.save_as_torch(path, save_adj_info=True)
                    mesh = fixture(cls, torch.float64)
                    mesh.update_node_normal()
                    mesh.update_element_area_and_normal()
                    mesh.load_from_torch(path)
                    self.assert_value_equal(mesh.node_to_node_adj_table, source.node_to_node_adj_table)
                    self.assert_value_equal(mesh.node_data, source.node_data)
                    self.assert_value_equal(mesh.element_data, source.element_data)
                    self.assert_value_equal(mesh.mesh_data, source.mesh_data)
                    self.assertIsNone(mesh.node_normal)
                    self.assertIsNone(mesh.element_area)

    def test_file_loader_replaces_old_state(self):
        with tempfile.TemporaryDirectory() as directory:
            for dtype in DTYPES:
                source = fixture(TriangleMesh, dtype)
                for extension, save, load in (
                    ('vtk', 'save_as_vtk', 'load_from_vtk'),
                    ('vtp', 'save_as_vtp', 'load_from_vtp'),
                    ('stl', None, 'load_from_stl'),
                ):
                    with self.subTest(dtype=dtype, extension=extension):
                        path = str(Path(directory) / ('mesh.' + extension))
                        if save is None:
                            writer = vtk.vtkSTLWriter()
                            writer.SetFileName(path)
                            writer.SetInputData(source.convert_to_vtk())
                            self.assertEqual(writer.Write(), 1)
                        else:
                            getattr(source, save)(path)
                        expected = TriangleMesh()
                        getattr(expected, load)(path, dtype)
                        mesh = fixture(TriangleMesh, dtype)
                        add_metadata(mesh)
                        mesh.update_node_normal()
                        getattr(mesh, load)(path, dtype)
                        self.assert_value_equal(vars(mesh), vars(expected))

    def test_mdk_load_replaces_node_set_and_restores_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'mesh.json')
            source = fixture(PolygonMesh, torch.float64)
            source.name = 'new mesh'
            source.node_set = {'new_nodes': [1, 2]}
            source.element_set = {'new_elements': [1]}
            source.save_as_mdk_json(path)
            mesh = fixture(PolygonMesh, torch.float64)
            add_metadata(mesh)
            mesh.load_from_mdk_json(path, torch.float64)
            self.assertEqual(mesh.name, source.name)
            self.assertEqual(mesh.node_set, source.node_set)
            self.assertEqual(mesh.element_set, source.element_set)
            self.assertEqual(mesh.mapping_from_node_name_to_index, {})
            self.assertEqual(mesh.mesh_data, {})

    def test_failed_reader_preserves_existing_mesh(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'invalid.pt')
            torch.save({'node': [[0., 0., 0.]]}, path)
            for reader in ('read_mesh_vtk', 'read_mesh_vtk_slow', 'load_from_torch'):
                with self.subTest(reader=reader):
                    mesh = fixture(TriangleMesh, torch.float64)
                    add_metadata(mesh)
                    mesh.update_node_normal()
                    before = deepcopy(vars(mesh))
                    with self.assertRaises(ValueError):
                        if reader == 'load_from_torch':
                            mesh.load_from_torch(path)
                        else:
                            getattr(mesh, reader)(vtk.vtkPolyData(), torch.float64)
                    self.assert_value_equal(vars(mesh), before)

    def test_polyline_initialize_replaces_state_including_empty_input(self):
        for dtype in DTYPES:
            curves = [torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=dtype)]
            for curve_list in (curves, [], [[]]):
                with self.subTest(dtype=dtype, empty=not curve_list or not len(curve_list[0])):
                    expected = PolylineMesh()
                    expected.initialize(curve_list)
                    mesh = fixture(PolylineMesh, dtype)
                    add_metadata(mesh)
                    mesh.update_edge_length()
                    mesh.initialize(curve_list)
                    self.assert_value_equal(vars(mesh), vars(expected))

    def test_compaction_preserves_element_data_and_general_metadata(self):
        mesh = fixture(TriangleMesh, torch.float64)
        mesh.element = torch.tensor([[3, 1, 2]])
        add_metadata(mesh)
        mesh.update_node_normal()
        preserved = {key: getattr(mesh, key) for key in
                     ('name', 'element_type', 'element_data', 'element_set', 'mesh_data')}
        kept = remove_unused_node(mesh, return_node_idx_list=True)
        self.assertEqual(kept.tolist(), [3, 1, 2])
        for key, value in preserved.items():
            self.assertIs(getattr(mesh, key), value)
        self.assertIsNone(mesh.element_area)
        self.assertIsNone(mesh.node_normal)

    def test_higher_order_conversion_resets_previous_state(self):
        for dtype in DTYPES:
            for cls, source_cls, method in ((Tri6Mesh, TriangleMesh, 'create_from_tri3'),
                                             (Tet10Mesh, TetrahedronMesh, 'create_from_tet4')):
                with self.subTest(dtype=dtype, cls=cls.__name__):
                    mesh = fixture(cls, dtype)
                    add_metadata(mesh)
                    fill_cache_slot(mesh)
                    source = fixture(source_cls, dtype)
                    expected = cls()
                    getattr(expected, method)(source)
                    getattr(mesh, method)(source)
                    self.assert_value_equal(vars(mesh), vars(expected))
