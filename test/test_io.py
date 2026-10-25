import importlib
import numpy as np
import pytest
import torch

from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh
from torch_fea_mesh.mesh.tetrahedron_mesh import TetrahedronMesh
from torch_fea_mesh.io.load_mesh_from_inp_file import read_abaqus_inp
from torch_fea_mesh.io.save_mesh_as_inp_file import save_polygon_mesh_to_inp, save_polyhedron_mesh_to_inp

pytestmark = pytest.mark.io


def test_torch_dictionary_round_trip(triangle, tmp_path):
    triangle.node_set['fixed'] = [0, 1]
    triangle.element_set['surface'] = [0]
    triangle.node_data['temperature'] = torch.arange(3, dtype=torch.float64).reshape(-1,1)
    triangle.element_data['stress'] = torch.ones((1,6), dtype=torch.float64)
    triangle.mesh_data['load'] = {'pressure': 2.0}
    triangle.build_edge()
    path = tmp_path / 'mesh.pt'
    triangle.save_as_torch(path)
    restored = TriangleMesh()
    restored.load_from_torch(path)
    torch.testing.assert_close(restored.node, triangle.node)
    assert torch.equal(restored.element, triangle.element)
    assert torch.equal(restored.edge, triangle.edge)
    assert restored.node_set == triangle.node_set
    assert restored.element_set == triangle.element_set
    assert restored.mesh_data == triangle.mesh_data
    torch.testing.assert_close(restored.node_data['temperature'], triangle.node_data['temperature'])
    torch.testing.assert_close(restored.element_data['stress'], triangle.element_data['stress'])


def test_minimal_legacy_torch_dictionary(triangle, tmp_path):
    path = tmp_path / 'legacy.pt'
    torch.save({'node': triangle.node, 'element': triangle.element,
                'node_set': {'fixed': [0]}, 'mesh_data': {'tag': 'legacy'}}, path)
    restored = TriangleMesh()
    restored.load_from_torch(path)
    torch.testing.assert_close(restored.node, triangle.node)
    assert restored.node_set['fixed'] == [0]
    assert restored.mesh_data['tag'] == 'legacy'


@pytest.mark.parametrize('kind', ['triangle', 'tetrahedron'])
def test_inp_writer_reader(kind, triangle, tetrahedron, tmp_path):
    mesh, writer = ((triangle, save_polygon_mesh_to_inp) if kind == 'triangle'
                    else (tetrahedron, save_polyhedron_mesh_to_inp))
    path = tmp_path / (kind+'.inp')
    writer(mesh, path)
    restored = read_abaqus_inp(path, remove_unused_node=True)
    np.testing.assert_allclose(restored['node'], mesh.node.numpy())
    assert [list(e) for e in restored['element']] == mesh.element.tolist()
    assert len(restored['element_type']) == 1


def test_vtk_fallback_writer_without_vtk_flag(triangle, tmp_path, monkeypatch):
    module = importlib.import_module('torch_fea_mesh.mesh.mesh')
    monkeypatch.setattr(module, '_Flag_VTK_IMPORT_', False)
    path = tmp_path/'triangle.vtk'
    triangle.save_as_vtk(str(path))
    text = path.read_text()
    assert 'DATASET POLYDATA' in text
    assert 'POINTS 3' in text
    assert 'POLYGONS 1 4' in text
    assert '3 0 1 2' in text


@pytest.mark.vtk
@pytest.mark.parametrize('kind', ['triangle', 'tetrahedron'])
def test_vtk_round_trip(kind, triangle, tetrahedron, tmp_path):
    pytest.importorskip('vtk')
    mesh = triangle if kind == 'triangle' else tetrahedron
    path = tmp_path/(kind+'.vtk')
    mesh.node_data['value'] = torch.arange(len(mesh.node),dtype=torch.float64).reshape(-1,1)
    mesh.save_as_vtk(str(path))
    restored = type(mesh)()
    restored.load_from_vtk(str(path), dtype=torch.float64)
    torch.testing.assert_close(restored.node, mesh.node)
    assert torch.equal(restored.element, mesh.element)
    torch.testing.assert_close(restored.node_data['value'], mesh.node_data['value'])
