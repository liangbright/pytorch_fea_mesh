"""MESH-18: misspelled public names are gone and the correct names work. NaN messages name the right normal.
create_quad_mesh_rectangle_in_circle and ReorderNodeInElement are exercised in test_generation.py and test_processing.py.
"""
import pytest
import torch
from torch_fea_mesh.generation import mesh_example
from torch_fea_mesh.mesh.quad_mesh import QuadMesh
from torch_fea_mesh.mesh.tri6_mesh import Tri6Mesh
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh
from torch_fea_mesh.processing import repair_mesh

F64 = torch.float64


def triangles():
    return TriangleMesh(torch.tensor([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], dtype=F64), [[0, 1, 2], [0, 2, 3]])


def test_old_names_are_gone():
    for owner, name in ((TriangleMesh, 'subdivid_to_tri_1to4'), (TriangleMesh, 'subdivid_to_tri_1to3'), (TriangleMesh, 'subdivid_to_tri'),
                        (QuadMesh, 'upate_element_flatness'), (mesh_example, 'create_quad_mesh_rectangle_in_cirlce'),
                        (repair_mesh, 'ReordeNodeInElement')):
        assert not hasattr(owner, name), name


def test_subdivide_to_tri_runs_the_method_of_its_mode():
    for mode, n_element, n_node, method in (('1to4', 8, 9, 'subdivide_to_tri_1to4'), ('1to3', 6, 6, 'subdivide_to_tri_1to3')):
        mesh = triangles().subdivide_to_tri(mode)
        direct = getattr(triangles(), method)()
        assert mesh.element.shape == (n_element, 3) and mesh.node.shape == (n_node, 3)
        assert torch.equal(mesh.element, direct.element) and torch.equal(mesh.node, direct.node)
    with pytest.raises(ValueError, match='unsupported mode'):
        triangles().subdivide_to_tri('1to2')


def test_nan_message_names_the_node_normal(capsys):
    node = torch.tensor([[float('nan'), 0, 0], [2, 0, 0], [0, 2, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=F64)
    for mesh, name in ((TriangleMesh(node[:3], [[0, 1, 2]]), 'TriangleMesh'), (Tri6Mesh(node, [[0, 1, 2, 3, 4, 5]]), 'Tri6Mesh')):
        mesh.update_node_normal()
        assert capsys.readouterr().out.strip() == f'error: nan in node_normal @ {name}:update_node_normal'
    QuadMesh(node[:4], [[0, 1, 2, 3]]).update_node_normal()
    assert capsys.readouterr().out.split('\n')[:2] == ['error: nan in node_normal @ QuadMesh:cal_node_normal',
                                                       'error: nan in node_normal @ QuadMesh:update_node_normal']
