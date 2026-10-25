import pytest
import torch

from torch_fea_mesh.mesh.quad_mesh import QuadMesh
from torch_fea_mesh.mesh.quad_triangle_mesh import QuadTriangleMesh
from torch_fea_mesh.mesh.tri6_mesh import Tri6Mesh
from torch_fea_mesh.mesh.tet10_mesh import Tet10Mesh
from torch_fea_mesh.mesh.polyline_mesh import PolylineMesh
from torch_fea_mesh.generation.mesh_example import create_hex_grid_mesh

pytestmark = pytest.mark.geometry


def test_triangle_geometry_and_boundary(triangle):
    triangle.update_element_area_and_normal()
    torch.testing.assert_close(triangle.element_area.reshape(-1), torch.tensor([.5], dtype=torch.float64))
    torch.testing.assert_close(triangle.element_normal, torch.tensor([[0., 0., 1.]], dtype=torch.float64))
    triangle.build_edge()
    assert {tuple(e) for e in triangle.edge.tolist()} == {(0, 1), (0, 2), (1, 2)}
    assert set(triangle.find_boundary_node()) == {0, 1, 2}


def test_tetrahedron_volume(tetrahedron):
    tetrahedron.update_element_volume()
    torch.testing.assert_close(tetrahedron.element_volume, torch.tensor([1/6], dtype=torch.float64))
    tetrahedron.build_edge()
    assert len(tetrahedron.edge) == 6
    assert len(tetrahedron.find_boundary_face()) == 4


def test_hexahedron_volume_and_boundary():
    mesh = create_hex_grid_mesh(2, 2, 2, dtype=torch.float64)
    mesh.update_element_volume()
    torch.testing.assert_close(mesh.element_volume, torch.ones(1, dtype=torch.float64))
    mesh.build_edge()
    assert len(mesh.edge) == 12
    assert len(mesh.find_boundary_face()) == 6


def test_quadratic_mesh_conversion(triangle, tetrahedron):
    tri6 = Tri6Mesh()
    tri6.create_from_tri3(triangle)
    assert tri6.node.shape == (6, 3)
    tri6.update_element_area_and_normal()
    torch.testing.assert_close(tri6.element_area.reshape(-1), torch.tensor([.5], dtype=torch.float64))
    restored = tri6.simplify_to_tri3()
    torch.testing.assert_close(restored.node, triangle.node)
    assert torch.equal(restored.element, triangle.element)
    tet10 = Tet10Mesh()
    tet10.create_from_tet4(tetrahedron)
    assert tet10.node.shape == (10, 3)
    tet10.update_element_volume()
    torch.testing.assert_close(tet10.element_volume, torch.tensor([1/6], dtype=torch.float64))
    assert len(tet10.find_boundary_face()) == 4
    restored = tet10.simplify_to_tet4()
    torch.testing.assert_close(restored.node, tetrahedron.node)
    assert torch.equal(restored.element, tetrahedron.element)


def test_mixed_surface_classification_and_area():
    node = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.],
                         [0., 1., 0.], [2., 0., 0.]], dtype=torch.float64)
    mesh = QuadTriangleMesh(node, [[0, 1, 2, 3], [1, 4, 2]])
    assert mesh.quad_element_idx == [0]
    assert mesh.tri_element_idx == [1]
    mesh.update_element_area_and_normal()
    torch.testing.assert_close(mesh.element_area.reshape(-1), torch.tensor([1., .5], dtype=torch.float64))


def test_polyline_edge_follows_segment_order():
    mesh = PolylineMesh(torch.zeros((4, 3), dtype=torch.float64), [[0, 1, 2], [2, 3]])
    mesh.build_edge()
    assert {tuple(e) for e in mesh.edge.tolist()} == {(0, 1), (1, 2), (2, 3)}
    assert set(mesh.find_boundary_node()) == {0, 3}


@pytest.mark.gpu
def test_triangle_area_cuda_matches_cpu(triangle):
    cls = type(triangle)
    cpu_area, cpu_normal = cls.cal_element_area_and_normal(triangle.node, triangle.element)
    area, normal = cls.cal_element_area_and_normal(triangle.node.cuda(), triangle.element.cuda())
    torch.testing.assert_close(area.cpu(), cpu_area)
    torch.testing.assert_close(normal.cpu(), cpu_normal)
