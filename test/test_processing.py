import pytest
import torch

from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh
from torch_fea_mesh.mesh.tri6_mesh import Tri6Mesh
from torch_fea_mesh.mesh.tet10_mesh import Tet10Mesh
from torch_fea_mesh.processing.mesh_processing import (
    segment_mesh_to_connected_region, simple_smoother, remove_unused_node,
)
from torch_fea_mesh.processing.polyhedron_mesh_processing import extract_surface_mesh
from torch_fea_mesh.processing.repair_mesh import ReordeNodeInElement

pytestmark = pytest.mark.processing


def test_connected_component():
    mesh = TriangleMesh(torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                                      [3.,0.,0.],[4.,0.,0.],[3.,1.,0.]], dtype=torch.float64),
                        [[0,1,2],[3,4,5]])
    result = segment_mesh_to_connected_region(mesh, adj="node")
    assert {tuple(sorted(component)) for component in result} == {(0,), (1,)}


def test_smoother_mean_and_mask():
    field = torch.tensor([[0.], [2.], [4.]], dtype=torch.float64)
    links = torch.tensor([[0,1],[1,0],[1,2],[2,1]])
    actual = simple_smoother(field, links, .5, torch.tensor([0.,1.,0.]), inplace=False)
    torch.testing.assert_close(actual, field)
    actual = simple_smoother(field, links, .5, None, inplace=False)
    torch.testing.assert_close(actual, torch.tensor([[1.],[2.],[3.]], dtype=torch.float64))


def test_surface_extraction_preserves_volume_boundary(tetrahedron):
    surface = extract_surface_mesh(tetrahedron)
    assert isinstance(surface, TriangleMesh)
    assert surface.element.shape == (4, 3)
    center = tetrahedron.node.mean(0)
    surface.update_element_area_and_normal()
    face_centers = surface.node[surface.element].mean(1)
    assert torch.all(((face_centers-center)*surface.element_normal).sum(-1) > 0)


def test_tet10_surface_uses_tri6(tetrahedron):
    volume = Tet10Mesh()
    volume.create_from_tet4(tetrahedron)
    surface = extract_surface_mesh(volume)
    assert isinstance(surface, Tri6Mesh)
    assert surface.element.shape == (4, 6)


def test_remove_unused_node_remaps_connectivity(triangle):
    triangle.node = torch.cat([triangle.node, torch.tensor([[5.,5.,5.]], dtype=torch.float64)])
    remove_unused_node(triangle)
    assert len(triangle.node) == 3
    assert torch.equal(triangle.element, torch.tensor([[0,1,2]]))


def test_repair_face_orientation():
    mesh = TriangleMesh(torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]], dtype=torch.float64),
                        [[0,1,2],[0,3,2]])
    ReordeNodeInElement(mesh, 0)
    mesh.update_element_area_and_normal()
    torch.testing.assert_close(mesh.element_normal[0], mesh.element_normal[1])
