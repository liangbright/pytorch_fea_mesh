import pytest
import torch

from torch_fea_mesh.generation.mesh_example import (
    create_quad_grid_mesh, create_hex_grid_mesh, create_quad_cylinder_mesh,
    create_quad_mesh_rectangle_in_rectangle, create_quad_mesh_rectangle_in_cirlce,
    create_quad_tri_mesh_circle_in_circle,
)

pytestmark = pytest.mark.generation


def test_grid_count_and_boundary_set():
    quad = create_quad_grid_mesh(3, 4, dtype=torch.float64)
    assert quad.node.shape == (12, 3)
    assert quad.element.shape == (6, 4)
    assert len(quad.node_set['boundary']) == 10
    hexa = create_hex_grid_mesh(3, 4, 2, dtype=torch.float64)
    assert hexa.node.shape == (24, 3)
    assert hexa.element.shape == (6, 8)
    hexa.update_element_volume()
    torch.testing.assert_close(hexa.element_volume, torch.ones(6, dtype=torch.float64))


def test_cylinder_geometry_and_end_set():
    mesh = create_quad_cylinder_mesh(3, 8, radius=2, height=4, dtype=torch.float64)
    assert mesh.element.shape == (16, 4)
    torch.testing.assert_close(torch.linalg.vector_norm(mesh.node[:,:2],dim=-1), torch.full((24,), 2., dtype=torch.float64))
    assert torch.all(mesh.node[mesh.node_set['boundary_lower'],2] == 0)
    assert torch.all(mesh.node[mesh.node_set['boundary_upper'],2] == 4)


@pytest.mark.parametrize("builder", [create_quad_mesh_rectangle_in_rectangle, create_quad_mesh_rectangle_in_cirlce])
@pytest.mark.parametrize("seal_hole", [False, True])
def test_rectangle_based_generator(builder, seal_hole):
    mesh = builder(n_ring=3, Nx=3, Ny=3, seal_hole=seal_hole)
    assert torch.isfinite(mesh.node).all()
    assert len(mesh.element) > 0
    assert 'ABCD' in mesh.node_set
    assert ('hole' in mesh.element_set) == seal_hole


@pytest.mark.parametrize("seal_hole", [False, True])
def test_circular_generator(seal_hole):
    mesh = create_quad_tri_mesh_circle_in_circle(n_circle=3, n_point_per_circle=8, seal_hole=seal_hole)
    assert len(mesh.node) == 24 + int(seal_hole)
    assert len(mesh.element) == 16 + (8 if seal_hole else 0)
    assert len(mesh.node_set['boundary']) == 8
    assert len(mesh.element_set['hole']) == (8 if seal_hole else 0)
