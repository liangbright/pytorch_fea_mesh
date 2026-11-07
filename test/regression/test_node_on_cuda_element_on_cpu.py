"""Point sampling, node normals and QuadMesh local surface flatness with node on CUDA and element on the CPU.

A list or numpy element goes on the device of node, but an element tensor stays on its own device,
so this mix can happen.

Results that come from torch_scatter are compared with a tolerance, not torch.equal: scatter_add_
on CUDA has no fixed order, so the last digits change between runs.
"""
import pytest
import torch

from torch_fea_mesh.mesh.quad_mesh import QuadMesh
from torch_fea_mesh.mesh.tri6_mesh import Tri6Mesh
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh

pytestmark = [pytest.mark.geometry, pytest.mark.gpu]

F64 = torch.float64
TOLERANCE = dict(rtol=1e-12, atol=1e-12)

sampling_classes = pytest.mark.parametrize('cls', [TriangleMesh, QuadMesh])
sampling_flags = pytest.mark.parametrize('flag', [False, True])
normal_cases = pytest.mark.parametrize('cls, angle_weighted, option', [
    (TriangleMesh, True, {}),
    (TriangleMesh, False, {}),
    (Tri6Mesh, True, {}),
    (Tri6Mesh, False, {}),
    (QuadMesh, True, {'normal_source': 'corner'}),
    (QuadMesh, False, {'normal_source': 'corner'}),
    (QuadMesh, True, {'normal_source': 'element'}),
    (QuadMesh, False, {'normal_source': 'element'}),
])

# Two separate elements. The second has 3 times the area of the first, in both classes.
SAMPLING_MESH = {
    TriangleMesh: ([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [7, 0, 0], [5, 1.5, 0]],
                   [[0, 1, 2], [3, 4, 5]]),
    QuadMesh: ([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [5, 0, 0], [7, 0, 0], [7, 1.5, 0], [5, 1.5, 0]],
               [[0, 1, 2, 3], [4, 5, 6, 7]]),
}


def sampling_mesh(cls):
    node, element = SAMPLING_MESH[cls]
    return torch.tensor(node, dtype=F64), torch.tensor(element)


def draw(cls, node, element, n_point, flag, seed=7):
    torch.manual_seed(seed)
    return cls.uniformly_sample_point(node, element, n_point, at_least_one_point_on_one_element=flag)


def height(x, y):
    return torch.sin(torch.as_tensor(x / 2, dtype=F64)) * torch.cos(torch.as_tensor(y / 2, dtype=F64))


def quad_grid(n=4):
    i, j = torch.meshgrid(torch.arange(n + 1, dtype=F64), torch.arange(n + 1, dtype=F64), indexing='ij')
    node = torch.stack([i, j, height(i, j)], dim=-1).reshape(-1, 3)
    k = torch.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    return node, torch.stack([k[:-1, :-1], k[1:, :-1], k[1:, 1:], k[:-1, 1:]], dim=-1).reshape(-1, 4)


def triangle_grid():
    node, quad = quad_grid()
    return node, torch.cat([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]], dim=0)


def tri6_grid():
    # Corners first, then the midpoints of edges 01, 12 and 20, on the same curved surface.
    node, triangle = triangle_grid()
    point = node.tolist()
    midpoint = {}

    def node_at_middle(a, b):
        key = (min(a, b), max(a, b))
        if key not in midpoint:
            x = (point[a][0] + point[b][0]) / 2
            y = (point[a][1] + point[b][1]) / 2
            midpoint[key] = len(point)
            point.append([x, y, height(x, y).item()])
        return midpoint[key]

    element = [[a, b, c, node_at_middle(a, b), node_at_middle(b, c), node_at_middle(c, a)]
               for a, b, c in triangle.tolist()]
    return torch.tensor(point, dtype=F64), torch.tensor(element)


GRID = {TriangleMesh: triangle_grid, Tri6Mesh: tri6_grid, QuadMesh: quad_grid}


# ------------------------------------------------------------------------------- Point sampling
@sampling_classes
@sampling_flags
def test_sampling_gives_the_same_points_as_with_both_on_cuda(cls, flag):
    node, element = sampling_mesh(cls)
    expected = draw(cls, node.cuda(), element.cuda(), 300, flag)
    x = draw(cls, node.cuda(), element, 300, flag)
    assert x.is_cuda and x.shape == (300, 3)
    assert torch.equal(x, expected)


@sampling_classes
@sampling_flags
def test_sample_point_keeps_element_on_cpu(cls, flag):
    node, element = sampling_mesh(cls)
    mesh = cls(node.cuda(), element)
    x = mesh.sample_point(300, flag)
    assert x.is_cuda and x.shape == (300, 3)
    assert mesh.element.device.type == 'cpu'
    assert torch.equal(mesh.element, element)


@sampling_classes
def test_sampled_points_follow_the_area_of_each_element(cls):
    node, element = sampling_mesh(cls)
    x = draw(cls, node.cuda(), element, 20000, False)
    assert abs((x[:, 0] >= 5).to(F64).mean().item() - 0.75) < 0.02


@sampling_classes
def test_sampling_one_point_on_each_element_when_nothing_is_picked_by_area(cls):
    node, element = sampling_mesh(cls)
    x = draw(cls, node.cuda(), element, 2, True)
    assert x.is_cuda
    assert x[0, 0] < 5 <= x[1, 0]


# --------------------------------------------------------------------------------- Node normals
@normal_cases
def test_node_normal_same_as_with_both_on_cuda(cls, angle_weighted, option):
    node, element = GRID[cls]()
    expected = cls.cal_node_normal(node.cuda(), element.cuda(), angle_weighted, **option)
    normal = cls.cal_node_normal(node.cuda(), element, angle_weighted, **option)
    assert normal.is_cuda and normal.shape == node.shape
    assert torch.allclose(normal, expected, **TOLERANCE)


@normal_cases
def test_node_normal_same_as_on_the_cpu(cls, angle_weighted, option):
    node, element = GRID[cls]()
    expected = cls.cal_node_normal(node, element, angle_weighted, **option)
    normal = cls.cal_node_normal(node.cuda(), element, angle_weighted, **option)
    assert torch.allclose(normal.cpu(), expected, **TOLERANCE)


@pytest.mark.parametrize('cls', [TriangleMesh, Tri6Mesh])
@pytest.mark.parametrize('angle_weighted', [True, False])
def test_node_normal_with_a_given_element_normal(cls, angle_weighted):
    node, element = GRID[cls]()
    expected = cls.cal_node_normal(node, element, angle_weighted)
    _, element_normal = cls.cal_element_area_and_normal(node.cuda(), element)
    normal = cls.cal_node_normal(node.cuda(), element, angle_weighted, element_normal)
    assert torch.allclose(normal.cpu(), expected, **TOLERANCE)


@normal_cases
def test_update_node_normal_keeps_element_on_cpu(cls, angle_weighted, option):
    node, element = GRID[cls]()
    mesh = cls(node.cuda(), element)
    mesh.update_node_normal(angle_weighted=angle_weighted, **option)
    assert mesh.node_normal.is_cuda
    assert mesh.element.device.type == 'cpu'
    assert torch.equal(mesh.element, element)


@normal_cases
def test_node_normal_same_gradient_as_with_both_on_cuda(cls, angle_weighted, option):
    node, element = GRID[cls]()
    node_a = node.cuda().requires_grad_(True)
    node_b = node.cuda().requires_grad_(True)
    cls.cal_node_normal(node_a, element.cuda(), angle_weighted, **option).sum().backward()
    cls.cal_node_normal(node_b, element, angle_weighted, **option).sum().backward()
    assert node_b.grad.abs().sum() > 0
    assert torch.allclose(node_a.grad, node_b.grad, **TOLERANCE)


# ------------------------------------------------------------------------ Local surface flatness
def test_flatness_same_as_with_both_on_cuda():
    node, element = quad_grid()
    expected = QuadMesh.cal_local_surface_flatness(node.cuda(), element.cuda())
    flatness = QuadMesh.cal_local_surface_flatness(node.cuda(), element)
    assert flatness.is_cuda and flatness.shape == (len(node),)
    assert torch.allclose(flatness, expected, **TOLERANCE)


def test_flatness_same_as_on_the_cpu():
    node, element = quad_grid()
    expected = QuadMesh.cal_local_surface_flatness(node, element)
    flatness = QuadMesh.cal_local_surface_flatness(node.cuda(), element)
    assert torch.allclose(flatness.cpu(), expected, **TOLERANCE)


def test_flatness_with_a_given_element_normal():
    node, element = quad_grid()
    expected = QuadMesh.cal_local_surface_flatness(node, element)
    element_normal = QuadMesh.cal_element_normal(node.cuda(), element)
    flatness = QuadMesh.cal_local_surface_flatness(node.cuda(), element, element_normal)
    assert torch.allclose(flatness.cpu(), expected, **TOLERANCE)


def test_flatness_is_one_on_a_flat_mesh():
    node, element = quad_grid()
    node[:, 2] = 0
    flatness = QuadMesh.cal_local_surface_flatness(node.cuda(), element)
    assert torch.allclose(flatness.cpu(), torch.ones(len(node), dtype=F64), **TOLERANCE)


def test_flatness_same_gradient_as_with_both_on_cuda():
    node, element = quad_grid()
    node_a = node.cuda().requires_grad_(True)
    node_b = node.cuda().requires_grad_(True)
    QuadMesh.cal_local_surface_flatness(node_a, element.cuda()).sum().backward()
    QuadMesh.cal_local_surface_flatness(node_b, element).sum().backward()
    assert node_b.grad.abs().sum() > 0
    assert torch.allclose(node_a.grad, node_b.grad, **TOLERANCE)
