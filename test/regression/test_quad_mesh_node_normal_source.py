"""QuadMesh node normal: the normal_source option."""
import pytest
import torch
from torch.linalg import cross

from torch_fea_mesh.mesh.quad_mesh import QuadMesh

pytestmark = pytest.mark.geometry

F32, F64 = torch.float32, torch.float64
E1 = torch.tensor([[0, 1, 2, 3]])
CONCAVE = [[0, 0, 0], [4, 0, 0], [1, 1, 0], [0, 4, 0]]  # reflex corner at x2
STRAIGHT = [[0, 0, 0], [2, 0, 0], [2, 2, 0], [1, 1, 0]]  # straight corner at x3
BOWTIE = [[0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0]]  # parallel diagonals, so no element normal


def warped_grid(n=5):
    g = torch.Generator().manual_seed(0)
    i, j = torch.meshgrid(torch.arange(n + 1, dtype=F64), torch.arange(n + 1, dtype=F64), indexing='ij')
    node = torch.stack([i, j, torch.sin(i / 2) * torch.cos(j / 2)], dim=-1).reshape(-1, 3)
    node = node + .1 * torch.randn(node.shape, dtype=F64, generator=g)
    k = torch.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    return node, torch.stack([k[:-1, :-1], k[1:, :-1], k[1:, 1:], k[:-1, 1:]], dim=-1).reshape(-1, 4)


def by_hand(x, element, source, angle_weighted):
    # a different method from the code: one corner at a time, with the angle from atan2
    total = torch.zeros_like(x)
    for e in element.tolist():
        whole = cross(x[e[2]] - x[e[0]], x[e[3]] - x[e[1]])
        for k, a in enumerate(e):
            v, w = x[e[(k + 1) % 4]] - x[a], x[e[k - 1]] - x[a]
            c = cross(v, w)
            normal = c if source == 'corner' else whole
            total[a] += normal / normal.norm() * (torch.atan2(c.norm(), v @ w) if angle_weighted else 1)
    return total / total.norm(dim=1, keepdim=True)


def test_each_source_against_a_sum_by_hand():
    x, element = warped_grid()
    for source in ('corner', 'element'):
        for flag in (True, False):
            expected = by_hand(x, element, source, flag)
            torch.testing.assert_close(QuadMesh.cal_node_normal(x, element, angle_weighted=flag, normal_source=source), expected)
            mesh = QuadMesh(x, element)
            mesh.update_node_normal(angle_weighted=flag, normal_source=source)
            torch.testing.assert_close(mesh.node_normal, expected)
    # the default is the corner normal, in both functions
    mesh = QuadMesh(x, element)
    mesh.update_node_normal()
    expected = QuadMesh.cal_node_normal(x, element, normal_source='corner')
    assert torch.equal(mesh.node_normal, expected) and torch.equal(QuadMesh.cal_node_normal(x, element), expected)


def test_flat_concave_quad_and_straight_corner():
    # z of the node normal at x0, x1, x2, x3
    for nodes, corner in ((CONCAVE, [1, 1, -1, 1]), (STRAIGHT, [1, 1, 1, 0])):
        x = torch.tensor(nodes, dtype=F64)
        assert QuadMesh.cal_node_normal(x, E1)[:, 2].tolist() == pytest.approx(corner)
        assert QuadMesh.cal_node_normal(x, E1, normal_source='element')[:, 2].tolist() == pytest.approx([1, 1, 1, 1])


def test_unknown_source_raises():
    x, element = warped_grid(1)
    for e in (element, torch.empty((0, 4), dtype=torch.int64)):
        with pytest.raises(ValueError):
            QuadMesh.cal_node_normal(x, e, normal_source='Corner')
    with pytest.raises(ValueError):
        QuadMesh(x, element).update_node_normal(normal_source='Element')


def test_element_source_keeps_the_dtype_and_takes_no_elements():
    x, element = warped_grid()
    normal = QuadMesh.cal_node_normal(x.to(F32), element, normal_source='element')
    assert normal.dtype == F32 and normal.shape == x.shape
    # a node with no element, and a mesh with no elements
    y = torch.cat([x, torch.tensor([[50., 50, 50]], dtype=F64)])
    assert QuadMesh.cal_node_normal(y, element, normal_source='element')[-1].abs().sum() == 0
    empty = QuadMesh.cal_node_normal(x, torch.empty((0, 4), dtype=torch.int64), normal_source='element')
    assert torch.equal(empty, torch.zeros_like(x))


def test_eps_reaches_the_element_source_in_both_functions():
    # at this scale the default eps would shorten the normals and change the angles; eps is still the 4th argument
    x, element = warped_grid()
    expected = QuadMesh.cal_node_normal(x, element, normal_source='element')
    torch.testing.assert_close(QuadMesh.cal_node_normal(x * 1e-14, element, True, 1e-40, 'element'), expected)
    mesh = QuadMesh(x * 1e-14, element)
    mesh.update_node_normal(eps=1e-40, normal_source='element')
    torch.testing.assert_close(mesh.node_normal, expected)


def test_element_source_gradient():
    x, element = warped_grid(2)
    assert torch.autograd.gradcheck(lambda y: QuadMesh.cal_node_normal(y, element, normal_source='element'), x.clone().requires_grad_(True))
    # finite on a quad with no element normal
    y = torch.tensor(BOWTIE, dtype=F64).requires_grad_(True)
    normal = QuadMesh.cal_node_normal(y, E1, normal_source='element')
    assert normal.abs().sum() == 0
    normal.sum().backward()
    assert torch.isfinite(y.grad).all()


def test_element_source_on_cuda():
    x, element = warped_grid()
    normal = QuadMesh.cal_node_normal(x.to('cuda'), element.to('cuda'), normal_source='element')
    assert normal.device.type == 'cuda'
    torch.testing.assert_close(normal.cpu(), QuadMesh.cal_node_normal(x, element, normal_source='element'))
