"""QuadTriangleMesh.update_node_normal (review items MESH-6 and MESH-11), and its normal_source option."""
import math

import pytest
import torch

from torch_fea_mesh.mesh.quad_mesh import QuadMesh
from torch_fea_mesh.mesh.quad_triangle_mesh import QuadTriangleMesh
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh

pytestmark = pytest.mark.geometry

F32, F64 = torch.float32, torch.float64
SOURCES = ('corner', 'element')


def warped_grid(n=5):
    # node, the quads of a warped grid, and the same nodes cut into triangles
    g = torch.Generator().manual_seed(0)
    i, j = torch.meshgrid(torch.arange(n + 1, dtype=F64), torch.arange(n + 1, dtype=F64), indexing='ij')
    node = torch.stack([i, j, torch.sin(i / 2) * torch.cos(j / 2)], dim=-1).reshape(-1, 3)
    node = node + .1 * torch.randn(node.shape, dtype=F64, generator=g)
    k = torch.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    quad = torch.stack([k[:-1, :-1], k[1:, :-1], k[1:, 1:], k[:-1, 1:]], dim=-1).reshape(-1, 4)
    return node, quad, torch.cat([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]])


def mixed_element(quad):
    # every third quad is cut into two triangles
    element = []
    for m, (a, b, c, d) in enumerate(quad.tolist()):
        element += [[a, b, c], [a, c, d]] if m % 3 == 0 else [[a, b, c, d]]
    return element


def by_hand(x, element, angle_weighted, source='corner'):
    # a different method from the code: one corner at a time, with the angle from atan2
    total = torch.zeros_like(x)
    for e in element:
        for k, a in enumerate(e):
            v = x[e[(k + 1) % len(e)]] - x[a]
            w = x[e[k - 1]] - x[a]
            c = torch.linalg.cross(v, w)
            normal = torch.linalg.cross(x[e[2]] - x[e[0]], x[e[3]] - x[e[1]]) if source == 'element' and len(e) == 4 else c
            total[a] += normal / normal.norm() * (torch.atan2(c.norm(), v @ w) if angle_weighted else 1)
    length = total.norm(dim=1, keepdim=True)
    return total / torch.where(length > 0, length, torch.ones_like(length))


def test_only_quads_or_only_triangles_gives_the_result_of_quad_mesh_or_triangle_mesh():
    x, quad, tri = warped_grid()
    for y in (x, x.to(F32)):
        for flag in (True, False):
            for source in SOURCES:
                for element in (quad, quad.tolist()):
                    mesh = QuadTriangleMesh(y, element)
                    mesh.update_node_normal(angle_weighted=flag, normal_source=source)
                    assert torch.equal(mesh.node_normal, QuadMesh.cal_node_normal(y, quad, angle_weighted=flag, normal_source=source))
                    assert mesh.node_normal.dtype == y.dtype
                for element in (tri, tri.tolist()):
                    mesh = QuadTriangleMesh(y, element)
                    mesh.update_node_normal(angle_weighted=flag, normal_source=source)
                    assert torch.equal(mesh.node_normal, TriangleMesh.cal_node_normal(y, tri, angle_weighted=flag))


def test_mixed_mesh_adds_every_corner_before_normalizing():
    # a box edge along x: 3 quads on the top (normal +z) and 1 triangle on the side (normal -y) share node 0,
    # and all 4 corners at node 0 are 60 degrees, so the sum is 3 of +z and 1 of -y
    ray = [[math.cos(t), math.sin(t), 0.] for t in (0, math.pi / 3, 2 * math.pi / 3, math.pi)]
    far = [[1.5 * math.cos(t), 1.5 * math.sin(t), 0.] for t in (math.pi / 6, math.pi / 2, 5 * math.pi / 6)]
    side = [.5, 0., -math.sqrt(3) / 2]
    x = torch.tensor([[0., 0, 0]] + ray + far + [side], dtype=F64)
    element = [[0, 1, 5, 2], [0, 2, 6, 3], [0, 3, 7, 4], [0, 8, 1]]
    expected = torch.tensor([0, -1, 3], dtype=F64) / math.sqrt(10)
    for flag in (True, False):
        torch.testing.assert_close(QuadTriangleMesh.cal_node_normal(x, element, angle_weighted=flag)[0], expected)
    # every node of a warped mesh, against a sum by hand
    x, quad, _ = warped_grid()
    element = mixed_element(quad)
    for flag in (True, False):
        for source in SOURCES:
            torch.testing.assert_close(QuadTriangleMesh.cal_node_normal(x, element, angle_weighted=flag, normal_source=source), by_hand(x, element, flag, source))
    # elements are classified first when the mesh has not done it yet
    late = QuadTriangleMesh()
    late.node, late.element = x, element
    late.update_node_normal()
    torch.testing.assert_close(late.node_normal, by_hand(x, element, True))


def test_normal_source_on_a_flat_concave_quad_with_a_triangle():
    # z of the node normal at x0, x1, x2, x3 and the triangle's node: the reflex corner at x2 flips only with 'corner'
    x = torch.tensor([[0, 0, 0], [4, 0, 0], [1, 1, 0], [0, 4, 0], [4, -4, 0]], dtype=F64)
    element = [[0, 1, 2, 3], [0, 4, 1]]
    assert QuadTriangleMesh.cal_node_normal(x, element)[2, 2] == pytest.approx(-1)
    assert QuadTriangleMesh.cal_node_normal(x, element, normal_source='element')[:, 2].tolist() == pytest.approx([1] * 5)


def test_unknown_source_raises():
    x = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=F64)
    for element in ([[0, 1, 2]], []):
        with pytest.raises(ValueError):
            QuadTriangleMesh.cal_node_normal(x, element, normal_source='Corner')


def test_no_elements_and_a_node_with_no_element():
    x = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [5, 5, 5]], dtype=F64)
    for element in ([], torch.empty((0, 4), dtype=torch.int64)):
        for source in SOURCES:
            mesh = QuadTriangleMesh(x, element)
            mesh.update_node_normal(normal_source=source)
            assert torch.equal(mesh.node_normal, torch.zeros_like(x)) and mesh.node_normal.dtype == F64
    normal = QuadTriangleMesh.cal_node_normal(x, [[0, 1, 2, 3], [0, 1, 2]])
    assert torch.equal(normal[4], torch.zeros(3, dtype=F64)) and torch.isfinite(normal).all()


def test_eps_reaches_every_part():
    # at this scale the default eps would shorten the normals and change the angles
    x, quad, _ = warped_grid()
    element = mixed_element(quad)
    for source in SOURCES:
        expected = QuadTriangleMesh.cal_node_normal(x, element, normal_source=source)
        torch.testing.assert_close(QuadTriangleMesh.cal_node_normal(x * 1e-14, element, eps=1e-40, normal_source=source), expected)
    # a large eps limits each part: quads, triangles, and the sum
    for part in ([[0, 1, 2, 3]], [[0, 1, 2]]):
        small = QuadTriangleMesh.cal_node_normal(x[:4] * 1e-3, part, angle_weighted=False, eps=1.0)
        assert small.norm(dim=1).max() < 1e-3


def test_gradient_and_quads_with_no_area():
    x, quad, _ = warped_grid(2)
    element = mixed_element(quad)
    for source in SOURCES:
        assert torch.autograd.gradcheck(lambda y: QuadTriangleMesh.cal_node_normal(y, element, normal_source=source), x.clone().requires_grad_(True))
    # a collapsed edge in each type: finite, as in QuadMesh and TriangleMesh
    y = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]], dtype=F64).requires_grad_(True)
    normal = QuadTriangleMesh.cal_node_normal(y, [[0, 1, 2, 3], [1, 2, 5], [1, 4, 5]])
    assert torch.isfinite(normal).all()


def test_nan_message_names_this_class(capsys):
    x = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=F64)
    x[0, 0] = float('nan')
    QuadTriangleMesh.cal_node_normal(x, [[0, 1, 2, 3], [0, 1, 2]])
    assert 'QuadTriangleMesh:update_node_normal' in capsys.readouterr().out


def test_node_normal_on_cuda():
    x, quad, _ = warped_grid()
    element = mixed_element(quad)
    for source in SOURCES:
        mesh = QuadTriangleMesh(x.to('cuda'), element)
        mesh.update_node_normal(normal_source=source)
        assert mesh.node_normal.device == mesh.node.device
        torch.testing.assert_close(mesh.node_normal.cpu(), QuadTriangleMesh.cal_node_normal(x, element, normal_source=source))
