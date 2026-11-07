"""Point sampling on TriangleMesh and QuadMesh (review items MESH-4 and MESH-5).

Uniformity is checked against a reference that doesn't use the sampler's formula:
points drawn uniformly in a bounding box, kept if a ray-casting test says they are inside.
"""
import math

import pytest
import torch

from torch_fea_mesh.mesh.quad_mesh import QuadMesh
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh

F64 = torch.float64
both = pytest.mark.parametrize('cls', [TriangleMesh, QuadMesh])

UNIT = {TriangleMesh: [[0, 0], [1, 0], [0, 1]], QuadMesh: [[0, 0], [1, 0], [1, 1], [0, 1]]}
AREA_3 = {TriangleMesh: [[5, 0], [7, 0], [5, 3]], QuadMesh: [[5, 0], [7, 0], [7, 1.5], [5, 1.5]]}
TRAPEZOID = [[0, 0], [3, 0], [2, 1], [1, 1]]
CONCAVE = [[0, 0], [4, 0], [1, 1], [0, 4]]  # the corner at [1, 1] points inward
SHAPES = {TriangleMesh: [[[0, 0], [4, 0.5], [-1, 2]]],
          QuadMesh: [UNIT[QuadMesh], TRAPEZOID, [[0, 0], [4, 0.5], [3, 3], [-1, 1.5]], CONCAVE]}
ANGLE = 0.7
ROTATION = torch.tensor([[1, 0, 0],
                         [0, math.cos(ANGLE), -math.sin(ANGLE)],
                         [0, math.sin(ANGLE), math.cos(ANGLE)]], dtype=F64)


def draw(cls, node, element, n_point, flag=False):
    return cls.uniformly_sample_point(node, element, n_point, at_least_one_point_on_one_element=flag)


def mesh_of(polys, dtype=F64):
    node = torch.tensor([[p[0], p[1], 0.0] for poly in polys for p in poly], dtype=dtype)
    return node, torch.arange(len(node)).reshape(len(polys), -1)


def numberings(poly):
    orders = [poly, poly[::-1]]
    return [order[start:] + order[:start] for order in orders for start in range(len(poly))]


def is_inside(points, poly):
    poly = torch.as_tensor(poly, dtype=F64)
    x, y = points[:, 0].to(F64), points[:, 1].to(F64)
    inside = torch.zeros_like(x, dtype=torch.bool)
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        crosses = (y0 > y) != (y1 > y)
        if y1 != y0:
            crosses = crosses & (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0)
        inside = inside ^ crosses
    return inside


def area_of(poly):
    p = torch.as_tensor(poly, dtype=F64)
    q = p.roll(-1, 0)
    return (0.5 * (p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]).sum()).abs().item()


def histogram(points, poly, bins=6):
    p = torch.as_tensor(poly, dtype=F64)
    low, high = p.min(dim=0).values, p.max(dim=0).values
    index = ((points[:, :2].to(F64) - low) / (high - low) * bins).long().clamp(0, bins - 1)
    counts = torch.bincount(index[:, 0] * bins + index[:, 1], minlength=bins * bins).to(F64)
    return counts / counts.sum()


def reference_histogram(poly, n_keep=1000000, bins=6):
    p = torch.as_tensor(poly, dtype=F64)
    low, high = p.min(dim=0).values, p.max(dim=0).values
    kept, total = [], 0
    while total < n_keep:
        candidate = low + torch.rand(n_keep, 2, dtype=F64) * (high - low)
        kept.append(candidate[is_inside(candidate, poly)])
        total += len(kept[-1])
    return histogram(torch.cat(kept)[:n_keep], poly, bins)


def graded_square(cls):
    """Unit square: many small elements on x < 0.5, and one large rectangle on x > 0.5."""
    def rectangle(x0, x1, y0, y1):
        a, b, c, d = [x0, y0], [x1, y0], [x1, y1], [x0, y1]
        return [[a, b, c, d]] if cls is QuadMesh else [[a, b, c], [a, c, d]]
    polys = [p for i in range(10) for j in range(10) for p in rectangle(0.05 * i, 0.05 * (i + 1), 0.1 * j, 0.1 * (j + 1))]
    n_left = len(polys)
    return polys + rectangle(0.5, 1.0, 0.0, 1.0), n_left


def left_share(cls, polys, n_point, flag, trials=300):
    node, element = mesh_of(polys)
    return sum((draw(cls, node, element, n_point, flag)[:, 0] < 0.5).double().mean().item() for _ in range(trials)) / trials


# ---------- how elements are picked ----------

@both
def test_default_picks_elements_by_area(cls):
    torch.manual_seed(0)
    polys, _ = graded_square(cls)
    M = len(polys)
    for n_point in (M, M + 1, 2 * M):
        assert left_share(cls, polys, n_point, False) == pytest.approx(0.5, abs=0.015)


@both
def test_true_gives_every_element_a_point_then_picks_by_area(cls):
    torch.manual_seed(1)
    polys, n_left = graded_square(cls)
    M = len(polys)
    node, element = mesh_of(polys)
    for n_point in (M, M + 1, 3 * M):
        counts = [is_inside(draw(cls, node, element, n_point, True), poly).sum().item() for poly in polys]
        assert min(counts) >= 1 and (n_point != M or max(counts) == 1)
    assert left_share(cls, polys, 2 * M, True) == pytest.approx((n_left + M * 0.5) / (2 * M), abs=0.015)
    for n_point in (0, 1, M - 1):
        with pytest.raises(ValueError, match='n_point'):
            draw(cls, node, element, n_point, True)


# ---------- where points land inside an element ----------

@both
def test_points_are_uniform_in_every_shape_and_numbering(cls):
    torch.manual_seed(2)
    for poly in SHAPES[cls]:
        reference = reference_histogram(poly)
        for numbered in numberings(poly):
            x = draw(cls, *mesh_of([numbered]), 300000)
            assert is_inside(x, poly).all(), numbered
            assert (histogram(x, poly) - reference).abs().max().item() < 0.004, numbered
        node, element = mesh_of([poly])
        back = draw(cls, node @ ROTATION.T, element, 300000) @ ROTATION
        assert back[:, 2].abs().max().item() < 1e-12
        assert (histogram(back, poly) - reference).abs().max().item() < 0.004, poly


def test_mixed_quads_only_some_needing_the_other_diagonal():
    torch.manual_seed(3)
    shapes = [UNIT[QuadMesh], CONCAVE, TRAPEZOID, numberings(CONCAVE)[1], numberings(CONCAVE)[6], numberings(CONCAVE)[2]]
    polys = [[[p[0] + 10.0 * i, p[1]] for p in poly] for i, poly in enumerate(shapes)]
    node, element = mesh_of(polys)
    for flag, n_point in ((False, 50), (True, 6), (True, 200000), (False, 2000000)):
        x = draw(QuadMesh, node, element, n_point, flag)
        which = ((x[:, 0] + 5) // 10).long()
        for i, poly in enumerate(polys):
            assert is_inside(x[which == i], poly).all()
    areas = torch.tensor([area_of(poly) for poly in polys], dtype=F64)
    assert (torch.bincount(which, minlength=len(polys)) / n_point - areas / areas.sum()).abs().max().item() < 0.002
    for i, poly in enumerate(polys):
        reference = reference_histogram(poly, n_keep=500000, bins=4)
        assert (histogram(x[which == i], poly, bins=4) - reference).abs().max().item() < 0.006


def test_warped_quad_points_are_on_two_flat_triangles():
    torch.manual_seed(4)
    for numbered in numberings([[0, 0, 0], [2, 0, 0.1], [2, 1, 0], [0, 1, 0.3]])[:4]:
        node = torch.tensor(numbered, dtype=F64)
        x = draw(QuadMesh, node, torch.tensor([[0, 1, 2, 3]]), 200000)
        a, b, c, d = node
        normal_012, normal_023 = torch.linalg.cross(b - a, c - a, dim=-1), torch.linalg.cross(c - a, d - a, dim=-1)
        length_012, length_023 = torch.linalg.vector_norm(normal_012), torch.linalg.vector_norm(normal_023)
        distance_012 = ((x - a) * normal_012 / length_012).sum(dim=1).abs()
        distance_023 = ((x - a) * normal_023 / length_023).sum(dim=1).abs()
        assert torch.minimum(distance_012, distance_023).max().item() < 1e-12
        assert (distance_012 < 1e-12).double().mean().item() == pytest.approx((length_012 / (length_012 + length_023)).item(), abs=0.004)


# ---------- meshes that are hard to sample ----------

@both
def test_mesh_without_a_valid_area_raises(cls):
    node, element = mesh_of([UNIT[cls], AREA_3[cls]])
    one_nan = node.clone()
    one_nan[0, 0] = float('nan')
    for bad_node, bad_element in ((node * 0, element), (one_nan, element), (node * float('inf'), element),
                                  (node * 1e200, element), (node, element[:0])):
        with pytest.raises(ValueError, match='area'):
            draw(cls, bad_node, bad_element, 5)


@both
def test_zero_area_element(cls):
    torch.manual_seed(5)
    good, zero = UNIT[cls], [[5, 5]] * len(UNIT[cls])
    for polys in ([good, zero], [zero, good]):
        x = draw(cls, *mesh_of(polys), 200000)
        assert (x[:, 0] <= 1).all()
    assert draw(cls, *mesh_of([good, zero]), 2, True)[1].tolist() == [5, 5, 0]
    assert draw(cls, *mesh_of([zero]), 1, True).tolist() == [[5, 5, 0]]


def test_quads_with_a_collapsed_corner_or_crossed_edges():
    torch.manual_seed(6)
    x = draw(QuadMesh, *mesh_of([[[0, 0], [1, 0], [0, 1], [0, 1]]]), 400000)
    assert x[:, 0].mean().item() == pytest.approx(1 / 3, abs=0.002)
    assert x[:, 0].var().item() == pytest.approx(1 / 18, abs=0.001)
    assert torch.isfinite(draw(QuadMesh, *mesh_of([[[0, 0], [2, 1], [2, 0], [0, 1.5]]]), 1000)).all()
    with pytest.raises(ValueError, match='area'):
        draw(QuadMesh, *mesh_of([[[0, 0], [1, 1], [1, 0], [0, 1]]]), 5)


@both
def test_tiny_elements_in_float32_are_picked_by_area(cls):
    torch.manual_seed(7)
    scale = 1e-25
    small, large = UNIT[cls], AREA_3[cls]
    node, element = mesh_of([small, large])
    x = draw(cls, (node * scale).to(torch.float32), element, 200000)
    assert x.dtype == torch.float32
    expected = area_of(small) / (area_of(small) + area_of(large))
    assert (x[:, 0] / scale < 2).double().mean().item() == pytest.approx(expected, abs=0.004)


def test_tiny_concave_quad_in_float32_is_uniform():
    torch.manual_seed(8)
    scale = 1e-25
    reference = reference_histogram(CONCAVE)
    for numbered in numberings(CONCAVE)[:2]:
        node, element = mesh_of([numbered])
        x = draw(QuadMesh, (node * scale).to(torch.float32), element, 200000)
        assert (histogram(x / scale, CONCAVE) - reference).abs().max().item() < 0.006


# ---------- the interface ----------

@both
def test_sample_point_and_the_default_setting(cls):
    polys, _ = graded_square(cls)
    node, element = mesh_of(polys)
    mesh = cls(node, element)
    for call, flag in ((lambda: mesh.sample_point(300), False), (lambda: cls.uniformly_sample_point(node, element, 300), False),
                       (lambda: mesh.sample_point(300, at_least_one_point_on_one_element=True), True)):
        torch.manual_seed(9)
        result = call()
        torch.manual_seed(9)
        assert torch.equal(result, draw(cls, node, element, 300, flag))


@both
def test_result_shape_and_dtype_and_unchanged_inputs(cls):
    for dtype in (torch.float32, F64):
        node, element = mesh_of([UNIT[cls], AREA_3[cls]], dtype)
        before = node.clone(), element.clone()
        for flag in (False, True):
            x = draw(cls, node, element, 7, flag)
            assert x.shape == (7, 3) and x.dtype == dtype
        assert torch.equal(node, before[0]) and torch.equal(element, before[1])
        assert draw(cls, node, element, 0).shape == (0, 3)


@both
def test_gradient_reaches_the_nodes(cls):
    torch.manual_seed(10)
    zero = [[5, 5]] * len(UNIT[cls])
    for polys, flag in (([UNIT[cls], AREA_3[cls]], False), ([UNIT[cls], zero], True)):
        node, element = mesh_of(polys)
        node.requires_grad_(True)
        draw(cls, node, element, 500, flag).sum().backward()
        # the weights of each point add up to 1, so each coordinate's gradients add up to n_point
        assert torch.isfinite(node.grad).all()
        assert node.grad.sum(dim=0).tolist() == pytest.approx([500.0, 500.0, 500.0], abs=1e-9)


@both
@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device')
def test_runs_on_cuda(cls):
    device = f'cuda:{torch.cuda.device_count() - 1}'
    node, element = mesh_of([UNIT[cls], AREA_3[cls]])
    for flag in (False, True):
        x = draw(cls, node.to(device), element.to(device), 300, flag)
        assert x.device.type == 'cuda' and x.shape == (300, 3)
