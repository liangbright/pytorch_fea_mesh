"""QuadMesh: area (review item MESH-17), element normal, flatness, names and messages (MESH-18), subdivide_to_quad.
Also update_element_normal on every surface mesh, which simple_smoother_for_mesh_element_normal calls.

Areas and normals are checked against a reference that doesn't use the code's formula:
the jacobian vector of the shape functions at the midpoints of a fine grid.
"""
import math

import pytest
import torch

from torch_fea_mesh.mesh.quad_mesh import QuadMesh
from torch_fea_mesh.mesh.quad_triangle_mesh import QuadTriangleMesh
from torch_fea_mesh.mesh.tri6_mesh import Tri6Mesh
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh
from torch_fea_mesh.processing.polygon_mesh_processing import simple_smoother_for_mesh_element_normal

pytestmark = pytest.mark.geometry

F32, F64 = torch.float32, torch.float64
E1 = torch.tensor([[0, 1, 2, 3]])
SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1]]
CONCAVE = [[0, 0], [4, 0], [1, 1], [0, 4]]  # reflex corner at index 2
BOWTIE = [[0, 0], [1, 1], [1, 0], [0, 1]]  # parallel diagonals, so no area
LIFTED = [[0, 0, 0], [1, 0, 0], [1, 1, .8], [0, 1, 0]]
SADDLE = [[0, 0, .5], [1, 0, -.5], [1, 1, .5], [0, 1, -.5]]
ROTATION = torch.tensor([[1, 0, 0],
                         [0, math.cos(.7), -math.sin(.7)],
                         [0, math.sin(.7), math.cos(.7)]], dtype=F64)


def flat(poly, dtype=F64, rotate=False):
    x = torch.tensor([[p[0], p[1], 0.0] for p in poly], dtype=F64)
    return (x @ ROTATION.T if rotate else x).to(dtype)


def warped(nodes):
    return torch.tensor(nodes, dtype=F64)


def starts(poly):
    return [poly[k:] + poly[:k] for k in range(len(poly))]


def one_point_area(x, element):
    a, b = x[element[:, 2]] - x[element[:, 0]], x[element[:, 3]] - x[element[:, 1]]
    return torch.linalg.cross(a, b, dim=1).norm(dim=1, keepdim=True) / 2


def fine_jacobian(x, n):
    t = (torch.arange(n, dtype=F64) + .5) / n
    r, s = t.reshape(-1, 1, 1), t.reshape(1, -1, 1)
    x0, x1, x2, x3 = x
    dr = (1 - s) * (x1 - x0) + s * (x2 - x3)
    ds = (1 - r) * (x3 - x0) + r * (x2 - x1)
    return torch.linalg.cross(dr.expand(n, n, 3), ds.expand(n, n, 3), dim=2).reshape(-1, 3)


def warped_grid(n=12):
    g = torch.Generator().manual_seed(0)
    i, j = torch.meshgrid(torch.arange(n + 1, dtype=F64), torch.arange(n + 1, dtype=F64), indexing='ij')
    node = torch.stack([i, j, torch.sin(i / 3) * torch.cos(j / 3)], dim=-1).reshape(-1, 3)
    node = node + .1 * torch.randn(node.shape, dtype=F64, generator=g)
    k = torch.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    return node, torch.stack([k[:-1, :-1], k[1:, :-1], k[1:, 1:], k[:-1, 1:]], dim=-1).reshape(-1, 4)


#%% area
def test_area_of_flat_quads():
    # square, trapezoid, concave and straight corner, from every starting node and in both directions
    for poly, expected in ((SQUARE, 1.0), ([[0, 0], [3, 0], [2, 1], [1, 1]], 2.0), (CONCAVE, 4.0), ([[0, 0], [2, 0], [2, 2], [1, 1]], 2.0)):
        for order in starts(poly) + starts(poly[::-1]):
            for dtype, rotate, tolerance in ((F64, False, 1e-13), (F32, True, 1e-5)):
                area, _ = QuadMesh.cal_element_area_and_normal(flat(order, dtype, rotate), E1)
                assert area.item() == pytest.approx(expected, rel=tolerance), poly
    # the fold must be handled below eps too: here the cross product at the centre is about 1e-16
    small, _ = QuadMesh.cal_element_area_and_normal(flat(CONCAVE) * 1e-8, E1)
    assert small.item() / 4e-16 == pytest.approx(1.0, rel=1e-13)
    collapsed, _ = QuadMesh.cal_element_area_and_normal(flat([[0, 0], [2, 0], [0, 2], [0, 2]]), E1)
    assert collapsed.item() == pytest.approx(2.0, rel=1e-13)
    # a bowtie keeps the net value
    assert QuadMesh.cal_element_area_and_normal(flat(BOWTIE), E1)[0].item() == 0
    bowtie = flat([[0, 0], [2, 2], [2, 0], [0, 1]])
    assert QuadMesh.cal_element_area_and_normal(bowtie, E1)[0].item() == pytest.approx(one_point_area(bowtie, E1).item(), rel=1e-13)


def test_area_of_warped_quads():
    # (nodes, relative tolerance): the errors of 2x2 Gauss points, measured
    for nodes, tolerance in ((LIFTED, 2e-4), (SADDLE, 1e-2)):
        x = warped(nodes)
        exact = fine_jacobian(x, 1000).norm(dim=1).mean().item()
        for order in starts([0, 1, 2, 3]):
            area, _ = QuadMesh.cal_element_area_and_normal(x, torch.tensor([order]))
            assert area.item() == pytest.approx(exact, rel=tolerance)
    x, element = warped_grid()
    area, _ = QuadMesh.cal_element_area_and_normal(x, element)
    fine = torch.stack([fine_jacobian(x[e], 300).norm(dim=1).mean() for e in element]).reshape(-1, 1)
    assert ((area - fine).abs() / fine).max() < 1e-4
    # any quad, folded or not: finite, and never below the one-point area
    noise = torch.randn(20000, 4, 3, dtype=F64, generator=torch.Generator().manual_seed(1))
    x = (flat(SQUARE)[None] + noise).reshape(-1, 3)
    element = torch.arange(len(x)).reshape(-1, 4)
    area, _ = QuadMesh.cal_element_area_and_normal(x, element)
    assert torch.isfinite(area).all() and (area >= one_point_area(x, element) * (1 - 1e-12)).all()


def test_area_gradient():
    # against a finite difference, on a warped quad, a warped concave quad, and at a small scale
    for nodes, scale in ((LIFTED, 1.0), ([[0, 0, 0], [4, 0, 0], [1, 1, .1], [0, 4, 0]], 1.0), (LIFTED, 1e-8)):
        x = warped(nodes) * scale
        y = x.clone().requires_grad_(True)
        QuadMesh.cal_element_area_and_normal(y, E1)[0].sum().backward()
        for i, j in ((2, 2), (0, 0), (3, 1)):
            step = 1e-6 * scale
            hi, lo = x.clone(), x.clone()
            hi[i, j] += step
            lo[i, j] -= step
            difference = (QuadMesh.cal_element_area_and_normal(hi, E1)[0] - QuadMesh.cal_element_area_and_normal(lo, E1)[0]).item() / (2 * step)
            assert y.grad[i, j].item() == pytest.approx(difference, rel=1e-5, abs=1e-5 * scale)
    # finite on quads with no area
    for poly in ([[0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [1, 0], [2, 0], [3, 0]], BOWTIE):
        y = flat(poly).requires_grad_(True)
        area, normal = QuadMesh.cal_element_area_and_normal(y, E1)
        (area.sum() + normal.sum()).backward()
        assert torch.isfinite(y.grad).all()


#%% element normal
def test_element_normal():
    # both functions give the same normal, keep the dtype, and return (M,1) and (M,3)
    x, element = warped_grid()
    for y in (x, x.to(F32)):
        area, normal = QuadMesh.cal_element_area_and_normal(y, element)
        assert torch.equal(QuadMesh.cal_element_normal(y, element), normal)
        assert area.shape == (len(element), 1) and normal.shape == (len(element), 3)
        assert area.dtype == normal.dtype == y.dtype
    no_node, no_element = torch.empty((0, 3), dtype=F64), torch.empty((0, 4), dtype=torch.int64)
    assert QuadMesh.cal_element_normal(no_node, no_element).shape == (0, 3)
    assert QuadMesh.cal_element_area_and_normal(no_node, no_element)[0].shape == (0, 1)
    # it is the area-weighted mean normal of the bilinear surface, from any starting node
    for x in (warped(LIFTED), warped(SADDLE), flat(CONCAVE)):
        mean = fine_jacobian(x, 400).sum(dim=0)
        for order in starts([0, 1, 2, 3]):
            torch.testing.assert_close(QuadMesh.cal_element_normal(x, torch.tensor([order]))[0], mean / mean.norm())


def test_element_normal_eps_and_no_area():
    # eps limits the length of cross(dxdu,dxdv), which is 2.5e-13 for this square
    x = flat(SQUARE) * 1e-6
    assert QuadMesh.cal_element_normal(x, E1).norm().item() == pytest.approx(.25)
    assert QuadMesh.cal_element_area_and_normal(x, E1)[1].norm().item() == pytest.approx(.25)
    assert QuadMesh.cal_element_normal(x, E1, eps=1e-14).norm().item() == pytest.approx(1)
    y = flat(BOWTIE).requires_grad_(True)
    normal = QuadMesh.cal_element_normal(y, E1)
    assert normal.abs().sum() == 0
    normal.sum().backward()
    assert torch.isfinite(y.grad).all()


def test_element_normal_keeps_digits_far_from_the_origin():
    # float32: adding coordinates before subtracting loses digits, so the bound fails if they are added first
    x, element = warped_grid()
    y = (x + 1000).to(F32)
    for normal in (QuadMesh.cal_element_normal(y, element), QuadMesh.cal_element_area_and_normal(y, element)[1]):
        exact = QuadMesh.cal_element_normal(y.to(F64), element)
        assert torch.linalg.cross(normal.to(F64), exact, dim=1).norm(dim=1).max() < 1e-6


#%% flatness
def test_normal_callers_do_not_compute_the_quad_area(monkeypatch):
    x, element = warped_grid()
    expected = QuadMesh.cal_local_surface_flatness(x, element, element_normal=QuadMesh.cal_element_area_and_normal(x, element)[1])

    def no_area(*args, **kwargs):
        raise AssertionError('the area is not needed')
    monkeypatch.setattr(QuadMesh, 'cal_element_area_and_normal', staticmethod(no_area))
    assert torch.equal(QuadMesh.cal_local_surface_flatness(x, element), expected)
    # eps reaches the normal: at this scale the default eps would shorten it
    torch.testing.assert_close(QuadMesh.cal_local_surface_flatness(x * 1e-8, element, eps=1e-30), expected)
    QuadMesh(x, element).update_element_normal()
    QuadTriangleMesh(x, element.tolist()).update_element_normal()


def test_flatness_marks_a_concave_quad_as_a_bad_shape():
    # by design: a concave quad is a bad shape, so it does not get 1 even when it is flat
    assert QuadMesh.cal_element_flatness(flat(SQUARE), E1).item() == pytest.approx(1.0, abs=1e-12)
    for order in starts(CONCAVE):
        assert QuadMesh.cal_element_flatness(flat(order), E1).item() == pytest.approx(0.5, abs=1e-12)
    assert QuadMesh.cal_element_flatness(flat(BOWTIE), E1).item() == pytest.approx(0.0, abs=1e-12)


#%% names and messages
def test_flatness_name_and_nan_message(capsys):
    assert not hasattr(QuadMesh, 'upate_element_flatness')
    mesh = QuadMesh(flat(SQUARE), E1)
    mesh.update_element_flatness()
    assert mesh.element_flatness.item() == 1
    x = flat(SQUARE)
    x[0, 0] = float('nan')
    QuadMesh(x, E1).update_node_normal()
    out = capsys.readouterr().out
    assert 'QuadMesh:update_node_normal' in out and 'TriangleMesh' not in out


#%% subdivide_to_quad
def two_quads(device='cpu'):
    node = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]], dtype=F64, device=device)
    return node, torch.tensor([[0, 1, 2, 3], [1, 4, 5, 2]], device=device)


def test_subdivide_keeps_the_area_and_takes_no_elements():
    mesh = QuadMesh(*two_quads()).subdivide_to_quad()
    assert mesh.element.dtype == torch.int64 and mesh.element.shape == (8, 4) and mesh.node.shape == (15, 3)
    mesh.update_element_area_and_normal()
    assert mesh.element_area.sum().item() == pytest.approx(2.0)
    empty = QuadMesh(torch.empty((0, 3), dtype=F64), torch.empty((0, 4), dtype=torch.int64)).subdivide_to_quad()
    assert empty.element.shape == (0, 4)


def test_subdivide_on_cuda_keeps_the_mesh_on_one_device():
    node, element = two_quads('cuda')
    for source in (QuadMesh(node, element), QuadMesh(node, element.cpu())):
        mesh = source.subdivide_to_quad()
        assert mesh.element.device == mesh.node.device == node.device
        assert mesh.sample_point(10).device == node.device
        mesh.update_node_normal()
        assert mesh.node_normal.device == node.device
        assert torch.equal(mesh.element.cpu(), QuadMesh(*two_quads()).subdivide_to_quad().element)


#%% update_element_normal
def test_update_element_normal_on_each_surface_mesh():
    x = warped([[0, 0, 0], [1, 0, 0], [1, 1, .1], [0, 1, 0], [2, 0, 0], [2, 1, 0]])
    mixed = [[0, 1, 2, 3], [1, 4, 2], [4, 5, 2]]
    triangle = TriangleMesh(x, torch.tensor([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]]))
    tri6 = Tri6Mesh()
    tri6.create_from_tri3(triangle)
    not_classified = QuadTriangleMesh()
    not_classified.node, not_classified.element = x, mixed
    for mesh in (not_classified, triangle, tri6, QuadTriangleMesh(x, mixed), QuadMesh(x, torch.tensor([[0, 1, 2, 3], [1, 4, 5, 2]]))):
        mesh.update_element_normal()
        normal = mesh.element_normal
        mesh.update_element_area_and_normal()
        assert torch.equal(normal, mesh.element_normal)
        # eps reaches the normal: at this scale the default eps would shorten it
        node = mesh.node
        mesh.node = node * 1e-8
        mesh.update_element_normal(eps=1e-30)
        torch.testing.assert_close(mesh.element_normal, normal)
        mesh.node = node
        simple_smoother_for_mesh_element_normal(mesh, .5, None, 1)
        torch.testing.assert_close(mesh.element_normal.norm(dim=1), torch.ones(len(mesh.element), dtype=F64))
