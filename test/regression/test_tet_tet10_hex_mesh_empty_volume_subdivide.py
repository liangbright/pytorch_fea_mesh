"""TetrahedronMesh, Tet10Mesh and HexahedronMesh: empty tables, element volume and subdivide.

A test named test_unchanged_... checks behaviour the fix kept, so it also passes on the files before the fix.
Every other test should fail on those files.
"""
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
import torch

from torch_fea_mesh.mesh.hexahedron_mesh import HexahedronMesh
from torch_fea_mesh.mesh.tet10_mesh import Tet10Mesh
from torch_fea_mesh.mesh.tetrahedron_mesh import TetrahedronMesh
from torch_fea_mesh.processing.polyhedron_mesh_processing import extract_surface_mesh

pytestmark = pytest.mark.geometry

F64 = torch.float64
F32 = torch.float32

UNIT_TET = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
UNIT_HEX = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
# natural coordinates of the C3D8 nodes
CORNER = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
# C3D10 midside node: its two corner nodes
MIDSIDE = {4: (0, 1), 5: (1, 2), 6: (0, 2), 7: (0, 3), 8: (1, 3), 9: (2, 3)}

# class, nodes per element, nodes per face
CLASSES = [(TetrahedronMesh, 4, 3), (Tet10Mesh, 10, 6), (HexahedronMesh, 8, 4)]
CLASS_IDS = ['tet4', 'tet10', 'hex8']

# For one element: its edges in the order of the element-to-edge table, and its faces as stored
SINGLE_ELEMENT_TABLES = {
    TetrahedronMesh: ([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
                      [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
    Tet10Mesh: ([[0, 4], [0, 5], [0, 6], [1, 4], [1, 7], [1, 8], [2, 7], [2, 5], [2, 9], [3, 6], [3, 8], [3, 9]],
                [[0, 2, 1, 5, 7, 4], [0, 1, 3, 4, 8, 6], [0, 3, 2, 6, 9, 5], [1, 2, 3, 7, 9, 8]]),
    HexahedronMesh: ([[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]],
                     [[0, 3, 2, 1], [0, 1, 5, 4], [3, 0, 4, 7], [1, 2, 6, 5], [2, 3, 7, 6], [4, 5, 6, 7]]),
}
# For two elements that share a face: number of edges, faces, boundary faces and boundary nodes
SHARED_FACE_COUNTS = {TetrahedronMesh: (9, 7, 6, 5), Tet10Mesh: (18, 7, 6, 14), HexahedronMesh: (20, 11, 10, 12)}


def make_mesh(cls, node, element, dtype=F64):
    node = torch.tensor(node, dtype=dtype)
    if cls is Tet10Mesh:
        mesh = Tet10Mesh()
        mesh.create_from_tet4(TetrahedronMesh(node, element))
        return mesh
    return cls(node, element)


def single_element(cls, dtype=F64):
    if cls is HexahedronMesh:
        return make_mesh(cls, UNIT_HEX, [list(range(8))], dtype)
    return make_mesh(cls, UNIT_TET, [[0, 1, 2, 3]], dtype)


def two_elements_sharing_a_face(cls, dtype=F64):
    if cls is HexahedronMesh:
        node = UNIT_HEX + [[0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2]]
        return make_mesh(cls, node, [list(range(8)), list(range(4, 12))], dtype)
    return make_mesh(cls, UNIT_TET + [[1, 1, 1]], [[0, 1, 2, 3], [1, 2, 3, 4]], dtype)


def two_disconnected_elements(cls):
    base, n = (UNIT_HEX, 8) if cls is HexahedronMesh else (UNIT_TET, 4)
    return make_mesh(cls, base + [[p[0] + 5, p[1], p[2]] for p in base], [list(range(n)), list(range(n, 2 * n))])


def volume_of(cls, x):
    x = torch.as_tensor(x)
    return cls.cal_element_volume(x, torch.arange(len(x)).reshape(1, -1)).item()


def warped_hex(seed, amplitude=0.2):
    generator = torch.Generator().manual_seed(seed)
    return torch.tensor(UNIT_HEX, dtype=F64) + amplitude * torch.rand(8, 3, generator=generator, dtype=F64)


def straight_tet10(corner=UNIT_TET, dtype=F64):
    # The nodes in element order: 4 corners, then the middle of each edge.
    x = torch.tensor(corner, dtype=dtype)
    return torch.cat([x, torch.stack([(x[i] + x[j]) / 2 for i, j in MIDSIDE.values()])])


def curved_tet10(seed, amplitude=0.15):
    generator = torch.Generator().manual_seed(seed)
    return straight_tet10() + amplitude * (torch.rand(10, 3, generator=generator, dtype=F64) - 0.5)


def exact_volume_of_hex(x):
    # Exact rational arithmetic, independent of the code under test. Write the element as
    # X(r,s,t)=a0+a1*r+a2*s+a3*t+a4*r*s+a5*s*t+a6*r*t+a7*r*s*t and integrate its Jacobian determinant term by term.
    x = [[Fraction(float(c)) for c in p] for p in x]

    def mode(sign):
        return [sum(Fraction(sign(*c)) * x[n][k] for n, c in enumerate(CORNER)) / 8 for k in range(3)]

    def triple(a, b, c):
        return (a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0]))
    a1 = mode(lambda r, s, t: r)
    a2 = mode(lambda r, s, t: s)
    a3 = mode(lambda r, s, t: t)
    a4 = mode(lambda r, s, t: r * s)
    a5 = mode(lambda r, s, t: s * t)
    a6 = mode(lambda r, s, t: r * t)
    return float(abs(8 * triple(a1, a2, a3) + Fraction(8, 3) * (triple(a1, a4, a6) + triple(a4, a2, a5) + triple(a6, a5, a3))))


def exact_volume_of_tet10(x):
    # Exact rational arithmetic, independent of the code under test. Multiply out det(dX/dr) as a polynomial in r,s,t,
    # then integrate each monomial over the reference tetrahedron: r^a s^b t^c gives a!b!c!/(a+b+c+3)!.
    x = [[Fraction(float(c)) for c in p] for p in x]

    def add(p, q, scale=1):
        out = dict(p)
        for key, value in q.items():
            out[key] = out.get(key, Fraction(0)) + scale * value
        return out

    def mul(p, q):
        out = {}
        for (a, b, c), u in p.items():
            for (d, e, f), v in q.items():
                key = (a + d, b + e, c + f)
                out[key] = out.get(key, Fraction(0)) + u * v
        return out

    def scaled(p, scale):
        return {key: scale * value for key, value in p.items()}

    one = {(0, 0, 0): Fraction(1)}
    r, s, t = {(1, 0, 0): Fraction(1)}, {(0, 1, 0): Fraction(1)}, {(0, 0, 1): Fraction(1)}
    L = [add(add(add(one, r, -1), s, -1), t, -1), r, s, t]
    dL = [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def dN(node, k):
        if node < 4:
            return scaled(add(scaled(L[node], 4), one, -1), dL[node][k])
        i, j = MIDSIDE[node]
        return add(scaled(L[i], 4 * dL[j][k]), scaled(L[j], 4 * dL[i][k]))

    J = [[{} for _ in range(3)] for _ in range(3)]
    for d, k, node in itertools.product(range(3), range(3), range(10)):
        J[d][k] = add(J[d][k], dN(node, k), x[node][d])
    det = {}
    for sign, (a, b, c) in ((1, (0, 1, 2)), (1, (1, 2, 0)), (1, (2, 0, 1)), (-1, (2, 1, 0)), (-1, (1, 0, 2)), (-1, (0, 2, 1))):
        det = add(det, mul(mul(J[0][a], J[1][b]), J[2][c]), sign)
    total = sum(v * Fraction(math.factorial(a) * math.factorial(b) * math.factorial(c), math.factorial(a + b + c + 3))
                for (a, b, c), v in det.items())
    return float(abs(total))


# ---------------------------------------------------------------- tables

@pytest.mark.parametrize('cls,width,face_width', CLASSES, ids=CLASS_IDS)
def test_mesh_with_no_elements(cls, width, face_width):
    for n_node, dtype in ((0, F64), (12, F64), (12, F32)):
        node = torch.rand(n_node, 3, dtype=dtype)
        new_mesh = lambda: cls(node, torch.empty((0, width), dtype=torch.int64))

        mesh = new_mesh()
        mesh.build_edge()
        mesh.build_face()
        assert (mesh.edge.shape, mesh.face.shape) == ((0, 2), (0, face_width))
        assert (mesh.edge.dtype, mesh.face.dtype) == (torch.int64, torch.int64)
        assert (mesh.element_to_edge_adj_table, mesh.element_to_face_adj_table) == ([], [])

        mesh = new_mesh()
        mesh.build_element_to_edge_adj_table()
        mesh.build_element_to_face_adj_table()
        assert (mesh.edge.shape, mesh.face.shape) == ((0, 2), (0, face_width))

        mesh = new_mesh()
        mesh.build_face_to_element_adj_table()
        mesh.build_edge_to_edge_adj_table()
        mesh.build_edge_to_element_adj_table()
        mesh.build_node_to_edge_adj_table()
        assert mesh.face_to_element_adj_table == mesh.edge_to_edge_adj_table == mesh.edge_to_element_adj_table == []
        assert mesh.node_to_edge_adj_table == [[] for _ in range(n_node)]

        mesh = new_mesh()
        mesh.build_node_to_node_adj_link()
        mesh.build_element_to_element_adj_link_face()
        assert mesh.node_to_node_adj_link.numel() == mesh.element_to_element_adj_link['face'].numel() == 0

        assert new_mesh().find_boundary_face() == new_mesh().find_boundary_node() == []

        mesh = new_mesh()
        mesh.update_edge_length()
        mesh.update_element_volume()
        assert (mesh.edge_length.shape, mesh.element_volume.shape) == ((0, 1), (0,))
        assert (mesh.edge_length.dtype, mesh.element_volume.dtype, mesh.node.dtype) == (dtype, dtype, dtype)

        surface = extract_surface_mesh(new_mesh())
        assert surface.element is None or len(surface.element) == 0

        # element=[] means the same as an empty tensor
        mesh = cls(node, [])
        mesh.build_edge()
        mesh.build_face()
        assert (mesh.element.shape, mesh.edge.shape, mesh.face.shape) == ((0, width), (0, 2), (0, face_width))


def test_tet10_from_a_tet4_mesh_with_no_elements():
    for n_node, built_edges in ((0, False), (12, False), (12, True)):
        node = torch.rand(n_node, 3, dtype=F64)
        tet4 = TetrahedronMesh(node, torch.empty((0, 4), dtype=torch.int64))
        if built_edges:
            tet4.build_edge()
        mesh = Tet10Mesh()
        mesh.create_from_tet4(tet4)
        assert mesh.element.shape == (0, 10)
        assert mesh.element.dtype == torch.int64
        assert torch.equal(mesh.node, node)
        mesh.build_edge()
        mesh.build_face()
        assert (mesh.edge.shape, mesh.face.shape) == ((0, 2), (0, 6))
        restored = mesh.simplify_to_tet4()
        assert isinstance(restored, TetrahedronMesh)
        assert (restored.node.shape, restored.element.shape) == ((0, 3), (0, 4))


@pytest.mark.parametrize('cls,width,face_width', CLASSES, ids=CLASS_IDS)
def test_unchanged_tables_of_a_mesh_with_elements(cls, width, face_width):
    edges, faces = SINGLE_ELEMENT_TABLES[cls]
    mesh = single_element(cls)
    mesh.build_edge()
    mesh.build_face()
    assert mesh.edge.tolist() == sorted(edges)
    assert mesh.face.tolist() == faces
    assert mesh.edge[mesh.element_to_edge_adj_table[0]].tolist() == edges
    assert len(mesh.element_to_face_adj_table[0]) == len(faces)
    assert sorted(mesh.face[mesh.element_to_face_adj_table[0]].tolist()) == sorted(faces)

    n_edge, n_face, n_boundary_face, n_boundary_node = SHARED_FACE_COUNTS[cls]
    mesh = two_elements_sharing_a_face(cls)
    assert len(mesh.find_boundary_face()) == n_boundary_face
    assert len(mesh.find_boundary_node()) == n_boundary_node
    mesh.build_edge()
    mesh.update_edge_length()
    assert (mesh.edge.shape, mesh.face.shape) == ((n_edge, 2), (n_face, face_width))
    assert mesh.edge_length.shape == (n_edge, 1)
    surface = extract_surface_mesh(mesh)
    assert surface.element.shape == (n_boundary_face, face_width)
    sub_mesh, node_idx_list = mesh.get_sub_mesh([1], return_node_idx_list=True)
    assert type(sub_mesh) is cls
    assert sub_mesh.element.shape == (1, width)
    assert torch.equal(sub_mesh.node, mesh.node[node_idx_list])

    # Disconnected elements share nothing, so the tables are twice as long and every face is on the boundary.
    mesh = two_disconnected_elements(cls)
    mesh.build_edge()
    assert (mesh.edge.shape[0], len(mesh.find_boundary_face())) == (2 * len(edges), 2 * len(faces))


def test_unchanged_tet10_from_tet4_and_back():
    tet4 = TetrahedronMesh(torch.tensor(UNIT_TET + [[1, 1, 1]], dtype=F64), [[0, 1, 2, 3], [1, 2, 3, 4]])
    mesh = Tet10Mesh()
    mesh.create_from_tet4(tet4)
    x = mesh.node[mesh.element]
    assert mesh.node.shape == (5 + 9, 3)
    for mid, (i, j) in MIDSIDE.items():
        torch.testing.assert_close(x[:, mid], (x[:, i] + x[:, j]) / 2)
    restored, node_idx_list = mesh.simplify_to_tet4(return_node_idx_list=True)
    assert torch.equal(restored.node, tet4.node)
    assert torch.equal(restored.element, tet4.element)
    assert node_idx_list.tolist() == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------- subdivide

def test_subdivide_is_not_implemented():
    for cls, width, _ in CLASSES:
        for mesh in (single_element(cls), cls(torch.empty((0, 3), dtype=F64), torch.empty((0, width), dtype=torch.int64))):
            node, element = mesh.node.clone(), mesh.element.clone()
            with pytest.raises(NotImplementedError, match=cls.__name__):
                mesh.subdivide()
            assert torch.equal(mesh.node, node)
            assert torch.equal(mesh.element, element)


# ---------------------------------------------------------------- hex8 volume

def test_hex_warped_volume_is_analytic_for_any_numbering():
    # The base and the top are height fields over the unit square, so the volume is mean(top)-mean(bottom).
    generator = torch.Generator().manual_seed(0)
    cases = [(torch.tensor([1., 1, 2, 1], dtype=F64), torch.zeros(4, dtype=F64)),
             (torch.tensor([1., 1.5, 1, 1], dtype=F64), torch.zeros(4, dtype=F64)),
             (2 + torch.rand(4, generator=generator, dtype=F64), torch.rand(4, generator=generator, dtype=F64))]
    for top, bottom in cases:
        x = torch.tensor(UNIT_HEX, dtype=F64)
        x[:4, 2], x[4:, 2] = bottom, top
        element = torch.tensor([[(j + k) % 4 for j in range(4)] + [4 + (j + k) % 4 for j in range(4)] for k in range(4)])
        expected = (top.mean() - bottom.mean()).expand(4)
        torch.testing.assert_close(HexahedronMesh.cal_element_volume(x, element), expected, rtol=1e-13, atol=0)


def test_hex_volume_is_exact_for_all_48_numberings():
    numbering = []
    for axes in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            numbering.append([CORNER.index(tuple(signs[k] * c[axes[k]] for k in range(3))) for c in CORNER])
    numbering = torch.tensor(numbering)
    assert len(set(map(tuple, numbering.tolist()))) == 48
    for seed in range(3):
        x = warped_hex(seed, amplitude=0.4)
        expected = torch.full((48,), exact_volume_of_hex(x), dtype=F64)
        torch.testing.assert_close(HexahedronMesh.cal_element_volume(x, numbering), expected, rtol=1e-13, atol=0)


def test_hex_warped_shared_face_only_moves_volume_between_elements():
    # Two stacked boxes. Moving the shared nodes takes volume from one and gives it to the other, so the sum stays 2.
    # The upper box's numbering is turned by one corner, so that the cut of a warped face would differ between the two.
    x = torch.tensor(UNIT_HEX + [[0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2]], dtype=F64)
    x[4:8, 2] += torch.tensor([0.1, -0.15, 0.12, -0.05], dtype=F64)
    upper = [4 + (j + 1) % 4 for j in range(4)] + [8 + (j + 1) % 4 for j in range(4)]
    volume = HexahedronMesh.cal_element_volume(x, torch.tensor([list(range(8)), upper]))
    assert volume.sum().item() == pytest.approx(2.0, rel=1e-13)


def test_hex_folded_element_gets_its_net_volume():
    # Node 6 is pushed through the opposite corner, so det(dX/dr) changes sign inside the element.
    x = torch.tensor(UNIT_HEX, dtype=F64)
    x[6] = torch.tensor([-0.5, -0.5, -0.5], dtype=F64)
    assert volume_of(HexahedronMesh, x) == pytest.approx(exact_volume_of_hex(x), rel=1e-13)


def test_unchanged_hex_volume_with_planar_faces():
    for size in ((1, 1, 1), (2, 3, 4), (1e-6, 1e-6, 1e-6), (1e6, 1e6, 1e6), (1, 1e-3, 1e3)):
        x = torch.tensor(UNIT_HEX, dtype=F64) * torch.tensor(size, dtype=F64)
        assert volume_of(HexahedronMesh, x) == pytest.approx(size[0] * size[1] * size[2], rel=1e-14)
    generator = torch.Generator().manual_seed(0)
    for _ in range(3):
        A = torch.rand(3, 3, generator=generator, dtype=F64)
        x = torch.tensor(UNIT_HEX, dtype=F64) @ A.T + torch.rand(3, generator=generator, dtype=F64)
        assert volume_of(HexahedronMesh, x) == pytest.approx(torch.linalg.det(A).abs().item(), rel=1e-11)
    # a truncated square pyramid
    a, b, h = 3, 1, 2
    x = [[-a / 2, -a / 2, 0], [a / 2, -a / 2, 0], [a / 2, a / 2, 0], [-a / 2, a / 2, 0],
         [-b / 2, -b / 2, h], [b / 2, -b / 2, h], [b / 2, b / 2, h], [-b / 2, b / 2, h]]
    assert volume_of(HexahedronMesh, torch.tensor(x, dtype=F64)) == pytest.approx(h * (a * a + a * b + b * b) / 3, rel=1e-14)
    # a hex collapsed to a wedge
    x = [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 1, 0], [0, 0, 1], [1, 0, 1], [0.5, 1, 1], [0.5, 1, 1]]
    assert volume_of(HexahedronMesh, torch.tensor(x, dtype=F64)) == pytest.approx(0.5, rel=1e-14)
    # a flat hex, and a mirrored numbering, whose volume is still positive
    x = torch.tensor(UNIT_HEX, dtype=F64)
    assert HexahedronMesh.cal_element_volume(x, torch.tensor([[4, 5, 6, 7, 0, 1, 2, 3]])).item() == pytest.approx(1.0, rel=1e-14)
    x[:, 2] = 0
    assert volume_of(HexahedronMesh, x) == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------- tet10 volume

def test_tet10_curved_volume_is_analytic():
    # One moved midside node: the volume changes by d.(A1+A2)/3, with A1 and A2 the area vectors of its 2 faces.
    corner = torch.tensor(UNIT_TET, dtype=F64)
    for mid, faces in ((9, ((0, 3, 2), (1, 2, 3))), (4, ((0, 2, 1), (0, 1, 3)))):
        for shift in ((0.3, 0.3, 0.3), (-0.1, 0.2, 0.05)):
            d = torch.tensor(shift, dtype=F64)
            x = straight_tet10()
            x[mid] += d
            area = sum(0.5 * torch.linalg.cross(corner[b] - corner[a], corner[c] - corner[a]) for a, b, c in faces)
            assert volume_of(Tet10Mesh, x) == pytest.approx(1 / 6 + (d * area).sum().item() / 3, rel=1e-13)
    # A tet10 reproduces a quadratic map exactly.
    for a in (0.25, -0.2):
        # x -> x+a*x^2 in one coordinate has det 1+2*a*x, so the volume is 1/6+a/12.
        x = straight_tet10()
        x[:, 0] = x[:, 0] + a * x[:, 0] ** 2
        assert volume_of(Tet10Mesh, x) == pytest.approx(1 / 6 + a / 12, rel=1e-13)
        # In all 3 coordinates the det is cubic, so this needs a rule that is exact up to cubic.
        # Over the unit tet: the integral of 1 = 1/6, x = 1/24, xy = 1/120, xyz = 1/720.
        b = 2 * a
        x = straight_tet10()
        x = x + a * x ** 2
        assert volume_of(Tet10Mesh, x) == pytest.approx(1 / 6 + 3 * b / 24 + 3 * b * b / 120 + b * b * b / 720, rel=1e-13)
    # Integer nodes must not turn the integration points into integers. Only a curved element shows that.
    # Corners at distance 2, and midside node 9 moved by (1,1,1): 8/6 + 4/3.
    x = torch.tensor([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 2, 2]])
    assert volume_of(Tet10Mesh, x) == pytest.approx(8 / 6 + 4 / 3, rel=1e-6)


def test_tet10_volume_is_exact_for_all_24_numberings():
    pairs = {frozenset(pair): node for node, pair in MIDSIDE.items()}
    numbering = torch.tensor([list(c) + [pairs[frozenset((c[a], c[b]))] for a, b in MIDSIDE.values()]
                              for c in itertools.permutations(range(4))])
    for seed in range(3):
        x = curved_tet10(seed)
        expected = torch.full((24,), exact_volume_of_tet10(x), dtype=F64)
        torch.testing.assert_close(Tet10Mesh.cal_element_volume(x, numbering), expected, rtol=1e-13, atol=0)


def test_tet10_inner_midside_node_only_moves_volume_between_elements():
    # 6 tets around the edge from node 0 to node 1, so that edge's midside node touches no boundary face.
    ring = [[math.cos(math.pi * k / 3), math.sin(math.pi * k / 3), 0.0] for k in range(6)]
    mesh = make_mesh(Tet10Mesh, [[0, 0, -1.0], [0, 0, 1.0]] + ring, [[0, 1, 2 + k, 2 + (k + 1) % 6] for k in range(6)])
    before = Tet10Mesh.cal_element_volume(mesh.node, mesh.element)
    x = mesh.node.clone()
    x[mesh.element[0, 4]] += torch.tensor([0.05, -0.1, 0.1], dtype=F64)
    after = Tet10Mesh.cal_element_volume(x, mesh.element)
    assert (after - before).abs().max() > 1e-3
    assert after.sum().item() == pytest.approx(before.sum().item(), rel=1e-13)


def test_tet10_folded_element_gets_its_net_volume():
    # A midside node pushed far through the element, so det(dX/dr) changes sign inside it.
    x = straight_tet10()
    x[9] = torch.tensor([-0.6, -0.6, -0.6], dtype=F64)
    assert volume_of(Tet10Mesh, x) == pytest.approx(exact_volume_of_tet10(x), rel=1e-13)


def test_unchanged_tet4_and_straight_tet10_volume():
    generator = torch.Generator().manual_seed(0)
    for _ in range(3):
        corner = torch.tensor(UNIT_TET, dtype=F64) + 0.3 * torch.rand(4, 3, generator=generator, dtype=F64)
        a, b, c = corner[1] - corner[0], corner[2] - corner[0], corner[3] - corner[0]
        expected = (torch.linalg.cross(a, b) * c).sum().abs().item() / 6
        assert volume_of(TetrahedronMesh, corner) == pytest.approx(expected, rel=1e-15)
        assert volume_of(Tet10Mesh, straight_tet10(corner.tolist())) == pytest.approx(expected, rel=1e-13)
    for scale in (1e-6, 1.0, 1e6):
        assert volume_of(Tet10Mesh, scale * straight_tet10()) == pytest.approx(scale ** 3 / 6, rel=1e-13)
    # a mirrored numbering, whose volume is still positive, and a flat element
    pairs = {frozenset(pair): node for node, pair in MIDSIDE.items()}
    c = (1, 0, 2, 3)
    element = torch.tensor([list(c) + [pairs[frozenset((c[a], c[b]))] for a, b in MIDSIDE.values()]])
    assert Tet10Mesh.cal_element_volume(straight_tet10(), element).item() == pytest.approx(1 / 6, rel=1e-13)
    x = straight_tet10()
    x[:, 2] = 0
    assert volume_of(Tet10Mesh, x) == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------- volume of each class: input and output

def test_unchanged_volume_dtype_shape_and_inputs():
    for cls, width, _ in CLASSES:
        for dtype in (F32, F64):
            mesh = two_elements_sharing_a_face(cls, dtype)
            mesh.update_element_volume()
            assert mesh.element_volume.dtype == dtype
            assert mesh.element_volume.shape == (2,)
            mesh.clear_geometry_derived_data()
            assert mesh.element_volume is None
            empty = cls.cal_element_volume(torch.empty((0, 3), dtype=dtype), torch.empty((0, width), dtype=torch.int64))
            assert (empty.shape, empty.dtype) == ((0,), dtype)
        mesh = single_element(cls)
        node, element = mesh.node.clone(), mesh.element.clone()
        volume = cls.cal_element_volume(mesh.node, mesh.element)
        assert torch.equal(mesh.node, node)
        assert torch.equal(mesh.element, element)
        assert torch.equal(cls.cal_element_volume(mesh.node, mesh.element.numpy()), volume)


def test_unchanged_volume_in_float32_far_from_the_origin():
    assert volume_of(HexahedronMesh, torch.tensor(UNIT_HEX, dtype=F32) + 1000) == pytest.approx(1.0, abs=1e-5)
    assert volume_of(Tet10Mesh, straight_tet10(dtype=F32) + 1000) == pytest.approx(1 / 6, abs=1e-5)


def test_unchanged_volume_with_integer_nodes():
    x = torch.tensor(UNIT_HEX)
    assert volume_of(HexahedronMesh, x) == pytest.approx(1.0, rel=1e-6)
    x = torch.tensor([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]])
    assert volume_of(Tet10Mesh, x) == pytest.approx(8 / 6, rel=1e-6)


def test_unchanged_volume_gradient():
    for cls, x in ((HexahedronMesh, warped_hex(3)), (Tet10Mesh, curved_tet10(2))):
        element = torch.arange(len(x)).reshape(1, -1)
        x = x.clone().requires_grad_(True)
        cls.cal_element_volume(x, element).sum().backward()
        assert x.grad.abs().sum() > 0
        h = 1e-6
        for n, k in itertools.product(range(len(x)), range(3)):
            xp = x.detach().clone(); xp[n, k] += h
            xm = x.detach().clone(); xm[n, k] -= h
            numeric = (cls.cal_element_volume(xp, element) - cls.cal_element_volume(xm, element)).item() / (2 * h)
            assert x.grad[n, k].item() == pytest.approx(numeric, abs=1e-8)
    # the gradient of a flat element is finite
    for cls, x in ((HexahedronMesh, torch.tensor(UNIT_HEX, dtype=F64)), (Tet10Mesh, straight_tet10())):
        x[:, 2] = 0
        x.requires_grad_(True)
        cls.cal_element_volume(x, torch.arange(len(x)).reshape(1, -1)).sum().backward()
        assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no GPU')
def test_unchanged_volume_on_cuda():
    for cls, _, _ in CLASSES:
        mesh = single_element(cls)
        volume = cls.cal_element_volume(mesh.node.to('cuda'), mesh.element.to('cuda'))
        assert volume.device.type == 'cuda'
        assert volume.item() == pytest.approx(1.0 if cls is HexahedronMesh else 1 / 6, rel=1e-14)
