"""Polygon meshes with no elements, and quad_to_tri('auto') on concave and warped quads."""
import math

import pytest
import torch

from torch_fea_mesh.mesh.polygon_mesh import PolygonMesh
from torch_fea_mesh.mesh.quad_mesh import QuadMesh
from torch_fea_mesh.mesh.quad_triangle_mesh import QuadTriangleMesh
from torch_fea_mesh.mesh.tri6_mesh import Tri6Mesh
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh

pytestmark = pytest.mark.geometry

F64 = torch.float64
F32 = torch.float32

GRID = torch.tensor([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0], [3, 0, 0], [3, 1, 0]], dtype=F64)
CLASS_AND_WIDTH = [(PolygonMesh, 3), (PolygonMesh, 4), (PolygonMesh, 8), (TriangleMesh, 3), (QuadMesh, 4),
                   (QuadTriangleMesh, 4), (Tri6Mesh, 6)]
CLASS_AND_WIDTH_ID = [f'{cls.__name__}-{width}' for cls, width in CLASS_AND_WIDTH]

# In both quads x2 is the reflex vertex (interior angle above 180 degrees), and the diagonal
# outside the quad is the shorter one. Their areas are 4.8 and 4.95.
CONCAVE_QUAD_A = [[-5., 0], [0, 0], [-0.2, 0.3], [0, 2]]
CONCAVE_QUAD_B = [[-10., 0], [0, 0], [-0.1, 0.1], [0, 1]]


def empty_element(width):
    return torch.empty((0, width), dtype=torch.int64)


def as3d(point, dtype=F64):
    x = torch.as_tensor(point, dtype=dtype)
    if x.shape[1] == 2:
        x = torch.cat([x, torch.zeros(len(x), 1, dtype=dtype)], dim=1)
    return x


def area_vector_of_polygon(x):
    # Newell's formula, which doesn't depend on how the polygon is cut into triangles.
    return 0.5 * torch.linalg.cross(x, x.roll(-1, 0), dim=1).sum(dim=0)


def area_vector_of_triangle(node, element):
    t = node[element]
    return 0.5 * torch.linalg.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=1)


def assert_split_is_right(node, quad_element, tri_element):
    """Both triangles face the way the quad faces, and they add up to the quad's area."""
    tri_element = torch.as_tensor(tri_element)
    assert tri_element.shape == (2, 3)
    assert sorted(tri_element.reshape(-1).unique().tolist()) == sorted(quad_element)
    total = area_vector_of_polygon(as3d(node)[quad_element].to(F64))
    triangle = area_vector_of_triangle(as3d(node).to(F64), tri_element)
    assert (triangle @ total > 0).all(), f'flipped or zero-area triangle: {triangle.tolist()} vs {total.tolist()}'
    assert torch.allclose(triangle.sum(dim=0), total, rtol=1e-9, atol=1e-9 * total.norm().item())


def split(node, element=None, mode='auto', cls=PolygonMesh, dtype=F64):
    node = as3d(node, dtype) if torch.as_tensor(node).shape[1] == 3 else torch.as_tensor(node, dtype=dtype)
    return cls(node, [[0, 1, 2, 3]] if element is None else element).quad_to_tri(mode)


def split_along_shorter_diagonal(node, element):
    """The rule for a quad with two valid cuts, in the node's own dtype. A tie goes to x1-x3."""
    element = torch.as_tensor(element)
    x = node[element]
    is_x0x2_shorter = ((x[:, 0] - x[:, 2])**2).sum(dim=1) < ((x[:, 1] - x[:, 3])**2).sum(dim=1)
    tri_element = []
    for (id0, id1, id2, id3), flag in zip(element.tolist(), is_x0x2_shorter.tolist()):
        tri_element += [[id0, id2, id3], [id0, id1, id2]] if flag else [[id0, id1, id3], [id1, id2, id3]]
    return torch.tensor(tri_element)


def relabel(point, shift, reverse):
    point = list(point)
    if reverse:
        point = point[::-1]
    return point[shift:] + point[:shift]


RELABELLED = [(name, quad, shift, reverse) for name, quad in [('A', CONCAVE_QUAD_A), ('B', CONCAVE_QUAD_B)]
              for shift in range(4) for reverse in (False, True)]
RELABELLED_ID = [f'{name}-shift{shift}-{"cw" if reverse else "ccw"}' for name, _, shift, reverse in RELABELLED]


def random_rotation(generator):
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=F64, generator=generator))
    return q


def random_concave_quad(count, seed):
    """A triangle with a point inside it as the reflex vertex, at a random position in the quad."""
    generator = torch.Generator().manual_seed(seed)
    quad_list = []
    while len(quad_list) < count:
        a, b, c = torch.randn(3, 2, dtype=F64, generator=generator)
        if abs(((b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0]).item()) < 0.05:
            continue
        w = torch.rand(3, dtype=F64, generator=generator) + 0.05
        reflex = (w[0] * a + w[1] * b + w[2] * c) / w.sum()
        quad = [a, b, c, reflex]
        if torch.rand(1, generator=generator).item() < 0.5:
            quad = quad[::-1]
        shift = int(torch.randint(0, 4, (1,), generator=generator))
        quad_list.append(torch.stack(quad[shift:] + quad[:shift]))
    return torch.stack(quad_list)


def random_convex_quad(count, seed):
    """Four points on an ellipse, in order, with some space between each pair."""
    generator = torch.Generator().manual_seed(seed)
    angle = torch.rand(count, 4, dtype=F64, generator=generator) * (math.pi / 2 - 0.2)
    angle = angle + torch.arange(4, dtype=F64) * math.pi / 2
    stretch = 0.2 + 2 * torch.rand(count, 2, dtype=F64, generator=generator)
    x = torch.stack([angle.cos(), angle.sin()], dim=2) * stretch[:, None, :]
    return x + torch.randn(count, 1, 2, dtype=F64, generator=generator)


def disjoint_mesh(quad):
    node = quad.reshape(-1, quad.shape[-1])
    return node, torch.arange(len(node)).reshape(-1, 4)


def is_convex_along_own_normal(x):
    # x: M x 4 x 3. Convex when seen along the quad's own normal: all four corners turn the same way.
    normal = torch.linalg.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 1], dim=1)
    turn = torch.linalg.cross(x - x.roll(1, 1), x.roll(-1, 1) - x, dim=2)
    return ((turn * normal[:, None]).sum(dim=2) > 0).all(dim=1)


def expected_edge(element_list, quadratic=False):
    table = []
    for element in element_list:
        if quadratic:
            half = len(element) // 2
            element = [v for pair in zip(element[:half], element[half:]) for v in pair]
        table.append([tuple(sorted((element[k], element[(k + 1) % len(element)]))) for k in range(len(element))])
    return sorted({pair for row in table for pair in row}), table


#%% meshes with no elements
@pytest.mark.parametrize('cls,width', CLASS_AND_WIDTH, ids=CLASS_AND_WIDTH_ID)
def test_empty_mesh_edge_table_has_shape_0_2(cls, width):
    mesh = cls(GRID, empty_element(width))
    mesh.build_edge()
    assert mesh.edge.shape == (0, 2)
    assert mesh.edge.dtype == torch.int64
    assert mesh.element_to_edge_adj_table == []


def test_empty_list_edge_table_has_shape_0_2():
    mesh = PolygonMesh(GRID, [[0, 1, 2]])
    mesh.element = []
    mesh.build_edge()
    assert mesh.edge.shape == (0, 2)
    assert mesh.element_to_edge_adj_table == []


@pytest.mark.parametrize('cls,width', CLASS_AND_WIDTH, ids=CLASS_AND_WIDTH_ID)
def test_update_edge_length_on_empty_mesh(cls, width):
    mesh = cls(GRID, empty_element(width))
    mesh.update_edge_length()
    assert mesh.edge_length.numel() == 0
    assert mesh.edge[:, 0].numel() == 0


def test_edge_table_of_0_by_0_element_array():
    mesh = PolygonMesh(GRID, empty_element(0))
    mesh.build_edge()
    assert mesh.edge.shape == (0, 2)
    assert mesh.element_to_edge_adj_table == []


@pytest.mark.parametrize('mode', [0, 1, 'auto'])
@pytest.mark.parametrize('cls,width', [c for c in CLASS_AND_WIDTH if c[1] != 3],
                         ids=[i for c, i in zip(CLASS_AND_WIDTH, CLASS_AND_WIDTH_ID) if c[1] != 3])
def test_quad_to_tri_on_empty_mesh(cls, width, mode):
    new_mesh = cls(GRID, empty_element(width)).quad_to_tri(mode)
    assert type(new_mesh) is PolygonMesh
    assert new_mesh.element.shape == (0, 3)
    assert torch.equal(new_mesh.node, GRID)
    assert new_mesh.node is not GRID


@pytest.mark.parametrize('mode', [0, 1, 'auto'])
def test_quad_to_tri_when_element_is_empty_list(mode):
    mesh = PolygonMesh(GRID, [[0, 1, 2]])
    mesh.element = []
    assert mesh.quad_to_tri(mode).element.shape == (0, 3)


@pytest.mark.parametrize('name', ['is_quad', 'is_tri', 'is_quad_tri'])
def test_type_check_when_element_is_empty_list(name):
    mesh = PolygonMesh(GRID, [[0, 1, 2]])
    mesh.element = []
    assert getattr(mesh, name)() is False


@pytest.mark.parametrize('width', [0, 2, 5, 6, 8])
def test_is_quad_tri_on_empty_tensor_of_other_width(width):
    assert PolygonMesh(GRID, empty_element(width)).is_quad_tri() is False


def test_type_check_on_empty_tensor_follows_the_width():
    for width, expected in [(3, (False, True, True)), (4, (True, False, True)), (6, (False, False, False))]:
        mesh = PolygonMesh(GRID, empty_element(width))
        assert (mesh.is_quad(), mesh.is_tri()) == expected[:2]
        if width != 6:
            assert mesh.is_quad_tri() is expected[2]


@pytest.mark.parametrize('cls,width', CLASS_AND_WIDTH, ids=CLASS_AND_WIDTH_ID)
def test_boundary_query_on_empty_mesh(cls, width):
    for call, expected in [(lambda m: m.find_boundary_node(), []), (lambda m: m.find_boundary_edge(), []),
                           (lambda m: m.find_boundary_node_and_edge(), ([], [])),
                           (lambda m: m.find_boundary_element('edge'), []),
                           (lambda m: m.find_boundary_element('node'), [])]:
        assert call(cls(GRID, empty_element(width))) == expected


#%% edge table of meshes with elements
EDGE_CASE = [
    ('single tri', PolygonMesh, [[0, 1, 2]], False),
    ('single quad', PolygonMesh, [[0, 1, 2, 3]], False),
    ('two disjoint', PolygonMesh, [[0, 1, 2], [3, 4, 5]], False),
    ('shared edge', PolygonMesh, [[0, 1, 2, 3], [1, 4, 5, 2]], False),
    ('ragged list', PolygonMesh, [[0, 1, 2, 3], [1, 4, 5]], False),
    ('triangle class', TriangleMesh, [[0, 1, 2], [0, 2, 3]], False),
    ('quad class', QuadMesh, [[0, 1, 2, 3], [1, 4, 5, 2]], False),
    ('quad tri class', QuadTriangleMesh, [[0, 1, 2, 3], [1, 4, 5]], False),
    ('tri6', Tri6Mesh, [[0, 1, 2, 3, 4, 5]], True),
    ('tri6 two', Tri6Mesh, [[0, 1, 2, 3, 4, 5], [1, 6, 2, 7, 0, 4]], True),
]


@pytest.mark.parametrize('name,cls,element,quadratic', EDGE_CASE, ids=[c[0] for c in EDGE_CASE])
def test_edge_table_on_mesh_with_elements(name, cls, element, quadratic):
    mesh = cls(GRID, element)
    mesh.build_edge()
    edge, table = expected_edge(element, quadratic)
    assert [tuple(e) for e in mesh.edge.tolist()] == edge
    assert mesh.edge.shape == (len(edge), 2)
    assert mesh.edge.dtype == torch.int64
    assert [[edge[i] for i in row] for row in mesh.element_to_edge_adj_table] == table


def test_boundary_of_single_triangle():
    mesh = PolygonMesh(GRID, [[0, 1, 2]])
    assert mesh.find_boundary_node() == [0, 1, 2]
    assert mesh.find_boundary_edge() == [0, 1, 2]
    assert mesh.find_boundary_element('edge') == [0]
    assert mesh.find_boundary_element('node') == [0]


#%% quad_to_tri('auto') on quads where the shorter diagonal is not a valid cut
@pytest.mark.parametrize('name,quad,shift,reverse', RELABELLED, ids=RELABELLED_ID)
def test_auto_concave_quad_in_all_numberings(name, quad, shift, reverse):
    point = relabel(quad, shift, reverse)
    assert_split_is_right(point, [0, 1, 2, 3], split(as3d(point)).element)


@pytest.mark.parametrize('name,quad,shift,reverse', RELABELLED[:8], ids=RELABELLED_ID[:8])
def test_auto_concave_quad_with_two_coordinates(name, quad, shift, reverse):
    point = relabel(quad, shift, reverse)
    assert_split_is_right(point, [0, 1, 2, 3], split(point).element)


@pytest.mark.parametrize('seed', range(5))
def test_auto_concave_quad_rotated_in_3d(seed):
    generator = torch.Generator().manual_seed(seed)
    node = as3d(CONCAVE_QUAD_A) @ random_rotation(generator).T + torch.randn(3, dtype=F64, generator=generator)
    assert_split_is_right(node, [0, 1, 2, 3], split(node).element)


@pytest.mark.parametrize('scale', [1e-30, 1e-12, 1e-6, 1e-3, 1e3, 1e6, 1e12, 1e30])
@pytest.mark.parametrize('dtype', [F32, F64], ids=['f32', 'f64'])
def test_auto_concave_quad_at_any_scale(scale, dtype):
    node = as3d(CONCAVE_QUAD_A, F64) * scale
    assert split(node.to(dtype), dtype=dtype).element.tolist() == [[0, 2, 3], [0, 1, 2]]


@pytest.mark.parametrize('height', [1e-6, 1e-3, 1e-2])
def test_auto_slightly_warped_concave_quad(height):
    generator = torch.Generator().manual_seed(5)
    for _ in range(50):
        node = as3d(CONCAVE_QUAD_A).clone()
        node[:, 2] = height * torch.randn(4, dtype=F64, generator=generator)
        assert split(node).element.tolist() == [[0, 2, 3], [0, 1, 2]]


@pytest.mark.parametrize('dtype', [F32, F64], ids=['f32', 'f64'])
def test_auto_concave_quad_keeps_the_node_dtype(dtype):
    new_mesh = split(as3d(CONCAVE_QUAD_A, dtype), dtype=dtype)
    assert new_mesh.node.dtype == dtype
    assert_split_is_right(new_mesh.node.to(F64), [0, 1, 2, 3], new_mesh.element)


@pytest.mark.parametrize('seed', range(4))
def test_auto_many_random_concave_quads_in_3d(seed):
    generator = torch.Generator().manual_seed(100 + seed)
    node, element = disjoint_mesh(as3d(random_concave_quad(2000, seed).reshape(-1, 2)).reshape(-1, 4, 3))
    node = node @ random_rotation(generator).T
    new_mesh = PolygonMesh(node, element).quad_to_tri('auto')
    assert new_mesh.element.shape == (4000, 3)
    for m in range(2000):
        assert_split_is_right(node, element[m].tolist(), new_mesh.element[2 * m:2 * m + 2])
    # This set has to contain quads where the shorter diagonal is wrong, or it proves nothing.
    assert not torch.equal(split_along_shorter_diagonal(node, element), new_mesh.element)


def test_auto_keeps_each_cut_with_its_element_in_a_mixed_mesh():
    # Triangles and pentagons between the quads must not move a cut to a different quad.
    node, quad_element = [], []
    for shift in (0, 1, 2, 3):
        quad_element.append(list(range(len(node), len(node) + 4)))
        node.extend(relabel(CONCAVE_QUAD_A, shift, False))
    node.extend([[9., 9], [10, 9], [10, 10], [9, 10], [8, 10]])
    # The quads are out of order on purpose: a cut must follow its element, not the sorted order.
    quad_element = [quad_element[1], quad_element[0], quad_element[3], quad_element[2]]
    element = [[16, 17, 18], quad_element[0], [16, 17, 18, 19, 20], quad_element[1], quad_element[2],
               [16, 18, 19], quad_element[3]]
    new = PolygonMesh(as3d(node), element).quad_to_tri('auto').element
    new = new if isinstance(new, list) else new.tolist()
    assert len(new) == 11
    assert new[0] == [16, 17, 18]
    assert new[3] == [16, 17, 18, 19, 20]
    assert new[8] == [16, 18, 19]
    for quad, position in zip(quad_element, (1, 4, 6, 9)):
        assert_split_is_right(as3d(node), quad, new[position:position + 2])


@pytest.mark.parametrize('cls', [PolygonMesh, QuadMesh, QuadTriangleMesh])
def test_auto_concave_quad_in_subclass(cls):
    new_mesh = cls(as3d(CONCAVE_QUAD_A), [[0, 1, 2, 3]]).quad_to_tri('auto')
    assert type(new_mesh) is PolygonMesh
    assert_split_is_right(as3d(CONCAVE_QUAD_A), [0, 1, 2, 3], new_mesh.element)


@pytest.mark.parametrize('point', [
    [[0., 0], [1, 0], [2, 0], [2, 2]],
    [[2., 2], [0, 0], [1, 0], [2, 0]],
    [[1., 0], [2, 0], [2, 2], [0, 0]],
    [[2., 0], [2, 2], [0, 0], [1, 0]],
], ids=['x1 on x0-x2', 'x2 on x1-x3', 'x0 on x1-x3', 'x3 on x0-x2'])
def test_auto_straight_corner_gives_no_zero_area_triangle(point):
    shorter = split_along_shorter_diagonal(as3d(point), [[0, 1, 2, 3]])
    assert (area_vector_of_triangle(as3d(point), shorter).norm(dim=1) == 0).any(), 'the shorter cut must be degenerate here'
    assert_split_is_right(point, [0, 1, 2, 3], split(as3d(point)).element)


def test_auto_concave_quad_on_cuda():
    node = as3d(CONCAVE_QUAD_A).to('cuda')
    new_mesh = PolygonMesh(node, torch.tensor([[0, 1, 2, 3]], device='cuda')).quad_to_tri('auto')
    assert new_mesh.node.device == node.device
    assert new_mesh.element.tolist() == [[0, 2, 3], [0, 1, 2]]


#%% quad_to_tri on quads where the shorter diagonal stays the cut
@pytest.mark.parametrize('dtype', [F32, F64], ids=['f32', 'f64'])
@pytest.mark.parametrize('seed', range(4))
def test_auto_random_convex_quads_take_the_shorter_diagonal(seed, dtype):
    generator = torch.Generator().manual_seed(200 + seed)
    node, element = disjoint_mesh(as3d(random_convex_quad(3000, seed).reshape(-1, 2)).reshape(-1, 4, 3))
    node = (node @ random_rotation(generator).T).to(dtype)
    assert torch.equal(PolygonMesh(node, element).quad_to_tri('auto').element, split_along_shorter_diagonal(node, element))


@pytest.mark.parametrize('dtype', [F32, F64], ids=['f32', 'f64'])
def test_auto_rotated_rectangles_compare_lengths_in_the_node_dtype(dtype):
    # In a rectangle the two diagonals are equal, so only rounding decides the comparison.
    generator = torch.Generator().manual_seed(7)
    corner = torch.tensor([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=F64)
    size = 0.1 + torch.rand(4000, 1, 2, dtype=F64, generator=generator)
    x = corner[None] * torch.cat([size, torch.ones(4000, 1, 1, dtype=F64)], dim=2)
    x = x @ random_rotation(generator).T + torch.randn(4000, 1, 3, dtype=F64, generator=generator)
    node, element = disjoint_mesh(x.to(dtype))
    assert torch.equal(PolygonMesh(node, element).quad_to_tri('auto').element, split_along_shorter_diagonal(node, element))


@pytest.mark.parametrize('height', [0.001, 0.01, 0.1, 0.3, 1.0, 3.0])
def test_auto_warped_quads_convex_along_their_normal_take_the_shorter_diagonal(height):
    generator = torch.Generator().manual_seed(11)
    x = as3d(random_convex_quad(20000, 3).reshape(-1, 2)).reshape(-1, 4, 3)
    x[:, :, 2] = height * torch.randn(20000, 4, dtype=F64, generator=generator)
    x = x[is_convex_along_own_normal(x)]
    assert len(x) > 1000
    node, element = disjoint_mesh(x)
    assert torch.equal(PolygonMesh(node, element).quad_to_tri('auto').element, split_along_shorter_diagonal(node, element))


@pytest.mark.parametrize('long_diagonal', ['x0-x2', 'x1-x3'])
def test_auto_roof_folded_more_than_90_degrees_takes_the_shorter_diagonal(long_diagonal):
    # A diamond seen from above. The normals along the x0-x2 ridge are more than 90 degrees apart.
    half_width = 0.5 if long_diagonal == 'x0-x2' else 5.0
    node = torch.tensor([[0., 0, 0], [1, -half_width, half_width * 2], [2, 0, 0], [1, half_width, half_width * 2]], dtype=F64)
    assert torch.equal(split(node).element, split_along_shorter_diagonal(node, [[0, 1, 2, 3]]))


def test_auto_concave_quad_where_shorter_diagonal_is_inside():
    point = [[0., 0], [4, 0], [0.5, 0.5], [0, 4]]
    new_mesh = split(as3d(point))
    assert torch.equal(new_mesh.element, split_along_shorter_diagonal(as3d(point), [[0, 1, 2, 3]]))
    assert_split_is_right(point, [0, 1, 2, 3], new_mesh.element)


NO_VALID_CUT = {
    'square, a tie': [[0., 0], [1, 0], [1, 1], [0, 1]],
    'bowtie': [[0., 0], [1, 1], [1, 0], [0, 1]],
    'long bowtie': [[0., 0], [5, 1], [5, 0], [0, 1]],
    'all nodes equal': [[1., 1]] * 4,
    'x0 equals x1': [[0., 0], [0, 0], [1, 0], [0, 1]],
    'x0 equals x2': [[0., 0], [1, 0], [0, 0], [0, 1]],
    'all on a line': [[0., 0], [1, 0], [2, 0], [3, 0]],
}


@pytest.mark.parametrize('name', NO_VALID_CUT)
def test_auto_quad_with_two_or_no_valid_cuts_takes_the_shorter_diagonal(name):
    node = as3d(NO_VALID_CUT[name])
    assert torch.equal(split(node).element, split_along_shorter_diagonal(node, [[0, 1, 2, 3]]))


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_auto_non_finite_node_does_not_raise(value):
    node = as3d(CONCAVE_QUAD_A).clone()
    node[1, 0] = value
    assert torch.equal(split(node).element, split_along_shorter_diagonal(node, [[0, 1, 2, 3]]))


@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('point', [CONCAVE_QUAD_A, [[0., 0], [1, 0], [1, 1], [0, 1]]], ids=['concave', 'square'])
def test_explicit_mode_takes_the_diagonal_it_names(mode, point):
    # A caller who names a diagonal gets that diagonal, even if it is outside the quad.
    assert split(as3d(point), mode=mode).element.tolist() == ([[0, 2, 3], [0, 1, 2]] if mode == 0 else [[0, 1, 3], [1, 2, 3]])


def test_invalid_mode_raises_on_a_quad():
    with pytest.raises(ValueError, match='mode is invalid'):
        PolygonMesh(GRID, [[0, 1, 2, 3]]).quad_to_tri('bad')


def test_triangle_tensor_is_returned_as_a_copy_for_any_mode():
    element = torch.tensor([[0, 1, 2], [0, 2, 3]])
    for mode in [0, 1, 'auto', 'bad']:
        new_mesh = PolygonMesh(GRID, element).quad_to_tri(mode)
        assert torch.equal(new_mesh.element, element)
        assert type(new_mesh) is PolygonMesh


def test_auto_passes_elements_with_other_node_counts_through():
    node = torch.rand(8, 3, dtype=F64)
    for element in ([[0, 1, 2, 3, 4]], [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]):
        assert PolygonMesh(node, element).quad_to_tri('auto').element.tolist() == element
    new = PolygonMesh(node, [[0, 1, 2], [0, 1, 2, 3, 4]]).quad_to_tri('auto').element
    assert new == [[0, 1, 2], [0, 1, 2, 3, 4]]


@pytest.mark.parametrize('make', [
    lambda e: e, lambda e: e.tolist(), lambda e: e.numpy(), lambda e: list(e),
], ids=['tensor', 'list of lists', 'ndarray', 'list of tensors'])
def test_auto_gives_the_same_split_for_any_element_container(make):
    element = torch.tensor([[0, 1, 2, 3], [1, 4, 5, 2], [4, 6, 7, 5]])
    expected = split_along_shorter_diagonal(GRID, element)
    assert torch.equal(PolygonMesh(GRID, make(element)).quad_to_tri('auto').element, expected)
    mesh = PolygonMesh(GRID, element)
    mesh.element = make(element)
    assert torch.equal(mesh.quad_to_tri('auto').element, expected)


def test_auto_leaves_the_mesh_alone_and_honours_clone_node():
    mesh = PolygonMesh(GRID.clone(), torch.tensor([[0, 1, 2, 3], [1, 4, 5, 2]]))
    node_before, element_before = mesh.node.clone(), mesh.element.clone()
    cloned, shared = mesh.quad_to_tri('auto'), mesh.quad_to_tri('auto', clone_node=False)
    assert cloned.node is not mesh.node and torch.equal(cloned.node, node_before)
    assert shared.node is mesh.node
    assert torch.equal(mesh.node, node_before) and torch.equal(mesh.element, element_before)


def test_auto_with_nodes_that_require_grad():
    node = as3d(CONCAVE_QUAD_A).requires_grad_(True)
    new_mesh = PolygonMesh(node, [[0, 1, 2, 3]]).quad_to_tri('auto')
    assert new_mesh.node.requires_grad
    new_mesh.node.sum().backward()
    assert torch.equal(node.grad, torch.ones_like(node))


def test_invalid_mode_with_no_quads_is_ignored():
    node = torch.rand(6, 3, dtype=F64)
    element = [[0, 1, 2, 3, 4]]
    assert PolygonMesh(node, element).quad_to_tri('bad').element.tolist() == element
