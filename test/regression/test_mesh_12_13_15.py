"""MESH-12: Tri6Mesh.subdivide_to_tri3 with no elements. MESH-15: subdivision with element as a python list.
MESH-13: find_boundary_face and find_boundary_node work only for TetrahedronMesh, Tet10Mesh and HexahedronMesh.
"""
import pytest
import torch
from torch_fea_mesh.mesh.hexahedron_mesh import HexahedronMesh
from torch_fea_mesh.mesh.polygon_mesh import PolygonMesh
from torch_fea_mesh.mesh.polyhedron_mesh import PolyhedronMesh
from torch_fea_mesh.mesh.quad_mesh import QuadMesh
from torch_fea_mesh.mesh.tet10_mesh import Tet10Mesh
from torch_fea_mesh.mesh.tetrahedron_mesh import TetrahedronMesh
from torch_fea_mesh.mesh.tri6_mesh import Tri6Mesh
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh
from torch_fea_mesh.processing.polyhedron_mesh_processing import extract_surface_mesh

F64 = torch.float64


def square_node():
    return torch.tensor([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], dtype=F64)


def triangles():
    return TriangleMesh(square_node(), [[0, 1, 2], [0, 2, 3]])


def two_quads():
    node = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]], dtype=F64)
    return QuadMesh(node, [[0, 1, 4, 3], [1, 2, 5, 4]])


def tri6():
    node = torch.tensor([[0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=F64)
    return Tri6Mesh(node, [[0, 1, 2, 3, 4, 5]])


# make the mesh, the method, the new class, nodes in a new element, new elements per element, new nodes
SUBDIVISIONS = [(triangles, 'subdivide_to_tri_1to4', TriangleMesh, 3, 4, 5),
                (triangles, 'subdivide_to_tri_1to3', TriangleMesh, 3, 3, 2),
                (triangles, 'subdivide_to_quad', QuadMesh, 4, 3, 7),
                (two_quads, 'subdivide_to_quad', QuadMesh, 4, 4, 9),
                (tri6, 'subdivide_to_tri3', TriangleMesh, 3, 4, 0)]


def as_tensor(element):
    return torch.as_tensor(element, dtype=torch.int64)


def test_subdivide_to_tri3_with_and_without_elements():
    mesh = tri6()
    children = [[0, 3, 5], [3, 4, 5], [1, 4, 3], [2, 5, 4]]
    for element, expected in ((mesh.element, children), (torch.empty((0, 6), dtype=torch.int64), []), ([], [])):
        mesh = Tri6Mesh(tri6().node, element)
        new_mesh = mesh.subdivide_to_tri3()
        assert type(new_mesh) is TriangleMesh
        assert new_mesh.element.shape == (len(expected), 3)
        assert new_mesh.element.tolist() == expected
        assert torch.equal(new_mesh.node, mesh.node) and new_mesh.node is not mesh.node
        assert mesh.subdivide_to_tri3(clone_node=False).node is mesh.node


def test_subdivide_gives_the_same_mesh_from_a_list_as_from_a_tensor():
    for make, method, new_class, width, n_child, n_new_node in SUBDIVISIONS:
        expected = getattr(make(), method)()
        mesh = make()
        element = mesh.element.tolist()
        mesh.element = element
        new_mesh = getattr(mesh, method)()
        assert type(new_mesh) is new_class, method
        assert torch.equal(new_mesh.node, expected.node), method
        assert torch.equal(as_tensor(new_mesh.element), as_tensor(expected.element)), method
        assert as_tensor(new_mesh.element).shape == (n_child * len(element), width), method
        assert new_mesh.node.shape[0] == mesh.node.shape[0] + n_new_node, method
        # area of a flat mesh is kept, so the children cover the parents
        new_mesh.update_element_area_and_normal()
        reference = make()
        reference.update_element_area_and_normal()
        assert new_mesh.element_area.sum().item() == pytest.approx(reference.element_area.sum().item()), method
        # the mesh itself is left alone
        assert mesh.element is element, method


def test_subdivide_rejects_a_mixed_list_or_the_wrong_width(tmp_path):
    # load_from_torch has no width check, so it can leave a mixed list
    node = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], dtype=F64)
    PolygonMesh(node, [[0, 1, 2, 3], [1, 4, 2]]).save_as_torch(str(tmp_path / 'mixed.pt'))
    loaded = QuadMesh()
    loaded.load_from_torch(str(tmp_path / 'mixed.pt'))
    assert isinstance(loaded.element, list)
    with pytest.raises(ValueError, match='must have 4 nodes'):
        loaded.subdivide_to_quad()
    for make, method, *_ in SUBDIVISIONS:
        mesh = make()
        width = mesh.element.shape[1]
        wrong = [list(range(width - 1)), list(range(width))]
        for element in (wrong, torch.tensor([list(range(width - 1))])):
            mesh = make()
            mesh.element = element
            with pytest.raises(ValueError, match='must have '+str(width)+' nodes'):
                getattr(mesh, method)()


def test_subdivide_of_an_uninitialized_mesh_says_so():
    for make, method, *_ in SUBDIVISIONS:
        with pytest.raises(ValueError, match='has no node or no element'):
            getattr(type(make())(), method)()


def test_subdivide_with_an_empty_list():
    for make, method, new_class, width, *_ in SUBDIVISIONS:
        mesh = make()
        old_width = mesh.element.shape[1]
        mesh.element = []
        assert mesh.make_element_tensor().shape == (0, old_width), method
        new_mesh = getattr(mesh, method)()
        assert type(new_mesh) is new_class, method
        assert as_tensor(new_mesh.element).shape == (0, width), method


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_subdivide_with_a_node_on_cuda():
    for make, method, *_ in SUBDIVISIONS:
        expected = getattr(make(), method)()
        for element in (make().element.tolist(), make().element.cuda()):
            mesh = make()
            mesh.node = mesh.node.cuda()
            mesh.element = element
            new_mesh = getattr(mesh, method)()
            assert new_mesh.node.is_cuda, method
            assert torch.equal(new_mesh.node.cpu(), expected.node), method
            assert torch.equal(as_tensor(new_mesh.element).cpu(), as_tensor(expected.element)), method


def tet_node():
    return torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=F64)


def test_find_boundary_is_not_supported_on_a_plain_polyhedron_mesh():
    for element in ([[0, 1, 2, 3], [1, 2, 3, 4]], []):
        mesh = PolyhedronMesh(tet_node(), element)
        for call in (mesh.find_boundary_face, mesh.find_boundary_node, mesh.build_face_to_element_adj_table,
                     lambda: mesh.build_element_to_element_adj_table('face')):
            with pytest.raises(NotImplementedError, match='TetrahedronMesh, Tet10Mesh and HexahedronMesh, not PolyhedronMesh'):
                call()


def test_find_boundary_still_works_on_tet_tet10_and_hex():
    hex_node = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=F64)
    for mesh, n_face, n_node in ((TetrahedronMesh(tet_node(), [[0, 1, 2, 3], [1, 2, 3, 4]]), 6, 5),
                                 (Tet10Mesh(torch.rand(10, 3, dtype=F64), [list(range(10))]), 4, 10),
                                 (HexahedronMesh(hex_node, [list(range(8))]), 6, 8)):
        assert len(mesh.find_boundary_face()) == n_face
        assert len(mesh.find_boundary_node()) == n_node


def test_extract_surface_mesh_still_works_on_a_plain_polyhedron_mesh():
    surface = extract_surface_mesh(PolyhedronMesh(tet_node(), [[0, 1, 2, 3], [1, 2, 3, 4]]))
    assert isinstance(surface, TriangleMesh)
    assert surface.element.shape == (6, 3)
