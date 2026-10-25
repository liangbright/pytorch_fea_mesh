import pytest

pytestmark = [pytest.mark.processing]

"""Public edge-cycle predicate and its use by the node-curve predicate."""
import unittest

import numpy as np
import torch

from torch_fea_mesh.processing.mesh_processing import is_edge_curve_closed, IsEdgeCurveClosed, is_node_curve_closed
from torch_fea_mesh.mesh.polyline_mesh import PolylineMesh
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh


DTYPES = (torch.float32, torch.float64)


def mesh_with_tail(dtype):
    nodes = torch.tensor([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[-1,-1,0]], dtype=dtype)
    return TriangleMesh(nodes, [[0,1,2],[0,3,4]])


def edge_index_list(mesh, pairs):
    return [mesh.get_edge_idx_from_node_pair(a,b) for a,b in pairs]


class EdgeCurvePredicateTest(unittest.TestCase):
    def test_selected_edge_open_loop_and_loop_with_tail(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                mesh = mesh_with_tail(dtype)
                path = edge_index_list(mesh, [(0,1),(1,2)])
                loop = edge_index_list(mesh, [(0,1),(1,2),(2,0)])
                tail = edge_index_list(mesh, [(0,3),(3,4)])
                self.assertIs(is_edge_curve_closed(mesh, path), False)
                self.assertIs(is_edge_curve_closed(mesh, loop), True)
                self.assertIs(is_edge_curve_closed(mesh, loop+tail), True)

    def test_duplicate_and_missing_edge_index_cannot_create_cycle(self):
        for dtype in DTYPES:
            mesh = mesh_with_tail(dtype)
            a,b,c = edge_index_list(mesh, [(0,1),(1,2),(2,0)])
            for edges in ([], [None], [a,a], [a,b,a,b,None]):
                with self.subTest(dtype=dtype, edges=edges):
                    self.assertIs(is_edge_curve_closed(mesh, edges), False)
            self.assertIs(is_edge_curve_closed(mesh, [a,b,c,a,b,c,None]), True)

    def test_container_order_alias_and_input_preservation(self):
        for dtype in DTYPES:
            mesh = mesh_with_tail(dtype)
            nodes, elements = mesh.node.clone(), mesh.element.clone()
            loop = edge_index_list(mesh, [(0,1),(1,2),(2,0)])
            for make_input in (list, tuple, set, np.array, torch.tensor):
                with self.subTest(dtype=dtype, input_type=make_input.__name__):
                    edges = make_input(loop[::-1])
                    before = list(edges)
                    self.assertIs(IsEdgeCurveClosed(mesh=mesh, edge_curve=edges), True)
                    self.assertEqual([int(i) for i in edges], [int(i) for i in before])
            torch.testing.assert_close(mesh.node, nodes)
            torch.testing.assert_close(mesh.element, elements)

    def test_edge_table_built_when_needed(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                mesh = TriangleMesh(torch.tensor([[0,0,0],[1,0,0],[0,1,0]], dtype=dtype), [[0,1,2]])
                self.assertIsNone(mesh.edge)
                self.assertIs(is_edge_curve_closed(mesh, []), False)
                self.assertIsNone(mesh.edge)
                self.assertIs(is_edge_curve_closed(mesh, [0,1,2]), True)
                self.assertEqual(tuple(mesh.edge.shape), (3,2))

    def test_invalid_index_rejection_without_edge_aliasing(self):
        mesh = mesh_with_tail(torch.float64)
        mesh.build_edge()
        for edges in ([-1], [len(mesh.edge)], [0,-1], [0,len(mesh.edge)]):
            with self.subTest(edges=edges):
                with self.assertRaisesRegex(ValueError, 'invalid edge index in edge_curve'):
                    is_edge_curve_closed(mesh, edges)

    def test_self_edge_does_not_count_as_a_closed_curve(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                mesh = PolylineMesh(torch.tensor([[0,0,0],[1,0,0],[0,1,0]], dtype=dtype),
                                    [[0,0],[0,1],[1,2],[2,0]])
                mesh.build_edge()
                self_edge = next(i for i, pair in enumerate(mesh.edge.tolist()) if pair == [0,0])
                self.assertIs(is_edge_curve_closed(mesh, [self_edge]), False)
                self.assertIs(is_edge_curve_closed(mesh, range(len(mesh.edge))), True)

    def test_node_curve_keyword_and_edge_curve_agree(self):
        curves = ([0,1,2], [0,1,2,0], [0,1,2,0,3,4], [0,1,3,4,0], [])
        for dtype in DTYPES:
            mesh = mesh_with_tail(dtype)
            for nodes in curves:
                with self.subTest(dtype=dtype, node_curve=nodes):
                    edges = edge_index_list(mesh, zip(nodes, nodes[1:]))
                    self.assertIs(is_node_curve_closed(mesh=mesh, node_curve=nodes),
                                  is_edge_curve_closed(mesh, edges))

    def test_cycle_in_one_component_is_enough(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                nodes = torch.zeros((7,3), dtype=dtype)
                pairs = [[0,1],[1,2],[2,0],[2,3],[4,5],[5,6]]
                mesh = PolylineMesh(nodes, pairs)
                self.assertIs(is_edge_curve_closed(mesh, edge_index_list(mesh, pairs)), True)
                forest = [[0,1],[1,2],[2,3],[4,5],[5,6]]
                self.assertIs(is_edge_curve_closed(mesh, edge_index_list(mesh, forest)), False)
