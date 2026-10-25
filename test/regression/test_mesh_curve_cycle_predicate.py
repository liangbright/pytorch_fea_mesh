import pytest

pytestmark = [pytest.mark.processing]

"""An ordered node sequence is closed if its selected edges include a cycle."""
import unittest

import numpy as np
import torch

from torch_fea_mesh.processing.mesh_processing import is_node_curve_closed, IsNodeCurveClosed
from torch_fea_mesh.mesh.polyline_mesh import PolylineMesh
from torch_fea_mesh.mesh.quad_mesh import QuadMesh
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh


DTYPES = (torch.float32, torch.float64)


def triangle_pair(dtype):
    nodes = torch.tensor([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[-1,-1,0]], dtype=dtype)
    return TriangleMesh(nodes, [[0,1,2],[0,3,4]])


class CurveCyclePredicateTest(unittest.TestCase):
    def test_open_closed_and_open_tail_example(self):
        examples = (([0,1,2], False), ([0,1,2,0], True),
                    ([0,1,2,0,3,4], True), ([4,3,0,1,2,0,3], True))
        for dtype in DTYPES:
            mesh = triangle_pair(dtype)
            for curve, expected in examples:
                with self.subTest(dtype=dtype, curve=curve):
                    self.assertIs(is_node_curve_closed(mesh, curve), expected)

    def test_unused_mesh_edge_does_not_close_open_sequence(self):
        for dtype in DTYPES:
            mesh = triangle_pair(dtype)
            for curve in ([0,2,1], [3,0,1,2], [0,3,4]):
                with self.subTest(dtype=dtype, curve=curve):
                    self.assertIs(is_node_curve_closed(mesh, curve), False)
            self.assertIs(is_node_curve_closed(mesh, [0,2,1,0]), True)

    def test_cycle_rejection_for_empty_singleton_self_pair_and_retraced_edge(self):
        cases = ([], [0], [0,0], [0,1,0], [0,1,0,1,0],
                 [0,1,2,1,0], [1,0,3,0,4])
        for dtype in DTYPES:
            mesh = triangle_pair(dtype)
            for curve in cases:
                with self.subTest(dtype=dtype, curve=curve):
                    self.assertIs(is_node_curve_closed(mesh, curve), False)

    def test_missing_edge_cannot_complete_cycle(self):
        cases = (([0,1,3,0], False), ([0,1,3,4,0], False),
                 ([0,1,2,0,1,3,4], True), ([1,3,4,0,3], True))
        for dtype in DTYPES:
            mesh = triangle_pair(dtype)
            self.assertIsNone(mesh.get_edge_idx_from_node_pair(1,3))
            for curve, expected in cases:
                with self.subTest(dtype=dtype, curve=curve):
                    self.assertIs(is_node_curve_closed(mesh, curve), expected)

    def test_quad_boundary_uses_supplied_order(self):
        cases = (([0,1,2,3], False), ([0,1,2,3,0], True),
                 ([0,2,1,3,0], False))
        for dtype in DTYPES:
            mesh = QuadMesh(torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=dtype), [[0,1,2,3]])
            for curve, expected in cases:
                with self.subTest(dtype=dtype, curve=curve):
                    self.assertIs(is_node_curve_closed(mesh, curve), expected)

    def test_polyline_loop_with_open_tail(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                mesh = PolylineMesh(triangle_pair(dtype).node, [[0,1],[1,2],[2,0],[2,3],[3,4]])
                self.assertIs(is_node_curve_closed(mesh, [0,1,2,0,2,3,4]), True)
                self.assertIs(is_node_curve_closed(mesh, [0,2,3,4]), False)

    def test_input_container_alias_and_geometry_preservation(self):
        for dtype in DTYPES:
            mesh = triangle_pair(dtype)
            nodes, elements = mesh.node.clone(), mesh.element.clone()
            for make_input in (list, tuple, np.array, torch.tensor):
                for values, expected in (([0,1,2], False), ([0,1,2,0], True)):
                    with self.subTest(dtype=dtype, container=make_input.__name__, values=values):
                        curve = make_input(values)
                        self.assertIs(IsNodeCurveClosed(mesh, curve), expected)
                        self.assertEqual([int(i) for i in curve], values)
            torch.testing.assert_close(mesh.node, nodes)
            torch.testing.assert_close(mesh.element, elements)

    def test_coincident_coordinate_does_not_imply_topological_closure(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                nodes = torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,0,0]], dtype=dtype)
                chain = PolylineMesh(nodes, [[0,1],[1,2],[2,3]])
                self.assertIs(is_node_curve_closed(chain, [0,1,2,3]), False)
                self.assertIs(is_node_curve_closed(chain, [0,1,2,3,0]), False)
                loop = PolylineMesh(nodes, [[0,1],[1,2],[2,3],[3,0]])
                self.assertIs(is_node_curve_closed(loop, [0,1,2,3,0]), True)
