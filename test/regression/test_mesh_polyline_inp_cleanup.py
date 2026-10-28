import pytest

pytestmark = [pytest.mark.io]

"""Regression coverage for polyline incidence and pre-relabel INP cleanup."""
from contextlib import redirect_stdout
import io
from pathlib import Path
from test.helper.output import output_directory
import unittest

import numpy as np
import torch

from torch_fea_mesh.io.load_mesh_from_inp_file import read_abaqus_inp, clean_data_remove_unused_node
from torch_fea_mesh.mesh.polyline_mesh import PolylineMesh


class PolylineAdjacencyTest(unittest.TestCase):
    def mesh(self, elements):
        return PolylineMesh(torch.zeros((7, 3), dtype=torch.float64), elements)

    def test_open_polyline_segment_pair_without_self_link(self):
        mesh = self.mesh([[0, 1, 2]])
        mesh.build_edge_to_element_adj_table()
        self.assertEqual(mesh.edge.tolist(), [[0, 1], [1, 2]])
        self.assertEqual(mesh.element_to_edge_adj_table, [[0, 1]])
        self.assertEqual(mesh.edge_to_element_adj_table, [[0], [0]])
        mesh.build_element_to_element_adj_link_edge()
        self.assertEqual(mesh.element_to_element_adj_link['edge'].numel(), 0)
        self.assertEqual(mesh.find_boundary_node(), [0, 2, 3, 4, 5, 6])

    def test_two_node_element_has_one_incidence(self):
        mesh = self.mesh([[0, 1], [1, 2], [5, 6]])
        mesh.build_edge_to_element_adj_table()
        self.assertEqual(mesh.element_to_edge_adj_table, [[0], [1], [2]])
        self.assertEqual(mesh.edge_to_element_adj_table, [[0], [1], [2]])

    def test_closed_polyline_has_explicit_closing_segment(self):
        mesh = self.mesh([[0, 1, 2, 0]])
        mesh.build_edge_to_element_adj_table()
        self.assertEqual(mesh.edge.tolist(), [[0, 1], [0, 2], [1, 2]])
        self.assertEqual(mesh.element_to_edge_adj_table, [[0, 2, 1]])
        self.assertEqual(mesh.edge_to_element_adj_table, [[0], [0], [0]])

    def test_ragged_polyline_shares_edge_without_extra_incidence(self):
        mesh = self.mesh([[0, 1, 2], [2, 1, 3, 4], [5, 6]])
        mesh.build_edge_to_element_adj_table()
        self.assertEqual(mesh.edge.tolist(), [[0, 1], [1, 2], [1, 3], [3, 4], [5, 6]])
        self.assertEqual(mesh.element_to_edge_adj_table, [[0, 1], [1, 2, 3], [4]])
        self.assertEqual(mesh.edge_to_element_adj_table, [[0], [0, 1], [1], [1], [2]])
        mesh.build_element_to_element_adj_link_edge()
        self.assertEqual(mesh.element_to_element_adj_link['edge'].tolist(), [[0, 1], [1, 0]])

    def test_empty_and_single_node_element_has_no_segment(self):
        for elements in (torch.empty((0, 2), dtype=torch.int64), [[0]],
                         torch.empty((1, 0), dtype=torch.int64)):
            with self.subTest(elements=elements):
                mesh = self.mesh(elements)
                mesh.build_edge_to_element_adj_table()
                self.assertEqual(tuple(mesh.edge.shape), (0, 2))
                self.assertEqual(mesh.element_to_edge_adj_table, [[] for _ in elements])
                self.assertEqual(mesh.edge_to_element_adj_table, [])


class INPPreRelabelCleanupTest(unittest.TestCase):
    def read(self, body, remove):
        with output_directory('polyline_inp_cleanup') as directory:
            path = Path(directory) / 'polyline_inp_cleanup_mesh.inp'
            path.write_text(body)
            with redirect_stdout(io.StringIO()):
                return read_abaqus_inp(str(path), remove_unused_node=remove)

    def body(self):
        return ('*Node\n40,0,1,0\n10,9,9,9\n30,1,0,0\n20,0,0,0\n'
                '*Element,type=S3\n7,20,30,40\n'
                '*Nset,nset=tip\n40\n*Nset,nset=mixed\n10,30\n'
                '*Nset,nset=unused\n10\n')

    def test_unsorted_noncontiguous_index_list_and_node_set_relabeling(self):
        data = self.read(self.body(), True)
        self.assertEqual(data['node_id'].tolist(), [20, 30, 40])
        self.assertEqual(data['element'], [[0, 1, 2]])
        self.assertEqual(data['node_set'], {'tip': [2], 'mixed': [1], 'unused': []})
        np.testing.assert_array_equal(data['node'], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])

    def test_disabled_cleanup_preserves_unused_node_and_node_set(self):
        data = self.read(self.body(), False)
        self.assertEqual(data['node_id'].tolist(), [10, 20, 30, 40])
        self.assertEqual(data['element'], [[1, 2, 3]])
        self.assertEqual(data['node_set'], {'tip': [3], 'mixed': [0, 2], 'unused': [0]})

    def test_helper_keeps_original_index_list_until_relabel(self):
        nodes = np.arange(12, dtype=np.float64).reshape(4, 3)
        ids = np.array([40, 10, 30, 20])
        elements = np.array([[20, 30, 40]], dtype=object)
        original = elements.copy()
        with redirect_stdout(io.StringIO()):
            kept, kept_ids, connectivity = clean_data_remove_unused_node(nodes, ids, elements)
        self.assertEqual(kept_ids.tolist(), [40, 30, 20])
        np.testing.assert_array_equal(kept, nodes[[0, 2, 3]])
        np.testing.assert_array_equal(connectivity, original)
        np.testing.assert_array_equal(elements, original)

    def test_cleanup_is_noop_when_every_node_is_used(self):
        body = '*Node\n3,0,1,0\n1,0,0,0\n2,1,0,0\n*Element,type=S3\n1,1,2,3\n*Nset,nset=tip\n3\n'
        a, b = self.read(body, False), self.read(body, True)
        for key in ('node', 'node_id', 'element_type'):
            np.testing.assert_array_equal(a[key], b[key])
        for key in ('element', 'element_id', 'node_set', 'element_set'):
            self.assertEqual(a[key], b[key])
