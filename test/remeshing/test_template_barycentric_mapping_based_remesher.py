"""Analytical, geometric, solver, and stopping regressions (stdlib unittest)."""
from copy import deepcopy
import unittest
from importlib.util import find_spec

if find_spec("scipy") is None:
    raise unittest.SkipTest("Install torch-fea-mesh[remeshing] for SciPy")
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

import torch_fea_mesh.processing.template_barycentric_mapping_based_remesher as remesh
from .template_barycentric_support import assert_same, mesh_snapshot


def policy(dtype=np.float64):
    p = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=dtype)
    return remesh._make_numerical_policy(p, np.array([[0, 1, 2]]), p[:, :2], np.dtype(dtype), 1.0)


def star(center=(0.4, 0.6, 0.2), dtype=torch.float64):
    node = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.], list(center)], dtype=dtype)
    return remesh.PolygonMesh(node=node, element=[[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])


SEGMENTS = [[0, 1], [1, 2], [2, 3], [3, 0]]


class MathematicalTests(unittest.TestCase):
    def test_original_face_metric_and_svd(self):
        uv = np.array([[0., 0.], [1., 0.], [0., 1.]])
        face = np.array([[0, 1, 2]])
        for sx, sy, expected in ((1, 1, 1), (3, 3, 3), (3, 1, np.sqrt(5))):
            source = np.c_[uv*np.array([sx, sy]), np.zeros(3)]
            actual = remesh._compute_local_stretch_face(source, face, uv, policy())
            self.assertAlmostEqual(actual[0], expected)
            jacobian = (source[1:]-source[0]).T
            sigma = np.linalg.svd(jacobian, compute_uv=False)
            self.assertAlmostEqual(actual[0], np.sqrt(np.mean(sigma*sigma)))

    def test_area_weighted_rms_without_smoothing(self):
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        areas, stretches = np.array([1., 3.]), np.array([2., 4.])
        actual = remesh._compute_local_stretch_node(faces, areas, stretches, 4)
        np.testing.assert_allclose(actual, [np.sqrt(13), 2, np.sqrt(13), 4])
        self.assertAlmostEqual(remesh._compute_average_local_stretch(areas, stretches), np.sqrt(13))
        np.testing.assert_allclose(remesh._compute_local_stretch_node(faces, areas*1e-20, stretches, 4), actual)

    def test_averaged_barycentric_center_and_opposite_vertex(self):
        for ring in (np.array([[2., 0.], [0., 1.], [-1., 0.], [0., -3.]]),
                     np.array([[2., 0.], [-1., 1.], [-1., -2.]])):
            source = np.vstack(([0., 0., 0.], np.c_[ring, np.zeros(len(ring))]))
            weights = remesh._averaged_barycentric_compute_vertex_weights(source, 0, np.arange(1, len(source)), policy())
            self.assertTrue(np.all(weights > 0))
            self.assertAlmostEqual(weights.sum(), 1)
            np.testing.assert_allclose(weights @ ring, [0, 0], atol=1e-14)
            if len(ring) == 3:
                expected = np.linalg.solve(np.vstack((ring.T, np.ones(3))), [0., 0., 1.])
                np.testing.assert_allclose(weights, expected, atol=1e-14)

    def test_mean_value_uses_actual_incident_faces(self):
        # Edge 0--1 has common neighbor 4 that is not in either incident face.
        faces = np.array([[0, 1, 2], [1, 0, 3], [0, 2, 4], [1, 4, 2]])
        mesh = remesh.PolygonMesh(node=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., -2., 0.], [2., 2., 1.]]), element=faces)
        mesh.build_edge(); mesh.build_node_to_node_adj_table(); mesh.build_edge_to_element_adj_table()
        self.assertIn(4, set(mesh.node_to_node_adj_table[0]) & set(mesh.node_to_node_adj_table[1]))
        edges = mesh.edge.numpy()
        ei = np.flatnonzero(np.all(edges == [0, 1], axis=1))[0]
        opposite = remesh._get_edge_opposite_vertices(faces, edges, mesh.edge_to_element_adj_table, ei)
        np.testing.assert_array_equal(opposite, [2, 3])
        weight = remesh._mean_value_compute_edge_weight(mesh.node.numpy(), 0, 1, opposite, policy())
        self.assertAlmostEqual(weight, 2.0)
        reverse = remesh._mean_value_compute_edge_weight(mesh.node.numpy(), 1, 0, opposite, policy())
        self.assertNotAlmostEqual(weight, reverse)
        with self.assertRaisesRegex(ValueError, "two incident"):
            remesh._mean_value_compute_edge_weight(mesh.node.numpy(), 0, 1, np.array([2]), policy())

    def test_cumulative_destination_weights(self):
        w = sp.csr_matrix(np.array([[0.2, 0.3, 0.5], [0.4, 0.5, 0.1]]))
        logs = w.copy(); logs.data = np.log(logs.data)
        s1, s2 = np.array([1., 4., 9.]), np.array([16., 1., 4.])
        l1, w1 = remesh._update_log_weight_by_stretch(logs, s1, 0.5)
        _, w2 = remesh._update_log_weight_by_stretch(l1, s2, 0.5)
        expected = w.toarray()/np.sqrt(s1*s2)
        expected /= expected.sum(axis=1)[:, None]
        np.testing.assert_allclose(w2.toarray(), expected)
        np.testing.assert_allclose(np.exp(logs.data), w.data)

    def test_solver_choices_same_system_and_precision(self):
        for dtype in (np.float64, np.float32):
            a = sp.csr_matrix(np.array([[1., -0.2, 0.], [-0.3, 1., -0.2], [-0.1, -0.4, 1.]], dtype=dtype))
            truth = np.array([[0.3, 0.4], [-0.2, 0.5], [0.7, -0.1]], dtype=dtype)
            b = a @ truth
            for solver in ("sparse_lu", "gmres", "bicgstab"):
                opts = remesh._validate_solver_options(solver, None, policy(dtype))
                actual, record = remesh._solve_parameterization_system(a, b, np.zeros_like(b), solver, opts, policy(dtype))
                self.assertEqual(actual.dtype, dtype)
                np.testing.assert_allclose(actual, truth, atol=1e-6 if dtype == np.float32 else 1e-12)
                self.assertTrue(all(c["residual_l2"] <= c["residual_limit"] for c in record["coordinates"]))

    def test_three_rhs_annular_field_solver(self):
        a = sp.csr_matrix([[1., -.2, 0.], [-.3, 1., -.2], [-.1, -.4, 1.]])
        truth = np.array([[.3, .4, .7], [-.2, .5, .4], [.7, -.1, .2]])
        for solver in ("sparse_lu", "gmres", "bicgstab"):
            result, record = remesh._solve_parameterization_system(a, a @ truth, np.zeros_like(truth), solver,
                remesh._validate_solver_options(solver, None, policy()), policy())
            np.testing.assert_allclose(result, truth, atol=1e-11)
            self.assertEqual(len(record["coordinates"]), 3)

    def test_solver_failure_and_invalid_options(self):
        with self.assertRaises(ValueError):
            remesh._validate_solver_options("cg", {}, policy())
        with self.assertRaises(ValueError):
            remesh._validate_solver_options("sparse_lu", {"restart": 10}, policy())
        with self.assertRaises(ValueError):
            remesh._validate_solver_options("gmres", {"maxiter": 0}, policy())
        a = sp.csr_matrix([[1., 1.], [2., 2.]])
        with self.assertRaisesRegex(ValueError, "factorization") as caught:
            remesh._solve_parameterization_system(a, np.ones((2, 2)), np.zeros((2, 2)), "sparse_lu", {"rtol": 1e-10, "permc_spec": "COLAMD"}, policy())
        self.assertIn("solve", caught.exception.diagnostics)
        with patch.object(remesh, "gmres", return_value=(np.zeros(2), 1)):
            with self.assertRaisesRegex(ValueError, "true-residual"):
                remesh._solve_parameterization_system(sp.eye(2, format="csr"), np.ones((2, 2)), np.zeros((2, 2)), "gmres",
                    remesh._validate_solver_options("gmres", {"preconditioner": "none"}, policy()), policy())

    def test_bicgstab_counts_converging_half_step(self):
        a = sp.eye(2, format="csr")
        b = np.ones((2, 2))
        _, record = remesh._solve_parameterization_system(a, b, np.zeros_like(b), "bicgstab",
                remesh._validate_solver_options("bicgstab", {"preconditioner": "none"}, policy()), policy())
        self.assertEqual([c["iterations"] for c in record["coordinates"]], [1, 1])


class GeometryTests(unittest.TestCase):
    def test_box_pair_search_matches_exhaustive(self):
        rng = np.random.default_rng(37)
        lower = rng.uniform(-1, 1, (55, 2))
        upper = lower+rng.uniform(.01, .4, (55, 2))
        expected = {(i, j) for i in range(55) for j in range(i+1, 55)
                    if np.all(lower[i] <= upper[j]) and np.all(lower[j] <= upper[i])}
        actual = remesh._find_overlapping_box_pairs(lower, upper, 0)
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(set(map(tuple, actual)), expected)

    def test_barycentric_and_degenerate_vertical_edges(self):
        tri = np.array([[0., 0.], [0., 2.], [1., 0.]])
        point, w, distance, status = remesh._closest_point_on_triangle(np.array([0.2, 0.4]), tri, policy())
        np.testing.assert_allclose(w, [0.6, 0.2, 0.2]); self.assertEqual(distance, 0)
        for scale in (1e-10, 1., 1e10):
            w = remesh._compute_barycentric_coordinates(np.array([0.2, 0.4])*scale, tri*scale, policy())
            np.testing.assert_allclose(w, [0.6, 0.2, 0.2], atol=1e-14)
        collinear = np.array([[0., 0.], [0., 2.], [0., 1.]])
        point, w, distance, status = remesh._closest_point_on_triangle(np.array([1., 0.5]), collinear, policy())
        np.testing.assert_allclose(point, [0, .5]); np.testing.assert_allclose(w @ collinear, point)
        self.assertEqual(status, "degenerate"); self.assertAlmostEqual(distance, 1)

    def test_locator_exhaustive_and_misleading_centroid(self):
        # The closest triangle has a centroid much farther away than triangle 1.
        triangles = [np.array([[0., 0.], [100., 0.], [0., 0.1]]), np.array([[0., 1.], [1., 1.], [0., 2.]])]
        rng = np.random.default_rng(2026)
        for _ in range(30):
            anchor = rng.uniform(-3, 3, 2)
            triangles.append(anchor + np.array([[0., 0.], [.2, 0.], [0., .3]]))
        nodes = np.concatenate(triangles); faces = np.arange(len(nodes)).reshape(-1, 3)
        locator = remesh._build_source_triangle_locator(nodes, faces, policy())
        for query in [np.array([0.05, -0.01]), *rng.uniform(-4, 4, (25, 2))]:
            found = remesh._find_source_triangle_for_template_point(query, locator, policy())
            all_distances = [remesh._closest_point_on_triangle(query, t, policy())[2] for t in triangles]
            self.assertAlmostEqual(found[3], min(all_distances), places=13)
            self.assertEqual(found[0], int(np.argmin(all_distances)))
        self.assertEqual(remesh._find_source_triangle_for_template_point(np.array([.05, -.01]), locator, policy())[0], 0)

    def test_outside_domain_errors_and_roundoff_projection(self):
        tri = np.array([[0., 0.], [1., 0.], [0., 1.]])
        face = np.array([[0, 1, 2]])
        p = np.c_[tri, np.zeros(3)]
        _, d = remesh._map_template_to_source_surface(p, face, tri, np.array([[.3, -1e-16]]), np.array([], dtype=int), np.empty((0, 3)), policy())
        self.assertEqual(d["query_status"], ["roundoff_projection"])
        self.assertTrue(np.all(d["barycentric_weight"] >= 0))
        with self.assertRaisesRegex(ValueError, "outside"):
            remesh._map_template_to_source_surface(p, face, tri, np.array([[.3, -.01]]), np.array([], dtype=int), np.empty((0, 3)), policy())

    def test_uv_overlap_including_shared_vertex_and_contact(self):
        cases = [(np.array([[0., 0.], [2., 0.], [0., 2.], [1., .2], [.2, 1.]]), [[0, 1, 2], [0, 3, 4]]),
                 (np.array([[0., 0.], [2., 0.], [0., 2.], [1., 0.], [2., -1.], [2., 0.]]), [[0, 1, 2], [3, 4, 5]])]
        for uv, faces in cases:
            with self.assertRaisesRegex(ValueError, "overlap|contacts"):
                remesh._validate_uv_mesh(uv, np.array(faces), None, policy())
        mesh = star(center=(.5, .5, 0))
        remesh._validate_uv_mesh(mesh.node.numpy()[:, :2], mesh.element.numpy(), None, policy())
        remesh._validate_uv_mesh(mesh.node.numpy()[:, :2]*[-1, 1], mesh.element.numpy(), None, policy())

    def test_flips_degeneracy_and_float32_sliver(self):
        for uv in (np.array([[0., 0.], [1., 0.], [0., 0.]]),
                   np.array([[0., 0.], [1., 0.], [.5, 1e-10]], dtype=np.float32)):
            with self.assertRaisesRegex(ValueError, "Degenerate"):
                remesh._validate_uv_mesh(uv, np.array([[0, 1, 2]]), None, policy(uv.dtype))
        mesh = star(center=(2, 2, 0))
        with self.assertRaisesRegex(ValueError, "Flipped"):
            remesh._validate_uv_mesh(mesh.node.numpy()[:, :2], mesh.element.numpy(), None, policy())

    def test_boundary_intersection_and_bowtie(self):
        p = np.array([[0., 0.], [1., 1.], [1., 0.], [0., 1.]])
        with self.assertRaisesRegex(ValueError, "crossing"):
            remesh._check_planar_edges(p, np.array([[0, 1], [1, 2], [2, 3], [3, 0]]), policy()["uv_distance_tolerance"], "test")
        mesh = remesh.PolygonMesh(node=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.]]), element=[[0, 1, 2], [0, 3, 4]])
        with self.assertRaisesRegex(ValueError, "vertex fan"):
            remesh._build_mesh_topology(mesh)


class PublicContractTests(unittest.TestCase):
    def test_initialization_name_rejects_removed_option(self):
        source, template = star(), star(center=(.5, .5, 0.))
        with self.assertRaisesRegex(ValueError, "averaged_barycentric, mean_value, annular, or untangle"):
            remesh.template_barycentric_mapping_based_remesh(
                source, SEGMENTS, template, SEGMENTS, initialization="shape_preserving")
        for method in ("averaged_barycentric", "mean_value", "untangle"):
            with self.subTest(initialization=method):
                output = remesh.template_barycentric_mapping_based_remesh(
                    source, SEGMENTS, template, SEGMENTS, initialization=method,
                    diffusion_coefficient_of_mesh_parameterization=0.)
                self.assertEqual(output.mesh_data["template_barycentric_mapping_based_remesh"]
                                 ["initialization_details"]["method"], method)

    def test_input_immutability_and_metadata(self):
        source, template = star(), star(center=(.5, .5, 0))
        source.build_edge(); template.build_node_to_node_adj_table()
        template.name = "kept"; template.node_set = {"arc": [0, 1]}
        template.mapping_from_node_name_to_index = {"corner": 0}
        template.mesh_data["original"] = {"value": [1, 2]}
        snapshots = [mesh_snapshot(source), mesh_snapshot(template)]
        output = remesh.template_barycentric_mapping_based_remesh(source, SEGMENTS, template, SEGMENTS)
        assert_same(source.__dict__, snapshots[0]); assert_same(template.__dict__, snapshots[1])
        self.assertEqual(output.name, "kept"); self.assertIsNone(output.edge)
        self.assertEqual(output.node_set, template.node_set)
        output.mesh_data["original"]["value"].append(3)
        self.assertEqual(template.mesh_data["original"]["value"], [1, 2])

    def test_no_free_nodes(self):
        node = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        mesh = remesh.PolygonMesh(node=node, element=[[0, 1, 2]])
        with patch.object(remesh, "_solve_parameterization_system", side_effect=AssertionError("must not solve")):
            output = remesh.template_barycentric_mapping_based_remesh(mesh, [[0, 1, 2, 0]], mesh, [[0, 1, 2, 0]])
        d = output.mesh_data["template_barycentric_mapping_based_remesh"]
        self.assertEqual(d["termination_reason"], "no_free_nodes")
        self.assertEqual(d["attempted_iterations"], 0)

    def test_unequal_boundary_sampling_exact_curve(self):
        # Rectangle with a subdivided bottom boundary versus an unsubdivided one.
        source = remesh.PolygonMesh(node=np.array([[0., 0., 0.], [.25, 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.], [.5, .5, .2]]),
            element=[[0, 1, 5], [1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 0, 5]])
        template = star(center=(.5, .5, 0))
        output = remesh.template_barycentric_mapping_based_remesh(source, [[0, 1, 2], [2, 3], [3, 4], [4, 0]], template, SEGMENTS, 0)
        np.testing.assert_array_equal(output.node[:4], source.node[[0, 2, 3, 4]])
        constraints = output.mesh_data["template_barycentric_mapping_based_remesh"]["constraints"]
        np.testing.assert_allclose(constraints["source_uv"][1], [-.25/np.sqrt(2), -.5/np.sqrt(2)])

    def test_conflicting_repeated_constraint(self):
        source, template = star(), star(center=(.5, .5, 0))
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            remesh.template_barycentric_mapping_based_remesh(source, SEGMENTS+[[0]], template, SEGMENTS+[[1]])

    def test_exact_landmark_overrides_agreeing_rounded_interpolation(self):
        source = np.array([[0., 0., 0.], [.3, 0., 0.], [1., 0., 0.]])
        template = np.array([[0., 0.], [.3+1e-15, 0.], [1., 0.]])
        for paths in ([[0, 1, 2], [1]], [[1], [0, 1, 2]]):
            constraints = remesh._find_boundary_constraints(source, template, paths, paths, policy(), source)
            np.testing.assert_array_equal(constraints["template_position_raw"][1], source[1])
            np.testing.assert_array_equal(constraints["source_uv"][1], template[1])

    def test_planar_affine_reproduction_and_nearly_convex_boundary(self):
        outline = np.array([[0., 0.], [1., 0.], [1., 1.], [.5, .98], [0., 1.], [.5, .5]])
        mesh = remesh.PolygonMesh(node=np.c_[outline, np.zeros(6)], element=[[0, 1, 5], [1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 0, 5]])
        segments = [[0, 1, 2, 3, 4, 0]]
        for initializer in ("averaged_barycentric", "mean_value"):
            output = remesh.template_barycentric_mapping_based_remesh(mesh, segments, mesh, segments, 0, initialization=initializer)
            np.testing.assert_allclose(output.node.numpy(), mesh.node.numpy(), atol=1e-13)

    def test_dtype_independent_scales_and_translation(self):
        for dtype in (torch.float64, torch.float32):
            base_source, base_template = star(), star(center=(.5, .5, 0))
            reference = remesh.template_barycentric_mapping_based_remesh(base_source, SEGMENTS, base_template, SEGMENTS, 0, computation_dtype=dtype).node.double().numpy()
            for source_scale, uv_scale in ((1e-6, 1e6), (1e6, 1e-6), (1., 1.)):
                source, template = star(), star(center=(.5, .5, 0))
                translation = np.array([100., -200., 300.])*source_scale
                source.node = source.node*source_scale + torch.tensor(translation)
                template.node[:, :2] = template.node[:, :2]*uv_scale + torch.tensor([50., -60.])*uv_scale
                output = remesh.template_barycentric_mapping_based_remesh(source, SEGMENTS, template, SEGMENTS, 0, computation_dtype=dtype)
                normalized = (output.node.double().numpy()-translation)/source_scale
                np.testing.assert_allclose(normalized, reference, atol=4e-5 if dtype == torch.float32 else 1e-11, rtol=1e-5 if dtype == torch.float32 else 1e-11)
                self.assertEqual(output.node.dtype, dtype)

    def test_invalid_inputs(self):
        source, template = star(), star(center=(.5, .5, 0))
        for kwargs in ({"max_iteration_of_mesh_parameterization": -1}, {"diffusion_coefficient_of_mesh_parameterization": 2},
                       {"tolerance_factor": 0}, {"computation_dtype": torch.float16}, {"initialization": "bad"}):
            with self.assertRaises(ValueError):
                remesh.template_barycentric_mapping_based_remesh(source, SEGMENTS, template, SEGMENTS, **kwargs)
        template.node[4, 2] = 1e-12
        with self.assertRaisesRegex(ValueError, "x, y, 0"):
            remesh.template_barycentric_mapping_based_remesh(source, SEGMENTS, template, SEGMENTS)

    def test_unrepresentable_float32_geometry_is_not_silently_promoted(self):
        source, template = star(), star(center=(.5, .5, 0))
        source.node[4] = torch.tensor([.5, 1e-9, 0.], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "unresolved source"):
            remesh.template_barycentric_mapping_based_remesh(source, SEGMENTS, template, SEGMENTS, computation_dtype=torch.float32)


class AnnularTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from .template_barycentric_support import load_cases
        cls.cases = {case["case_id"]: case for case in load_cases()}

    def run_annular(self, case, cap=2):
        from .template_barycentric_support import verify_result
        output = remesh.template_barycentric_mapping_based_remesh(case["source"], case["source_segments"],
            case["template"], case["template_segments"], max_iteration_of_mesh_parameterization=cap, initialization="annular",
            diffusion_coefficient_of_mesh_parameterization=0.5)
        verify_result(case, output)
        return output

    def test_tube_has_valid_seed_and_improving_refinement(self):
        case = self.cases["test_tube"]
        snapshots = [mesh_snapshot(case[role]) for role in ("source", "template")]
        output = self.run_annular(case)
        d = output.mesh_data["template_barycentric_mapping_based_remesh"]
        self.assertEqual(d["accepted_iterations"], 2)
        self.assertLess(d["stretch_history"][-1], d["stretch_history"][0])
        self.assertTrue(d["initialization_details"]["extension_to_original_initializer"])
        self.assertLess(d["initialization_details"]["seed_equilibrium_max_change"], 1e-10)
        self.assertEqual(len(d["solver_records"][0]["coordinates"]), 3)
        for role, snapshot in zip(("source", "template"), snapshots):
            assert_same(case[role].__dict__, snapshot)

    def test_original_initializer_is_not_silently_changed(self):
        case = self.cases["test_tube"]
        with self.assertRaisesRegex(ValueError, "Flipped UV faces") as caught:
            remesh.template_barycentric_mapping_based_remesh(case["source"], case["source_segments"], case["template"], case["template_segments"],
                initialization="averaged_barycentric", diffusion_coefficient_of_mesh_parameterization=0.5)
        self.assertEqual(len(caught.exception.diagnostics["face_indices"]), 183)

    def test_independent_of_node_face_order_and_uv_frame(self):
        case = deepcopy(self.cases["test_tube"])
        reference = self.run_annular(case, cap=0).node.numpy()
        source = case["source"]
        rng = np.random.default_rng(734)
        permutation = rng.permutation(len(source.node))
        old_to_new = np.empty(len(permutation), dtype=int); old_to_new[permutation] = np.arange(len(permutation))
        source.node = source.node[permutation].clone()
        source.element = torch.tensor(old_to_new[source.element.numpy()[rng.permutation(len(source.element))]])
        source.node_set = {name: old_to_new[ids].tolist() for name, ids in source.node_set.items()}
        source.mapping_from_node_name_to_index = {name: int(old_to_new[i]) for name, i in source.mapping_from_node_name_to_index.items()}
        source.node_data = {name: value[permutation].clone() for name, value in source.node_data.items()}
        case["source_segments"] = [old_to_new[ids].tolist() for ids in case["source_segments"]]
        rotation = torch.tensor([[np.cos(.73), -np.sin(.73)], [np.sin(.73), np.cos(.73)]], dtype=torch.float64)
        case["template"].node[:, :2] = case["template"].node[:, :2] @ rotation.T * 1e-4 + torch.tensor([8., -3.])
        actual = self.run_annular(case, cap=0).node.numpy()
        np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=1e-10)

    def test_hole_with_singleton_interior_landmark(self):
        case = self.cases["test_hole"]
        output = self.run_annular(case, cap=0)
        np.testing.assert_array_equal(output.node[208].numpy(), case["source"].node[136].numpy())

    def test_non_circular_annular_template(self):
        case = deepcopy(self.cases["test_tube"])
        case["template"].node[:, 0] *= 1.3
        case["template"].node[:, 1] *= .9
        output = self.run_annular(case, cap=0)
        self.assertEqual(len(output.node), 288)

    def test_reject_nonannular_or_non_star_shaped_domain(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            remesh.template_barycentric_mapping_based_remesh(star(), SEGMENTS, star(center=(.5, .5, 0)), SEGMENTS, initialization="annular")
        inner = np.array([[-2., -2.], [2., -2.], [2., -1.], [-1., -1.], [-1., 1.], [2., 1.], [2., 2.], [-2., 2.]])
        outer = np.array([[-3., -3.], [3., -3.], [3., 3.], [-3., 3.]])
        with self.assertRaisesRegex(ValueError, "star-shaped"):
            remesh._annular_prepare_boundary_loops(np.r_[inner, outer], {"loops": [list(range(8)), list(range(8, 12))]}, policy())

    def test_ray_radius_uses_polygon_edges(self):
        vertex = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]])
        angles = np.mod(np.arctan2(vertex[:, 1], vertex[:, 0]), 2*np.pi)
        direction = np.array([[1., 0.], [1/np.sqrt(2), 1/np.sqrt(2)], [0., -1.]])
        radius = remesh._annular_compute_boundary_radius(direction, {"vertex": vertex, "angle": angles}, policy())
        np.testing.assert_allclose(radius, [1., np.sqrt(2), 1.], atol=1e-14)


class UntangleTests(unittest.TestCase):
    def test_energy_is_finite_and_has_correct_gradient_across_inversions(self):
        face = torch.tensor([[0, 1, 2]])
        reference = torch.eye(2, dtype=torch.float64)[None]
        area = torch.ones(1, dtype=torch.float64)
        for height in (-1., 0., 1.):
            uv = torch.tensor([[0., 0.], [1., 0.], [.2, height]], dtype=torch.float64, requires_grad=True)
            def energy(x):
                return remesh._untangle_compute_energy(x, face, reference, area, 1, .1, 1.)[0]
            self.assertTrue(torch.autograd.gradcheck(energy, (uv,), eps=1e-6, atol=1e-5))
            self.assertTrue(torch.isfinite(energy(uv)))

    def test_stable_negative_chi_and_precision(self):
        for dtype in (torch.float32, torch.float64):
            uv = torch.tensor([[0., 0.], [1., 0.], [0., -1.]], dtype=dtype, requires_grad=True)
            energy, determinant = remesh._untangle_compute_energy(uv, torch.tensor([[0, 1, 2]]),
                torch.eye(2, dtype=dtype)[None], torch.ones(1, dtype=dtype), 1, 1e-8, 1.)
            energy.backward()
            self.assertTrue(torch.isfinite(energy) and torch.isfinite(uv.grad).all())
            self.assertEqual(energy.dtype, dtype)
            self.assertEqual(float(determinant[0].detach()), -1.)

    def test_valid_original_initialization_is_unchanged(self):
        s, t = star(), star(center=(.5, .5, 0))
        args = (s, SEGMENTS, t, SEGMENTS)
        original = remesh.template_barycentric_mapping_based_remesh(*args, initialization="averaged_barycentric",
            diffusion_coefficient_of_mesh_parameterization=0.5)
        repaired = remesh.template_barycentric_mapping_based_remesh(*args, initialization="untangle",
            diffusion_coefficient_of_mesh_parameterization=0.5)
        np.testing.assert_array_equal(original.node, repaired.node)
        a, b = [o.mesh_data['template_barycentric_mapping_based_remesh'] for o in (original, repaired)]
        np.testing.assert_array_equal(a['final_weight_matrix'].toarray(), b['final_weight_matrix'].toarray())
        self.assertEqual(a['stretch_history'], b['stretch_history'])
        self.assertFalse(b['initialization_details']['applied'])

    def test_repair_with_reversed_orientation_and_fixed_coordinates(self):
        for sign in (1, -1):
            source = star()
            p, f = source.node.numpy(), source.element.numpy()
            topology = remesh._build_mesh_topology(source)
            uv = p[:, :2].copy(); uv[:, 1] *= sign
            uv[4] = [1.5, .5*sign]
            fixed = np.arange(4)
            with torch.no_grad():
                result, d = remesh._untangle_repair_parameterization(p, f, topology, uv, fixed, policy(),
                    remesh._untangle_validate_options(None))
            np.testing.assert_array_equal(result[fixed], uv[fixed])
            self.assertEqual(d['orientation'], sign)
            self.assertEqual(d['termination_reason'], 'valid_uv')

    def test_fixed_invalid_triangle_reports_error(self):
        source = star(); p, f = source.node.numpy(), source.element.numpy()
        uv = p[:, :2].copy(); uv[4] = [1.5, .5]
        with self.assertRaisesRegex(ValueError, 'cannot move'):
            remesh._untangle_repair_parameterization(p, f, remesh._build_mesh_topology(source), uv, np.array([0,1,2,4]),
                                    policy(), remesh._untangle_validate_options(None))

    def test_interior_landmark_remains_fixed_during_repair(self):
        from .template_barycentric_support import load_cases
        c = next(c for c in load_cases() if c['case_id']=='test_hole')
        out = remesh.template_barycentric_mapping_based_remesh(c['source'], c['source_segments'], c['template'], c['template_segments'],
            max_iteration_of_mesh_parameterization=0)
        d = out.mesh_data['template_barycentric_mapping_based_remesh']
        p, uv, f = d['source_node_normalized'], d['source_uv_normalized'].copy(), c['source'].element.numpy()
        fixed = d['constraints']['source_index']; uv[d['free_node_index'][4]] += [.2, .15]
        with self.assertRaises(ValueError): remesh._validate_uv_mesh(uv, f, None, d['numerical_policy'])
        result, repaired = remesh._untangle_repair_parameterization(p, f, remesh._build_mesh_topology(c['source']), uv, fixed,
            d['numerical_policy'], remesh._untangle_validate_options(None))
        np.testing.assert_array_equal(result[fixed], uv[fixed])
        self.assertIn(136, fixed)
        self.assertEqual(repaired['termination_reason'], 'valid_uv')

    def test_options_and_budget_failure(self):
        for opts in ({'max_outer_iterations':0}, {'max_inner_iterations':True}, {'area_weight':-1}, {'unknown':1}):
            with self.assertRaises(ValueError): remesh._untangle_validate_options(opts)
        with self.assertRaisesRegex(ValueError,'only'):
            remesh.template_barycentric_mapping_based_remesh(star(), SEGMENTS, star(center=(.5,.5,0)), SEGMENTS,
                initialization="averaged_barycentric", untangle_options={})
        source = star(); p, f = source.node.numpy(), source.element.numpy()
        uv = p[:, :2].copy(); uv[4] = [1.5, .5]
        # Freeze the optimizer to exercise failure reporting independently of
        # a solver's fortuitous success within one iteration.
        with patch.object(torch.optim.LBFGS, 'step', return_value=None):
            with self.assertRaisesRegex(ValueError,'Feasibility remains undetermined') as caught:
                remesh._untangle_repair_parameterization(p,f,remesh._build_mesh_topology(source),uv,np.arange(4),policy(),
                    remesh._untangle_validate_options({'max_outer_iterations':1,'max_inner_iterations':1}))
        self.assertEqual(caught.exception.diagnostics['untangling']['termination_reason'],'no_valid_initialization_found')


class StoppingTests(unittest.TestCase):
    def controlled_run(self, metrics, cap=10, alpha=0.5, invalid=False, nonfinite=False):
        source = star(); p = source.node.numpy(); face = source.element.numpy()
        topology = remesh._build_mesh_topology(star())
        fixed = np.arange(4); fixed_uv = p[:4, :2]
        options = {"initialization": "averaged_barycentric", "linear_solver": "sparse_lu", "solver_options": {}, "alpha": alpha, "max_iteration": cap}
        coordinates = [np.array([[.4, .5]]), np.array([[.45, .55]]), np.array([[.55, .45]])]
        if invalid:
            coordinates[1] = np.array([[2., 2.]])
        if nonfinite:
            coordinates[1] = np.array([[np.nan, .5]])
        counter = [0]
        def solver(*args):
            k = min(counter[0], len(coordinates)-1); counter[0] += 1
            return coordinates[k].copy(), {"mock_solve": counter[0]}
        with patch.object(remesh, "_solve_parameterization_system", side_effect=solver), \
             patch.object(remesh, "_compute_average_local_stretch", side_effect=metrics) as metric:
            uv, d = remesh._parameterize_source_mesh(p, face, topology, fixed, fixed_uv, options, policy())
        return uv, d, counter[0], metric.call_count

    def test_reject_worsening_parameterization(self):
        uv, d, solves, _ = self.controlled_run([10., 8., 9.])
        np.testing.assert_array_equal(uv[4], [.45, .55])
        self.assertEqual(d["stretch_history"], [10., 8.])
        self.assertEqual(d["candidate_stretch_history"], [8., 9.])
        self.assertEqual(d["attempted_iterations"], 2); self.assertEqual(d["accepted_iterations"], 1)
        self.assertEqual(solves, 3)

    def test_reject_worsening_final_candidate(self):
        for cap in (1, 10):
            uv, d, solves, _ = self.controlled_run([10., 12.], cap=cap)
            np.testing.assert_array_equal(uv[4], [.4, .5])
            self.assertEqual(d["stretch_history"], [10.]); self.assertEqual(solves, 2)
            self.assertEqual(d["termination_reason"], "stretch_increase")

    def test_accept_equal_stretch_candidate_and_honor_cap(self):
        uv, d, solves, _ = self.controlled_run([10., 10., 10.], cap=2)
        np.testing.assert_array_equal(uv[4], [.55, .45])
        self.assertEqual(d["accepted_iterations"], 2); self.assertEqual(solves, 3)
        self.assertEqual(d["termination_reason"], "iteration_limit")

    def test_zero_cap_and_zero_alpha(self):
        for cap, alpha in ((0, .5), (10, 0.)):
            _, d, solves, _ = self.controlled_run([10.], cap=cap, alpha=alpha)
            self.assertEqual(solves, 1); self.assertEqual(d["attempted_iterations"], 0)
            self.assertEqual(d["termination_reason"], "initialization_only")

    def test_invalid_candidate_rejected_without_metric(self):
        uv, d, solves, metrics = self.controlled_run([10.], invalid=True)
        np.testing.assert_array_equal(uv[4], [.4, .5])
        self.assertEqual(d["candidate_stretch_history"], [None]); self.assertEqual(metrics, 1)
        self.assertEqual(d["termination_reason"], "invalid_uv")

    def test_nonfinite_candidate_raises(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.controlled_run([10.], nonfinite=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
