"""Automatic selection regressions: geometry, constraints and search behavior."""
from copy import deepcopy
import unittest
from importlib.util import find_spec

if find_spec("scipy") is None:
    raise unittest.SkipTest("Install torch-fea-mesh[remeshing] for SciPy")
from unittest.mock import patch

import numpy as np
import torch

import torch_fea_mesh.processing.template_barycentric_mapping_based_remesher as algorithm
from .template_barycentric_support import DIAGNOSTIC_KEY, mesh_snapshot, assert_same
from .test_template_barycentric_mapping_based_remesher import policy, star, SEGMENTS


class AutoDiffusionTests(unittest.TestCase):
    def test_auto_selection_uses_shared_refinement_for_every_initializer(self):
        from .template_barycentric_support import load_cases
        tube = next(case for case in load_cases() if case['case_id'] == 'test_tube')
        for method in ('averaged_barycentric', 'mean_value', 'annular', 'untangle'):
            with self.subTest(initialization=method):
                if method == 'annular':
                    args = (tube['source'], tube['source_segments'], tube['template'], tube['template_segments'])
                else:
                    args = (star(), SEGMENTS, star(center=(.5, .5, 0.)), SEGMENTS)
                expected = algorithm.template_barycentric_mapping_based_remesh(
                    *args, initialization=method, diffusion_coefficient_of_mesh_parameterization=.5,
                    max_iteration_of_mesh_parameterization=2)
                with patch.object(algorithm, '_initialize_source_parameterization',
                                  wraps=algorithm._initialize_source_parameterization) as initialize, \
                     patch.object(algorithm, '_refine_parameterization_by_stretch',
                                  wraps=algorithm._refine_parameterization_by_stretch) as refine, \
                     patch.object(algorithm, '_map_template_to_source_surface',
                                  wraps=algorithm._map_template_to_source_surface) as transfer:
                    actual = algorithm.template_barycentric_mapping_based_remesh(
                        *args, initialization=method, max_iteration_of_mesh_parameterization=2)
                self.assertEqual(initialize.call_count, 1)
                self.assertEqual(refine.call_count, 1)
                self.assertGreaterEqual(transfer.call_count, 1)
                assert_same(actual.node, expected.node)
                diagnostics = actual.mesh_data[DIAGNOSTIC_KEY]
                self.assertEqual(diagnostics['auto_diffusion']['selected_coefficient'], .5)
                self.assertEqual(diagnostics['stretch_history'], expected.mesh_data[DIAGNOSTIC_KEY]['stretch_history'])

    def call(self, **kwargs):
        return algorithm.template_barycentric_mapping_based_remesh(
            star(), SEGMENTS, star(center=(.5, .5, 0.)), SEGMENTS, **kwargs)

    def test_default_matches_numeric_and_uses_one_initialization(self):
        s, t = star(), star(center=(.5, .5, 0.))
        snapshots = [mesh_snapshot(m) for m in (s, t)]
        expected = algorithm.template_barycentric_mapping_based_remesh(s, SEGMENTS, t, SEGMENTS,
            diffusion_coefficient_of_mesh_parameterization=0.5)
        with patch.object(algorithm, '_initialize_source_parameterization', wraps=algorithm._initialize_source_parameterization) as initialize:
            actual = algorithm.template_barycentric_mapping_based_remesh(s, SEGMENTS, t, SEGMENTS)
        assert_same(actual.node, expected.node)
        a, b = actual.mesh_data[DIAGNOSTIC_KEY], expected.mesh_data[DIAGNOSTIC_KEY]
        np.testing.assert_array_equal(a['source_uv_normalized'], b['source_uv_normalized'])
        self.assertEqual(a['stretch_history'], b['stretch_history'])
        self.assertEqual(a['auto_diffusion']['trial_count'], 1)
        self.assertEqual(a['auto_diffusion']['selected_coefficient'], .5)
        self.assertEqual(a['options']['initialization'], 'untangle')
        self.assertEqual(a['auto_diffusion']['requested_coefficient'], 'auto')
        self.assertEqual(initialize.call_count, 1)
        for snapshot, mesh in zip(snapshots, (s, t)):
            assert_same(snapshot, mesh.__dict__)

    def test_boundary_and_interior_landmarks_exact_in_both_dtypes(self):
        for dtype in (torch.float32, torch.float64):
            s, t = star(), star(center=(.65, .35, 0.))
            result = algorithm.template_barycentric_mapping_based_remesh(s, SEGMENTS, t, SEGMENTS,
                landmark_of_source_mesh=[4, 0], landmark_of_template_mesh=[4, 0],
                diffusion_coefficient_of_mesh_parameterization='auto', initialization='untangle',
                computation_dtype=dtype)
            self.assertTrue(torch.equal(result.node, s.node.to(dtype)))
            d = result.mesh_data[DIAGNOSTIC_KEY]
            np.testing.assert_array_equal(d['source_uv_normalized'][d['constraints']['source_index']], d['constraints']['source_uv'])
            self.assertEqual(d['auto_diffusion']['trial_count'], 1)

    def test_invalid_options_and_numeric_options_conflict(self):
        for alpha in ('automatic', None, True, -.1, 1.1):
            with self.assertRaisesRegex(ValueError, 'must be in'):
                algorithm.template_barycentric_mapping_based_remesh(star(), SEGMENTS, star(center=(.5, .5, 0)), SEGMENTS,
                    diffusion_coefficient_of_mesh_parameterization=alpha)
        for value in (0, -1, True, 1.2):
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                self.call(auto_diffusion_options={'max_trials': value})
        with self.assertRaisesRegex(ValueError, 'only max_trials'):
            self.call(auto_diffusion_options={'unknown': 1})
        with self.assertRaisesRegex(ValueError, 'only to automatic'):
            algorithm.template_barycentric_mapping_based_remesh(star(), SEGMENTS, star(center=(.5, .5, 0)), SEGMENTS,
                diffusion_coefficient_of_mesh_parameterization=0.5, auto_diffusion_options={'max_trials': 2})

    def rejected_quality(self, *args):
        return {'passed': False, 'failed_corner_count': 1, 'minimum_corner_margin': -.1,
                'minimum_signed_corner_sine': -.1, 'failed_corners': [{'element_index': 0, 'corner_index': 0}]}

    def test_failed_search_is_bounded_and_resets_uv_and_weights(self):
        initial_states = []
        refine = algorithm._refine_parameterization_by_stretch
        def record(*args, **kwargs):
            initial_states.append((args[4].copy(), args[5]['final_weight_matrix'].toarray().copy()))
            return refine(*args, **kwargs)
        with patch.object(algorithm, '_output_corner_quality', side_effect=self.rejected_quality), \
             patch.object(algorithm, '_refine_parameterization_by_stretch', side_effect=record), \
             patch.object(algorithm, '_initialize_source_parameterization', wraps=algorithm._initialize_source_parameterization) as initialize:
            with self.assertRaisesRegex(ValueError, 'Feasibility remains undetermined') as caught:
                self.call(auto_diffusion_options={'max_trials': 6})
        d = caught.exception.diagnostics['auto_diffusion']
        self.assertEqual(d['trial_count'], 6)
        self.assertEqual([t['coefficient'] for t in d['trials'][:5]], [.5, 0., .25, .75, 1.])
        self.assertEqual(initialize.call_count, 1)
        for uv, weight in initial_states[1:]:
            np.testing.assert_array_equal(uv, initial_states[0][0])
            np.testing.assert_array_equal(weight, initial_states[0][1])
        self.assertIsNone(d['selected_coefficient'])

    def test_earlier_acceptable_iterate_survives_later_bad_corners(self):
        check = algorithm._output_corner_quality
        count = [0]
        def quality(*args):
            count[0] += 1
            return check(*args) if count[0] == 1 else self.rejected_quality()
        with patch.object(algorithm, '_output_corner_quality', side_effect=quality), \
             patch.object(algorithm, '_compute_average_local_stretch', side_effect=[3., 2., 1.]):
            output = self.call(max_iteration_of_mesh_parameterization=2, auto_diffusion_options={'max_trials': 1})
        d = output.mesh_data[DIAGNOSTIC_KEY]
        self.assertEqual(d['auto_diffusion']['selected_iteration'], 0)
        self.assertEqual(d['auto_diffusion']['total_refinement_attempts'], 2)
        self.assertEqual(d['stretch_history'], [3.])
        self.assertEqual(d['accepted_iterations'], 0)
        self.assertEqual(d['termination_reason'], 'auto_selected_earlier_iterate')

    def test_zero_iteration_cap_does_not_repeat_identical_trials(self):
        with patch.object(algorithm, '_output_corner_quality', side_effect=self.rejected_quality):
            with self.assertRaises(ValueError) as caught:
                self.call(max_iteration_of_mesh_parameterization=0)
        self.assertEqual(caught.exception.diagnostics['auto_diffusion']['trial_count'], 1)
        self.assertEqual(caught.exception.diagnostics['auto_diffusion']['total_refinement_attempts'], 0)

    def test_one_failed_coefficient_does_not_abort_other_trials(self):
        refine = algorithm._refine_parameterization_by_stretch
        def fail_default(*args, **kwargs):
            if args[6]['alpha'] == .5:
                algorithm._fail('simulated coefficient-dependent solver failure', attempted_iterations=1)
            return refine(*args, **kwargs)
        with patch.object(algorithm, '_refine_parameterization_by_stretch', side_effect=fail_default):
            output = self.call(auto_diffusion_options={'max_trials': 2})
        d = output.mesh_data[DIAGNOSTIC_KEY]['auto_diffusion']
        self.assertEqual(d['selected_coefficient'], 0.)
        self.assertEqual(d['trials'][0]['termination_reason'], 'trial_failed')
        self.assertEqual(d['total_refinement_attempts'], 1)

    def test_intersection_rejection_cannot_return_success(self):
        with patch.object(algorithm, '_check_output_intersections', return_value={'passed': False, 'element_pairs_with_improper_contact': [(0, 1)]}):
            with self.assertRaisesRegex(ValueError, 'no acceptable output') as caught:
                self.call(auto_diffusion_options={'max_trials': 1})
        records = caught.exception.diagnostics['auto_diffusion']['trials'][0]['iterates']
        self.assertTrue(all(not r['passed'] for r in records))

    def test_positive_concave_collapsed_and_bowtie_quads(self):
        square = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        for dtype in (np.float32, np.float64):
            p = policy(dtype)
            good = algorithm._output_corner_quality(square, [[0, 1, 2, 3]], p)
            self.assertTrue(good['passed'])
            self.assertAlmostEqual(good['minimum_signed_corner_sine'], 1.)
            for coordinates, face in ((square*np.array([1, 0, 1]), [0, 1, 2, 3]),
                                      (square, [0, 2, 1, 3]),
                                      (np.array([[0, 0, 0], [1, 0, 0], [.2, .2, 0], [0, 1, 0]]), [0, 1, 2, 3])):
                result = algorithm._output_corner_quality(coordinates, [face], p)
                self.assertFalse(result['passed'])
                self.assertGreater(result['failed_corner_count'], 0)

    def test_corner_tolerance_depends_on_dtype_and_geometry(self):
        q = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1e-6, 0.], [0., 1., 0.]])
        self.assertTrue(algorithm._output_corner_quality(q, [[0, 1, 2, 3]], policy(np.float64))['passed'])
        self.assertFalse(algorithm._output_corner_quality(q, [[0, 1, 2, 3]], policy(np.float32))['passed'])
        translated = q + [1e10, 0, 0]
        self.assertFalse(algorithm._output_corner_quality(translated, [[0, 1, 2, 3]], policy(np.float64))['passed'])

    def test_intersection_geometry_including_shared_features(self):
        a = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        examples = [
            (a+[0, 0, 1], [3, 4, 5], False),
            (np.array([[.2, .2, -1], [.2, .2, 1], [.8, .2, 0]]), [3, 4, 5], True),
            (a*.4+[.1, .1, 0], [3, 4, 5], True),
            (np.array([[0., 0., 0.], [-1., 0., 0.], [0., -1., 0.]]), [0, 4, 5], False),
            (np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]]), [0, 1, 5], False),
            (np.array([[0., 0., 0.], [1., 0., 0.], [.2, .2, 0.]]), [0, 1, 5], True),
            (a.copy(), [0, 1, 2], True)]
        for b, second, bad in examples:
            result = algorithm._check_output_intersections(np.r_[a, b], [[0, 1, 2], second], policy())
            self.assertEqual(not result['passed'], bad)

    def test_positive_quad_corners_do_not_replace_intersection_check(self):
        a = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        faces = [[0, 1, 2, 3], [4, 5, 6, 7]]
        nodes = np.r_[a, a+[.25, .25, 0.]]
        self.assertTrue(algorithm._output_corner_quality(nodes, faces, policy())['passed'])
        result = algorithm._check_output_intersections(nodes, faces, policy())
        self.assertFalse(result['passed'])
        self.assertEqual(result['element_pairs_with_improper_contact'], [(0, 1)])
        self.assertTrue(result['both_quad_diagonals_checked'])


if __name__ == '__main__':
    unittest.main()
