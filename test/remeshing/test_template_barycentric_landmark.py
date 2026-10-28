"""Public landmark correspondence tests, including shared boundary vertices."""
from copy import deepcopy
import unittest
from importlib.util import find_spec

if find_spec("scipy") is None:
    raise unittest.SkipTest("Install torch-fea-mesh[remeshing] for SciPy")

import numpy as np
import torch

import torch_fea_mesh.processing.template_barycentric_mapping_based_remesher as remesh
from .template_barycentric_support import mesh_snapshot, assert_same
from .test_template_barycentric_mapping_based_remesher import star, SEGMENTS

KEY = 'template_barycentric_mapping_based_remesh'


def boundary_fan(bottom, curved):
    node = np.array([[x, 0., 0.] for x in bottom]+[[1., 1., 0.], [0., 1., 0.], [.5, .5, .2 if curved else 0.]])
    center = len(node)-1
    mesh = remesh.PolygonMesh(node=node, element=[[i, (i+1)%center, center] for i in range(center)])
    last = len(bottom)-1
    return mesh, [list(range(len(bottom))), [last, last+1], [last+1, last+2], [last+2, 0]]


class LandmarkInputTests(unittest.TestCase):
    def test_interior_pair_with_different_node_indices_and_dtype(self):
        for dtype in (torch.float64, torch.float32):
            source, template = star(), star(center=(.65, .35, 0.))
            # Put the template interior vertex at index 2 rather than 4.
            permutation = np.array([0, 1, 4, 3, 2])
            inverse = np.argsort(permutation)
            template.node = template.node[permutation].clone()
            template.element = torch.tensor(inverse[template.element.numpy()])
            paths = [inverse[s].tolist() for s in SEGMENTS]
            source_ids, template_ids = torch.tensor([4]), np.array([2], dtype=np.int64)
            snapshots = [mesh_snapshot(m) for m in (source, template)]
            output = remesh.template_barycentric_mapping_based_remesh(source, SEGMENTS, template, paths,
                landmark_of_source_mesh=source_ids, landmark_of_template_mesh=template_ids,
                initialization='untangle', computation_dtype=dtype)
            np.testing.assert_array_equal(output.node[2].numpy(), source.node[4].to(dtype).numpy())
            d = output.mesh_data[KEY]
            position = np.flatnonzero(d['constraints']['source_index'] == 4)[0]
            np.testing.assert_array_equal(d['source_uv_normalized'][4], d['constraints']['source_uv'][position])
            self.assertNotIn(4, d['free_node_index'])
            self.assertEqual(d['query_status'][2], 'constraint')
            self.assertEqual(d['landmark_correspondence']['interior_pair_count'], 1)
            for snapshot, mesh in zip(snapshots, (source, template)):
                assert_same(snapshot, mesh.__dict__)
            np.testing.assert_array_equal(source_ids, [4]); np.testing.assert_array_equal(template_ids, [2])

    def test_omitted_and_empty_inputs_preserve_legacy_result(self):
        s, t = star(), star(center=(.5, .5, 0.))
        reference = remesh.template_barycentric_mapping_based_remesh(s, SEGMENTS, t, SEGMENTS)
        explicit = remesh.template_barycentric_mapping_based_remesh(s, SEGMENTS, t, SEGMENTS,
            landmark_of_source_mesh=[], landmark_of_template_mesh=[])
        np.testing.assert_array_equal(reference.node, explicit.node)
        assert_same(reference.mesh_data[KEY]['source_uv_normalized'], explicit.mesh_data[KEY]['source_uv_normalized'])

    def test_joint_reordering_duplicates_and_existing_singleton(self):
        s, t = star(), star(center=(.6, .4, 0.))
        reference = None
        for source_ids, template_ids in (([4, 0, 2], [4, 0, 2]), ([2, 4, 0], [2, 4, 0]),
                                          ([4, 4, 2, 0], [4, 4, 2, 0])):
            output = remesh.template_barycentric_mapping_based_remesh(s, SEGMENTS+[[4]], t, SEGMENTS+[[4]],
                landmark_of_source_mesh=source_ids, landmark_of_template_mesh=template_ids)
            if reference is None:
                reference = output
            else:
                np.testing.assert_array_equal(output.node, reference.node)
                for field in ('source_uv_normalized', 'constraints', 'landmark_correspondence', 'stretch_history'):
                    assert_same(output.mesh_data[KEY][field], reference.mesh_data[KEY][field])
                np.testing.assert_array_equal(output.mesh_data[KEY]['final_weight_matrix'].toarray(),
                                              reference.mesh_data[KEY]['final_weight_matrix'].toarray())
            self.assertEqual(output.mesh_data[KEY]['landmark_correspondence']['unique_pair_count'], 3)
            np.testing.assert_array_equal(output.node[4], s.node[4])

    def test_boundary_landmark_splits_unequal_sampling_and_arc_fractions(self):
        source, ss = boundary_fan([0., .1, .25, 1.], True)
        template, ts = boundary_fan([0., .6, .8, .9, 1.], False)
        before = deepcopy((ss, ts))
        output = remesh.template_barycentric_mapping_based_remesh(source, ss, template, ts,
            landmark_of_source_mesh=[2], landmark_of_template_mesh=[1])
        manual_s = [[0, 1, 2], [2, 3]]+ss[1:]
        manual_t = [[0, 1], [1, 2, 3, 4]]+ts[1:]
        reference = remesh.template_barycentric_mapping_based_remesh(source, manual_s, template, manual_t)
        np.testing.assert_array_equal(output.node, reference.node)
        np.testing.assert_array_equal(output.node[1], source.node[2])
        d = output.mesh_data[KEY]
        self.assertEqual(d['landmark_correspondence']['additional_boundary_segments'], 1)
        np.testing.assert_allclose(d['source_uv'][2], template.node[1, :2], rtol=0, atol=1e-15)
        assert_same((ss, ts), before)

    def test_closed_boundary_is_split_without_changing_anchor(self):
        source, _ = boundary_fan([0., .25, 1.], True)
        template, _ = boundary_fan([0., .6, 1.], False)
        paths = [[0, 1, 2, 3, 4, 0]]
        output = remesh.template_barycentric_mapping_based_remesh(source, paths, template, paths,
            landmark_of_source_mesh=[1], landmark_of_template_mesh=[1])
        manual = [[0, 1], [1, 2, 3, 4, 0]]
        reference = remesh.template_barycentric_mapping_based_remesh(source, manual, template, manual)
        np.testing.assert_array_equal(output.node, reference.node)
        np.testing.assert_array_equal(output.node[[0, 1]], source.node[[0, 1]])

    def test_invalid_landmark_inputs(self):
        s, t = star(), star(center=(.5, .5, 0.))
        for source_ids, template_ids, message in (
            ([4], None, 'together'), (None, [4], 'together'), ([4], [], 'same length'),
            ([True], [4], 'invalid node index'), ([4.0], [4], 'invalid node index'),
            ([4], [-1], 'invalid node index'), ([5], [4], 'invalid node index'),
            ([[4]], [[4]], 'invalid node index'), (np.array([[4]]), [4], 'one-dimensional'),
            ([4], torch.tensor([4.] ), 'invalid node index'), ({4}, [4], 'one-dimensional'),
            ([4, 4], [4, 0], 'one-to-one'), ([0, 1], [0, 0], 'one-to-one'),
            ([4], [0], 'boundary landmark'), ([0], [4], 'boundary landmark')):
            with self.subTest(source=source_ids, template=template_ids):
                with self.assertRaisesRegex(ValueError, message):
                    remesh.template_barycentric_mapping_based_remesh(s, SEGMENTS, t, SEGMENTS,
                        landmark_of_source_mesh=source_ids, landmark_of_template_mesh=template_ids)

    def test_conflicting_shared_endpoint_and_singleton(self):
        s, t = star(), star(center=(.5, .5, 0.))
        with self.assertRaisesRegex(ValueError, 'conflicts with a boundary endpoint'):
            remesh.template_barycentric_mapping_based_remesh(s, SEGMENTS, t, SEGMENTS,
                landmark_of_source_mesh=[0], landmark_of_template_mesh=[1])
        source, ss = boundary_fan([0., .25, .75, 1.], True)
        template, ts = boundary_fan([0., .3, .8, 1.], False)
        with self.assertRaisesRegex(ValueError, 'singleton constraint'):
            remesh.template_barycentric_mapping_based_remesh(source, ss+[[1]], template, ts+[[1]],
                landmark_of_source_mesh=[1], landmark_of_template_mesh=[2])

    def test_boundary_landmark_order_must_follow_paired_paths(self):
        source, ss = boundary_fan([0., .25, .75, 1.], True)
        template, ts = boundary_fan([0., .3, .8, 1.], False)
        with self.assertRaisesRegex(ValueError, 'order conflicts'):
            remesh.template_barycentric_mapping_based_remesh(source, ss, template, ts,
                landmark_of_source_mesh=[1, 2], landmark_of_template_mesh=[2, 1])

    def test_testdata2_unsplit_boundary_and_explicit_named_indices(self):
        from .template_barycentric_support import load_cases
        case = next(c for c in load_cases() if c['case_id'] == 'testdata2')
        source, template = case['source'], case['template']
        names = ['Hinge', 'Tip', 'LC', 'RC']
        source_ids = [source.mapping_from_node_name_to_index[n] for n in names]
        template_ids = [template.mapping_from_node_name_to_index[n] for n in names]
        paths = ['LC_to_T_to_RC', 'RC_to_H', 'LC_to_H']
        output = remesh.template_barycentric_mapping_based_remesh(source, [source.node_set[n] for n in paths],
            template, [template.node_set[n] for n in paths], initialization='untangle',
            landmark_of_source_mesh=source_ids, landmark_of_template_mesh=template_ids)
        reference = remesh.template_barycentric_mapping_based_remesh(source, case['source_segments'], template,
            case['template_segments'], initialization='untangle')
        np.testing.assert_array_equal(output.node, reference.node)
        np.testing.assert_array_equal(output.node[template_ids], source.node[source_ids])
        self.assertEqual(output.mesh_data[KEY]['landmark_correspondence']['additional_boundary_segments'], 1)


if __name__ == '__main__':
    unittest.main()
