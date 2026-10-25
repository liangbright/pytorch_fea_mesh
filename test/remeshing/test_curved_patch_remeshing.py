import pytest

pytestmark = [pytest.mark.remeshing, pytest.mark.processing, pytest.mark.vtk]
pytest.importorskip("vtk", reason="VTK is required for this test module")

from test.helper.output import output_dir
"""Curved enclosure regressions: topology, scheduling, protection and full runs."""
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import torch_fea_mesh.processing.isotropic_triangle_surface_remesher as r
from test.helper.mesh_test_io import save_mesh_pair
from test.remeshing.test_isotropic_triangle_surface_remesher import context, attempt
from test.helper.curved_patch_example import curved_disk, folded_disk, nested_curved_disk, ambiguous_two_sided_disk, curved_disk_with_hole, rotated_example, repairable_curved_patch


def collect(ctx, metadata):
    m,t,f,b,ref,p,tol = ctx
    cycle = dict(vertices=metadata['cycle'],edge=metadata['edge'],revision=t['revision'])
    return r._collect_enclosed_face(m,t,cycle,tol)


class CurvedPatchTest(unittest.TestCase):
    def test_curved_and_folded_disk_include_nonincident_face(self):
        for builder in (curved_disk, folded_disk):
            source,meta=builder()
            ctx=context(source,length=meta['target'])
            region,reason=collect(ctx,meta)
            self.assertIsNotNone(region,reason)
            self.assertEqual(region['method'],'curved_disk')
            self.assertEqual(region['faces'],meta['expected_enclosed'])
            self.assertEqual(region['euler_characteristic'],1)
            self.assertGreater(len(region['faces']),len(ctx[1]['edge_to_faces'][meta['edge']]))
            self.assertFalse(region['faces'] & meta['expected_exterior'])
        source,meta=folded_disk()
        self.assertGreater(float(source.node[3,0]),float(source.node[:3,0].max()))

    def test_real_repair_first_then_exact_half_gate(self):
        source,meta=folded_disk()
        ctx=context(source,length=meta['target'])
        m,t,*_=ctx
        before_x,before_f=m.node.clone(),m.element.clone()
        original=r._try_repair_collapse_patch
        with patch.object(r,'_try_repair_collapse_patch',wraps=original) as repair:
            for i in range(5):
                self.assertEqual(attempt(ctx,meta['edge'],i),(False,'deferred_after_repair_failure'))
                self.assertTrue(torch.equal(m.node,before_x));self.assertTrue(torch.equal(m.element,before_f))
                self.assertEqual(t['revision'],0)
            self.assertEqual(attempt(ctx,meta['edge'],5),(True,'enclosed_region_collapsed'))
            self.assertEqual(repair.call_count,6)
        self.assertEqual(len(m.element),1)
        self.assertEqual(t['deletions'][0]['iteration'],6)
        r._check_adjacency_consistency(m,t)

    def test_complete_boundary_edge_on_curved_disk(self):
        source,meta=curved_disk(exterior=False)
        for edge in [(0,1),(0,2),(1,2)]:
            ctx=context(source,length=6.)
            self.assertEqual(attempt(ctx,edge,0),(False,'deferred_after_repair_failure'))
            self.assertEqual(attempt(ctx,edge,5),(True,'enclosed_region_collapsed'))
            self.assertEqual(len(ctx[0].element),0)
            r._check_adjacency_consistency(ctx[0],ctx[1])

    def test_curved_repair_success_prevents_deletion_in_each_phase(self):
        source,meta=repairable_curved_patch()
        for i in (0,5):
            ctx=context(source,length=meta['target'])
            with patch.object(r,'_collect_enclosed_face',side_effect=AssertionError('deletion attempted after repair')):
                self.assertEqual(attempt(ctx,meta['edge'],i,False),(True,'repaired_collapsed'))
            self.assertFalse(ctx[1]['deletions'])
            r._check_adjacency_consistency(ctx[0],ctx[1])

    def test_nested_chord_and_exterior_rewiring(self):
        source,meta=nested_curved_disk(100)
        ctx=context(source,length=meta['target'],boundary=False,points=meta['feature_points'])
        region,reason=collect(ctx,meta)
        self.assertEqual(reason,'ok')
        self.assertEqual(region['faces'],meta['expected_enclosed'])
        self.assertEqual(len(region['support_faces']),199)
        self.assertEqual(len(region['nested_chord_faces']),99)
        m,t,*_=ctx
        self.assertEqual(attempt(ctx,(0,1),5,False),(True,'enclosed_region_collapsed'))
        survivor=next(v for v in (0,1) if t['vertex_to_faces'][v])
        removed=1-survivor
        expected={tuple(sorted(survivor if v==removed else int(v) for v in source.element[i]))
                  for i in meta['expected_exterior'] if not {0,1}<=set(source.element[i].tolist())}
        self.assertEqual({tuple(sorted(f)) for f in m.element.tolist()},expected)
        self.assertEqual(len(m.element),6)
        self.assertTrue(torch.equal(m.node[2:],source.node[2:]))
        r._check_adjacency_consistency(m,t)
        _,_,f,b,ref,p,tol=ctx
        output=r._finalize_output(m,t,source,ref,p,f,b,tol)
        save_mesh_pair(source,output,output_dir()/'features/selected_edge_nested_100',
                       feature_point_index_list=meta['feature_points'])

    def test_orientation_and_numbering_independence(self):
        source,meta=rotated_example(nested_curved_disk,8)
        rng=np.random.default_rng(813)
        permutation=rng.permutation(len(source.node))
        inverse=np.argsort(permutation)
        face_order=rng.permutation(len(source.element))
        mesh=r.TriangleMesh(source.node[permutation],inverse[source.element.numpy()][face_order])
        revised=dict(meta,cycle=tuple(int(inverse[v]) for v in meta['cycle']),
                     edge=r._edge(*(int(inverse[v]) for v in meta['edge'])))
        expected={i for i,old in enumerate(face_order) if old in meta['expected_enclosed']}
        region,reason=collect(context(mesh,length=meta['target']),revised)
        self.assertIsNotNone(region,reason)
        self.assertEqual(region['faces'],expected)

    def test_cap_pair_deletion_requires_real_repair_failure(self):
        source,meta=ambiguous_two_sided_disk()
        ctx=context(source,length=meta['target'])
        region,reason=collect(ctx,meta)
        self.assertEqual(reason,'ok');self.assertEqual(region['method'],'curved_two_caps')
        self.assertEqual(region['faces'],set(range(6)))
        self.assertEqual(region['cap_count'],2)
        before_x,before_f=ctx[0].node.clone(),ctx[0].element.clone()
        original=r._try_repair_collapse_patch
        with patch.object(r,'_try_repair_collapse_patch',wraps=original) as repair:
            for i in range(5):
                self.assertEqual(attempt(ctx,meta['edge'],i,False),(False,'deferred_after_repair_failure'))
                self.assertTrue(torch.equal(ctx[0].node,before_x))
                self.assertTrue(torch.equal(ctx[0].element,before_f))
            self.assertEqual(attempt(ctx,meta['edge'],5,False),(True,'enclosed_region_collapsed'))
            self.assertEqual(repair.call_count,6)
        m,t,f,b,ref,p,tol=ctx
        self.assertEqual(len(m.element),0)
        r._check_adjacency_consistency(m,t)
        out=r._finalize_output(m,t,source,ref,p,f,b,tol)
        self.assertEqual(tuple(out.node.shape),(0,3));self.assertEqual(tuple(out.element.shape),(0,3))
        self.assertEqual(t['deletions'][0]['enclosures'][0]['cap_count'],2)

    def test_flat_and_curved_cap_pair_is_supported(self):
        source,meta=curved_disk(exterior=False)
        source.element=torch.cat((source.element,torch.tensor([[1,0,2]])))
        ctx=context(source,length=meta['target'])
        region,reason=collect(ctx,meta)
        self.assertEqual(reason,'ok');self.assertEqual(region['faces'],set(range(4)))
        self.assertEqual(attempt(ctx,meta['edge'],0,False),(False,'deferred_after_repair_failure'))
        self.assertEqual(attempt(ctx,meta['edge'],5,False),(True,'enclosed_region_collapsed'))
        self.assertEqual(len(ctx[0].element),0)

    def test_two_cap_deletion_preserves_unrelated_component(self):
        source,meta=ambiguous_two_sided_disk()
        source.node=torch.cat((source.node,torch.tensor([[10.,0,0],[11.,0,0],[10.,1,0]],dtype=torch.float64)))
        source.element=torch.cat((source.element,torch.tensor([[5,6,7]])))
        ctx=context(source,length=meta['target'])
        self.assertEqual(attempt(ctx,meta['edge'],5,False),(True,'enclosed_region_collapsed'))
        m,t,f,b,ref,p,tol=ctx
        out=r._finalize_output(m,t,source,ref,p,f,b,tol)
        self.assertEqual(len(out.element),1)
        self.assertTrue(torch.equal(out.node,source.node[5:]))
        self.assertTrue(torch.equal(out.element,torch.tensor([[0,1,2]])))

    def test_two_cap_deletion_cannot_erase_feature_edge(self):
        source,meta=ambiguous_two_sided_disk()
        ctx=context(source,length=meta['target'],curves=[[2,3]])
        self.assertFalse(attempt(ctx,meta['edge'],5,False)[0])
        self.assertEqual(ctx[1]['revision'],0)
        self.assertTrue(torch.equal(ctx[0].element,source.element))

    def test_full_public_remesher_cap_pair_respects_half(self):
        source,meta=ambiguous_two_sided_disk()
        early=r.isotropic_triangle_surface_remesh(source,meta['target'],False,max_iter=1)
        self.assertGreater(len(early.element),0)
        self.assertFalse(early.mesh_data['remeshing_report']['region_deletions'])
        late=r.isotropic_triangle_surface_remesh(source,meta['target'],False,max_iter=10)
        self.assertEqual(tuple(late.node.shape),(0,3));self.assertEqual(tuple(late.element.shape),(0,3))
        deletions=late.mesh_data['remeshing_report']['region_deletions']
        self.assertEqual(deletions[0]['iteration'],6)
        self.assertEqual(deletions[0]['enclosures'][0]['cap_count'],2)

    def test_cap_junction_and_incoherent_pair_remain_unresolved(self):
        source,meta=ambiguous_two_sided_disk()
        source.node=torch.cat((source.node,torch.tensor([[.2,1.,2.]],dtype=torch.float64)))
        source.element=torch.cat((source.element,torch.tensor([[0,1,5],[1,2,5],[2,0,5]])))
        region,reason=collect(context(source,length=meta['target']),meta)
        self.assertIsNone(region);self.assertEqual(reason,'ambiguous_cycle_sides')
        source,meta=ambiguous_two_sided_disk()
        source.element[3:]=source.element[3:].flip(1)
        region,reason=collect(context(source,length=meta['target']),meta)
        self.assertIsNone(region);self.assertEqual(reason,'ambiguous_cap_orientation')

    def test_hole_and_internal_branch_are_not_guessed(self):
        source,meta=curved_disk_with_hole()
        region,reason=collect(context(source,length=meta['target']),meta)
        self.assertIsNone(region)
        self.assertEqual(reason,'disk_boundary_mismatch')
        source,meta=nested_curved_disk(8)
        new=len(source.node)
        source.node=torch.cat((source.node,torch.tensor([[2.,.7,-4.]],dtype=torch.float64)))
        source.element=torch.cat((source.element,torch.tensor([[0,4,new]])))
        region,reason=collect(context(source,length=meta['target']),meta)
        self.assertIsNone(region,reason)

    def test_full_patch_feature_protection(self):
        source,meta=curved_disk()
        ctx=context(source,length=meta['target'],curves=[[2,3]])
        before=ctx[0].element.clone()
        accepted,_=attempt(ctx,meta['edge'],5)
        self.assertFalse(accepted)
        self.assertTrue(torch.equal(ctx[0].element,before))
        self.assertEqual(ctx[1]['revision'],0)

    def test_full_public_remesher_nested_chord(self):
        source,meta=nested_curved_disk(100)
        before_x,before_f=source.node.clone(),source.element.clone()
        output=r.isotropic_triangle_surface_remesh(source,meta['target'],False,max_iter=10,
            feature_point_index_list=meta['feature_points'])
        deletions=output.mesh_data['remeshing_report']['region_deletions']
        self.assertEqual(len(deletions),1)
        self.assertEqual(deletions[0]['iteration'],6)
        self.assertEqual(deletions[0]['enclosed_face_count'],298)
        self.assertTrue(deletions[0]['repair_failure'])
        self.assertEqual(deletions[0]['enclosures'][0]['method'],'curved_disk')
        self.assertEqual(len(output.element),6)
        self.assertTrue(torch.equal(source.node,before_x));self.assertTrue(torch.equal(source.element,before_f))
        save_mesh_pair(source,output,output_dir()/'features/nested_curved_100',
                       feature_point_index_list=meta['feature_points'])

    def test_projection_disabled_on_full_curved_run(self):
        source,meta=nested_curved_disk(8)
        with patch.object(r,'_project_point',side_effect=AssertionError('projection disabled')):
            output=r.isotropic_triangle_surface_remesh(source,meta['target'],False,max_iter=10,
                feature_point_index_list=meta['feature_points'],flag_project_to_source_mesh=False)
        self.assertEqual(output.mesh_data['remeshing_report']['region_deletions'][0]['iteration'],6)
        save_mesh_pair(source,output,output_dir()/'features/nested_curved_8_no_projection',
                       feature_point_index_list=meta['feature_points'])

    def test_unprotected_nested_vertex_compaction(self):
        source,meta=nested_curved_disk(100)
        output=r.isotropic_triangle_surface_remesh(source,meta['target'],False,max_iter=10)
        deletions=output.mesh_data['remeshing_report']['region_deletions']
        self.assertEqual(len(deletions),1)
        self.assertEqual(deletions[0]['iteration'],6)
        self.assertEqual(deletions[0]['enclosed_face_count'],298)
        self.assertEqual(len(set(output.element.reshape(-1).tolist())),len(output.node))
        self.assertEqual(len(output.node),5)
        self.assertEqual(len(output.element),4)
