import pytest

pytestmark = [pytest.mark.remeshing, pytest.mark.processing, pytest.mark.vtk]
pytest.importorskip("vtk", reason="VTK is required for this test module")

from test.helper.output import output_dir
"""Curved patch regressions: nests, scheduling, protection and full runs."""
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import torch_fea_mesh.processing.isotropic_triangle_surface_remesher as r
from test.helper.mesh_test_io import save_mesh_pair
from test.remeshing.test_isotropic_triangle_surface_remesher import context, attempt
from test.helper.curved_patch_example import curved_disk, folded_disk, nested_curved_disk, ambiguous_two_sided_disk, curved_disk_with_hole, rotated_example, repairable_curved_patch


def flat_nest(n=8):
    """The exact nest from the owner's description, flattened into one plane."""
    source,meta=nested_curved_disk(n,chords=False)
    source.node[:,2]=0.
    return source,meta


def nests(ctx,edge):
    m,t,f,b,ref,p,tol=ctx
    current=r._classify_collapse_neighborhood(t,edge,f,b)
    found=[r._largest_nest(t,edge,anchor,current['common'],current['incident'],p['repair_max_faces'])
           for anchor in sorted(current['incident'])]
    return [nest for nest in found if nest is not None]


class CurvedPatchTest(unittest.TestCase):
    def test_curved_and_folded_disk_include_nonincident_face(self):
        for builder in (curved_disk, folded_disk):
            source,meta=builder()
            ctx=context(source,length=meta['target'])
            w,disk,other=nests(ctx,meta['edge'])[0]
            self.assertEqual(w,meta['cycle'][2])
            self.assertEqual(disk['faces'],meta['expected_enclosed'])
            self.assertGreater(len(disk['faces']),len(ctx[1]['edge_to_faces'][meta['edge']]))
            self.assertFalse(disk['faces'] & meta['expected_exterior'])
        source,meta=folded_disk()
        self.assertGreater(float(source.node[3,0]),float(source.node[:3,0].max()))

    def test_real_repair_first_then_exact_half_gate(self):
        source,meta=curved_disk(height=.2)
        ctx=context(source,length=meta['target'])
        m,t,*_=ctx
        before_x,before_f=m.node.clone(),m.element.clone()
        original=r._try_repair_collapse_patch
        with patch.object(r,'_try_repair_collapse_patch',wraps=original) as repair:
            for i in range(5):
                self.assertEqual(attempt(ctx,meta['edge'],i),(False,'deferred_after_repair_failure'))
                self.assertTrue(torch.equal(m.node,before_x));self.assertTrue(torch.equal(m.element,before_f))
                self.assertEqual(t['revision'],0)
            self.assertEqual(attempt(ctx,meta['edge'],5),(True,'ordinary_collapsed'))
            self.assertEqual(repair.call_count,6)
        self.assertEqual(len(m.element),1)
        self.assertEqual(t['replacements'][0]['iteration'],6)
        self.assertEqual(t['replacements'][0]['outer_triangle'],list(meta['cycle']))
        r._check_adjacency_consistency(m,t)

    def test_disk_over_the_limit_is_left_alone(self):
        for builder in (curved_disk, folded_disk):
            source,meta=builder()
            ctx=context(source,length=meta['target'])
            m,t,*_=ctx
            before_x,before_f=m.node.clone(),m.element.clone()
            self.assertFalse(attempt(ctx,meta['edge'],5)[0])
            self.assertTrue(torch.equal(m.node,before_x));self.assertTrue(torch.equal(m.element,before_f))
            self.assertFalse(t['replacements'])

    def test_lone_disk_keeps_one_element_on_each_outer_edge(self):
        source,meta=curved_disk(height=.2,exterior=False)
        for edge in [(0,1),(0,2),(1,2)]:
            ctx=context(source,length=6.)
            self.assertEqual(attempt(ctx,edge,0),(False,'deferred_after_repair_failure'))
            self.assertEqual(attempt(ctx,edge,5),(True,'nested_triangles_replaced'))
            self.assertEqual(len(ctx[0].element),1)
            r._check_adjacency_consistency(ctx[0],ctx[1])

    def test_curved_repair_success_prevents_replacement_in_each_phase(self):
        source,meta=repairable_curved_patch()
        for i in (0,5):
            ctx=context(source,length=meta['target'])
            with patch.object(r,'_replace_nested_triangles',side_effect=AssertionError('replacement attempted after repair')):
                self.assertEqual(attempt(ctx,meta['edge'],i,False),(True,'repaired_collapsed'))
            self.assertFalse(ctx[1]['replacements'])
            r._check_adjacency_consistency(ctx[0],ctx[1])

    def test_nested_chain_and_exterior_rewiring(self):
        source,meta=flat_nest(100)
        ctx=context(source,length=meta['target'],boundary=False)
        w,disk,other=nests(ctx,(0,1))[0]
        self.assertEqual(w,meta['cycle'][2])
        self.assertEqual(disk['faces'],meta['expected_enclosed'])
        self.assertEqual(len(disk['faces']),199)
        m,t,*_=ctx
        self.assertEqual(attempt(ctx,(0,1),5,False),(True,'ordinary_collapsed'))
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
        save_mesh_pair(source,output,output_dir()/'features/selected_edge_nested_100')

    def test_feature_points_keep_the_nest(self):
        source,meta=flat_nest(100)
        ctx=context(source,length=meta['target'],boundary=False,points=meta['feature_points'])
        m,t,*_=ctx
        before=m.element.clone()
        self.assertFalse(attempt(ctx,(0,1),5,False)[0])
        self.assertTrue(torch.equal(m.element,before))
        self.assertTrue(all(t['vertex_to_faces'][v] for v in meta['feature_points']))

    def test_orientation_and_numbering_independence(self):
        source,meta=rotated_example(flat_nest,8)
        rng=np.random.default_rng(813)
        permutation=rng.permutation(len(source.node))
        inverse=np.argsort(permutation)
        face_order=rng.permutation(len(source.element))
        mesh=r.TriangleMesh(source.node[permutation],inverse[source.element.numpy()][face_order])
        cycle=tuple(int(inverse[v]) for v in meta['cycle'])
        edge=r._edge(*(int(inverse[v]) for v in meta['edge']))
        expected={i for i,old in enumerate(face_order) if old in meta['expected_enclosed']}
        ctx=context(mesh,length=meta['target'])
        found={w:disk['faces'] for w,disk,_ in nests(ctx,edge)}
        self.assertEqual(found[cycle[2]],expected)

    def test_two_sided_disk_is_never_erased(self):
        tall,meta=ambiguous_two_sided_disk()
        thin,_=ambiguous_two_sided_disk()
        thin.node[3,2]=.1;thin.node[4,2]=-.1
        for source in (tall,thin):
            ctx=context(source,length=meta['target'])
            before_x,before_f=ctx[0].node.clone(),ctx[0].element.clone()
            original=r._try_repair_collapse_patch
            with patch.object(r,'_try_repair_collapse_patch',wraps=original) as repair:
                for i in range(5):
                    self.assertEqual(attempt(ctx,meta['edge'],i,False),(False,'deferred_after_repair_failure'))
                    self.assertTrue(torch.equal(ctx[0].node,before_x))
                    self.assertTrue(torch.equal(ctx[0].element,before_f))
                self.assertFalse(attempt(ctx,meta['edge'],5,False)[0])
                self.assertEqual(repair.call_count,6)
            m,t,f,b,ref,p,tol=ctx
            self.assertEqual(len(m.element),6)
            r._check_adjacency_consistency(m,t)
            out=r._finalize_output(m,t,source,ref,p,f,b,tol)
            self.assertEqual(tuple(out.element.shape),(6,3))
            self.assertFalse(t['replacements'])

    def test_flat_and_curved_pair_is_left_alone(self):
        source,meta=curved_disk(exterior=False)
        source.element=torch.cat((source.element,torch.tensor([[1,0,2]])))
        ctx=context(source,length=meta['target'])
        self.assertEqual(attempt(ctx,meta['edge'],0,False),(False,'deferred_after_repair_failure'))
        self.assertFalse(attempt(ctx,meta['edge'],5,False)[0])
        self.assertEqual(len(ctx[0].element),4)

    def test_two_sided_disk_and_unrelated_component_are_kept(self):
        source,meta=ambiguous_two_sided_disk()
        source.node=torch.cat((source.node,torch.tensor([[10.,0,0],[11.,0,0],[10.,1,0]],dtype=torch.float64)))
        source.element=torch.cat((source.element,torch.tensor([[5,6,7]])))
        ctx=context(source,length=meta['target'])
        self.assertFalse(attempt(ctx,meta['edge'],5,False)[0])
        m,t,f,b,ref,p,tol=ctx
        out=r._finalize_output(m,t,source,ref,p,f,b,tol)
        self.assertEqual(len(out.element),7)
        self.assertTrue(torch.equal(out.node,source.node))

    def test_two_sided_disk_keeps_feature_edge(self):
        source,meta=ambiguous_two_sided_disk()
        ctx=context(source,length=meta['target'],curves=[[2,3]])
        self.assertFalse(attempt(ctx,meta['edge'],5,False)[0])
        self.assertEqual(ctx[1]['revision'],0)
        self.assertTrue(torch.equal(ctx[0].element,source.element))

    def test_full_public_remesher_never_erases_a_two_sided_disk(self):
        source,meta=ambiguous_two_sided_disk()
        for max_iter in (1,10):
            out=r.isotropic_triangle_surface_remesh(source,meta['target'],False,max_iter=max_iter)
            self.assertGreater(len(out.element),0)
            self.assertFalse(out.mesh_data['remeshing_report']['nested_replacements'])

    def test_three_sheets_and_incoherent_pair_are_left_alone(self):
        source,meta=ambiguous_two_sided_disk()
        source.node=torch.cat((source.node,torch.tensor([[.2,1.,2.]],dtype=torch.float64)))
        source.element=torch.cat((source.element,torch.tensor([[0,1,5],[1,2,5],[2,0,5]])))
        ctx=context(source,length=meta['target'])
        before=ctx[0].element.clone()
        self.assertFalse(attempt(ctx,meta['edge'],5,False)[0])
        self.assertTrue(torch.equal(ctx[0].element,before))
        source,meta=ambiguous_two_sided_disk()
        source.element[3:]=source.element[3:].flip(1)
        ctx=context(source,length=meta['target'])
        before=ctx[0].element.clone()
        self.assertFalse(attempt(ctx,meta['edge'],5,False)[0])
        self.assertTrue(torch.equal(ctx[0].element,before))

    def test_hole_and_internal_branch_are_not_guessed(self):
        source,meta=curved_disk_with_hole()
        ctx=context(source,length=meta['target'])
        self.assertEqual(nests(ctx,meta['edge']),[])
        source,meta=flat_nest(8)
        new=len(source.node)
        source.node=torch.cat((source.node,torch.tensor([[2.,.7,-4.]],dtype=torch.float64)))
        source.element=torch.cat((source.element,torch.tensor([[0,4,new]])))
        m,t,f,b,ref,p,tol=context(source,length=meta['target'])
        disk,status=r._search_nest_side(t,(0,1),0,meta['cycle'][2],p['repair_max_faces'])
        self.assertIsNone(disk)
        self.assertEqual(status,'not_disk')

    def test_full_patch_feature_protection(self):
        source,meta=curved_disk()
        ctx=context(source,length=meta['target'],curves=[[2,3]])
        before=ctx[0].element.clone()
        accepted,_=attempt(ctx,meta['edge'],5)
        self.assertFalse(accepted)
        self.assertTrue(torch.equal(ctx[0].element,before))
        self.assertEqual(ctx[1]['revision'],0)

    def test_full_public_remesher_protected_nest_stays(self):
        source,meta=flat_nest(100)
        before_x,before_f=source.node.clone(),source.element.clone()
        output=r.isotropic_triangle_surface_remesh(source,meta['target'],False,max_iter=10,
            feature_point_index_list=meta['feature_points'])
        self.assertFalse(output.mesh_data['remeshing_report']['nested_replacements'])
        self.assertEqual(len(output.element),len(source.element))
        self.assertEqual(len(set(output.element.reshape(-1).tolist())),len(output.node))
        self.assertTrue(torch.equal(source.node,before_x));self.assertTrue(torch.equal(source.element,before_f))
        save_mesh_pair(source,output,output_dir()/'features/nested_curved_100',
                       feature_point_index_list=meta['feature_points'])

    def test_projection_disabled_on_full_run(self):
        source,meta=flat_nest(8)
        with patch.object(r,'_project_point',side_effect=AssertionError('projection disabled')):
            output=r.isotropic_triangle_surface_remesh(source,meta['target'],False,max_iter=10,
                flag_project_to_source_mesh=False)
        self.assertGreater(len(output.element),0)
        save_mesh_pair(source,output,output_dir()/'features/nested_curved_8_no_projection')

    def test_unprotected_nested_vertex_compaction(self):
        source,meta=flat_nest(100)
        output=r.isotropic_triangle_surface_remesh(source,meta['target'],False,max_iter=10)
        self.assertEqual(len(set(output.element.reshape(-1).tolist())),len(output.node))
        self.assertEqual(len(output.node),5)
        self.assertEqual(len(output.element),4)
