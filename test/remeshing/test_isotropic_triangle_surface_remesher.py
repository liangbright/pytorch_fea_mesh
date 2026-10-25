import pytest

pytestmark = [pytest.mark.remeshing, pytest.mark.processing, pytest.mark.vtk]
pytest.importorskip("vtk", reason="VTK is required for this test module")

from test.helper.output import output_dir
"""Remesher regression tests against the installed torch_fea_mesh package."""
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import torch_fea_mesh.processing.isotropic_triangle_surface_remesher as r
from test.helper.mesh_test_io import save_mesh_pair


def figure(dtype=torch.float64):
    return r.TriangleMesh(torch.tensor([[0,0,0],[4,0,0],[1.5,4,0],[1.8,1.5,0]],dtype=dtype),
                          [[0,1,3],[1,2,3],[2,0,3]])


def grid(n=4, dtype=torch.float64, curved=False):
    node = [[i, j, .1*(i*i+j*j) if curved else 0] for j in range(n) for i in range(n)]
    faces = []
    for j in range(n-1):
        for i in range(n-1):
            a=j*n+i
            faces.extend([(a,a+1,a+n+1),(a,a+n+1,a+n)])
    return r.TriangleMesh(torch.tensor(node,dtype=dtype),faces)


def context(mesh, length=10., boundary=True, points=None, edges=None, curves=None, max_iter=10, project=True):
    working=r._copy_source_mesh(mesh)
    t=r._initialize_topology(working)
    p=r._make_parameter(length,1.4,.7,max_iter,boundary,project)
    tol=r._compute_tolerance(working,length)
    f=r._initialize_feature_constraint(mesh,working,t,points,edges,curves)
    ref,b=r._prepare_initial_mesh(working,t,f,p,tol)
    return working,t,f,b,ref,p,tol


def attempt(ctx, edge, iteration, boundary_only=True):
    m,t,f,b,ref,p,tol=ctx
    return r._attempt_collapse(m,t,edge,f,b,ref,p,tol,{},boundary_only=boundary_only,
                                iteration_index=iteration)


class RemesherTest(unittest.TestCase):
    def test_phase_policy(self):
        for total, allowed in [(0,[]),(1,[]),(3,[2]),(9,[5,6,7,8]),(10,[5,6,7,8,9])]:
            self.assertEqual([i for i in range(total) if r._deletion_phase_allowed(i,total)],allowed)
            self.assertFalse(r._deletion_phase_allowed(-1,total))
            self.assertFalse(r._deletion_phase_allowed(total,total))

    def test_early_defer_late_delete_each_outer_edge(self):
        for edge in [(0,1),(1,2),(0,2)]:
            ctx=context(figure())
            m,t,f,b,ref,p,tol=ctx
            oldx,oldf=m.node.clone(),m.element.clone()
            with patch.object(r,'_try_repair_collapse_patch',return_value=(None,'controlled_failure')) as repair:
                for i in range(5):
                    self.assertEqual(attempt(ctx,edge,i),(False,'deferred_after_repair_failure'))
                    self.assertTrue(torch.equal(m.node,oldx));self.assertTrue(torch.equal(m.element,oldf))
                    self.assertEqual(t['revision'],0)
                self.assertEqual(attempt(ctx,edge,5),(True,'enclosed_region_collapsed'))
                self.assertEqual(repair.call_count,6)
            self.assertEqual(len(m.element),0)
            r._check_adjacency_consistency(m,t)
            out=r._finalize_output(m,t,figure(),ref,p,f,b,tol)
            self.assertEqual(tuple(out.node.shape),(0,3));self.assertEqual(tuple(out.element.shape),(0,3))

    def test_real_failed_repair_then_deletion(self):
        ctx=context(figure())
        ctx[5]['repair_distance']=1e-8
        self.assertEqual(attempt(ctx,(0,1),0),(False,'deferred_after_repair_failure'))
        self.assertEqual(attempt(ctx,(0,1),5),(True,'enclosed_region_collapsed'))

    def test_real_repair_succeeds_without_deletion(self):
        source=r.TriangleMesh(torch.tensor([[0,0,0],[4,0,0],[4,4,0],[0,4,0],
            [1.95,2,0],[2.05,2,0],[2.3,3,0],[2.11,2.3,0]],dtype=torch.float64),
            [(0,1,5),(0,5,4),(0,4,3),(3,4,6),(3,6,2),(2,6,5),(2,5,1),
             (4,5,7),(5,6,7),(6,4,7)])
        for iteration in (0,5):
            ctx=context(source,length=.2)
            accepted,reason=attempt(ctx,(4,5),iteration,boundary_only=False)
            self.assertTrue(accepted,reason)
            self.assertEqual(reason,'repaired_collapsed')
            self.assertEqual(ctx[1]['counts']['repair_collapse'],1)
            self.assertFalse(ctx[1]['deletions'])
            r._check_adjacency_consistency(ctx[0],ctx[1])

    def test_complete_large_incidence_and_nested_region(self):
        source=r.TriangleMesh(torch.tensor([[0,0,0],[1,0,0]]+[[.5,i+1,0] for i in range(100)],dtype=torch.float64),
                              [[0,1,i+2] for i in range(100)])
        ctx=context(source,boundary=False)
        m,t,f,b,ref,p,tol=ctx
        self.assertEqual(len(t['edge_to_faces'][(0,1)]),100)
        regions=[]
        for cycle in r._find_enclosing_cycle(m,t,(0,1),tol):
            region,_=r._collect_enclosed_face(m,t,cycle,tol)
            if region:regions.append(region)
        chosen,reason=r._select_outermost_enclosed_region(regions,(0,1))
        self.assertEqual(reason,'ok');self.assertEqual(len(chosen),1)
        self.assertEqual(len(chosen[0]['faces']),100)
        out=r.isotropic_triangle_surface_remesh(source,1.,False,max_iter=0)
        self.assertEqual(len(out.element),100)

    def test_nonincident_interior_and_exterior_preserved(self):
        source=figure()
        source.node=torch.cat((source.node,torch.tensor([[5,-2,0]],dtype=torch.float64)))
        source.element=torch.cat((source.element,torch.tensor([[1,4,2]])))
        ctx=context(source)
        m,t,f,b,ref,p,tol=ctx
        cycle={'vertices':(0,1,2),'edge':(0,1),'revision':0}
        region,why=r._collect_enclosed_face(m,t,cycle,tol)
        self.assertIsNotNone(region,why)
        self.assertEqual(region['faces'],{0,1,2})
        # One selected interior face does not contain the collapsing edge.
        self.assertGreater(len(region['faces']),len(t['edge_to_faces'][(0,1)]))

    def test_protected_edge_blocks_indirect_region_erasure(self):
        ctx=context(figure(),curves=[[1,3]])
        with patch.object(r,'_try_repair_collapse_patch',return_value=(None,'controlled_failure')):
            accepted,reason=attempt(ctx,(0,2),5)
        self.assertFalse(accepted)
        self.assertEqual(len(ctx[0].element),3)
        self.assertEqual(ctx[1]['revision'],0)

    def test_protected_isolated_point_survives_deletion(self):
        source=figure()
        ctx=context(source,points=[3])
        with patch.object(r,'_try_repair_collapse_patch',return_value=(None,'controlled_failure')):
            self.assertTrue(attempt(ctx,(0,1),5)[0])
        m,t,f,b,ref,p,tol=ctx
        out=r._finalize_output(m,t,source,ref,p,f,b,tol)
        self.assertEqual(len(out.element),0);self.assertEqual(len(out.node),1)
        self.assertTrue(torch.equal(out.node[0],source.node[3]))
        save_mesh_pair(source,out,output_dir()/'features/isolated_point',
                       feature_point_index_list=[3])

    def test_feature_constraint_source_edge_order_and_curve_union(self):
        source=grid()
        source.build_edge()
        source.edge=source.edge.flip(0).clone() # Explicit caller table order is authoritative.
        chosen=tuple(source.edge[0].tolist())
        original=source.node.clone();original_edge=source.edge.clone()
        out=r.isotropic_triangle_surface_remesh(source,.6,True,max_iter=2,
            feature_edge_index_list=[0,0],feature_point_index_list=[5],feature_curve_list=[[0,1,2],[],[15]])
        out.build_edge()
        mapping={tuple(p):i for i,p in enumerate(out.node.tolist())}
        for v in {0,1,2,5,*chosen}:
            self.assertIn(tuple(original[v].tolist()),mapping)
        output_edges={tuple(e) for e in out.edge.tolist()}
        for a,b in [chosen,(0,1),(1,2)]:
            self.assertIn(r._edge(mapping[tuple(original[a].tolist())],mapping[tuple(original[b].tolist())]),output_edges)
        self.assertTrue(torch.equal(source.node,original));self.assertTrue(torch.equal(source.edge,original_edge))
        save_mesh_pair(source,out,output_dir()/'features/source_edge_order',
                       feature_edge_index_list=[0,0],feature_point_index_list=[5])

    def test_empty_curve_does_not_protect_single_vertex(self):
        ctx=context(grid(),curves=[[],[5]])
        self.assertFalse(ctx[2]['points']);self.assertFalse(ctx[2]['edges'])
        with self.assertRaises(ValueError):
            r.isotropic_triangle_surface_remesh(grid(),1.,feature_curve_list=[[0,15]])

    def test_frozen_boundary_and_dtype(self):
        for dtype in (torch.float32,torch.float64):
            source=grid(dtype=dtype)
            original=source.node.clone();faces=source.element.clone()
            source.build_edge();ids,_=source.find_boundary_node_and_edge()
            boundary_points={tuple(original[v].tolist()) for v in ids}
            out=r.isotropic_triangle_surface_remesh(source,.7,False,max_iter=2)
            self.assertEqual(out.node.dtype,dtype)
            self.assertTrue(boundary_points <= {tuple(p) for p in out.node.tolist()})
            self.assertTrue(torch.equal(source.node,original));self.assertTrue(torch.equal(source.element,faces))
            self.assertNotEqual(out.node.data_ptr(),source.node.data_ptr())

    def test_projection_false_never_repositions_through_locator(self):
        with patch.object(r,'_project_point',side_effect=AssertionError('projection called')):
            out=r.isotropic_triangle_surface_remesh(grid(curved=True),1.,True,max_iter=2,
                                                    flag_project_to_source_mesh=False)
        self.assertGreater(len(out.element),0)

    def test_enabled_projection_is_used(self):
        original=r._project_point
        with patch.object(r,'_project_point',wraps=original) as project:
            r.isotropic_triangle_surface_remesh(grid(curved=True),1.,True,max_iter=1)
            self.assertGreater(project.call_count,0)

    def test_incremental_adjacency_after_every_edit(self):
        commit=r._commit_edit
        kinds=set()
        def checked(m,t,edit,f,b):
            commit(m,t,edit,f,b)
            kinds.add(edit['kind'])
            r._check_adjacency_consistency(m,t)
        with patch.object(r,'_commit_edit',side_effect=checked):
            r.isotropic_triangle_surface_remesh(grid(),.65,True,max_iter=2)
        self.assertTrue({'split','collapse','flip','move'} <= kinds,kinds)

    def test_normal_iteration_does_not_rebuild_adjacency(self):
        original=r._initialize_topology
        with patch.object(r,'_initialize_topology',wraps=original) as initialize:
            r.isotropic_triangle_surface_remesh(grid(),.8,False,max_iter=3)
            # Initial topology and independent final casting validation only.
            self.assertEqual(initialize.call_count,2)

    def test_zero_iteration_count_and_parameter_validation(self):
        source=grid()
        out=r.isotropic_triangle_surface_remesh(source,.1,False,max_iter=0)
        self.assertEqual(out.mesh_data['remeshing_report']['iterations'],[])
        self.assertFalse(out.mesh_data['remeshing_report']['region_deletions'])
        for kwargs in [dict(max_iter=-1),dict(max_iter=True),dict(target_edge_length_ratio_min=2),
                       dict(feature_point_index_list=[100]),dict(target_edge_length_ratio_max=float('inf'))]:
            with self.assertRaises((ValueError,TypeError)):
                r.isotropic_triangle_surface_remesh(source,1.,**kwargs)
        with self.assertRaises(TypeError):
            r.isotropic_triangle_surface_remesh(source,1.,flag_process_boundary=1)

    def test_empty_input_and_invalid_geometry(self):
        empty=r._new_mesh([],[])
        out=r.isotropic_triangle_surface_remesh(empty,1.)
        self.assertEqual(tuple(out.element.shape),(0,3))
        duplicate=r.TriangleMesh(torch.tensor([[0.,0,0],[1,0,0],[0,1,0]]),[[0,1,2],[2,0,1]])
        with self.assertRaises(ValueError):r.isotropic_triangle_surface_remesh(duplicate,1.)
        degenerate=r.TriangleMesh(torch.tensor([[0.,0,0],[1,0,0],[2,0,0]]),[[0,1,2]])
        with self.assertRaises(ValueError):r.isotropic_triangle_surface_remesh(degenerate,1.)

    def test_determinism_and_nondefault_ratio(self):
        source=grid()
        args=dict(max_iter=2,target_edge_length_ratio_min=.6,target_edge_length_ratio_max=1.6)
        a=r.isotropic_triangle_surface_remesh(source,.8,False,**args)
        b=r.isotropic_triangle_surface_remesh(source,.8,False,**args)
        self.assertTrue(torch.equal(a.node,b.node));self.assertTrue(torch.equal(a.element,b.element))
        self.assertAlmostEqual(a.mesh_data['remeshing_report']['parameters']['max_edge_length'],1.28)

    def test_target_edge_length_below_minimum(self):
        source=grid()
        for target in (8e-9,1e-6,np.nextafter(1e-5,0.)):
            for iterations in (0,1):
                with self.subTest(target=target,max_iter=iterations):
                    with patch.object(r,'_copy_source_mesh',wraps=r._copy_source_mesh) as copy:
                        with self.assertRaisesRegex(ValueError,'target_edge_length must be at least 1e-5') as error:
                            r.isotropic_triangle_surface_remesh(source,target,max_iter=iterations)
                        copy.assert_not_called()
                    self.assertIn('rescale source_mesh.node and target_edge_length by the same factor',
                                  str(error.exception))

    def test_minimum_supported_target_edge_length(self):
        for dtype in (torch.float32,torch.float64):
            for target in (1e-5,np.nextafter(1e-5,np.inf)):
                with self.subTest(dtype=dtype,target=target):
                    source=grid(dtype=dtype)
                    source.node*=target
                    out=r.isotropic_triangle_surface_remesh(source,target,False,max_iter=2,
                                                            flag_project_to_source_mesh=False)
                    self.assertGreater(len(out.element),0)
                    self.assertEqual(out.node.dtype,dtype)
                    out.update_element_area_and_normal()
                    self.assertTrue(torch.all(out.element_area>0))
                    self.assertGreater(out.mesh_data['remeshing_report']['operations'].get('split',0),0)

    def test_stale_or_early_deletion_proposal_rejected(self):
        ctx=context(figure())
        m,t,f,b,ref,p,tol=ctx
        cycle={'vertices':(0,1,2),'edge':(0,1),'revision':0}
        region,_=r._collect_enclosed_face(m,t,cycle,tol)
        evidence=dict(iteration_index=5,revision=0,repair_failure='test_failure',problematic=True)
        edit=r._propose_enclosed_region_collapse(m,t,(0,1),[region],f,b,ref,p,evidence)[0]
        self.assertTrue(r._validate_edit(m,t,edit,f,b,ref,p,tol)[0])
        edit['deletion_evidence']['iteration_index']=4
        self.assertFalse(r._validate_edit(m,t,edit,f,b,ref,p,tol)[0])
        edit['deletion_evidence']['iteration_index']=5
        edit['deletion_evidence']['revision']=-1
        self.assertFalse(r._validate_edit(m,t,edit,f,b,ref,p,tol)[0])
        self.assertEqual(t['revision'],0);self.assertEqual(len(m.element),3)

    def test_final_pass_cannot_advance_phase(self):
        for total in (0,1,3,10):
            ctx=context(grid(),max_iter=total)
            m,t,f,b,ref,p,tol=ctx
            with patch.object(r,'_process_boundary'),patch.object(r,'_equalize_valence'),\
                 patch.object(r,'_split_long_edge'),patch.object(r,'_improve_triangle_angle'),\
                 patch.object(r,'_relax_vertex'),patch.object(r,'_collapse_short_edge') as collapse:
                r._run_remeshing_iteration(m,t,f,b,ref,p,tol,{})
            final=collapse.call_args_list[-1].kwargs
            self.assertTrue(final['is_final_pass'])
            self.assertEqual(final['iteration_index'],total-1)
            self.assertEqual(r._deletion_phase_allowed(final['iteration_index'],total),total>1)
