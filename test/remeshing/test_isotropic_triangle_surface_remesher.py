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
from test.helper.curved_patch_example import repairable_curved_patch
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


def boundary_coordinates(mesh):
    node,element=r._as_numpy(mesh.node),r._as_numpy(mesh.element)
    counts={}
    for a,b,c in element.tolist():
        for edge in ((a,b),(b,c),(c,a)):
            key=tuple(sorted(edge))
            counts[key]=counts.get(key,0)+1
    return {tuple(sorted(map(tuple,np.round(node[list(key)],10)+0.))) for key,count in counts.items() if count==1}


def hexagon_and_rhombus():
    """Return a mesh with two short interior edges. All its other edges are close to length 10.

    The hexagon's short edge joins its two inner vertices, so it can collapse. The rhombus's
    short edge is its diagonal, so both ends are on the boundary, which stays fixed by default.
    """
    hexagon=[[10*np.cos(a),10*np.sin(a),0] for a in np.arange(6)*np.pi/3]+[[-.5,0,0],[.5,0,0]]
    rhombus=[[39.5,0,0],[40.5,0,0],[40,10,0],[40,-10,0]]
    faces=[(6,7,1),(7,0,1),(6,1,2),(6,2,3),(6,3,4),(6,4,5),(7,6,5),(7,5,0),(8,9,10),(9,8,11)]
    return r.TriangleMesh(torch.tensor(hexagon+rhombus,dtype=torch.float64),faces)


def hexagon_with_inner_chain():
    """Return a hexagon with three inner vertices in a row, joined by two short edges.

    Collapsing one short edge removes the other and leaves a new short edge in its place.
    """
    hexagon=[[10*np.cos(a),10*np.sin(a),0] for a in np.arange(6)*np.pi/3]+[[-.5,0,0],[0,0,0],[.5,0,0]]
    faces=[(8,0,1),(7,8,1),(6,7,1),(6,1,2),(6,2,3),(6,3,4),(6,4,7),(7,4,5),(7,5,8),(8,5,0)]
    return r.TriangleMesh(torch.tensor(hexagon,dtype=torch.float64),faces)


def surface_area(mesh):
    node,element=r._as_numpy(mesh.node),r._as_numpy(mesh.element)
    corners=node[element]
    return np.linalg.norm(np.cross(corners[:,1]-corners[:,0],corners[:,2]-corners[:,0]),axis=1).sum()/2


def context(mesh, length=10., boundary=True, points=None, edges=None, curves=None, max_iter=10, project=True):
    working=r._copy_source_mesh(mesh)
    t=r._initialize_topology(working)
    p=r._make_parameter(length,1.4,.7,max_iter,boundary,project)
    tol=r._compute_tolerance(working,length)
    f=r._initialize_feature_constraint(mesh,working,t,points,edges,curves)
    ref,b=r._prepare_initial_mesh(working,t,p,tol)
    return working,t,f,b,ref,p,tol


def attempt(ctx, edge, iteration, boundary_only=True):
    m,t,f,b,ref,p,tol=ctx
    return r._attempt_collapse(m,t,edge,f,b,p,tol,boundary_only=boundary_only,
                                iteration_index=iteration)


class RemesherTest(unittest.TestCase):
    def test_phase_policy(self):
        for total, allowed in [(0,[]),(1,[]),(3,[2]),(9,[5,6,7,8]),(10,[5,6,7,8,9])]:
            self.assertEqual([i for i in range(total) if r._replacement_phase_allowed(i,total)],allowed)
            self.assertFalse(r._replacement_phase_allowed(-1,total))
            self.assertFalse(r._replacement_phase_allowed(total,total))

    def test_early_defer_late_replace_each_outer_edge(self):
        for edge in [(0,1),(1,2),(0,2)]:
            ctx=context(figure())
            m,t,f,b,ref,p,tol=ctx
            oldx,oldf=m.node.clone(),m.element.clone()
            with patch.object(r,'_try_repair_collapse_patch',return_value=(None,'controlled_failure')) as repair:
                for i in range(5):
                    self.assertEqual(attempt(ctx,edge,i),(False,'deferred_after_repair_failure'))
                    self.assertTrue(torch.equal(m.node,oldx));self.assertTrue(torch.equal(m.element,oldf))
                    self.assertEqual(t['revision'],0)
                self.assertEqual(attempt(ctx,edge,5),(True,'nested_triangles_replaced'))
                self.assertEqual(repair.call_count,6)
            self.assertEqual(len(m.element),1)
            r._check_adjacency_consistency(m,t)
            out=r._finalize_output(m,t,figure(),ref,p,f,b,tol)
            self.assertEqual(tuple(out.node.shape),(3,3));self.assertEqual(tuple(out.element.shape),(1,3))

    def test_real_failed_repair_then_replacement(self):
        ctx=context(figure())
        ctx[5]['repair_distance']=1e-8
        self.assertEqual(attempt(ctx,(0,1),0),(False,'deferred_after_repair_failure'))
        self.assertEqual(attempt(ctx,(0,1),5),(True,'nested_triangles_replaced'))

    def test_real_repair_succeeds_without_replacement(self):
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
            self.assertFalse(ctx[1]['replacements'])
            r._check_adjacency_consistency(ctx[0],ctx[1])

    def test_edge_with_more_than_two_elements_is_not_replaced(self):
        source=r.TriangleMesh(torch.tensor([[0,0,0],[1,0,0]]+[[.5,i+1,0] for i in range(100)],dtype=torch.float64),
                              [[0,1,i+2] for i in range(100)])
        ctx=context(source,boundary=False)
        m,t,f,b,ref,p,tol=ctx
        self.assertEqual(len(t['edge_to_faces'][(0,1)]),100)
        self.assertEqual(r._replace_nested_triangles(m,t,(0,1),f,b,p,tol,iteration_index=5,repair_failure='test')[0],0)
        self.assertEqual(len(m.element),100)
        out=r.isotropic_triangle_surface_remesh(source,1.,False,max_iter=0)
        self.assertEqual(len(out.element),100)

    def test_nonincident_interior_and_exterior_preserved(self):
        source=figure()
        source.node=torch.cat((source.node,torch.tensor([[5,-2,0]],dtype=torch.float64)))
        source.element=torch.cat((source.element,torch.tensor([[1,4,2]])))
        ctx=context(source)
        m,t,f,b,ref,p,tol=ctx
        current=r._classify_collapse_neighborhood(t,(0,1),f,b)
        w,disk,_=r._largest_nest(t,(0,1),0,current['common'],current['incident'],p['repair_max_faces'])
        self.assertEqual(w,2)
        self.assertEqual(disk['faces'],{0,1,2})
        # One selected interior face does not contain the collapsing edge.
        self.assertGreater(len(disk['faces']),len(t['edge_to_faces'][(0,1)]))
        self.assertEqual(r._replace_nested_triangles(m,t,(0,1),f,b,p,tol,iteration_index=5,repair_failure='test')[0],1)
        self.assertEqual({tuple(sorted(face)) for face in m.element.tolist()},{(0,1,2),(1,2,4)})

    def test_protected_edge_blocks_indirect_replacement(self):
        ctx=context(figure(),curves=[[1,3]])
        with patch.object(r,'_try_repair_collapse_patch',return_value=(None,'controlled_failure')):
            accepted,reason=attempt(ctx,(0,2),5)
        self.assertFalse(accepted)
        self.assertEqual(len(ctx[0].element),3)
        self.assertEqual(ctx[1]['revision'],0)

    def test_protected_inner_point_blocks_replacement(self):
        source=figure()
        ctx=context(source,points=[3])
        with patch.object(r,'_try_repair_collapse_patch',return_value=(None,'controlled_failure')):
            self.assertFalse(attempt(ctx,(0,1),5)[0])
        m,t,f,b,ref,p,tol=ctx
        out=r._finalize_output(m,t,source,ref,p,f,b,tol)
        self.assertEqual(len(out.element),3);self.assertEqual(len(out.node),4)
        self.assertTrue(torch.equal(out.node[3],source.node[3]))
        self.assertTrue(3 in out.element)
        save_mesh_pair(source,out,output_dir()/'features/protected_inner_point',
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
        def checked(m,t,edit,b):
            commit(m,t,edit,b)
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

    def test_boundary_processing_is_off_by_default(self):
        source=grid(7,curved=True)
        default=r.isotropic_triangle_surface_remesh(source,1.5,max_iter=5)
        frozen=r.isotropic_triangle_surface_remesh(source,1.5,False,max_iter=5)
        processed=r.isotropic_triangle_surface_remesh(source,1.5,True,max_iter=5)
        self.assertIs(default.mesh_data['remeshing_report']['parameters']['flag_process_boundary'],False)
        self.assertTrue(torch.equal(default.node,frozen.node))
        self.assertTrue(torch.equal(default.element,frozen.element))
        self.assertEqual(boundary_coordinates(default),boundary_coordinates(source))
        self.assertNotEqual(boundary_coordinates(processed),boundary_coordinates(source))

    def test_failures_name_the_short_edge_left_and_why(self):
        # With no iterations only the final pass runs: the hexagon's edge collapses, the rhombus's can't.
        out=r.isotropic_triangle_surface_remesh(hexagon_and_rhombus(),10.,max_iter=0)
        report=out.mesh_data['remeshing_report']
        failures=report['failures']
        self.assertNotIn('collapse_outcomes',report)
        self.assertAlmostEqual(failures.pop('area_ratio'),1.)
        self.assertEqual(failures,dict(error=None,short_edges=1,short_edge_reasons={'protected_endpoints':1},long_edges=0))
        self.assertEqual(report['operations'],{'collapse':1})
        self.assertEqual(len(out.element),8)

    def test_failures_are_empty_when_every_edge_meets_the_limits(self):
        failures=r.isotropic_triangle_surface_remesh(grid(),1.1,max_iter=2).mesh_data['remeshing_report']['failures']
        self.assertAlmostEqual(failures.pop('area_ratio'),1.)
        self.assertEqual(failures,dict(error=None,short_edges=0,short_edge_reasons={},long_edges=0))

    def test_failures_call_a_fixed_boundary_edge_not_tried(self):
        source=r.TriangleMesh(torch.tensor([[0,0,0],[.1,0,0],[0,5,0]],dtype=torch.float64),[[0,1,2]])
        failures=r.isotropic_triangle_surface_remesh(source,5.).mesh_data['remeshing_report']['failures']
        self.assertEqual(failures['short_edge_reasons'],{'not_tried':1})

    def test_failures_keep_the_last_real_try_on_an_edge(self):
        ctx=context(hexagon_and_rhombus())
        m,t,f,b,ref,p,tol=ctx
        results=[(False,'neighbor_length'),(False,'not_short'),(False,'edge_removed'),(False,'classification_changed')]
        for result in results:
            with patch.object(r,'_collect_candidate_edge',return_value=[(8,9)]),                 patch.object(r,'_attempt_collapse',return_value=result):
                r._collapse_short_edge(m,t,f,b,p,tol,boundary_only=False,iteration_index=0)
        self.assertEqual(t['collapse_failures'],{(8,9):'neighbor_length'})

    def test_failures_match_the_output_mesh(self):
        source,meta=repairable_curved_patch()
        out=r.isotropic_triangle_surface_remesh(source,meta['target'],max_iter=2)
        failures=out.mesh_data['remeshing_report']['failures']
        node,element=r._as_numpy(out.node),r._as_numpy(out.element)
        edges=np.array(sorted({tuple(sorted(e)) for a,b,c in element.tolist() for e in ((a,b),(b,c),(c,a))}))
        lengths=np.linalg.norm(node[edges[:,0]]-node[edges[:,1]],axis=1)/meta['target']
        self.assertGreater(failures['short_edges'],0)
        self.assertEqual(failures['short_edges'],int((lengths<=.7).sum()))
        self.assertEqual(failures['long_edges'],int((lengths>1.4).sum()))
        self.assertEqual(sum(failures['short_edge_reasons'].values()),failures['short_edges'])
        self.assertAlmostEqual(failures['area_ratio'],surface_area(out)/surface_area(source))

    def test_failures_show_lost_area(self):
        out=r.isotropic_triangle_surface_remesh(grid(),5.,True,max_iter=5)
        self.assertAlmostEqual(out.mesh_data['remeshing_report']['failures']['area_ratio'],surface_area(out)/9.)
        self.assertLess(out.mesh_data['remeshing_report']['failures']['area_ratio'],.1)

    def test_refused_replacement_reaches_the_report(self):
        source,meta=repairable_curved_patch()
        with patch.object(r,'_element_crosses_surface',return_value=True):
            out=r.isotropic_triangle_surface_remesh(source,meta['target'],max_iter=2)
        reasons=out.mesh_data['remeshing_report']['failures']['short_edge_reasons']
        self.assertTrue(any(reason.endswith(', element_crosses_surface') for reason in reasons),reasons)

    def test_error_during_the_run_carries_the_report(self):
        with patch.object(r,'_validate_result',side_effect=RuntimeError('broken output')):
            with self.assertRaisesRegex(RuntimeError,'^broken output$') as caught:
                r.isotropic_triangle_surface_remesh(hexagon_and_rhombus(),10.,max_iter=2)
        report=caught.exception.diagnostics
        self.assertEqual(report['failures']['error'],'RuntimeError: broken output')
        self.assertEqual(report['failures']['short_edges'],1)
        self.assertEqual(report['failures']['short_edge_reasons'],{'protected_endpoints':1})
        self.assertEqual(report['operations']['collapse'],1)
        self.assertEqual(len(report['iterations']),2)
        self.assertIsNone(report['output_faces'])
        self.assertIsNone(report['output_nodes'])
        self.assertEqual(report['input_faces'],10)

    def test_error_from_the_mesh_carries_the_report(self):
        source=r.TriangleMesh(torch.tensor([[0,0,0],[1,0,0],[0,1,0]],dtype=torch.float64),[[0,1,2],[0,1,2]])
        with self.assertRaisesRegex(ValueError,'duplicate triangle faces') as caught:
            r.isotropic_triangle_surface_remesh(source,1.)
        self.assertEqual(caught.exception.diagnostics['failures']['error'],'ValueError: duplicate triangle faces')

    def test_error_in_an_argument_has_no_report(self):
        with self.assertRaises(ValueError) as caught:
            r.isotropic_triangle_surface_remesh(grid(),1.,max_iter=-1)
        self.assertFalse(hasattr(caught.exception,'diagnostics'))

    def test_zero_iteration_count_and_parameter_validation(self):
        source=grid()
        out=r.isotropic_triangle_surface_remesh(source,.1,False,max_iter=0)
        self.assertEqual(out.mesh_data['remeshing_report']['iterations'],[])
        self.assertFalse(out.mesh_data['remeshing_report']['nested_replacements'])
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

    def test_stale_or_early_replacement_proposal_rejected(self):
        ctx=context(figure())
        m,t,f,b,ref,p,tol=ctx
        current=r._classify_collapse_neighborhood(t,(0,1),f,b)
        w,disk,_=r._largest_nest(t,(0,1),0,current['common'],current['incident'],p['repair_max_faces'])
        edit=r._propose_nested_replacement(t,(0,1),0,w,disk,iteration_index=5,repair_failure='test_failure')
        self.assertTrue(r._validate_edit(m,t,edit,f,b,p,tol)[0])
        edit['iteration_index']=4
        self.assertEqual(r._validate_edit(m,t,edit,f,b,p,tol),(False,'replacement_not_permitted'))
        edit['iteration_index']=5
        edit['repair_failure']=None
        self.assertEqual(r._validate_edit(m,t,edit,f,b,p,tol),(False,'replacement_not_permitted'))
        edit['repair_failure']='test_failure'
        edit['revision']=-1
        self.assertEqual(r._validate_edit(m,t,edit,f,b,p,tol),(False,'stale_edit'))
        self.assertEqual(t['revision'],0);self.assertEqual(len(m.element),3)

    def test_final_pass_tries_short_edges_that_form_during_it(self):
        out=r.isotropic_triangle_surface_remesh(hexagon_with_inner_chain(),10.,max_iter=0)
        report=out.mesh_data['remeshing_report']
        self.assertEqual(report['operations'],{'collapse':2})
        self.assertEqual(report['failures']['short_edges'],0)
        self.assertEqual(len(out.element),6)

    def test_other_passes_leave_new_short_edges_for_the_next_pass(self):
        m,t,f,b,ref,p,tol=context(hexagon_with_inner_chain())
        r._collapse_short_edge(m,t,f,b,p,tol,boundary_only=False,iteration_index=0)
        self.assertEqual(t['counts']['collapse'],1)
        self.assertEqual(sum(t['lengths'][edge]<=p['min_edge_length'] for edge in t['edge_to_faces']),1)

    def test_final_pass_only_repairs_the_edges_it_started_with(self):
        m,t,f,b,ref,p,tol=context(hexagon_with_inner_chain())
        with patch.object(r,'_attempt_collapse',wraps=r._attempt_collapse) as tries:
            r._collapse_short_edge(m,t,f,b,p,tol,boundary_only=False,iteration_index=0,include_new_edges=True)
        self.assertEqual([(call.args[2],call.kwargs['allow_repair']) for call in tries.call_args_list],
                         [((6,7),True),((7,8),True),((6,8),False)])

    def test_final_pass_gives_each_edge_one_try(self):
        m,t,f,b,ref,p,tol=context(hexagon_with_inner_chain())
        with patch.object(r,'_collect_candidate_edge',return_value=[(6,7)]),\
             patch.object(r,'_attempt_collapse',side_effect=[(False,'neighbor_length')]) as tries:
            r._collapse_short_edge(m,t,f,b,p,tol,boundary_only=False,iteration_index=0,include_new_edges=True)
        self.assertEqual(tries.call_count,1)

    def test_final_pass_cannot_advance_phase(self):
        for total in (0,1,3,10):
            ctx=context(grid(),max_iter=total)
            m,t,f,b,ref,p,tol=ctx
            with patch.object(r,'_process_boundary'),patch.object(r,'_equalize_valence'),\
                 patch.object(r,'_split_long_edge'),patch.object(r,'_improve_triangle_angle'),\
                 patch.object(r,'_relax_vertex'),patch.object(r,'_collapse_short_edge') as collapse:
                r._run_remeshing_iteration(m,t,f,b,ref,p,tol)
            self.assertEqual(collapse.call_count,total+1)
            final=collapse.call_args_list[-1].kwargs
            self.assertIs(final['include_new_edges'],True)
            self.assertFalse(any('include_new_edges' in call.kwargs for call in collapse.call_args_list[:-1]))
            self.assertEqual(final['iteration_index'],total-1)
            self.assertEqual(r._replacement_phase_allowed(final['iteration_index'],total),total>1)
