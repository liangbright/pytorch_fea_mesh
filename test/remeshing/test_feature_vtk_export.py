import pytest

pytestmark = [pytest.mark.remeshing, pytest.mark.processing, pytest.mark.vtk]
pytest.importorskip("vtk", reason="VTK is required for this test module")

from test.helper.output import output_dir
"""Feature VTK exports must match source numbering and remain inspectable."""
from pathlib import Path
import unittest

import torch
import vtk
from torch_fea_mesh.mesh.mesh import Mesh
from torch_fea_mesh.mesh.polyline_mesh import PolylineMesh
import torch_fea_mesh.processing.isotropic_triangle_surface_remesher as r
from test.helper.mesh_test_io import save_mesh_pair
from test.remeshing.test_isotropic_triangle_surface_remesher import grid


class FeatureExportTest(unittest.TestCase):
    def test_original_and_missing_edge(self):
        for reverse in (False, True):
            with self.subTest(reversed_source_table=reverse):
                source=grid(3,dtype=torch.float32,curved=True)
                reference=grid(3,dtype=torch.float32,curved=True)
                reference.build_edge()
                if reverse:
                    source.edge=reference.edge.flip(0).clone()
                    reference.edge=source.edge.clone()
                before=source.node.clone()
                source_table=None if source.edge is None else source.edge.clone()
                edge_ids=[3,0,3]
                point_ids=[4,0,4]
                output=r.isotropic_triangle_surface_remesh(source,.6,True,max_iter=2,
                    feature_point_index_list=point_ids,feature_edge_index_list=edge_ids)
                name='reversed_source_edges' if reverse else 'missing_source_edges'
                paths=save_mesh_pair(source,output,output_dir()/'features'/name,
                    feature_point_index_list=point_ids,feature_edge_index_list=edge_ids)
                lines=PolylineMesh()
                data=lines.load_from_vtk(paths['feature_edges_file'],torch.float64,return_mesh_vtk=True)
                self.assertEqual(data.GetNumberOfLines(),2)
                self.assertEqual(data.GetNumberOfPolys(),0)
                for i, source_id in enumerate([3,0]):
                    # The library's ASCII VTK writer rounds decimal coordinates.
                    self.assertTrue(torch.allclose(lines.node[lines.element[i]],
                        before[reference.edge[source_id]].to(torch.float64),rtol=1e-10,atol=1e-10))
                    self.assertEqual(data.GetCellData().GetArray('source_edge_index').GetTuple1(i),source_id)
                points=Mesh.load_vtk_file(paths['feature_points_file'],'polyline')
                self.assertEqual(points.GetNumberOfPoints(),2)
                self.assertEqual(points.GetNumberOfVerts(),2)
                for i, source_id in enumerate([4,0]):
                    self.assertEqual(points.GetPoint(i),tuple(before[source_id].tolist()))
                    self.assertEqual(points.GetCellType(i),vtk.VTK_VERTEX)
                    self.assertEqual(points.GetPointData().GetArray('source_point_index').GetTuple1(i),source_id)
                self.assertTrue(torch.equal(source.node,before))
                if source_table is None:
                    self.assertIsNone(source.edge)
                else:
                    self.assertTrue(torch.equal(source.edge,source_table))

    def test_explicit_empty_feature_index_list(self):
        source=grid(3)
        output=r.isotropic_triangle_surface_remesh(source,.8,False,max_iter=1,
            feature_point_index_list=[],feature_edge_index_list=[])
        paths=save_mesh_pair(source,output,output_dir()/'features/empty_selections',
            feature_point_index_list=[],feature_edge_index_list=[])
        for key in ('feature_points_file','feature_edges_file'):
            data=Mesh.load_vtk_file(paths[key],'polyline')
            self.assertEqual(data.GetNumberOfPoints(),0)
            self.assertEqual(data.GetNumberOfCells(),0)
