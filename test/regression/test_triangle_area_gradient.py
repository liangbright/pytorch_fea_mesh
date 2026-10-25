import pytest

pytestmark = [pytest.mark.geometry]

"""Area and clamped-normal derivatives for supported float32/float64 meshes."""
import unittest

import torch

from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh


def geometry(parameter, scale, tilt=0.):
    zero=parameter*0
    return torch.stack((torch.stack((zero,zero,zero)),
                        torch.stack((zero+scale,zero,zero)),
                        torch.stack((zero,parameter*scale,zero+tilt*scale))))


class TriangleAreaGradientTest(unittest.TestCase):
    def test_area_gradient_matches_analytic_and_finite_difference(self):
        faces=torch.tensor([[0,1,2]])
        for dtype in (torch.float32,torch.float64):
            for scale in (1.,1e-5,5e-7,1e-8):
                with self.subTest(dtype=dtype,scale=scale):
                    parameter=torch.tensor(1.,dtype=dtype,requires_grad=True)
                    area,_=TriangleMesh.cal_element_area_and_normal(geometry(parameter,scale),faces)
                    area.sum().backward()
                    expected=torch.tensor(.5*scale*scale,dtype=dtype)
                    tolerance=1e-6 if dtype==torch.float32 else 1e-12
                    torch.testing.assert_close(area.squeeze(),expected,rtol=tolerance,atol=0)
                    torch.testing.assert_close(parameter.grad,expected,rtol=tolerance,atol=0)
                    step=1e-3 if dtype==torch.float32 else 1e-5
                    values=[TriangleMesh.cal_element_area_and_normal(
                        geometry(torch.tensor(t,dtype=dtype),scale),faces)[0].sum()
                        for t in (1-step,1+step)]
                    finite_difference=(values[1]-values[0])/(2*step)
                    torch.testing.assert_close(parameter.grad,finite_difference,
                                               rtol=1e-3 if dtype==torch.float32 else 1e-8,atol=0)

    def test_normal_gradient_matches_finite_difference_for_each_clamp_branch(self):
        faces=torch.tensor([[0,1,2]])
        for dtype in (torch.float32,torch.float64):
            for scale in (1e-5,5e-7):
                with self.subTest(dtype=dtype,scale=scale):
                    parameter=torch.tensor(1.,dtype=dtype,requires_grad=True)
                    _,normal=TriangleMesh.cal_element_area_and_normal(geometry(parameter,scale,.3),faces)
                    normal[0,2].backward()
                    step=1e-3 if dtype==torch.float32 else 1e-5
                    values=[TriangleMesh.cal_element_area_and_normal(
                        geometry(torch.tensor(t,dtype=dtype),scale,.3),faces)[1][0,2]
                        for t in (1-step,1+step)]
                    finite_difference=(values[1]-values[0])/(2*step)
                    torch.testing.assert_close(parameter.grad,finite_difference,
                                               rtol=1e-3 if dtype==torch.float32 else 1e-8,atol=0)

    def test_forward_clamp_orientation_and_source_immutability(self):
        for dtype in (torch.float32,torch.float64):
            for scale in (1.,1e-8):
                with self.subTest(dtype=dtype,scale=scale):
                    nodes=torch.tensor([[0.,0,0],[scale,0,0],[0,scale,0]],dtype=dtype)
                    original=nodes.clone()
                    faces=torch.tensor([[0,1,2],[0,2,1]])
                    area,normal=TriangleMesh.cal_element_area_and_normal(nodes,faces)
                    tolerance=1e-6 if dtype==torch.float32 else 1e-12
                    expected_area=torch.full((2,1),.5*scale*scale,dtype=dtype)
                    magnitude=scale*scale/max(scale*scale,1e-12)
                    expected_normal=torch.tensor([[0.,0,magnitude],[0.,0,-magnitude]],dtype=dtype)
                    torch.testing.assert_close(area,expected_area,rtol=tolerance,atol=0)
                    torch.testing.assert_close(normal,expected_normal,rtol=tolerance,atol=0)
                    self.assertTrue(torch.equal(nodes,original))

    def test_finite_geometry_for_degenerate_triangle_and_empty_mesh(self):
        for dtype in (torch.float32,torch.float64):
            for coordinates in ([[0.,0,0],[0.,0,0],[0.,0,0]],
                                [[0.,0,0],[1.,0,0],[2.,0,0]]):
                with self.subTest(dtype=dtype,coordinates=coordinates):
                    nodes=torch.tensor(coordinates,dtype=dtype,requires_grad=True)
                    area,normal=TriangleMesh.cal_element_area_and_normal(nodes,torch.tensor([[0,1,2]]))
                    self.assertTrue(torch.equal(area,torch.zeros_like(area)))
                    self.assertTrue(torch.equal(normal,torch.zeros_like(normal)))
                    (area.sum()+normal.sum()).backward()
                    self.assertTrue(torch.isfinite(nodes.grad).all())
            area,normal=TriangleMesh.cal_element_area_and_normal(
                torch.empty((0,3),dtype=dtype),torch.empty((0,3),dtype=torch.int64))
            self.assertEqual(area.shape,(0,1))
            self.assertEqual(normal.shape,(0,3))
