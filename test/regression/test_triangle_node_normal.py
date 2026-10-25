import pytest

pytestmark = [pytest.mark.geometry]

"""Node-normal forward and gradient checks for float32/float64 meshes."""
import unittest

import torch

from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh


def folded_patch(parameter, separation):
    zero=parameter*0
    nodes=torch.stack((torch.stack((zero,zero,zero)),
                       torch.stack((zero,zero,zero+1)),
                       torch.stack((zero,zero+1,zero)),
                       torch.stack((parameter*separation,zero+1,zero))))
    return nodes,torch.tensor([[0,1,2],[1,0,3]])


def folded_normal_value(parameter, separation, weighted, cached):
    nodes,faces=folded_patch(parameter,separation)
    normals=TriangleMesh.cal_element_normal(nodes,faces) if cached else None
    return TriangleMesh.cal_node_normal(nodes,faces,weighted,normals)[0,:2].sum()


class TriangleNodeNormalTest(unittest.TestCase):
    def test_gradient_with_and_without_normal_cancellation(self):
        for dtype in (torch.float32,torch.float64):
            for weighted in (False,True):
                for cached in (False,True):
                    for separation in (.2,5e-13):
                        with self.subTest(dtype=dtype,weighted=weighted,cached=cached,
                                          separation=separation):
                            parameter=torch.tensor(1.,dtype=dtype,requires_grad=True)
                            value=folded_normal_value(parameter,separation,weighted,cached)
                            value.backward()
                            # Use float64 for finite differences: subtracting
                            # nearby float32 outputs amplifies rounding error.
                            step=1e-5
                            values=[folded_normal_value(torch.tensor(t,dtype=torch.float64),
                                    separation,weighted,cached) for t in (1-step,1+step)]
                            finite_difference=((values[1]-values[0])/(2*step)).to(dtype)
                            torch.testing.assert_close(parameter.grad,finite_difference,
                                rtol=2e-3 if dtype==torch.float32 else 1e-8,atol=0)

    def test_planar_normal_orientation_isolated_node_and_ownership(self):
        for dtype in (torch.float32,torch.float64):
            nodes=torch.tensor([[0.,0,0],[1.,0,0],[1.,1,0],[0.,1,0],[2.,2,2]],dtype=dtype)
            original=nodes.clone()
            for weighted in (False,True):
                for reverse in (False,True):
                    with self.subTest(dtype=dtype,weighted=weighted,reverse=reverse):
                        faces=torch.tensor([[0,1,2],[0,2,3]])
                        if reverse:
                            faces=faces.flip(1)
                        element_normals=TriangleMesh.cal_element_normal(nodes,faces)
                        saved=element_normals.clone()
                        result=TriangleMesh.cal_node_normal(nodes,faces,weighted,element_normals)
                        expected=torch.zeros_like(nodes)
                        expected[:4,2]=-1. if reverse else 1.
                        torch.testing.assert_close(result,expected,rtol=0,atol=0)
                        self.assertTrue(result.is_contiguous())
                        self.assertTrue(torch.equal(nodes,original))
                        self.assertTrue(torch.equal(element_normals,saved))

    def test_exact_cancellation_has_finite_forward_and_backward(self):
        for dtype in (torch.float32,torch.float64):
            for weighted in (False,True):
                with self.subTest(dtype=dtype,weighted=weighted):
                    nodes,faces=folded_patch(torch.tensor(1.,dtype=dtype),0.)
                    nodes.requires_grad_()
                    result=TriangleMesh.cal_node_normal(nodes,faces,weighted)
                    self.assertTrue(torch.equal(result[:2],torch.zeros((2,3),dtype=dtype)))
                    self.assertTrue(torch.isfinite(result).all())
                    result.sum().backward()
                    self.assertTrue(torch.isfinite(nodes.grad).all())

    def test_empty_connectivity(self):
        for dtype in (torch.float32,torch.float64):
            for weighted in (False,True):
                for count in (0,3):
                    with self.subTest(dtype=dtype,weighted=weighted,node_count=count):
                        nodes=torch.zeros((count,3),dtype=dtype)
                        faces=torch.empty((0,3),dtype=torch.int64)
                        result=TriangleMesh.cal_node_normal(nodes,faces,weighted)
                        self.assertEqual(result.shape,(count,3))
                        self.assertTrue(torch.equal(result,torch.zeros_like(nodes)))
