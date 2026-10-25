import pytest

pytestmark = [pytest.mark.geometry]

"""Clamped-angle derivatives, forward behavior and compatibility checks."""
import contextlib
import io
import unittest

import numpy as np
import torch

from torch_fea_mesh.utility.geometry_utility import cal_angle_between_3d_vector, ComputeAngleBetweenTwoVectorIn3D


def angle_value(parameters, scale_a, scale_b, return_cos):
    zero=parameters.sum()*0
    a=torch.stack((parameters[0]*scale_a,zero+.5*scale_a,zero+.1*scale_a))
    b=torch.stack((zero+.4*scale_b,parameters[1]*scale_b,zero+.3*scale_b))
    with contextlib.redirect_stdout(io.StringIO()):
        return cal_angle_between_3d_vector(a,b,return_cos).sum()


class GeometryAngleGradientTest(unittest.TestCase):
    def test_vector_gradient_under_length_clamping(self):
        for dtype in (torch.float32,torch.float64):
            for return_cos in (False,True):
                for scale_a,scale_b in ((1.,1.),(1e-5,2e-5),(2e-13,1.),
                                       (1.,3e-13),(2e-13,3e-13)):
                    with self.subTest(dtype=dtype,return_cos=return_cos,
                                      scale_a=scale_a,scale_b=scale_b):
                        parameters=torch.ones(2,dtype=dtype,requires_grad=True)
                        angle_value(parameters,scale_a,scale_b,return_cos).backward()
                        # Evaluate finite differences in float64 to avoid
                        # subtractive rounding in the float32 reference.
                        expected=[]
                        step=1e-5
                        for i in range(2):
                            lo=torch.ones(2,dtype=torch.float64)
                            hi=lo.clone()
                            lo[i]-=step
                            hi[i]+=step
                            low=angle_value(lo,scale_a,scale_b,return_cos)
                            high=angle_value(hi,scale_a,scale_b,return_cos)
                            expected.append((high-low)/(2*step))
                        torch.testing.assert_close(parameters.grad,torch.stack(expected).to(dtype),
                                                   rtol=1e-5 if dtype==torch.float32 else 1e-8,atol=0)

    def test_numpy_and_torch_forward_result_and_input_ownership(self):
        for dtype in (np.float32,np.float64):
            a=np.array([[1.,0,0],[1.,0,0],[0.,0,0],[2e-13,0,0],[1.,2,3]],dtype=dtype)
            b=np.array([[1.,0,0],[-1.,0,0],[1.,0,0],[3e-13,3e-13,0],[4.,-2,1]],dtype=dtype)
            aa,bb=torch.from_numpy(a.copy()),torch.from_numpy(b.copy())
            with contextlib.redirect_stdout(io.StringIO()):
                cosine=cal_angle_between_3d_vector(a,b,True)
                numpy_angle=cal_angle_between_3d_vector(a,b)
                tensor_cosine=cal_angle_between_3d_vector(aa,bb,True)
                tensor_angle=cal_angle_between_3d_vector(aa,bb)
            np.testing.assert_allclose(tensor_cosine.numpy(),cosine,rtol=1e-6 if dtype==np.float32 else 1e-12,atol=0)
            np.testing.assert_allclose(numpy_angle,np.arccos(cosine),rtol=0,atol=0)
            # Preserve the existing torch acos endpoint guard, which differs
            # deliberately from NumPy's exact endpoint angles.
            eps=1e-7 if dtype==np.float32 else 1e-16
            expected=np.arccos(np.clip(cosine,-1+eps,1-eps))
            np.testing.assert_allclose(tensor_angle.numpy(),expected,
                                       rtol=1e-6 if dtype==np.float32 else 1e-12,atol=0)
            np.testing.assert_array_equal(aa.numpy(),a)
            np.testing.assert_array_equal(bb.numpy(),b)

    def test_finite_gradient_for_zero_parallel_and_antiparallel_input(self):
        for dtype in (torch.float32,torch.float64):
            for return_cos in (False,True):
                with self.subTest(dtype=dtype,return_cos=return_cos):
                    a=torch.tensor([[0.,0,0],[0.,0,0],[1.,0,0],[1.,0,0]],
                                   dtype=dtype,requires_grad=True)
                    b=torch.tensor([[0.,0,0],[1.,0,0],[1.,0,0],[-1.,0,0]],
                                   dtype=dtype,requires_grad=True)
                    with contextlib.redirect_stdout(io.StringIO()):
                        value=cal_angle_between_3d_vector(a,b,return_cos)
                    self.assertTrue(torch.isfinite(value).all())
                    value.sum().backward()
                    self.assertTrue(torch.isfinite(a.grad).all())
                    self.assertTrue(torch.isfinite(b.grad).all())

    def test_single_vector_empty_batch_and_legacy_alias(self):
        self.assertIs(ComputeAngleBetweenTwoVectorIn3D,cal_angle_between_3d_vector)
        for dtype in (torch.float32,torch.float64):
            for return_cos in (False,True):
                a=torch.tensor([1.,0,0],dtype=dtype)
                b=torch.tensor([0.,1,0],dtype=dtype)
                result=ComputeAngleBetweenTwoVectorIn3D(a,b,return_cos)
                self.assertEqual(result.shape,(1,))
                expected=0. if return_cos else np.pi/2
                torch.testing.assert_close(result,torch.tensor([expected],dtype=dtype))
                empty=torch.empty((0,3),dtype=dtype,requires_grad=True)
                result=cal_angle_between_3d_vector(empty,empty,return_cos)
                self.assertEqual(result.shape,(0,))
                result.sum().backward()
                self.assertEqual(empty.grad.shape,(0,3))
