import pytest
import torch

from torch_fea_mesh.transform.rotation_transform import RotationTransform3D
from torch_fea_mesh.transform.rigid_transform import RigidTransform3D
from torch_fea_mesh.transform.similarity_transform import SimilarityTransform3D
from torch_fea_mesh.transform.rodrigue_rotation_transform import RodrigueRotationTransform3D

pytestmark = pytest.mark.transform


@pytest.mark.parametrize("cls", [RotationTransform3D, RigidTransform3D, SimilarityTransform3D])
def test_identity_and_input_gradient(cls):
    transform = cls().double()
    points = torch.tensor([[1.,2.,3.],[-2.,.5,4.]], dtype=torch.float64, requires_grad=True)
    result = transform(points)
    torch.testing.assert_close(result, points)
    gradient, = torch.autograd.grad(result.sum(), points)
    torch.testing.assert_close(gradient, torch.ones_like(points))


def test_z_rotation_and_parameter_gradient():
    point = torch.tensor([[1.,0.,0.]], dtype=torch.float64)
    angle = torch.tensor([0.,0.,torch.pi/2], dtype=torch.float64, requires_grad=True)
    result = RotationTransform3D.rotate(point, angle)
    torch.testing.assert_close(result, torch.tensor([[0.,1.,0.]], dtype=torch.float64), atol=1e-14, rtol=0)
    assert torch.autograd.gradcheck(lambda a: RotationTransform3D.rotate(point, a), (angle,))


def test_rigid_and_similarity_translation_convention():
    points = torch.tensor([[3.,4.,5.]], dtype=torch.float64)
    for cls, scale in [(RigidTransform3D, 1.), (SimilarityTransform3D, 2.)]:
        transform = cls().double()
        with torch.no_grad():
            transform.t.copy_(torch.tensor([[1.,2.,3.]], dtype=torch.float64))
            if cls is SimilarityTransform3D:
                transform.s.fill_(scale)
        torch.testing.assert_close(transform(points), (points-transform.t)*scale)


def test_same_angle_gives_same_R():
    angle = torch.tensor([.3,-.5,.7], dtype=torch.float64)
    R = RotationTransform3D.cal_R(angle)
    c, s = torch.cos(angle), torch.sin(angle)
    Rx = torch.tensor([[1,0,0],[0,c[0],-s[0]],[0,s[0],c[0]]], dtype=torch.float64)
    Ry = torch.tensor([[c[1],0,s[1]],[0,1,0],[-s[1],0,c[1]]], dtype=torch.float64)
    Rz = torch.tensor([[c[2],-s[2],0],[s[2],c[2],0],[0,0,1]], dtype=torch.float64)
    torch.testing.assert_close(R, Rz@Ry@Rx)
    points = torch.tensor([[3.,4.,5.],[-2.,.5,4.]], dtype=torch.float64)
    t = torch.tensor([[1.,2.,3.]], dtype=torch.float64)
    for cls, scale in [(RigidTransform3D, 1.), (SimilarityTransform3D, 2.)]:
        assert torch.equal(cls.cal_R(angle), R)
        transform = cls().double()
        with torch.no_grad():
            transform.angle.copy_(angle)
            transform.t.copy_(t)
            if cls is SimilarityTransform3D:
                transform.s.fill_(scale)
        torch.testing.assert_close(transform(points), scale*(points-t)@R.T)


def test_dtype_input():
    points = torch.tensor([[1.,2.,3.]], dtype=torch.float64)
    for cls in (RotationTransform3D, RigidTransform3D, SimilarityTransform3D, RodrigueRotationTransform3D):
        assert all(p.dtype == torch.get_default_dtype() for p in cls().parameters())
        transform = cls(dtype=torch.float64)
        assert all(p.dtype == torch.float64 for p in transform.parameters())
        assert transform(points).dtype == torch.float64


def test_rodrigues_static_rotation_about_origin():
    point = torch.tensor([[2.,1.,0.]], dtype=torch.float64)
    axis = torch.tensor([0.,0.,1.], dtype=torch.float64)
    origin = torch.tensor([[1.,1.,0.]], dtype=torch.float64)
    result = RodrigueRotationTransform3D.rotate(point, axis, torch.tensor(torch.pi/2, dtype=torch.float64), origin)
    torch.testing.assert_close(result, torch.tensor([[1.,2.,0.]], dtype=torch.float64))


def test_rodrigues_forward(triangle):
    transform = RodrigueRotationTransform3D().double()
    torch.testing.assert_close(transform(triangle.node), triangle.node)


def test_rodrigues_forward_about_nonzero_origin():
    transform = RodrigueRotationTransform3D().double()
    with torch.no_grad():
        transform.axis.copy_(torch.tensor([0., 0., 1.], dtype=torch.float64))
        transform.angle.fill_(torch.pi / 2)
        transform.origin.copy_(torch.tensor([[1., 2., 3.]], dtype=torch.float64))

    points = torch.tensor([[2., 2., 3.], [1., 3., 3.], [1., 2., 5.]],
                          dtype=torch.float64, requires_grad=True)
    expected = torch.tensor([[1., 3., 3.], [0., 2., 3.], [1., 2., 5.]],
                            dtype=torch.float64)
    result = transform(points)
    torch.testing.assert_close(result, expected, atol=1e-14, rtol=0)

    # The chosen pivot and every point on its rotation axis remain fixed.
    torch.testing.assert_close(transform(transform.origin), transform.origin)
    # Rotation must still carry gradients through the trainable pivot.
    origin_gradient, = torch.autograd.grad(result.sum(), transform.origin)
    torch.testing.assert_close(origin_gradient,
                               torch.tensor([[0., 6., 0.]], dtype=torch.float64),
                               atol=1e-14, rtol=0)
