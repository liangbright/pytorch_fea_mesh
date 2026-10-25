import pytest
import torch

pytestmark = pytest.mark.chamfer


@pytest.fixture
def chamfer_distance_module():
    pytest.importorskip('torch_geometric')
    from torch_geometric.nn.pool import knn
    # Only skip an explicitly missing optional backend; runtime errors should fail.
    try:
        knn(torch.zeros((1,3)), torch.zeros((1,3)), 1)
    except ImportError as error:
        pytest.skip(str(error))
    from torch_fea_mesh.utility import chamfer_distance
    return chamfer_distance


def test_chamfer_known_distance_and_gradient(chamfer_distance_module):
    source = torch.tensor([[0.,0.,0.],[2.,0.,0.]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0.,1.,0.],[2.,1.,0.]], dtype=torch.float64)
    actual = chamfer_distance_module.cal_chamfer_distance(source,target,'mean',True)
    torch.testing.assert_close(actual,torch.tensor(1.,dtype=torch.float64))
    gradient, = torch.autograd.grad(actual,source)
    torch.testing.assert_close(gradient,torch.tensor([[0.,-1.,0.],[0.,-1.,0.]],dtype=torch.float64))


def test_batched_chamfer_matches_independent_execution(chamfer_distance_module):
    source = torch.tensor([[[0.,0.,0.],[2.,0.,0.]], [[10.,0.,0.],[12.,0.,0.]]],dtype=torch.float64)
    target = source + torch.tensor([0.,1.,0.],dtype=torch.float64)
    actual = chamfer_distance_module.cal_chamfer_distance_batch(source,target,'mean',True)
    expected = torch.stack([chamfer_distance_module.cal_chamfer_distance(a,b,'mean',True) for a,b in zip(source,target)]).mean()
    torch.testing.assert_close(actual, expected)
