import pytest

pytestmark = [pytest.mark.remeshing, pytest.mark.processing, pytest.mark.vtk]
pytest.importorskip("vtk", reason="VTK is required for this test module")

"""Feature indices of any integer width are converted to int64. Anything else is refused."""
import torch

import torch_fea_mesh.processing.isotropic_triangle_surface_remesher as r
from test.remeshing.test_isotropic_triangle_surface_remesher import grid


def remesh(**features):
    return r.isotropic_triangle_surface_remesh(grid(), 0.6, max_iter=2, **features)


def test_every_unsigned_width_gives_the_same_result_as_python_ints():
    expected = remesh(
        feature_point_index_list=[5], feature_edge_index_list=[0], feature_curve_list=[[0, 1, 2]]
    )
    for dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        output = remesh(
            feature_point_index_list=torch.tensor([5], dtype=dtype),
            feature_edge_index_list=torch.tensor([0], dtype=dtype),
            feature_curve_list=[torch.tensor([0, 1, 2], dtype=dtype)],
        )
        assert torch.equal(output.node, expected.node)
        assert torch.equal(output.element, expected.element)
        assert output.element.dtype == torch.int64


def test_values_and_types_that_cannot_be_indices_are_refused():
    with pytest.raises(ValueError, match='out-of-range'):
        remesh(feature_point_index_list=torch.tensor([2**63], dtype=torch.uint64))
    for dtype in (torch.bool, torch.float32):
        with pytest.raises(TypeError, match='integer indices'):
            remesh(feature_point_index_list=torch.tensor([1], dtype=dtype))


def test_none_in_a_curve_list_means_no_curve():
    expected = remesh()
    output = remesh(feature_curve_list=[None])
    assert torch.equal(output.node, expected.node)
    assert torch.equal(output.element, expected.element)
