"""Optional local dataset check; the committed suite needs no external files."""
import pytest
import torch

pytestmark = [pytest.mark.local_data, pytest.mark.vtk, pytest.mark.remeshing]


def test_local_triangle_surface(local_data_dir, tmp_path):
    path = local_data_dir / 'triangle_surface.vtk'
    if not path.is_file():
        pytest.skip('Optional data/triangle_surface.vtk was not provided')
    pytest.importorskip('vtk')
    from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh
    from torch_fea_mesh.processing.isotropic_triangle_surface_remesher import isotropic_triangle_surface_remesh
    source = TriangleMesh()
    source.load_from_vtk(str(path), dtype=torch.float64)
    assert len(source.element) > 0
    source.build_edge()
    lengths = torch.linalg.vector_norm(source.node[source.edge[:,0]]-source.node[source.edge[:,1]], dim=1)
    target = max(float(lengths.median()), 1e-5)
    before = source.node.clone()
    output = isotropic_triangle_surface_remesh(source,target,False,max_iter=1)
    torch.testing.assert_close(source.node,before)
    assert torch.isfinite(output.node).all()
    assert 'remeshing_report' in output.mesh_data
    output.save_as_vtk(str(tmp_path/'remeshed.vtk'))
