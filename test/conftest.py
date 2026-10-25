"""Shared synthetic fixtures and options for user-triggered test runs."""
from pathlib import Path

import pytest
import torch

from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh
from torch_fea_mesh.mesh.tetrahedron_mesh import TetrahedronMesh


def pytest_addoption(parser):
    parser.addoption("--data-dir", default=None, help="Directory of optional local test data")


def pytest_collection_modifyitems(items):
    # External pytest hook name: preserve its spelling for hook discovery.
    for item in items:
        if "cuda" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
        if item.get_closest_marker("gpu") and not torch.cuda.is_available():
            item.add_marker(pytest.mark.skip(reason="CUDA device unavailable"))


@pytest.fixture(scope="session", autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def temporary_remeshing_output(request, tmp_path, monkeypatch):
    if request.node.get_closest_marker("remeshing"):
        monkeypatch.setenv("TORCH_FEA_MESH_TEST_OUTPUT", str(tmp_path))


@pytest.fixture
def local_data_dir(request):
    path = request.config.getoption("--data-dir")
    return Path(path).expanduser().resolve() if path else request.config.rootpath / "data"


@pytest.fixture
def triangle():
    return TriangleMesh(torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                                    dtype=torch.float64), [[0, 1, 2]])


@pytest.fixture
def tetrahedron():
    return TetrahedronMesh(torch.tensor([[0., 0., 0.], [1., 0., 0.],
                                        [0., 1., 0.], [0., 0., 1.]],
                                       dtype=torch.float64), [[0, 1, 2, 3]])
