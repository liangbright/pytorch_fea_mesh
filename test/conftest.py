"""Shared synthetic fixtures and options for user-triggered test runs."""
from pathlib import Path

import pytest
import torch

from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh
from torch_fea_mesh.mesh.tetrahedron_mesh import TetrahedronMesh
from test.helper.data import input_dir
from test.helper.output import create_run_output, test_output_path


def pytest_addoption(parser):
    parser.addoption("--data-dir", default=None, help="Test input root (default: data/test/data)")


def pytest_configure(config):
    if not config.option.collectonly:
        config._mesh_test_output = create_run_output()


def pytest_report_header(config):
    if hasattr(config, '_mesh_test_output'):
        return f'Test outputs: {config._mesh_test_output}'


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if hasattr(config, '_mesh_test_output'):
        terminalreporter.write_line(f'Test outputs retained in: {config._mesh_test_output}')


def pytest_collection_modifyitems(items):
    # External pytest hook name: preserve its spelling for hook discovery.
    for item in items:
        if item.path.name.startswith("test_template_barycentric"):
            item.add_marker(pytest.mark.remeshing)
            item.add_marker(pytest.mark.processing)
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


@pytest.fixture(scope="session", autouse=True)
def configured_test_data(request):
    with pytest.MonkeyPatch.context() as patch:
        path = request.config.getoption('--data-dir')
        if path:
            patch.setenv('TORCH_FEA_MESH_TEST_DATA', str(Path(path).expanduser().resolve()))
        yield input_dir()


@pytest.fixture
def test_output_dir(request):
    return test_output_path(request.config._mesh_test_output, request.node.nodeid)


@pytest.fixture(autouse=True)
def retained_test_output(test_output_dir, monkeypatch):
    monkeypatch.setenv('TORCH_FEA_MESH_TEST_OUTPUT', str(test_output_dir))


@pytest.fixture
def local_data_dir(configured_test_data):
    return configured_test_data


@pytest.fixture
def triangle():
    return TriangleMesh(torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                                    dtype=torch.float64), [[0, 1, 2]])


@pytest.fixture
def tetrahedron():
    return TetrahedronMesh(torch.tensor([[0., 0., 0.], [1., 0., 0.],
                                        [0., 1., 0.], [0., 0., 1.]],
                                       dtype=torch.float64), [[0, 1, 2, 3]])
