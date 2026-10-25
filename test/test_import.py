import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch_fea_mesh

pytestmark = pytest.mark.packaging


def run_python(code):
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    result = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True,
                            text=True, env=environment, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_root_does_not_import_optional_module():
    run_python('import sys, torch_fea_mesh; '
               'assert "vtk" not in sys.modules; '
               'assert "torch_geometric" not in sys.modules; '
               'assert torch_fea_mesh.__version__ == "0.1.0"')


def test_core_import_without_optional_dependency():
    run_python('''
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'vtk', 'vtkmodules', 'torch_geometric'}:
            raise ModuleNotFoundError('Optional dependency blocked for test: '+fullname)
sys.meta_path.insert(0, BlockOptional())
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh
from torch_fea_mesh.processing.mesh_processing import simple_smoother
from torch_fea_mesh.generation.mesh_example import create_quad_grid_mesh
mesh = create_quad_grid_mesh(2, 2)
assert mesh.element.shape == (1, 4)
assert 'vtk' not in sys.modules
assert 'torch_geometric' not in sys.modules
''')


@pytest.mark.parametrize('reverse', [False, True])
def test_each_available_module_import_in_fresh_process(reverse):
    package = Path(torch_fea_mesh.__file__).parent
    names = ['torch_fea_mesh.' + '.'.join(p.relative_to(package).with_suffix('').parts)
             for p in package.rglob('*.py') if p.name != '__init__.py']
    if importlib.util.find_spec('vtk') is None:
        names.remove('torch_fea_mesh.processing.isotropic_triangle_surface_remesher')
    if importlib.util.find_spec('torch_geometric') is None:
        names.remove('torch_fea_mesh.utility.chamfer_distance')
    names.sort(reverse=reverse)
    run_python('import importlib\nfor name in '+repr(names)+':\n    importlib.import_module(name)')


def test_canonical_mesh_class_in_each_consumer():
    from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh
    from torch_fea_mesh.processing.polygon_mesh_processing import TriangleMesh as ProcessingTriangle
    assert TriangleMesh is ProcessingTriangle
    if importlib.util.find_spec('vtk'):
        from torch_fea_mesh.processing.isotropic_triangle_surface_remesher import TriangleMesh as RemeshTriangle
        assert TriangleMesh is RemeshTriangle
