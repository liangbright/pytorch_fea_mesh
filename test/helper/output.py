"""Persistent, isolated output folders for pytest and unittest runs."""
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import re
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_standalone_output = None


def create_run_output():
    run_id = 'test_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
    path = PROJECT_ROOT / 'data/test' / run_id / 'output'
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_output_path(root, node_id):
    """Keep the test module and case visible in the output path."""
    module, *case = node_id.split('::')
    prefix = Path(module).stem
    name = re.sub(r'[^A-Za-z0-9_.-]+', '_', '_'.join(case)).strip('_')
    path = root / prefix / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir():
    global _standalone_output
    configured = os.environ.get('TORCH_FEA_MESH_TEST_OUTPUT')
    if configured:
        return Path(configured)
    if _standalone_output is None:
        _standalone_output = create_run_output()
    return _standalone_output


@contextmanager
def output_directory(prefix):
    """Retain each unittest invocation's files, including repeated subcases."""
    path = output_dir() / (prefix + '_' + uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    yield path
