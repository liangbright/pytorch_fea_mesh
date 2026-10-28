"""Locate separately shared test inputs without depending on the working directory."""
import os
from pathlib import Path
from unittest import SkipTest


def input_dir():
    configured = os.environ.get('TORCH_FEA_MESH_TEST_DATA')
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / 'data/test/data'


def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise SkipTest(f'Local dataset file unavailable: {path}. See data/data_record.json.')
    return path
