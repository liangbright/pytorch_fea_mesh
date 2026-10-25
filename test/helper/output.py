"""Temporary output directory supplied by pytest; never the source checkout."""
import os
from pathlib import Path

def output_dir():
    return Path(os.environ["TORCH_FEA_MESH_TEST_OUTPUT"])
