# Installation

Python 3.10 or newer is required. Use a Python environment with compatible
PyTorch and compiled extension builds.

## Required dependencies

The package requires NumPy, PyTorch, and `torch-scatter`.

1. Install the PyTorch build for your Python version and CPU/CUDA environment
   using the [official PyTorch instructions](https://pytorch.org/get-started/locally/).
2. Install a matching `torch-scatter` build using the
   [PyG extension installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
   Match the extension wheel to your PyTorch and CPU/CUDA build, including when
   running a CUDA-enabled PyTorch installation on the CPU.
3. From the project root, install this package:

```bash
python -m pip install .
```

The import name is `torch_fea_mesh`; the distribution name is `torch-fea-mesh`.

## Optional features

| Extra | Purpose | Installation |
| --- | --- | --- |
| `vtk` | VTK I/O, VTK-based processing, and isotropic remeshing | `python -m pip install '.[vtk]'` |
| `remeshing` | Template barycentric remeshing with SciPy | `python -m pip install '.[remeshing]'` |
| `chamfer` | Chamfer distance with PyTorch Geometric and its KNN backend | `python -m pip install '.[chamfer]'` |
| `dev` | pytest, build tools, and distribution checks | `python -m pip install -e '.[dev]'` |

For `chamfer`, install a matching `pyg-lib` build using the PyG extension guide
before installing the extra. This extra requires PyTorch Geometric 2.8 or newer
and `pyg-lib` 0.6 or newer.

VTK is optional for core mesh operations. Importing the isotropic remesher
requires VTK.

## Development setup

After installing matching `torch-scatter` and `pyg-lib` builds, install all test
dependencies in editable mode:

```bash
python -m pip install -e '.[dev,vtk,chamfer]'
python -m pytest test/ -v
```

CUDA tests also require an accessible CUDA device. See [testing](testing.md) for
partial runs and optional dependency skips.

## Build a distribution

With the `dev` extra installed:

```bash
python -m build
python -m twine check dist/*
```

The wheel contains the library. The source distribution also contains tests,
examples, and documentation. Local `data/` files are excluded from both.

To check an installed wheel, use a separate environment with matching numerical
dependencies and run from outside the project directory. This ensures the source
folder cannot shadow the installed package; see [wheel validation](testing.md#wheel-validation).
