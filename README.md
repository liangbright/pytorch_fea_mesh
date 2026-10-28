# PyTorch-FEA-Mesh

Mesh data structures and geometry processing in PyTorch for finite element
analysis and AI/ML. The repository is `pytorch_fea_mesh`; the Python package is
`torch_fea_mesh`.

The library provides:

- Surface, volume, and polyline mesh containers, including linear and quadratic
  triangle and tetrahedral elements.
- Connectivity queries, smoothing, repair, surface extraction, and isotropic
  triangle surface remeshing.
- Rotation, rigid, and similarity transforms, plus structured mesh generation.
- Torch serialization, INP and VTK I/O, geometry utilities, and Chamfer distance.

## Installation

Python 3.10 or newer is required. Install PyTorch and a matching `torch-scatter`
build, then install the package from the project root:

```bash
python -m pip install .
```

`torch-scatter` is required. VTK-based features, including isotropic remeshing,
are available through the `vtk` extra:

```bash
python -m pip install '.[vtk]'
```

Chamfer distance requires the `chamfer` extra and a matching `pyg-lib` build.
See [installation](doc/installation.md) for dependencies, optional features,
development setup, and distribution builds.

## Quickstart

```python
import torch
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh

mesh = TriangleMesh(
    torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=torch.float64),
    [[0, 1, 2]],
)
mesh.update_element_area_and_normal()
print(mesh.element_area)  # one triangle with area 0.5
print(mesh.element_normal)
```

With VTK installed, remesh the triangle surface:

```python
from torch_fea_mesh.processing.isotropic_triangle_surface_remesher import (
    isotropic_triangle_surface_remesh,
)

remeshed = isotropic_triangle_surface_remesh(
    mesh,
    target_edge_length=0.4,
    flag_process_boundary=False,
    max_iter=2,
)
```

Template barycentric remeshing is available with the `remeshing` extra. See
`example/remesh_template_barycentric.py` for boundary and interior landmark examples.

See the [API overview](doc/api.md), [remeshing guide](doc/remeshing.md), and
[runnable examples](example/README.md).

## Project layout

```text
torch_fea_mesh/
  mesh/         # Mesh containers and geometry methods
  processing/   # Connectivity, smoothing, repair, and remeshing
  io/           # INP and VTK readers and writers
  transform/    # Rotation, rigid, and similarity transforms
  generation/   # Synthetic mesh generators
  utility/      # Geometry utilities and Chamfer distance
test/           # Automated tests
example/        # Runnable examples with generated inputs
doc/            # Installation, API, architecture, and testing guides
data/           # Optional local data; excluded from version control and builds
```

## Testing

Install the development dependencies, then run all tests or select a subset:

```bash
python -m pip install -e '.[dev]'
python -m pytest test/ -v
python -m pytest test/test_io.py -v
python -m pytest test/remeshing/ -v
```

Optional-feature and CUDA tests report skips when their dependencies or hardware
are unavailable. The [testing guide](doc/testing.md) explains test selection,
coverage, and optional local data.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for naming conventions and development
practices, [architecture](doc/architecture.md) for package responsibilities, and
the [changelog](CHANGELOG.md) for version information.
