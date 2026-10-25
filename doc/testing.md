# Testing

The test suite uses pytest to run numerical assertions, regression checks, and
import checks. Install the package and development dependencies as described in
[installation](installation.md). Core tests require NumPy, PyTorch, and
`torch-scatter`; optional VTK and Chamfer tests need their corresponding extras.

## Run all or selected tests

Run these commands from the project root:

```bash
# Full suite, with skip reasons in the summary.
python -m pytest test/ -v -ra

# One module or one test.
python -m pytest test/test_io.py -v
python -m pytest test/test_mesh_geometry.py::test_tetrahedron_volume -v

# Remeshing tests or selected categories.
python -m pytest test/remeshing/ -v
python -m pytest test/ -m 'geometry or transform' -v
python -m pytest test/ -m 'remeshing and not local_data' -v
python -m pytest test/ -k 'tetrahedron or hexahedron' -v

# Preview test selection.
python -m pytest test/ --collect-only -q
```

Markers are registered in `pyproject.toml`: `geometry`, `processing`, `io`,
`transform`, `generation`, `vtk`, `chamfer`, `remeshing`, `gpu`, `slow`,
`local_data`, and `packaging`. Tests explicitly skip when their required optional
feature dependency, CUDA device, or local input is unavailable. Review those
reasons when assessing coverage.

The suite includes `unittest.TestCase` regressions collected by pytest. Synthetic
mesh helpers live under `test/helper/`. Generated files, including VTK exports,
use temporary directories. Tests are included in the source distribution.

## Local test data

The default suite needs no external dataset. Optional local files belong in the
root `data/` directory, which is ignored by version control and excluded from
package builds. Create the directory if needed:

```bash
mkdir -p data
```

To exercise the optional local remeshing check, place a nonempty triangular
surface in `data/triangle_surface.vtk`, then run:

```bash
python -m pytest test/test_local_data.py -v

# Use a sibling directory containing triangle_surface.vtk.
python -m pytest test/test_local_data.py --data-dir ../local_data -v
```

The check runs one remeshing iteration using the median source edge length,
freezes the boundary, checks source immutability, and writes the result to a
temporary directory. Input files are never modified. Quadrilateral meshes need
to be triangulated before use with this triangle-specific test.

## Transform checks

The transform suite checks numerical results and gradients. Rodrigues tests
cover identity rotation, a 90-degree rotation around a nonzero origin, fixed
points on the rotation axis, and gradients through the origin:

```bash
python -m pytest test/test_transform.py -k rodrigues -v
```

## Wheel validation

Build a wheel and install it normally in a separate environment with matching
dependencies. Copy `test/` and `pyproject.toml` into a separate test directory,
then run `python -m pytest test/ -v` from there. Running outside the source
project prevents local files from masking missing wheel contents.
