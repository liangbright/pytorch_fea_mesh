# Architecture

`torch_fea_mesh` groups functionality into six subpackages. Import classes and
functions from their defining modules; package initializers are kept minimal.

| Subpackage | Responsibility |
| --- | --- |
| `mesh` | Mesh containers, geometry, adjacency, mesh fields, and class-based I/O |
| `processing` | Connectivity, smoothing, repair, surface operations, and remeshing |
| `io` | Standalone INP and VTK readers and writers |
| `transform` | Rotation, rigid, and similarity transformations |
| `generation` | Synthetic surface and volume meshes |
| `utility` | Angle geometry and Chamfer distance |

Internal imports are relative to the package. Mesh classes use I/O and geometry
helpers; processing operates on mesh classes; generators construct meshes and
may use processing helpers. Examples and tests live outside the runtime package.

Mesh containers hold node coordinates, element connectivity, node/element sets,
user data, and cached geometry or adjacency information. Concrete mesh classes
provide element-specific calculations. See [the API overview](api.md).

The isotropic remesher operates on `TriangleMesh` and maintains dynamic topology
through split, collapse, and flip operations. Its mutable state belongs to each
call. See [remeshing](remeshing.md) for geometry constraints, feature protection,
and output behavior.

Tests use synthetic geometry and temporary output directories. Optional local
input files belong in `data/`, which is ignored by version control and excluded
from package distributions.
