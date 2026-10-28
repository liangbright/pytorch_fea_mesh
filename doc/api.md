# API overview

Import public classes and functions from their defining modules.

## Mesh containers

```python
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh
from torch_fea_mesh.mesh.tetrahedron_mesh import TetrahedronMesh
from torch_fea_mesh.mesh.hexahedron_mesh import HexahedronMesh
```

| Class | Geometry |
| --- | --- |
| `Mesh` | Base mesh container |
| `PolygonMesh` | Polygon surfaces |
| `PolyhedronMesh` | Polyhedral volume meshes |
| `PolylineMesh` | Line segments and polylines |
| `TriangleMesh` | Three-node triangles |
| `Tri6Mesh` | Six-node quadratic triangles |
| `QuadMesh` | Four-node quadrilaterals |
| `QuadTriangleMesh` | Mixed quadrilateral and triangular surfaces |
| `TetrahedronMesh` | Four-node tetrahedra |
| `Tet10Mesh` | Ten-node quadratic tetrahedra |
| `HexahedronMesh` | Eight-node hexahedra |

`node` stores coordinates as a PyTorch tensor. `element` stores connectivity as a
tensor or, where supported, ragged lists. Meshes also provide `node_set`,
`element_set`, `node_data`, `element_data`, and `mesh_data` for labels and user data.
Methods such as `build_edge()` and `update_element_area_and_normal()` populate
adjacency or geometry fields on the appropriate mesh classes.

## Processing

```python
from torch_fea_mesh.processing.mesh_processing import simple_smoother_for_mesh
from torch_fea_mesh.processing.polyhedron_mesh_processing import extract_surface_mesh
from torch_fea_mesh.processing.isotropic_triangle_surface_remesher import (
    isotropic_triangle_surface_remesh,
)
from torch_fea_mesh.processing.template_barycentric_mapping_based_remesher import (
    template_barycentric_mapping_based_remesh,
)
```

Processing includes connectivity queries, smoothing, repair, surface extraction,
clipping, and remeshing. VTK-based operations require the `vtk` extra. The
[remeshing guide](remeshing.md) documents isotropic and template barycentric remeshing,
constraints, optional dependencies, and source immutability.

## Generation

Import generators from `torch_fea_mesh.generation.mesh_example`:

- `create_quad_grid_mesh`
- `create_hex_grid_mesh`
- `create_quad_cylinder_mesh`
- `create_quad_mesh_rectangle_in_rectangle`
- `create_quad_mesh_rectangle_in_cirlce`
- `create_quad_tri_mesh_circle_in_circle`

For a structured volume mesh:

```python
from torch_fea_mesh.generation.mesh_example import create_hex_grid_mesh

mesh = create_hex_grid_mesh(2, 2, 2)
mesh.update_element_volume()
print(mesh.element_volume)
```

## Transforms

The `transform` subpackage provides `RotationTransform3D`, `RigidTransform3D`,
`SimilarityTransform3D`, and `RodrigueRotationTransform3D`.

Rigid transforms use the mathematical convention `R * (x - t)`; similarity
transforms additionally apply scale. Rodrigues rotation supports both
`rotate(x, axis, angle, origin)` and module calls such as `transform(point)`.
The module uses its trainable `origin` parameter as the rotation pivot.

## I/O and utilities

Mesh classes provide Torch save/load methods and format-specific I/O methods.
Standalone INP and VTK helpers are available in `torch_fea_mesh.io`. See the
[I/O example](../example/mesh_io.py).

`torch_fea_mesh.utility.geometry_utility` provides angle calculations.
`torch_fea_mesh.utility.chamfer_distance` provides Chamfer distance functions and
requires the optional `chamfer` dependencies.

Volume-mesh subdivision is not implemented. Consult each class and method for
supported operations and argument details.
