# Examples

Install the package using the [installation guide](../doc/installation.md), then
run these scripts from the project root. Each example generates its own input.

| Script | Demonstrates |
| --- | --- |
| `basic_mesh.py` | Triangle area, normals, and boundary nodes |
| `generate_mesh.py` | Structured quad and hex grids |
| `transform_mesh.py` | Rotation around the z axis |
| `mesh_io.py` | Torch save/load with mesh fields |
| `remesh_surface.py` | Triangle surface remeshing with a fixed boundary |

```bash
python example/basic_mesh.py
python example/generate_mesh.py --output data/example
python example/transform_mesh.py
python example/mesh_io.py --output data/example
python example/remesh_surface.py --output data/example --iterations 2
```

The remeshing example requires VTK. Generated output under `data/` remains local
and is excluded from version control and package builds. See the
[remeshing guide](../doc/remeshing.md) for feature constraints and parameters.
