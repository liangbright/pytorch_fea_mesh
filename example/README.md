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
| `remesh_template_barycentric.py` | Template mapping with boundary and optional interior landmarks |

```bash
python example/basic_mesh.py
python example/generate_mesh.py
python example/transform_mesh.py
python example/mesh_io.py
python example/remesh_surface.py --iterations 2
python example/remesh_template_barycentric.py
python example/remesh_template_barycentric.py --landmark
```

The remeshing examples require VTK; template barycentric mapping also needs the
`remeshing` extra. Default output folders are resolved from the repository,
independently of the working directory:

- `data/example/generate_mesh_output/`
- `data/example/mesh_io_output/`
- `data/example/isotropic_remesher_output/`
- `data/example/template_barycentric_output/`

Filenames carry the matching example prefix. Use `--output` to choose another
output folder. The current examples generate their inputs in code and need no
external files. Future shared example inputs belong in `data/example/data/`,
with descriptive example/case prefixes and entries in `data/data_record.json`.
The inventory records only paths relative to `data/` and their purposes. Input
files and generated outputs remain local and are excluded from Git and package
builds. See the
[remeshing guide](../doc/remeshing.md) for feature constraints and parameters.
