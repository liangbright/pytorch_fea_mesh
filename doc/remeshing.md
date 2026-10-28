# Isotropic triangle surface remeshing

Install the `vtk` extra. Import the public function from processing:

```python
from torch_fea_mesh.processing.isotropic_triangle_surface_remesher import (
    isotropic_triangle_surface_remesh,
)

result = isotropic_triangle_surface_remesh(
    source_mesh,
    target_edge_length=0.5,
    flag_process_boundary=False,
    feature_point_index_list=None,
    feature_edge_index_list=None,
    feature_curve_list=None,
    target_edge_length_ratio_max=1.4,
    target_edge_length_ratio_min=0.7,
    max_iter=10,
    flag_project_to_source_mesh=True,
)
```

The input and output are `TriangleMesh` objects. The first three arguments can
be positional; the remaining arguments are keyword-only.

- Target edge length must be finite and at least `1e-5`. For smaller scales,
  rescale the geometry and target length together.
- Feature-point indices refer to input nodes. Feature-edge indices refer to the
  original source edge-table order. A missing edge table is built on a private
  copy; a stale or inconsistent table is rejected.
- Feature curves list consecutive input vertices connected by existing edges.
  Repeat the first vertex to close a curve. Feature edges and their endpoints
  remain protected from topology edits.
- `flag_process_boundary=False` freezes boundary geometry. Disabling projection
  avoids closest-point repositioning, but the implementation still requires VTK.
- Working geometry uses independent CPU float64 arrays. Output restores source
  coordinate dtype/device and connectivity device. This is not differentiable:
  an autograd graph is not retained through remeshing.
- The input mesh and feature lists are unchanged. Source fields are not copied
  onto changed output topology; diagnostics are stored in
  `result.mesh_data['remeshing_report']`.

For problematic collapses, bounded local repair is attempted first. Failed
attempts are deferred during the first `ceil(max_iter / 2)` iterations. Only
later iterations may remove a certified enclosed region as a last resort;
feature protection takes precedence. Geometry checks are bounded/sampled, not a
general proof of global surface equivalence.

`max_iter=0` still permits the implementation's final interior-collapse pass; it
does not mean a guaranteed identity operation. The final pass does not advance
the enclosed-region-deletion phase.

The regression tests cover phase scheduling, repair and deletion,
feature numbering/protection, frozen boundaries, source immutability, dtype,
projection, dynamic adjacency, curved enclosures, and feature VTK exports:

```bash
python -m pytest test/remeshing/ -v
```

See the [synthetic example](../example/remesh_surface.py) and the
[testing guide](testing.md) for optional checks against local data.

## Template barycentric mapping

Install `torch-fea-mesh[remeshing]` for SciPy. Add the `vtk` extra for VTK export.

```python
from torch_fea_mesh.processing.template_barycentric_mapping_based_remesher import (
    template_barycentric_mapping_based_remesh,
)

result = template_barycentric_mapping_based_remesh(
    source_mesh, source_boundary_segment_list,
    template_mesh, template_boundary_segment_list,
    landmark_of_source_mesh=None,
    landmark_of_template_mesh=None,
)
```

The source is a connected, consistently oriented triangular `PolygonMesh` with
boundary and genus zero. The template is a planar polygon mesh with z equal to
zero. Output preserves template connectivity and metadata. Pair ordered boundary
paths with corresponding direction and endpoints, covering every boundary edge.
Sampling counts can differ: positions correspond by relative arc length. Repeat
the initial vertex for a closed path. Paired landmark indices fix additional
correspondences.

Initialization choices are `untangle` (default), `averaged_barycentric`,
`mean_value`, and `annular`; see the function docstring for their assumptions.
Automatic diffusion selection supports triangle and quad templates. Set
`diffusion_coefficient_of_mesh_parameterization` to a number in [0, 1] to use a
fixed coefficient. Solver choices are `sparse_lu`, `gmres`, and `bicgstab`.
Calculations use CPU NumPy/SciPy arrays and are not differentiable. Inputs are
preserved. Diagnostics are stored in
`result.mesh_data['template_barycentric_mapping_based_remesh']`.

Run the boundary mapping and interior landmark examples from the repository:

```sh
python example/remesh_template_barycentric.py --output /tmp/barycentric_boundary
python example/remesh_template_barycentric.py --landmark --output /tmp/barycentric_landmark
python -m pytest test/remeshing/test_template_barycentric*.py
```

The regression tests bundle their input fixtures and do not need the original
development checkout or generated result files.
