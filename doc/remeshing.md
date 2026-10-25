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
