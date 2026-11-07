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
    flag_use_coverage_then_tangential=False,
)
```

The input and output are `TriangleMesh` objects. The first three arguments can
be positional; the remaining arguments are keyword-only.

- Target edge length must be finite and at least `1e-5`. For smaller scales,
  rescale the geometry and target length together.
- Every source triangle must have an area. A triangle with three nodes on a line, or with two
  nodes at the same place, raises `ValueError: mesh has a degenerate triangle with no area`.
  The remesher doesn't repair it, and the report is in the error's `diagnostics` attribute.
  No area means at or below `scale**2 * 1e-14`, where `scale` is the larger of
  `target_edge_length` and the diagonal of the bounding box of all nodes, including unused
  ones. So a very large target or a node far away can reject a valid triangle. To find them:

  ```python
  from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh

  node = source_mesh.node.double()
  scale = max((node.amax(0) - node.amin(0)).norm().item(), target_edge_length)
  area, _ = TriangleMesh.cal_element_area_and_normal(node, source_mesh.element)
  no_area = (area.ravel() <= scale**2 * 1e-14).nonzero().ravel()
  ```

  To fix the source, merge nodes at the same place and drop the triangles that then repeat a
  node. For three different nodes on a line, flip the longest edge with the triangle next to it.
- Feature-point indices refer to input nodes. Feature-edge indices refer to the
  original source edge-table order. A missing edge table is built on a private
  copy; a stale or inconsistent table is rejected.
- Feature curves list consecutive input vertices connected by existing edges.
  Repeat the first vertex to close a curve. Feature edges and their endpoints
  remain protected from topology edits.
- `flag_process_boundary=False`, the default, freezes boundary geometry. Disabling projection
  avoids closest-point repositioning, but the implementation still requires VTK.
- `flag_use_coverage_then_tangential` chooses how interior vertices are smoothed. By default
  (`False`), a vertex moves to the plain average of its neighbors, which pulls curved surfaces and
  thin parts in further with each iteration. With `True`, a vertex moves by coverage-based
  smoothing, and in the last two iterations it slides within its tangent plane instead, unless it
  is on a sharp crease or corner. With `max_iter=2`, tangential smoothing runs only in the last
  iteration, and with `max_iter=1` it doesn't run. This reads the source mesh even when
  `flag_project_to_source_mesh=False`.
- Working geometry uses independent CPU float64 arrays. For a float32 source, each new position
  is rounded to float32 when it is proposed, so the checks see what is written. Output restores
  the source coordinate dtype, and its node and element are both on the source node's device,
  whatever torch's default device is. This is not differentiable:
  an autograd graph is not retained through remeshing.
- The input mesh and feature lists are unchanged. Source fields are not copied
  onto changed output topology; diagnostics are stored in
  `result.mesh_data['remeshing_report']`.

For problematic collapses, bounded local repair is attempted first. Failed
attempts are deferred during the first `ceil(max_iter / 2)` iterations. Only
later iterations may use nested replacement as a last resort: the nested
triangles on one side of the short edge (u, v), bounded by a triangle of edges
(u, v, w), become the one element (u, v, w), and the collapse is tried again as
an ordinary one. Feature protection takes precedence.

A replacement happens only if the whole nest is within `0.1 * target_edge_length`
of (u, v, w). The surface keeps its Euler value, boundary loops and components,
so a mesh cannot be erased. Nothing changes if any of these holds: the search for
the nest goes over 256 faces, the other side is also within the limit, or the new
element would pass through a face it shares no vertex with.

The limit applies to each replacement, not across edits, and the merge of u and v
has no distance check, like any other short edge. Other geometry checks are
bounded/sampled, not a general proof of global surface equivalence. Records are in
`result.mesh_data['remeshing_report']['nested_replacements']`.

`result.mesh_data['remeshing_report']['failures']` says what the run couldn't do:

- `error`: the error that stopped the run, or `None`.
- `short_edges` and `long_edges`: how many edges are still outside the length limits.
- `short_edge_reasons`: how many short edges ended with each reason, taken from the
  last collapse tried on each edge. Two reasons separated by a comma name the repair
  failure and then why replacement was refused. `not_tried` means no collapse was
  tried on the edge, which is usually on a boundary. `link_obstruction` means the edge
  formed during the last pass and is blocked, so repair and replacement weren't tried.
  `nested_triangles_replaced` means the nest was replaced, but the collapse tried
  after that failed.
- `area_ratio`: the output area divided by the source area. The run doesn't judge
  this number, so a value far from 1 is for you to check.

If the run stops with an error once remeshing has started, the same report is in the
error's `diagnostics` attribute. Errors in the arguments have no report.

`max_iter=0` still permits the implementation's final interior-collapse pass; it
does not mean a guaranteed identity operation. The final pass does not advance
the nested-replacement phase. It also tries the short edges that form during it, with
an ordinary collapse only, because a repair there can replace a short edge with another
one forever.

The regression tests cover phase scheduling, repair and nested replacement,
feature numbering/protection, frozen boundaries, source immutability, dtype,
projection, dynamic adjacency, curved nests, and feature VTK exports:

```bash
python -m pytest test/remeshing/ -v
```

See the [synthetic example](../example/remesh_surface.py) and the
[testing guide](testing.md) for optional checks against local data.

## Template barycentric mapping

Install the `vtk` extra for VTK export.

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

The regression tests load separately shared inputs from
`data/test/data/template_barycentric_fixture/`. The tracked
`data/data_record.json` lists their relative paths and purposes. Missing inputs
produce explicit skips; see the [testing guide](testing.md) for the layout.
