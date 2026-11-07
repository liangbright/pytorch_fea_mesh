# Changelog

## 0.1.0 - 2026-11-07

- Mesh containers for polygon, polyline, triangle, quad, mixed quad/triangle,
  tetrahedral, and hexahedral geometry, including quadratic triangle and
  tetrahedral elements.
- Connectivity and geometry operations, smoothing, repair, surface extraction,
  and isotropic triangle surface remeshing with feature protection.
- Rotation, rigid, and similarity transforms, synthetic mesh generation, and
  Chamfer distance utilities.
- Torch serialization and INP/VTK I/O.
- Automated tests, runnable examples, and installation and API documentation.

### Changes since the first commit (2026-10-25)

- `RotationTransform3D`, `RigidTransform3D`, `SimilarityTransform3D` and
  `RodrigueRotationTransform3D` take a new `dtype` input for their parameters. The default
  `None` keeps the old behaviour, and `.double()` still works.
- `RigidTransform3D` and `SimilarityTransform3D` now use `RotationTransform3D.cal_R`, so the
  same `angle` gives the same `R = Rz * Ry * Rx` in all three. Their results change for non-zero
  angles: negate a saved `angle` to get the old rotation.
- Misspelled public names are corrected, and the old names no longer exist:
  `TriangleMesh.subdivid_to_tri_1to4`, `subdivid_to_tri_1to3` and `subdivid_to_tri` are now
  `subdivide_to_tri_1to4`, `subdivide_to_tri_1to3` and `subdivide_to_tri`;
  `create_quad_mesh_rectangle_in_cirlce` is now `create_quad_mesh_rectangle_in_circle`;
  `ReordeNodeInElement` is now `ReorderNodeInElement`; and the name
  `QuadMesh.upate_element_flatness` is gone, so use `update_element_flatness`. Calls that use
  an old name must change.
- The CamelCase aliases of processing and geometry functions are removed, so only the
  snake_case names remain. Calls that use an alias must change to the function it named. Most
  of these are the same name in snake_case, for example `MergeMesh` is now `merge_mesh`. The
  others are `PointToMeshDistance` (now `cal_point_to_mesh_distance`), `MeshToMeshDistance`
  (`cal_mesh_to_mesh_distance`), `ComputeCurvature` (`cal_curvature`), `Subdivision`
  (`subdivide`) and `ComputeAngleBetweenTwoVectorIn3D` (`cal_angle_between_3d_vector`), and
  `ComputeAngleBetweenTwoVectorIn3D_slow` is renamed `cal_angle_between_3d_vector_slow`. The
  removed names are: in `mesh_processing`, `FindConnectedRegion`,
  `SegmentMeshToConnectedRegion`, `SimpleSmoother`, `SimpleSmootherForMesh`,
  `IsEdgeCurveClosed`, `IsNodeCurveClosed`, `TracePolyline`, `MergeMesh`, `RemoveUnusedNode`,
  `FindNearestNode` and `FindNeighborNode`, which `polygon_mesh_processing` also no longer
  imports; in `polyhedron_mesh_processing`, `ExtractSurfaceElement` and `ExtractSurfaceMesh`;
  in `polygon_mesh_processing`, `TraceMeshBoundaryCurve`, `FindMeshBoundaryCurve`,
  `ExtractRegionEnclosedByCurve`, `SegmentMeshByCurve`, `MergeMeshOnBoundary`,
  `SimpleSmootherForMeshNodeNormal`, `SimpleSmootherForMeshElementNormal`,
  `SimpleSmootherForQuadMesh`, `CutMeshByCurve`, `ProjectPointToMesh`, `SmoothAndProject`,
  `PointToMeshDistance`, `MeshToMeshDistance`, `ConvertPolygonMeshToTriangleMesh`,
  `ClipMeshByPlane`, `ClipMeshByNodeAttribute`, `SliceMeshByPlane`, `ComputeCurvature`,
  `FillHole`, `FindDijkstraGraphGeodesicPath` and `Subdivision`, and its import of
  `ComputeAngleBetweenTwoVectorIn3D`; and in `geometry_utility`,
  `ComputeAngleBetweenTwoVectorIn3D`.
- Isotropic remesher: a float32 source no longer raises at the end of a finished run, and
  `failures` now matches the output. Each new position is rounded to float32 when it is proposed,
  so the checks see what is written. Float32 results can change; float64 results don't.
- Isotropic remesher: checking an edit no longer computes corner angles, which makes runs faster
  and removes most of the "L2Norm <= eps" lines.
- Isotropic remesher: its tensors no longer follow torch's default device, so
  `torch.set_default_device` no longer makes it fail. The output's node and element are both on
  the source node's device; before, the element stayed on the source element's device.
- Isotropic remesher: a refused nested replacement whose search went over the face budget now
  reports `nest_over_budget` instead of `no_nest`.
- Isotropic remesher: feature indices in a tensor of any integer type are now converted to
  int64, as numpy arrays and lists already were, so uint16, uint32 and uint64 tensors no longer
  raise.
- Isotropic remesher: new option `flag_use_coverage_then_tangential`, off by default. Plain
  averaging pulls curved surfaces and thin parts in further with each iteration. With this option
  on, interior vertices use coverage-based smoothing and then, in the last two iterations,
  tangential smoothing. With `max_iter=2`, tangential smoothing runs only in the last iteration, and
  with `max_iter=1` it doesn't run.
- Isotropic remesher: checking an edit against protected points and edges no longer visits
  all of them, so runs with many protected items or a frozen boundary are faster. The output
  doesn't change.
- Isotropic remesher: blocked short edges now use nested replacement instead of
  deleting enclosed regions, which could erase closed surfaces. The report field
  `region_deletions` is now `nested_replacements`.
- Isotropic remesher: `flag_process_boundary` now defaults to `False`, so the
  boundary stays fixed unless you turn boundary processing on. Calls that relied
  on the old default must now pass `flag_process_boundary=True`.
- Isotropic remesher: the report field `collapse_outcomes` is replaced by `failures`,
  which gives the edges still outside the length limits, why the short ones failed,
  the change in area, and any error. A run that stops with an error once remeshing
  has started now keeps its report in the error's `diagnostics` attribute.
- Isotropic remesher: the final collapse pass now also tries the short edges that form
  during it, with an ordinary collapse only, so some output meshes change.
- `TriangleMesh` and `QuadMesh` sampling: `uniformly_sample_point` and `sample_point`
  now pick every point's element by area, so a small element may get no point. The old
  guarantee of one point per element when `n_point` exceeds the element count is now the
  option `at_least_one_point_on_one_element=True`, which raises `ValueError` if `n_point`
  is below the element count. Calls that relied on the old behaviour must now pass it.
- `QuadMesh` sampling: points are now uniform on flat quads of any shape and stay inside
  concave quads, because each quad is cut into two triangles. On a warped quad, points
  lie on the two flat triangles, not on the curved surface.
- `TriangleMesh` and `QuadMesh` sampling: picking elements is faster, and the same seed
  now gives different points. A mesh without a valid area raises `ValueError` instead of
  `RuntimeError`, and `n_point=0` returns an empty result.
- Fallback VTK writers (used without VTK, or with VTK 9.0): polylines keep `element_data`,
  field names are encoded as VTK's writer does, and labelled pyramids and 15 and 20-node
  solids get their real cell types. Other solid node counts, and field or node arrays of the
  wrong shape, now raise `ValueError` instead of writing a bad file.
- `polygon_mesh_processing`: a `KeyboardInterrupt` during the optional VTK import is no
  longer swallowed, and the message for a failed import now gives the reason.
- `save_as_vtk` and `save_as_vtp` now raise `RuntimeError` when VTK can't write the file, such as
  for a missing folder or a full disk. They used to return with no file and only a log line.
- `TriangleMesh`, `QuadMesh`, `Tri6Mesh`, `TetrahedronMesh`, `Tet10Mesh` and `HexahedronMesh` now
  raise `ValueError` for elements with another number of nodes, in the constructor, `copy` and the
  VTK, VTP and STL loaders. They used to give wrong geometry with no error. `load_from_torch` still
  restores a mesh exactly as it was saved.
- `load_from_vtk` and `load_vtk_file` now raise `ValueError` for a legacy VTK file that ends before
  the data its headers declare, instead of returning missing or wrong values. A file with no line
  break at its end loads with a warning, because its last value may be cut. A file that VTK reads
  is not refused, so a cut between two complete sections, or right after the header of `POINT_DATA`
  or `CELL_DATA`, can't be seen: such a file loads without those sections.
- A polygon mesh never mixes linear and quadratic elements. `merge_mesh` of a `Tri6Mesh` with a
  `TriangleMesh` or `QuadMesh`, and a label such as `TRI6` on a linear polygon mesh, now raise
  `ValueError`. Merging two `Tri6Mesh` still works.
- VTK voxels and pixels, which come from image data, are now read in the node order of a hex and a
  quad. They used to give crossed faces, with a volume or area of 0, and a voxel was saved as a
  wrong hex. In a `PolyhedronMesh`, a voxel now gets the label `HEX8`, so it can be saved.
- `get_sub_mesh` now raises `ValueError` for a boolean mask, which it never supported: it used to raise
  an unrelated error or select the wrong elements. A one-entry tensor and a CUDA tensor now work as indices.
- `Mesh.copy` now converts its input as the constructor does, so an integer numpy `node` becomes float32
  and an int32 `element` becomes int64. An unsupported input raises `ValueError`, not `NotImplementedError`.
- A list or numpy `element` now goes on the device of `node`, in the constructor and in `copy`. So the
  subdivisions and `quad_to_tri` of a mesh on CUDA return an `element` on CUDA.
- `save_as_torch` now saves `name`. `load_from_torch` restores it, and takes the saved `mesh_type` if it says
  more than the mesh's own: `polygon` becomes `polygon_tri6`, and `polyhedron_hex8` stays as it is.
- `QuadTriangleMesh.cal_element_corner_angle(node, element, ...)`: the third argument is now `return_cos`, as in
  the other classes, not `eps`. Calls that pass `eps` by position must now use `eps=`.
- `QuadTriangleMesh` now raises `ValueError` for an element that isn't a quad or a triangle before the mesh
  changes, so a failed load keeps the mesh as it was. It also has `cal_element_normal`, and `load_from_vtk`
  has `return_mesh_vtk`.
- `PolylineMesh.initialize` and `from_curve` now keep the device and the gradient of a curve given as a tensor.
- The constructor and `copy` now accept elements whose rows are tensors or numpy arrays, such as
  `list(element)`. They used to raise `TypeError` in `QuadTriangleMesh`, and fail later on other classes.
- `read_mesh_vtk_slow` is removed. `read_mesh_vtk` now raises `ValueError` for a VTK object other than
  `vtkPolyData` or `vtkUnstructuredGrid`, such as `vtkImageData`, instead of reading it with that function.
- `self.element` is now an int64 tensor or a nested list of python int, wherever the class sets it: the
  constructor, `copy`, the readers and `load_from_torch`. Any integer type is converted, and a float or a bool
  now raises `ValueError`. A float used to be cut silently, so `1.5` became node 1.
- `read_mesh_vtk`, and so `load_from_vtk`, now raises `ValueError` for a 2D VTK cell whose node order doesn't fit
  a polygon mesh. A linear polygon mesh refuses cells that list their corners first, such as
  `VTK_QUADRATIC_TRIANGLE`. A `Tri6Mesh` reads only `VTK_QUADRATIC_TRIANGLE`, so it refuses a 6-node `VTK_POLYGON`.
  These used to give wrong edges or a wrong area with no error, so files that loaded this way no longer load.
- `get_sub_mesh` now works when `element_type` is a list, which `load_from_torch` can restore. It used to raise
  `TypeError`.
- `load_from_torch` takes a new `dtype` input for `node` and for the floating-point tensors in `node_data` and
  `element_data`. The default `None` keeps the types in the file. `load` now passes its `dtype` on for `.pt` and
  `.pth` files, where it used to print a warning and ignore it, so `load(path, 'float64')` on a float32 file now
  gives float64.
- These now work with `node` on CUDA and an `element` tensor on the CPU, and `element` stays on the CPU:
  `sample_point` and `uniformly_sample_point` of `TriangleMesh` and `QuadMesh`, `update_node_normal` and
  `cal_node_normal` of `TriangleMesh`, `Tri6Mesh` and `QuadMesh`, and `QuadMesh.cal_local_surface_flatness`. They
  used to raise `RuntimeError`. A `node` on the CPU with an `element` on CUDA still raises `RuntimeError`.
- `get_sub_mesh` of a plain `PolygonMesh` or `PolyhedronMesh` now keeps a `mesh_type` that says more, such as
  `polygon_tri6` or `polyhedron_hex8`, which `load_from_torch` can set. It used to reset it, so a `polygon_tri6`
  sub-mesh got wrong edges with no error, and `save_as_vtk` raised `ValueError` on a `polyhedron_hex8` sub-mesh.
- `merge_mesh` of two meshes with the same `mesh_type` now keeps it when the result is a plain `PolygonMesh` or
  `PolyhedronMesh`, such as `polygon_tri6` or `polyhedron_hex8`. It used to reset it, so merged `polygon_tri6` meshes
  got wrong edges with no error, and `save_as_vtk` raised `ValueError` on merged `polyhedron_hex8` meshes. A plain
  `polygon_tri6` mesh with a `Tri6Mesh` now works, where it used to raise `ValueError`. Two plain meshes with
  different `mesh_type`s now give the same result as their classes: a plain `polygon_tri6` mesh with a linear polygon
  mesh now raises `ValueError`, so these merges, which used to give wrong edges with no error, no longer run. A plain
  `polyhedron_hex8` mesh with a plain `polyhedron_tet4` mesh now gets labels, so it can be saved.
- `merge_mesh` now works with `meshA` and `meshB` on different devices, and the output is on the device of
  `meshA.node`. It used to raise `RuntimeError`.
- The fallback VTK writers now behave as the VTK path does, because both use the new `Mesh.make_vtk_cells`. They save
  `polygon_tri6` and `polygon_quad8` meshes, which used to raise `ValueError`. They now raise `ValueError` for a label
  that does not fit its element, for the wrong number of labels, and for a solid whose shape no label or `mesh_type`
  says, such as an 8-node element in a plain `PolyhedronMesh`. These used to be written, some with the wrong cell type,
  so such saves no longer run. A file that cannot be written now raises `RuntimeError`, where it used to raise `OSError`,
  and a partly written file is removed. A number as `filename` now raises `TypeError`, where it used to write the mesh
  to that file descriptor.
- `Mesh.load` now needs `dtype`, so `load(filename)` raises `TypeError`. It used to default to float32.
  `load(filename, None)` now keeps the type in the file, so a float64 VTK file loads as float64, where it used to load
  as float32. Calls that left out `dtype` must now pass it.
- The module `torch_fea_mesh.io.save_polyline_as_vtk_file` is removed, with `save_curve_as_vtk`,
  `convert_curve_to_polyline` and `save_polyline_as_vtk`. Use `PolylineMesh.from_curve(curve).save_as_vtk(filename)`.
- `find_nearest_node` with several points now returns one entry per point, with `None` for a point that has no node
  within `distance_threshold`. It used to leave such points out, so the entries couldn't be matched to the points.
- `subdivide()` of `TetrahedronMesh`, `Tet10Mesh` and `HexahedronMesh` now raises `NotImplementedError`. It used to do
  nothing and return `None`.
- `PolylineMesh.initialize`: the argument `curve_list` is now `curve`, so `initialize(curve_list=...)` raises
  `TypeError`. It now also takes a single curve, as `from_curve` does. A list of points now gives float64 in
  `initialize` and `from_curve`, where it used to give float32. A tensor or a numpy array gives the same type as
  before. So a float32 curve stays float32 as a 2D tensor, but becomes float64 as a list of 1D tensors, which used to
  raise `ValueError`.
- `HexahedronMesh.cal_element_volume` now integrates the trilinear element, so a hex with warped faces gets a different
  volume. `Tet10Mesh.cal_element_volume` now uses the midside nodes, where it used to use only the four corners, so a
  curved element gets a different volume. A hex with flat faces and a tet10 with straight edges keep their volume.
- `QuadMesh.cal_element_area_and_normal`: the area now comes from 2x2 Gauss points on the bilinear surface of the quad,
  so a warped quad gets a different area. A flat quad keeps its area.
- `QuadMesh.cal_node_normal` with `angle_weighted=True` now weights each corner by its angle. It used to divide by the
  sum of the angles of the quad, so concave and warped quads get different node normals. `QuadTriangleMesh.update_node_normal`
  now adds the corners of quads and triangles together. It used to add one unit normal from the quads and one from the
  triangles, so a node shared by both gets a different normal.
- `quad_to_tri('auto')` now cuts along the shorter diagonal that is inside the quad. It used to take the shorter
  diagonal even outside a concave quad, which gave a triangle facing the other way. A convex quad gets the same cut.
- `clip_mesh_by_plane` and `clip_mesh_by_node_attribute` with `return_clipped_output=True` and `clean_output=False`:
  both outputs now have no unused nodes, and their nodes are renumbered. They used to keep every node, used or not.
  With one output, nothing changes.
- `cal_curvature` now triangulates the mesh first, because `vtkCurvatures` handles only triangles, so the curvature of
  a quad mesh changes. A triangle mesh gets the same mean curvature. `'gaussian'`, `'max'` and `'min'` now work, where
  they used to raise `AttributeError`, and `'gauss'` is also accepted.
- `merge_mesh` now returns the closest class of both inputs: two `TriangleMesh` give a `TriangleMesh`, a `TriangleMesh`
  and a `QuadMesh` give a `PolygonMesh`, and a `HexahedronMesh` and a `TetrahedronMesh` give a `PolyhedronMesh`. It used
  to return a plain `Mesh`, and `merge_mesh_on_boundary` used to return a `PolygonMesh`. Meshes of different dimensions
  now raise `ValueError`.
- `read_mesh_vtk` now drops cells of another dimension and prints how many, so a line in a polygon mesh no longer
  becomes an element. A triangle strip is now split into triangles, where it used to become one polygon. An array that
  isn't numeric is skipped with a message, where it used to raise `AttributeError`. `load_from_vtk` of a file that
  declares zero points now gives an empty mesh, where it used to raise `ValueError`.
- Saving a solid with `element_type` now works, where it used to raise `NotImplementedError` for every label. An unknown
  label is only metadata, so the element is saved as if it had no label. A label of another dimension, such as `S4` on a
  `PolyhedronMesh`, now raises `ValueError` in the constructor, and a label that needs another number of nodes raises
  `ValueError` when saving. `get_sub_mesh` of a plain `PolygonMesh` or `PolyhedronMesh` now keeps `element_type`.
- `save_as_vtk` through VTK now writes 17 significant digits, so float64 nodes and fields load back exactly, and files
  are larger.
- `simple_smoother_for_mesh`, `simple_smoother_for_quad_mesh` and `smooth_and_project` now clear derived values such as
  `node_normal` after moving nodes, because they would be out of date. Adjacency tables stay. `smooth_and_project` now
  takes `mask=None`, where it used to raise `AttributeError`.
- VTK-based functions in `polygon_mesh_processing`, such as `clip_mesh_by_plane`, `cal_curvature`, `fill_hole` and
  `subdivide`, now raise `ValueError` for a mesh with quadratic elements, such as a `Tri6Mesh`. Clipping one used to
  return an empty mesh with no error.
- `extract_surface_mesh` now works on a plain `PolyhedronMesh` of tet4 and hex8 elements, where it used to raise
  `AttributeError`. `find_boundary_node` of such a mesh now raises `NotImplementedError`, not `AttributeError`.
- Clearer errors: `simple_smoother_for_mesh_node_normal` and `simple_smoother_for_mesh_element_normal` raise `TypeError`
  for a plain `PolygonMesh`, and `ValueError` if the mesh has no normal to smooth. `project_point_to_mesh` raises
  `ValueError` for a mesh with no elements, where it used to return a meaningless result.
- `cal_distance`, `cal_chamfer_distance` and their batch versions now raise `ValueError` for points with NaN or inf,
  where they used to return NaN, and for inputs with no points, on different devices or with different numbers of
  coordinates.
- New inputs and methods: `update_element_normal` on `TriangleMesh`, `Tri6Mesh`, `QuadMesh` and `QuadTriangleMesh`;
  `QuadMesh.cal_element_normal`; `normal_source` on `update_node_normal` and `cal_node_normal` of `QuadMesh` and
  `QuadTriangleMesh`; `clear_adjacency_data` on `clear_geometry_derived_data`; `follow_element_orientation` on
  `trace_mesh_boundary_curve` and `find_mesh_boundary_curve`; `return_node_idx_list` on `extract_surface_mesh`;
  and `return_cos` on `QuadTriangleMesh.update_element_corner_angle`. See the [API overview](doc/api.md).
- SciPy 1.14 or newer is now a required dependency, so `pip install torch-fea-mesh` installs it, and the `remeshing`
  extra is removed. Template barycentric remeshing and `flag_use_coverage_then_tangential` no longer need an extra.
  An install that still asks for `[remeshing]` works, with a warning from pip that the extra doesn't exist.
