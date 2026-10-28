"""Fixed-boundary parameterization and barycentric template remeshing.

Initialize source UV coordinates, reduce stretch with cumulative weight updates,
and map a planar template onto the source surface. Boundary and landmark
constraints remain fixed throughout. Automatic diffusion selection evaluates
mapped output geometry while choosing a coefficient and accepted iteration.

Initialization methods share weight builders, a constrained linear solver,
stretch refinement, and template mapping. Their docstrings describe the
construction, supported geometry, numerical assumptions, and reference papers.
"""

from copy import deepcopy
from functools import partial
import heapq
import math
from numbers import Integral, Real

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres, spilu, splu
from scipy.special import logsumexp
from scipy.optimize import linprog
import torch

from ..mesh.polygon_mesh import PolygonMesh

# Public interface


def template_barycentric_mapping_based_remesh(
    source_mesh: PolygonMesh,
    boundary_segment_list_of_source_mesh: list[list[int]],
    template_mesh: PolygonMesh,
    boundary_segment_list_of_template_mesh: list[list[int]],
    diffusion_coefficient_of_mesh_parameterization: float | str = "auto",
    max_iteration_of_mesh_parameterization: int = 10,
    *,
    landmark_of_source_mesh: list[int] | None = None,
    landmark_of_template_mesh: list[int] | None = None,
    initialization: str = "untangle",
    linear_solver: str = "sparse_lu",
    solver_options: dict | None = None,
    untangle_options: dict | None = None,
    auto_diffusion_options: dict | None = None,
    computation_dtype: torch.dtype = torch.float64,
    tolerance_factor: float = 1.0,
) -> PolygonMesh:
    """Map a planar polygon template onto a triangular source surface.

    Boundary paths and paired landmarks define fixed correspondences. Initialize
    source UV coordinates, refine them by stretch, and interpolate source positions
    at the template vertices. The output keeps template connectivity and metadata.

    Parameters
    ----------
    source_mesh : PolygonMesh
        Connected, consistently oriented triangular surface with finite 3D
        coordinates. The mesh must be manifold, genus zero, and have a boundary.
    boundary_segment_list_of_source_mesh : list of lists of int
        Ordered vertex paths covering every source boundary edge. A closed path
        repeats its first vertex. A singleton supplies a fixed correspondence.
    template_mesh : PolygonMesh
        Connected planar polygon mesh whose vertices have z equal to zero.
        Boundary loops must correspond one-to-one with the source loops.
    boundary_segment_list_of_template_mesh : list of lists of int
        Paths paired with the source paths, with corresponding direction and
        endpoints. Sampling counts may differ; positions match by relative arc
        length. Singleton paths must be paired with singletons.
    diffusion_coefficient_of_mesh_parameterization : float or "auto", default "auto"
        A value in [0, 1] sets one coefficient; zero keeps the initial map.
        Automatic selection supports triangle and quad templates. It tries 0.5
        first and returns its final accepted state if output geometry passes.
        Otherwise it samples additional coefficients and selects a passing state
        by stretch, with corner margin resolving ties. Each trial resets UV
        coordinates and cumulative weights to the same initialization.
    max_iteration_of_mesh_parameterization : int, default 10
        Maximum candidate refinement solves per coefficient trial. Zero keeps
        initialization. Diagnostics also report total work across automatic trials.
    landmark_of_source_mesh, landmark_of_template_mesh : sequence of int, optional
        Equal-length one-dimensional node-index lists paired by position. Integer
        NumPy arrays and PyTorch tensors are accepted. Supply both or neither.
        Identical repeated pairs are deduplicated; each node has one counterpart.
        Boundary landmarks must match boundary landmarks and follow paired path
        order. They split paths before arc-length correspondence is computed.
        Interior landmarks fix source UV coordinates at the matched template UV.
        Output landmark positions copy source coordinates in the output dtype.
    initialization : str, default "untangle"
        "averaged_barycentric" computes local ring weights and solves for UV.
        "mean_value" uses incident triangle angles and edge lengths.
        "annular" constructs direction and radial fields between two nested
        loops strictly star-shaped about the inner polygon's centroid, then
        computes planar mean-value weights.
        "untangle" starts with averaged-barycentric initialization and repairs
        invalid geometry through constrained optimization. A repaired map uses
        planar mean-value weights. All methods share the same stretch refinement.
    linear_solver : str, default "sparse_lu"
        Sparse LU, "gmres", or "bicgstab" for the same constrained equations.
    solver_options : dict, optional
        Common options are positive "rtol" and nonnegative "atol". Sparse LU
        accepts "permc_spec". Iterative solvers accept "maxiter" and a
        "preconditioner" of "ilu" or "none". ILU uses "drop_tol" and
        "fill_factor". GMRES also accepts "restart". Defaults and accepted
        combinations are specified in ``_validate_solver_options``.
    untangle_options : dict, optional
        Settings for "untangle": positive integer "max_outer_iterations" (80),
        "max_inner_iterations" (100), and "history_size" (20), plus positive
        "area_weight" (1.0). These budgets control nonlinear repair separately
        from the public refinement limit.
    auto_diffusion_options : dict, optional
        Settings for automatic selection: positive integer "max_trials" (9).
        The budget includes the initial trial at 0.5.
    computation_dtype : torch.dtype, default torch.float64
        torch.float32 or torch.float64 for computation and output positions.
        Coordinate reductions use float64 where specified by the numerical policy.
    tolerance_factor : float, default 1.0
        Positive multiplier for precision- and geometry-dependent tolerances.

    Returns
    -------
    PolygonMesh
        A new mesh on the source device with template connectivity, numbering,
        and copied metadata. ``mesh_data['template_barycentric_mapping_based_remesh']``
        contains UV coordinates, constraints, weights, stretch and solver records,
        mapping details, and automatic-selection records when requested.

    Raises
    ------
    ValueError
        Inputs or options are invalid, geometry is unresolved or unsupported,
        constraints conflict, a solve fails, or the search cannot produce a valid
        map or acceptable automatic output within its budget. Exceptions carry
        structured diagnostics when the failing stage provides them.

    Notes
    -----
    Boundaries must be simple, disjoint, and correctly nested. Concave boundaries,
    holes, and interior landmarks require a feasible embedding under the fixed
    constraints. Every initialization and accepted refinement state is validated.
    Automatic output checks require positive resolved corners and valid contacts
    among straight-triangle representations, including both diagonals of quads.
    Those checks apply to the piecewise-planar representation of each element.
    The finite repair and coefficient-search budgets can fail on feasible inputs.

    Inputs remain unchanged. Numerical work runs on the CPU, with NumPy/SciPy
    linear algebra and PyTorch for nonlinear repair. The returned mesh is detached
    from autograd. Each method's docstring cites its reference papers.
    """
    inputs = _prepare_remeshing_inputs(
        source_mesh,
        boundary_segment_list_of_source_mesh,
        template_mesh,
        boundary_segment_list_of_template_mesh,
        diffusion_coefficient_of_mesh_parameterization,
        max_iteration_of_mesh_parameterization,
        landmark_of_source_mesh=landmark_of_source_mesh,
        landmark_of_template_mesh=landmark_of_template_mesh,
        initialization=initialization,
        linear_solver=linear_solver,
        solver_options=solver_options,
        untangle_options=untangle_options,
        auto_diffusion_options=auto_diffusion_options,
        computation_dtype=computation_dtype,
        tolerance_factor=tolerance_factor,
    )
    source_nodes_normalized = inputs["source"]
    source_element = inputs["data"]["source_element"]
    constraints = inputs["constraints"]
    options, numerical_policy = inputs["options"], inputs["policy"]
    try:
        initial_uv, initial_diagnostics = _initialize_source_parameterization(
            source_nodes_normalized,
            source_element,
            inputs["source_topology"],
            constraints["source_index"],
            constraints["source_uv"],
            options,
            numerical_policy,
        )
        if inputs["automatic"]:
            source_uv, diagnostics, output_node, transfer = _select_diffusion_coefficient(
                initial_uv,
                initial_diagnostics,
                inputs,
                options,
                numerical_policy,
                inputs["auto_settings"],
            )
            options["alpha"] = diagnostics["auto_diffusion"]["selected_coefficient"]
        else:
            source_uv, diagnostics = _refine_parameterization_by_stretch(
                source_nodes_normalized,
                source_element,
                constraints["source_index"],
                constraints["source_uv"],
                initial_uv,
                initial_diagnostics,
                options,
                numerical_policy,
            )
    except ValueError as error:
        error.diagnostics = {
            "attempted_iterations": 0,
            **getattr(error, "diagnostics", {}),
            **inputs["transforms"],
            "numerical_policy": numerical_policy,
            "options": options,
            "landmark_correspondence": inputs["landmark_info"],
        }
        raise
    if not inputs["automatic"]:
        output_node, transfer = _map_template_to_source_surface(
            inputs["data"]["source_node_raw"],
            source_element,
            source_uv,
            inputs["template_uv"],
            constraints["template_index"],
            constraints["template_position_raw"],
            numerical_policy,
        )
    diagnostics = _assemble_output_diagnostics(
        inputs,
        source_uv,
        diagnostics,
        transfer,
        source_mesh,
        template_mesh,
        computation_dtype,
    )
    return _build_output_mesh(
        template_mesh, output_node, source_mesh.node.device, computation_dtype, diagnostics
    )


# Input preparation and fixed constraints


def _prepare_remeshing_inputs(
    source_mesh: PolygonMesh,
    boundary_segment_list_of_source_mesh: list[list[int]],
    template_mesh: PolygonMesh,
    boundary_segment_list_of_template_mesh: list[list[int]],
    diffusion_coefficient_of_mesh_parameterization: float | str = "auto",
    max_iteration_of_mesh_parameterization: int = 10,
    *,
    landmark_of_source_mesh: list[int] | None = None,
    landmark_of_template_mesh: list[int] | None = None,
    initialization: str = "untangle",
    linear_solver: str = "sparse_lu",
    solver_options: dict | None = None,
    untangle_options: dict | None = None,
    auto_diffusion_options: dict | None = None,
    computation_dtype: torch.dtype = torch.float64,
    tolerance_factor: float = 1.0,
) -> dict:
    """Validate meshes and options, normalize geometry, and build fixed constraints.

    Return a dictionary containing private mesh data, normalized coordinates,
    topology, numerical policy, solver options, coordinate transforms, and
    boundary/landmark correspondence. Inputs and their metadata remain unchanged.
    """
    diffusion_coefficient = diffusion_coefficient_of_mesh_parameterization
    automatic = isinstance(diffusion_coefficient, str) and diffusion_coefficient == "auto"
    iteration_limit = max_iteration_of_mesh_parameterization
    if not automatic and (
        isinstance(diffusion_coefficient, bool)
        or not isinstance(diffusion_coefficient, Real)
        or not math.isfinite(diffusion_coefficient)
        or not 0 <= diffusion_coefficient <= 1
    ):
        _fail("diffusion_coefficient_of_mesh_parameterization must be in [0,1] or 'auto'")
    if auto_diffusion_options is not None and not automatic:
        _fail("auto_diffusion_options applies only to automatic diffusion")
    auto_settings = _auto_diffusion_validate_options(auto_diffusion_options)
    if (
        isinstance(iteration_limit, bool)
        or not isinstance(iteration_limit, Integral)
        or iteration_limit < 0
    ):
        _fail("max_iteration_of_mesh_parameterization must be a nonnegative integer")
    if initialization not in ("averaged_barycentric", "mean_value", "annular", "untangle"):
        _fail("initialization must be averaged_barycentric, mean_value, annular, or untangle")
    if untangle_options is not None and initialization != "untangle":
        _fail("untangle_options applies only to initialization='untangle'")
    untangle_settings = _untangle_validate_options(untangle_options)
    if (
        isinstance(tolerance_factor, bool)
        or not isinstance(tolerance_factor, Real)
        or not math.isfinite(tolerance_factor)
        or tolerance_factor <= 0
    ):
        _fail("tolerance_factor must be positive and finite")
    data = _prepare_mesh_data(source_mesh, template_mesh, computation_dtype)
    if automatic and any(len(face) not in (3, 4) for face in data["template_element"]):
        _fail("Automatic diffusion supports triangle and quad templates")
    source_topology, template_topology = _build_mesh_topology(
        data["source_private"]
    ), _build_mesh_topology(data["template_private"])
    options = {
        "alpha": 0.5 if automatic else float(diffusion_coefficient),
        "max_iteration": int(iteration_limit),
        "initialization": initialization,
        "linear_solver": linear_solver,
        "untangle_options": untangle_settings,
    }
    source_segments, template_segments = _validate_boundary_correspondence(
        data,
        source_topology,
        template_topology,
        boundary_segment_list_of_source_mesh,
        boundary_segment_list_of_template_mesh,
        options,
    )
    source_segments, template_segments, landmark_info = _prepare_landmark_correspondence(
        landmark_of_source_mesh,
        landmark_of_template_mesh,
        source_segments,
        template_segments,
        source_topology,
        template_topology,
        len(data["source_node_raw"]),
        len(data["template_node_raw"]),
    )
    source_nodes_normalized, template_uv, transforms = _normalize_geometry(
        data["source_node_raw"],
        data["source_element"],
        data["template_node_raw"][:, :2],
        data["numpy_dtype"],
    )
    numerical_policy = _make_numerical_policy(
        source_nodes_normalized,
        data["source_element"],
        template_uv,
        data["numpy_dtype"],
        float(tolerance_factor),
    )
    options["solver_options"] = _validate_solver_options(
        linear_solver, solver_options, numerical_policy
    )
    source_double_area, source_area_tolerance = _triangle_measures(
        source_nodes_normalized[data["source_element"]], numerical_policy
    )
    invalid_faces = np.flatnonzero(source_double_area <= source_area_tolerance)
    if len(invalid_faces):
        _fail(
            "Degenerate or unresolved source triangles",
            face_indices=invalid_faces.tolist(),
            double_area=source_double_area[invalid_faces],
            tolerance=source_area_tolerance[invalid_faces],
        )
    for label, node, topology in (
        ("Source", source_nodes_normalized, source_topology),
        ("Template", template_uv, template_topology),
    ):
        edges = node[topology["edge"]]
        lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
        limits = (
            2
            * numerical_policy["roundoff_factor"]
            * np.max(abs(edges), axis=(1, 2))
            * math.sqrt(node.shape[1])
        )
        bad_edges = np.flatnonzero(lengths <= limits)
        if len(bad_edges):
            _fail(f"{label}: zero-length or unresolved mesh edges", edge_indices=bad_edges.tolist())
    _check_planar_edges(
        template_uv,
        template_topology["boundary_edges"],
        numerical_policy["uv_distance_tolerance"],
        "Template boundary",
    )
    constraints = _find_boundary_constraints(
        source_nodes_normalized,
        template_uv,
        source_segments,
        template_segments,
        numerical_policy,
        data["source_node_raw"],
    )
    prescribed = np.zeros((len(source_nodes_normalized), 2), dtype=source_nodes_normalized.dtype)
    prescribed[constraints["source_index"]] = constraints["source_uv"]
    _check_planar_edges(
        prescribed,
        source_topology["boundary_edges"],
        numerical_policy["uv_distance_tolerance"],
        "Prescribed source boundary",
    )
    free_template = np.setdiff1d(np.arange(len(template_uv)), constraints["template_index"])
    _check_domain_coverage(
        template_uv, free_template, prescribed, source_topology["boundary_edges"], numerical_policy
    )
    return {
        "data": data,
        "source": source_nodes_normalized,
        "template_uv": template_uv,
        "source_topology": source_topology,
        "template_topology": template_topology,
        "transforms": transforms,
        "policy": numerical_policy,
        "options": options,
        "constraints": constraints,
        "landmark_info": landmark_info,
        "source_area_bound": source_area_tolerance,
        "automatic": automatic,
        "auto_settings": auto_settings,
    }


def _prepare_mesh_data(source_mesh, template_mesh, computation_dtype):
    """Copy and validate mesh coordinates, connectivity, and computation precision.

    The source contains triangles and the template contains planar polygons.
    Return raw float64 coordinate copies, connectivity, private PolygonMesh
    objects, and the NumPy computation dtype. Input meshes remain unchanged.
    """
    if computation_dtype not in (torch.float32, torch.float64):
        _fail("computation_dtype must be torch.float32 or torch.float64")
    result = {
        "numpy_dtype": np.dtype(np.float32 if computation_dtype == torch.float32 else np.float64)
    }
    for role, mesh in (("source", source_mesh), ("template", template_mesh)):
        if not isinstance(mesh, PolygonMesh):
            _fail(f"{role}_mesh must be a PolygonMesh")
        node_positions = mesh.node.detach().cpu().to(torch.float64).numpy().copy()
        faces = _copy_mesh_faces(mesh)
        if (
            node_positions.ndim != 2
            or node_positions.shape[1] != 3
            or len(node_positions) < 3
            or not np.isfinite(node_positions).all()
        ):
            _fail(f"{role}: expected finite (n,3) coordinates")
        if not faces:
            _fail(f"{role}: empty mesh")
        for face_index, face_nodes in enumerate(faces):
            if (
                len(face_nodes) < 3
                or len(set(face_nodes)) != len(face_nodes)
                or any(
                    isinstance(node_index, bool)
                    or not isinstance(node_index, Integral)
                    or node_index < 0
                    or node_index >= len(node_positions)
                    for node_index in face_nodes
                )
            ):
                _fail(f"{role}: invalid connectivity in element {face_index}")
        if len({tuple(sorted(face_nodes)) for face_nodes in faces}) != len(faces):
            _fail(f"{role}: duplicate faces")
        if role == "source" and any(len(face_nodes) != 3 for face_nodes in faces):
            _fail("Source must be triangular; triangulate quads explicitly before calling")
        if role == "template" and np.any(node_positions[:, 2] != 0):
            _fail("Every template node must be [x, y, 0]")
        private = PolygonMesh(node=node_positions.copy(), element=deepcopy(faces))
        result[role + "_private"] = private
        result[role + "_node_raw"] = node_positions
        result[role + "_element"] = np.asarray(faces, dtype=np.int64) if role == "source" else faces
    return result


def _build_mesh_topology(mesh):
    """Build connectivity and boundary loops on a private mesh copy.

    Check connectivity, edge incidence, vertex fans, and genus-zero topology.
    Return adjacency tables, boundary membership, loop indices, edge lookup,
    and winding conflicts. PolygonMesh adjacency caches are built in place.
    """
    mesh.build_edge()
    mesh.build_node_to_node_adj_table()
    mesh.build_node_to_element_adj_table()
    mesh.build_edge_to_element_adj_table()
    boundary_node, boundary_edge_index = mesh.find_boundary_node_and_edge()
    edge = mesh.edge.numpy().copy()
    face = _copy_mesh_faces(mesh)
    incidence = deepcopy(mesh.edge_to_element_adj_table)
    adjacency = deepcopy(mesh.node_to_node_adj_table)
    if any(not neighbors for neighbors in adjacency):
        _fail("Unused mesh nodes are not supported")
    if len(_find_connected_vertices(adjacency, 0)) != len(mesh.node):
        _fail("Mesh must be connected")
    nonmanifold_edges = [k for k, faces in enumerate(incidence) if len(faces) not in (1, 2)]
    if nonmanifold_edges:
        _fail("Nonmanifold edge incidence", edge_indices=nonmanifold_edges)
    boundary_node_set = set(boundary_node)
    if not boundary_node_set:
        _fail("Closed meshes require an external cut")
    links = [dict() for _ in mesh.node]
    directed = {}
    for face_index, face_nodes in enumerate(face):
        for k, vertex_index in enumerate(face_nodes):
            a, b = face_nodes[k - 1], face_nodes[(k + 1) % len(face_nodes)]
            links[vertex_index].setdefault(a, []).append(b)
            links[vertex_index].setdefault(b, []).append(a)
            directed.setdefault(tuple(sorted((vertex_index, b))), []).append((vertex_index, b))
    for vertex_index, graph in enumerate(links):
        degrees = [len(a) for a in graph.values()]
        valid = (
            (degrees.count(1) == 2 and all(d in (1, 2) for d in degrees))
            if vertex_index in boundary_node_set
            else all(d == 2 for d in degrees)
        )
        if not valid or len(_find_connected_vertices(graph, next(iter(graph)))) != len(graph):
            _fail(f"Nonmanifold vertex fan at node {vertex_index}")
    boundary_edges = edge[boundary_edge_index]
    boundary_graph = {i: [] for i in boundary_node}
    for a, b in boundary_edges:
        boundary_graph[a].append(int(b))
        boundary_graph[b].append(int(a))
    if any(len(n) != 2 for n in boundary_graph.values()):
        _fail("Boundary must consist of simple topological loops")
    loops, component = [], {}
    for start in sorted(boundary_graph):
        if start in component:
            continue
        loop, previous_node, current_node = [start], None, start
        while True:
            next_node = min(n for n in boundary_graph[current_node] if n != previous_node)
            if next_node == start:
                break
            loop.append(next_node)
            previous_node, current_node = current_node, next_node
        for vertex_index in loop:
            component[vertex_index] = len(loops)
        loops.append(loop)
    euler_characteristic = len(mesh.node) - len(edge) + len(face)
    if euler_characteristic != 2 - len(loops):
        _fail(
            "Only genus-zero patches are supported",
            euler=euler_characteristic,
            boundary_loops=len(loops),
        )
    conflicts = [list(e) for e, dirs in directed.items() if len(dirs) == 2 and dirs[0] == dirs[1]]
    return {
        "edge": edge,
        "edge_to_element_adj_table": incidence,
        "node_to_node_adj_table": adjacency,
        "node_to_element_adj_table": deepcopy(mesh.node_to_element_adj_table),
        "edge_lookup": {tuple(e): k for k, e in enumerate(edge)},
        "boundary_nodes": np.asarray(sorted(boundary_node_set), dtype=np.int64),
        "boundary_edges": boundary_edges,
        "boundary_edge_set": set(map(tuple, boundary_edges)),
        "loops": loops,
        "boundary_component": component,
        "vertex_links": links,
        "orientation_conflicts": conflicts,
        "euler_characteristic": euler_characteristic,
    }


def _normalize_geometry(source_node, source_element, template_uv, numpy_dtype):
    """Normalize source and template coordinates for numerical computation.

    Scale the source to unit total triangle area and the template to unit
    bounding-box diagonal. Return arrays in the requested dtype and the
    float64 centers and scales needed to restore physical coordinates.
    """

    def center_scale(x):
        """Compute a finite bounding-box center and diagonal length."""
        low, high = x.min(0), x.max(0)
        center = low / 2 + high / 2
        span = high / 2 - low / 2
        scale = float(np.linalg.norm(span) * 2)
        if not np.isfinite(scale) or scale <= 0:
            _fail("Nonpositive or unrepresentable geometry extent")
        return center, scale

    center, source_extent = center_scale(source_node)
    source_scaled = (source_node - center) / source_extent
    area = _compute_source_element_area(source_scaled, source_element)
    area_sum = float(np.sum(area, dtype=np.float64))
    source_scale = source_extent * math.sqrt(area_sum)
    uv_center, uv_scale = center_scale(template_uv)
    if not np.isfinite(source_scale) or source_scale <= 0:
        _fail("Nonpositive or unrepresentable source area scale")
    source_nodes_normalized = ((source_node - center) / source_scale).astype(numpy_dtype)
    template_uv_normalized = ((template_uv - uv_center) / uv_scale).astype(numpy_dtype)
    return (
        source_nodes_normalized,
        template_uv_normalized,
        {
            "source_center": center,
            "source_scale": source_scale,
            "uv_center": uv_center,
            "uv_scale": uv_scale,
            "source_area": source_scale**2,
        },
    )


def _make_numerical_policy(
    source_node_normalized, source_element, template_uv_normalized, numpy_dtype, tolerance_factor
):
    """Derive geometry tolerances from computation precision and normalized extent.

    Return roundoff bounds, UV distance tolerance, barycentric conditioning
    limits, and reduction precision. ``tolerance_factor`` scales the bounds.
    """
    eps = float(np.finfo(numpy_dtype).eps)
    f = 64.0 * tolerance_factor * eps
    extent = float(np.linalg.norm(np.ptp(template_uv_normalized.astype(np.float64), axis=0)))
    m = float(np.max(np.abs(template_uv_normalized)))
    return {
        "eps": eps,
        "tolerance_factor": tolerance_factor,
        "roundoff_safety_multiplier": 64.0,
        "roundoff_factor": f,
        "dtype": np.dtype(numpy_dtype).name,
        "uv_extent": extent,
        "uv_distance_tolerance": f * (extent + 4 * math.sqrt(2) * m),
        "barycentric_condition_limit": 1e-2,
        "reduction_dtype": "float64",
    }


def _validate_boundary_correspondence(
    data, source_topology, template_topology, source_segments, template_segments, options
):
    """Check source winding, boundary paths, and one-to-one loop correspondence.

    Return checked source and template segment lists in their supplied order.
    Paired paths must agree on singleton status and explicit closure.
    """
    if source_topology["orientation_conflicts"]:
        _fail("Source winding is inconsistent", edges=source_topology["orientation_conflicts"])
    if len(source_topology["loops"]) != len(template_topology["loops"]):
        _fail("Source and template must have the same number of boundary loops")
    if len(source_segments) != len(template_segments) or not source_segments:
        _fail("The two boundary-segment lists must have the same nonzero length")
    source_boundary_segments = _validate_boundary_segments(
        source_segments, source_topology, len(data["source_node_raw"]), "Source"
    )
    template_boundary_segments = _validate_boundary_segments(
        template_segments, template_topology, len(data["template_node_raw"]), "Template"
    )
    forward, backward = {}, {}
    for segment_index, (source_segment, template_segment) in enumerate(
        zip(source_boundary_segments, template_boundary_segments)
    ):
        if (len(source_segment) == 1) != (len(template_segment) == 1):
            _fail(f"Segment {segment_index}: a singleton must be paired with a singleton")
        if len(source_segment) > 1:
            if (source_segment[0] == source_segment[-1]) != (
                template_segment[0] == template_segment[-1]
            ):
                _fail(f"Segment {segment_index}: closure differs between source and template")
            source_loop, template_loop = (
                source_topology["boundary_component"][source_segment[0]],
                template_topology["boundary_component"][template_segment[0]],
            )
            if (
                forward.setdefault(source_loop, template_loop) != template_loop
                or backward.setdefault(template_loop, source_loop) != source_loop
            ):
                _fail("Boundary loops must correspond bijectively")
    return source_boundary_segments, template_boundary_segments


def _validate_boundary_segments(segments, topology, node_count, label):
    """Validate ordered boundary paths and require coverage of every boundary edge.

    Singleton paths specify fixed vertices. Closed paths repeat their starting
    vertex. Return copied index lists; malformed indices or paths raise ValueError.
    """
    checked, covered = [], set()
    for segment_index, indices in enumerate(segments):
        node_indices = list(indices)
        if not node_indices or any(
            isinstance(i, bool) or not isinstance(i, Integral) or i < 0 or i >= node_count
            for i in node_indices
        ):
            _fail(f"{label} segment {segment_index}: invalid indices")
        if len(node_indices) > 1:
            closed = node_indices[0] == node_indices[-1]
            unique = node_indices[:-1] if closed else node_indices
            if len(set(unique)) != len(unique) or (closed and len(unique) < 3):
                _fail(f"{label} segment {segment_index}: repeated nodes or invalid closure")
            for a, b in zip(node_indices[:-1], node_indices[1:]):
                edge = tuple(sorted((a, b)))
                if edge not in topology["boundary_edge_set"]:
                    _fail(
                        f"{label} segment {segment_index} does not follow a boundary edge: {edge}"
                    )
                covered.add(edge)
        checked.append(node_indices)
    if covered != topology["boundary_edge_set"]:
        _fail(f"{label}: boundary segments must cover every boundary edge")
    return checked


def _prepare_landmark_correspondence(
    source_landmarks,
    template_landmarks,
    source_segments,
    template_segments,
    source_topology,
    template_topology,
    source_node_count,
    template_node_count,
):
    """Merge paired node landmarks with boundary paths before arc-length mapping.

    Entry k on each side specifies a pair. Pair order is immaterial; sorting
    unique pairs makes joint permutations deterministic. Identical repeated
    pairs are allowed, but each node must have one counterpart. Existing path
    endpoints/singletons must agree. Landmarks inside a boundary path split both
    paired paths, so differing arc-length fractions still match exactly.
    Interior mesh landmarks are added as singleton constraints. The helper
    preserves input coordinates, paths, topology, and mesh metadata.
    """
    if (source_landmarks is None) != (template_landmarks is None):
        _fail("landmark_of_source_mesh and landmark_of_template_mesh must be supplied together")
    source_ids = (
        []
        if source_landmarks is None
        else _validate_landmark_indices(
            source_landmarks, source_node_count, "landmark_of_source_mesh"
        )
    )
    template_ids = (
        []
        if template_landmarks is None
        else _validate_landmark_indices(
            template_landmarks, template_node_count, "landmark_of_template_mesh"
        )
    )
    if len(source_ids) != len(template_ids):
        _fail("The two landmark lists must have the same length")
    forward, backward = {}, {}
    for source_index, template_index in zip(source_ids, template_ids):
        if (
            forward.setdefault(source_index, template_index) != template_index
            or backward.setdefault(template_index, source_index) != source_index
        ):
            _fail(
                "Landmarks must have one-to-one correspondence",
                source_index=source_index,
                template_index=template_index,
            )
    pairs = sorted(forward.items())
    boundary_pairs, interior_pairs = [], []
    for source_index, template_index in pairs:
        source_end = source_index in source_topology["boundary_component"]
        template_end = template_index in template_topology["boundary_component"]
        if source_end != template_end:
            _fail(
                "A boundary landmark must match a boundary landmark, and an interior landmark an interior landmark",
                source_index=source_index,
                template_index=template_index,
            )
        (boundary_pairs if source_end else interior_pairs).append((source_index, template_index))
    for source_segment, template_segment in zip(source_segments, template_segments):
        endpoints = (
            [(source_segment[0], template_segment[0])]
            if len(source_segment) == 1
            else [
                (source_segment[0], template_segment[0]),
                (source_segment[-1], template_segment[-1]),
            ]
        )
        for source_index, template_index in endpoints:
            if (source_index in forward and forward[source_index] != template_index) or (
                template_index in backward and backward[template_index] != source_index
            ):
                _fail(
                    "Landmark correspondence conflicts with a boundary endpoint or singleton constraint",
                    existing_source_index=source_index,
                    existing_template_index=template_index,
                )
    source_boundary_segments, template_boundary_segments = [], []
    additional_segments = 0
    for segment, (source_segment, template_segment) in enumerate(
        zip(source_segments, template_segments)
    ):
        if len(source_segment) == 1:
            source_boundary_segments.append(source_segment)
            template_boundary_segments.append(template_segment)
            continue
        # On a closed path the repeated anchor belongs to offset zero; retain
        # the last offset separately as the other endpoint of the final piece.
        source_offsets = {
            v: k
            for k, v in enumerate(
                source_segment[:-1] if source_segment[0] == source_segment[-1] else source_segment
            )
        }
        template_offsets = {
            v: k
            for k, v in enumerate(
                template_segment[:-1]
                if template_segment[0] == template_segment[-1]
                else template_segment
            )
        }
        anchors = {0: 0, len(source_segment) - 1: len(template_segment) - 1}
        for source_index, template_index in boundary_pairs:
            in_source, in_template = (
                source_index in source_offsets,
                template_index in template_offsets,
            )
            if in_source != in_template:
                _fail(
                    "Boundary landmarks must lie on corresponding boundary segments",
                    segment=segment,
                    source_index=source_index,
                    template_index=template_index,
                )
            if in_source:
                anchors[source_offsets[source_index]] = template_offsets[template_index]
        ordered = sorted(anchors.items())
        if any(b[1] <= a[1] for a, b in zip(ordered[:-1], ordered[1:])):
            _fail(
                "Boundary landmark order conflicts with the paired segment direction",
                segment=segment,
            )
        for (source_start, template_start), (source_end, template_end) in zip(
            ordered[:-1], ordered[1:]
        ):
            source_boundary_segments.append(source_segment[source_start : source_end + 1])
            template_boundary_segments.append(template_segment[template_start : template_end + 1])
        additional_segments += len(ordered) - 2
    for source_index, template_index in interior_pairs:
        if not any(
            source_segment == [source_index] and template_segment == [template_index]
            for source_segment, template_segment in zip(
                source_boundary_segments, template_boundary_segments
            )
        ):
            source_boundary_segments.append([source_index])
            template_boundary_segments.append([template_index])
    info = {
        "pairing": "matching list positions; joint pair order is immaterial",
        "source_index": np.asarray(
            [source_segment for source_segment, template_segment in pairs], dtype=np.int64
        ),
        "template_index": np.asarray(
            [template_segment for source_segment, template_segment in pairs], dtype=np.int64
        ),
        "unique_pair_count": len(pairs),
        "boundary_pair_count": len(boundary_pairs),
        "interior_pair_count": len(interior_pairs),
        "additional_boundary_segments": additional_segments,
    }
    return source_boundary_segments, template_boundary_segments, info


def _validate_landmark_indices(value, node_count, label):
    """Read a one-dimensional integer node-index list without changing it."""
    if torch.is_tensor(value):
        if value.ndim != 1:
            _fail(f"{label} must be a one-dimensional list of node indices")
        ids = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        if value.ndim != 1:
            _fail(f"{label} must be a one-dimensional list of node indices")
        ids = value.tolist()
    elif isinstance(value, (list, tuple)):
        ids = list(value)
    else:
        _fail(f"{label} must be a one-dimensional list of node indices")
    if any(
        isinstance(i, bool) or not isinstance(i, Integral) or i < 0 or i >= node_count for i in ids
    ):
        _fail(f"{label} contains an invalid node index")
    return [int(i) for i in ids]


def _find_boundary_constraints(
    source_node_normalized,
    template_uv_normalized,
    source_segments,
    template_segments,
    numerical_policy,
    source_node_raw=None,
):
    """Pair boundary positions by relative arc length and enforce exact landmarks.

    Source coordinates and template UV coordinates are normalized. Optional
    raw source coordinates provide physical output positions. Return source
    fixed indices and UV values, template fixed indices and physical positions,
    and constraint counts. Conflicting repeated constraints raise ValueError.
    Endpoint and singleton values take precedence after agreement is checked.
    """
    source_fixed, template_fixed = {}, {}
    roundoff_factor = numerical_policy["roundoff_factor"]
    source_tolerance = roundoff_factor * (
        np.linalg.norm(np.ptp(source_node_normalized, axis=0))
        + 4 * np.max(abs(source_node_normalized))
    )
    # Check agreement in normalized coordinates. Physical source positions
    # give exact endpoints and accurate output interpolation.
    source_raw = source_node_normalized if source_node_raw is None else source_node_raw
    template_normalized_fixed = {}

    def assign(table, index, value, tolerance):
        """Insert a copied constraint after checking agreement with an existing value."""
        if index in table and np.linalg.norm(table[index] - value) > tolerance:
            _fail(
                f"Conflicting repeated constraint at node {index}",
                existing=table[index],
                incoming=value,
            )
        table.setdefault(index, value.copy())

    for source_segment, template_segment in zip(source_segments, template_segments):
        if len(source_segment) == 1:
            prescribed_uv, prescribed_xyz = (
                template_uv_normalized[template_segment],
                source_raw[source_segment],
            )
            normalized_xyz = source_node_normalized[source_segment]
        else:
            for positions in (
                source_node_normalized[source_segment],
                template_uv_normalized[template_segment],
            ):
                edge_length = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                bound = 2 * roundoff_factor * np.max(abs(positions)) * math.sqrt(positions.shape[1])
                if np.any(edge_length <= bound):
                    _fail("Boundary edge cannot be resolved at computation precision")
            source_arc_fraction = _compute_relative_curve_length(
                source_node_normalized[source_segment]
            )
            template_arc_fraction = _compute_relative_curve_length(
                template_uv_normalized[template_segment]
            )
            prescribed_uv = _interpolate_curve_by_relative_length(
                template_uv_normalized[template_segment], template_arc_fraction, source_arc_fraction
            )
            prescribed_xyz = _interpolate_curve_by_relative_length(
                source_raw[source_segment], source_arc_fraction, template_arc_fraction
            )
            normalized_xyz = _interpolate_curve_by_relative_length(
                source_node_normalized[source_segment], source_arc_fraction, template_arc_fraction
            )
        for i, q in zip(source_segment, prescribed_uv):
            assign(source_fixed, i, q, numerical_policy["uv_distance_tolerance"])
        for i, p, pn in zip(template_segment, prescribed_xyz, normalized_xyz):
            assign(template_normalized_fixed, i, pn, source_tolerance)
            template_fixed.setdefault(i, p.copy())
    # A shared node may first appear inside a longer path and later as an exact
    # endpoint/landmark. After checking agreement above, direct assignments take
    # precedence over rounding-different interpolants, independent of path order.
    exact_source, exact_template = set(), set()
    for source_segment, template_segment in zip(source_segments, template_segments):
        endpoints = (
            [(source_segment[0], template_segment[0])]
            if len(source_segment) == 1
            else [
                (source_segment[0], template_segment[0]),
                (source_segment[-1], template_segment[-1]),
            ]
        )
        for source_indices, template_indices in endpoints:
            if source_indices not in exact_source:
                source_fixed[source_indices] = template_uv_normalized[template_indices].copy()
                exact_source.add(source_indices)
            if template_indices not in exact_template:
                template_fixed[template_indices] = source_raw[source_indices].copy()
                exact_template.add(template_indices)
    source_indices, template_indices = np.array(sorted(source_fixed), dtype=np.int64), np.array(
        sorted(template_fixed), dtype=np.int64
    )
    return {
        "source_index": source_indices,
        "source_uv": np.array([source_fixed[i] for i in source_indices]),
        "template_index": template_indices,
        "template_position_raw": np.array([template_fixed[i] for i in template_indices]),
        "source_constraint_count": len(source_indices),
        "template_constraint_count": len(template_indices),
    }


def _compute_cumulative_curve_length(curve_position):
    """Return cumulative arc length along a finite, nondegenerate polyline.

    The first entry is zero. Lengths use the input coordinate dtype, and
    every step must be positive and representable in that precision.
    """
    length = np.linalg.norm(np.diff(curve_position, axis=0), axis=1)
    if not np.isfinite(length).all() or np.any(length <= 0):
        _fail("Boundary curve has a zero-length or nonfinite segment")
    cumulative = np.concatenate(
        (np.zeros(1, dtype=curve_position.dtype), np.cumsum(length, dtype=curve_position.dtype))
    )
    if not np.isfinite(cumulative).all() or np.any(np.diff(cumulative) <= 0):
        _fail("Cumulative curve lengths cannot be resolved at computation precision")
    return cumulative


def _compute_relative_curve_length(curve_position):
    """Return cumulative polyline arc length normalized to the interval [0, 1]."""
    length = _compute_cumulative_curve_length(curve_position)
    return length / length[-1]


def _interpolate_curve_by_relative_length(
    curve_position, curve_relative_length, query_relative_length
):
    """Interpolate polyline positions at relative arc-length queries.

    Inputs use matching ordered arc fractions. Endpoints are copied exactly.
    Return an array with one position per query and the curve's coordinate dimension.
    """
    index = np.searchsorted(curve_relative_length, query_relative_length, side="right") - 1
    index = np.clip(index, 0, len(curve_position) - 2)
    fraction = (query_relative_length - curve_relative_length[index]) / (
        curve_relative_length[index + 1] - curve_relative_length[index]
    )
    result = curve_position[index] + fraction[:, None] * (
        curve_position[index + 1] - curve_position[index]
    )
    result[query_relative_length == 0] = curve_position[0]
    result[query_relative_length == 1] = curve_position[-1]
    return result


# Initialization selection and shared helpers


def _initialize_source_parameterization(
    source_node_normalized: np.ndarray,
    source_element: np.ndarray,
    source_topology: dict,
    constrained_node_indices: np.ndarray,
    constrained_uv: np.ndarray,
    options: dict,
    numerical_policy: dict,
) -> tuple[np.ndarray, dict]:
    """Select an initializer and prepare its state for shared stretch refinement.

    Return normalized source UV coordinates and diagnostics containing the
    initial weights, their logarithms, validation results, and stretch values.
    Every initializer keeps the supplied boundary and landmark coordinates fixed.
    Method-specific references are given in each initializer's docstring.
    """
    initializers = {
        "averaged_barycentric": _averaged_barycentric_initialize,
        "mean_value": _mean_value_initialize,
        "annular": _annular_initialize,
        "untangle": _untangle_initialize,
    }
    initialization = options["initialization"]
    if initialization not in initializers:
        _fail("initialization must be averaged_barycentric, mean_value, annular, or untangle")
    result = initializers[initialization](
        source_node_normalized,
        source_element,
        source_topology,
        constrained_node_indices,
        constrained_uv,
        options,
        numerical_policy,
    )
    source_uv = result["source_uv"]
    weights = result["initial_weight_matrix"]
    log_weights = weights.copy()
    log_weights.data = np.log(log_weights.data)
    area = _compute_source_element_area(source_node_normalized, source_element)
    face_stretch = _compute_local_stretch_face(
        source_node_normalized,
        source_element,
        source_uv,
        numerical_policy,
    )
    average_stretch = _compute_average_local_stretch(area, face_stretch)
    diagnostics = {
        "free_node_index": np.setdiff1d(
            np.arange(len(source_node_normalized)), constrained_node_indices
        ),
        "solver_records": result["solver_records"],
        "stretch_history": [average_stretch],
        "candidate_stretch_history": [],
        "candidate_records": [],
        "attempted_iterations": 0,
        "accepted_iterations": 0,
        "max_iterations": options["max_iteration"],
        "termination_reason": "initialization_only",
        "initialization_details": result["initialization_details"],
        "initial_uv_validation": result["uv_validation"],
        "final_uv_validation": result["uv_validation"],
        "face_stretch": face_stretch,
        "final_weight_matrix": weights,
        "final_log_weight_matrix": log_weights,
    }
    return source_uv, diagnostics


def _initialization_result(source_uv, weights, solver_records, details):
    """Collect the common initializer fields before geometric validation.

    The weight matrix contains free-node rows. Validation fills ``uv_validation``
    before the result reaches shared refinement. The arrays retain their dtype.
    """
    return {
        "source_uv": source_uv,
        "initial_weight_matrix": weights,
        "uv_validation": None,
        "solver_records": solver_records,
        "initialization_details": details,
    }


def _validate_initialization_result(
    result, source_element, numerical_policy, expected_orientation=None
):
    """Validate an initial UV map and attach its state to a geometric failure."""
    try:
        result["uv_validation"] = _validate_uv_mesh(
            result["source_uv"],
            source_element,
            expected_orientation,
            numerical_policy,
        )
    except ValueError as error:
        error.diagnostics = {
            **getattr(error, "diagnostics", {}),
            "stage": "initialization",
            "source_uv_normalized": result["source_uv"].copy(),
            "solver_records": result["solver_records"],
            "attempted_iterations": 0,
        }
        raise
    return result


def _solve_initial_weighted_map(
    source_nodes, constrained_node_indices, constrained_uv, weights, options, numerical_policy
):
    """Solve the initial two-coordinate map with free vertices starting at zero."""
    source_uv = np.zeros((len(source_nodes), 2), dtype=source_nodes.dtype)
    source_uv[constrained_node_indices] = constrained_uv
    free_node_indices = np.setdiff1d(np.arange(len(source_nodes)), constrained_node_indices)
    solver_records = []
    if len(free_node_indices):
        source_uv = _solve_parameterization_with_fixed_constraints(
            weights,
            free_node_indices,
            constrained_node_indices,
            constrained_uv,
            source_uv,
            options,
            numerical_policy,
            solver_records,
            stage="initialization",
        )
    return source_uv, solver_records


def _build_vertex_weight_matrix(
    source_nodes,
    source_element,
    source_topology,
    free_node_indices,
    compute_vertex_weights,
    numerical_policy,
):
    """Assemble normalized sparse rows in free-node order and sorted column order.

    ``compute_vertex_weights`` receives a vertex and its cyclic neighbor indices.
    The returned CSR matrix has shape (free_node_count, source_node_count).
    """
    row_offsets, column_indices, values = [0], [], []
    for node_index in free_node_indices:
        neighbors = _order_vertex_one_ring(source_element, source_topology, int(node_index))
        weights = compute_vertex_weights(source_nodes, node_index, neighbors, numerical_policy)
        order = np.argsort(neighbors)
        column_indices.extend(neighbors[order])
        values.extend(weights[order])
        row_offsets.append(len(column_indices))
    return sp.csr_matrix(
        (np.asarray(values, dtype=source_nodes.dtype), column_indices, row_offsets),
        shape=(len(free_node_indices), len(source_nodes)),
    )


# Averaged-barycentric initialization


def _averaged_barycentric_initialize(
    source_nodes: np.ndarray,
    source_element: np.ndarray,
    source_topology: dict,
    constrained_node_indices: np.ndarray,
    constrained_uv: np.ndarray,
    options: dict,
    numerical_policy: dict,
) -> dict:
    """Initialize a source UV map using averaged local barycentric weights.

    Local source neighborhoods supply positive weights. Solving their equilibrium
    equations fixes boundary and landmark coordinates at ``constrained_uv``.
    Return the common initialization result with a validated (node_count, 2)
    UV map, a sparse weight matrix, solver records, and method details.

    The source must have nondegenerate triangles and closed free-vertex fans.
    Concave domains, holes, and interior constraints require full UV validation;
    an invalid or numerically unresolved map raises ValueError.

    References
    ----------
    Michael S. Floater (1997), Parametrization and smooth approximation of
    surface triangulations, section 6.
    https://www.cs.jhu.edu/~misha/Fall09/Floater97.pdf
    """
    result = _averaged_barycentric_solve_initial_map(
        source_nodes,
        source_element,
        source_topology,
        constrained_node_indices,
        constrained_uv,
        options,
        numerical_policy,
    )
    return _validate_initialization_result(result, source_element, numerical_policy)


def _averaged_barycentric_solve_initial_map(
    source_nodes: np.ndarray,
    source_element: np.ndarray,
    source_topology: dict,
    constrained_node_indices: np.ndarray,
    constrained_uv: np.ndarray,
    options: dict,
    numerical_policy: dict,
) -> dict:
    """Construct a weighted UV map that the caller can validate or repair.

    The returned initialization result has ``uv_validation=None``. All fixed
    coordinates are assigned exactly; free coordinates solve the weight equations.
    Weight construction follows Michael S. Floater (1997), Parametrization and
    smooth approximation of surface triangulations, section 6.
    https://www.cs.jhu.edu/~misha/Fall09/Floater97.pdf
    """
    free_node_indices = np.setdiff1d(np.arange(len(source_nodes)), constrained_node_indices)
    weights = _averaged_barycentric_build_weight_matrix(
        source_nodes,
        source_element,
        source_topology,
        free_node_indices,
        numerical_policy,
    )
    source_uv, solver_records = _solve_initial_weighted_map(
        source_nodes,
        constrained_node_indices,
        constrained_uv,
        weights,
        options,
        numerical_policy,
    )
    details = {"method": options["initialization"], "extension_to_original_initializer": False}
    return _initialization_result(source_uv, weights, solver_records, details)


def _averaged_barycentric_build_weight_matrix(
    source_nodes: np.ndarray,
    source_element: np.ndarray,
    source_topology: dict,
    free_node_indices: np.ndarray,
    numerical_policy: dict,
) -> sp.csr_matrix:
    """Build averaged-barycentric weights from the supplied 3D vertex neighborhoods.

    Rows correspond to free vertices and columns to all source vertices.
    Each row sums to one. Boundary vertices must be constrained.

    References
    ----------
    Michael S. Floater (1997), Parametrization and smooth approximation of
    surface triangulations, section 6.
    https://www.cs.jhu.edu/~misha/Fall09/Floater97.pdf
    """
    return _build_vertex_weight_matrix(
        source_nodes,
        source_element,
        source_topology,
        free_node_indices,
        _averaged_barycentric_compute_vertex_weights,
        numerical_policy,
    )


def _averaged_barycentric_compute_vertex_weights(
    source_node_normalized, node_index, ordered_neighbor_index, numerical_policy
):
    """Compute averaged barycentric weights in a flattened vertex neighborhood.

    Preserve relative neighbor radii and scale consecutive angles to a full turn.
    Each opposite ray supplies a barycentric representation of the center;
    averaging these representations gives positive weights that sum to one.
    The source coordinates have shape (node_count, 3), and neighbor indices
    follow a closed cyclic fan. Return weights in the same dtype and order.
    Degenerate rings or unresolved intersections raise ValueError.

    References
    ----------
    Michael S. Floater (1997), Parametrization and smooth approximation of
    surface triangulations, section 6, equations (12)-(15).
    https://www.cs.jhu.edu/~misha/Fall09/Floater97.pdf
    """
    neighbor_vectors = (
        source_node_normalized[ordered_neighbor_index] - source_node_normalized[node_index]
    )
    radius = np.linalg.norm(neighbor_vectors, axis=1)
    following = np.roll(neighbor_vectors, -1, axis=0)
    angles = np.arctan2(
        np.linalg.norm(np.cross(neighbor_vectors, following), axis=1),
        np.sum(neighbor_vectors * following, axis=1),
    )
    if np.any(radius <= 0) or np.any(angles <= 0):
        _fail(f"Degenerate local ring at node {node_index}")
    theta = np.r_[np.zeros(1, dtype=neighbor_vectors.dtype), np.cumsum(angles[:-1])] * (
        2 * np.pi / np.sum(angles)
    )
    normalized_radius = radius / np.max(radius)
    ring = np.column_stack(
        (normalized_radius * np.cos(theta), normalized_radius * np.sin(theta))
    ).astype(neighbor_vectors.dtype)
    weights = np.zeros(len(normalized_radius), dtype=neighbor_vectors.dtype)
    roundoff_tolerance = numerical_policy["roundoff_factor"]
    for neighbor_offset in range(len(normalized_radius)):
        ray = -ring[neighbor_offset] / normalized_radius[neighbor_offset]
        # Vertex ties use an exact two-point representation; the loop order
        # makes ties deterministic without double-counting the adjacent edges.
        hits = np.flatnonzero(
            (abs(_cross_2d(ring, ray)) <= roundoff_tolerance * (1 + np.linalg.norm(ring, axis=1)))
            & (ring @ ray > 0)
        )
        if len(hits):
            edge_start = int(hits[0])
            distance = float(np.dot(ring[edge_start], ray))
            vertex_weight = distance / (distance + float(normalized_radius[neighbor_offset]))
            weights[neighbor_offset] += vertex_weight / len(normalized_radius)
            weights[edge_start] += (1 - vertex_weight) / len(normalized_radius)
            continue
        found = False
        for edge_start in range(len(normalized_radius)):
            edge_end = (edge_start + 1) % len(normalized_radius)
            edge = ring[edge_end] - ring[edge_start]
            matrix = np.column_stack((ray, -edge))
            determinant = _cross_2d(matrix[:, 0], matrix[:, 1])
            if abs(determinant) <= roundoff_tolerance * np.linalg.norm(edge):
                continue
            distance, fraction = np.linalg.solve(matrix, ring[edge_start])
            if distance > 0 and 0 <= fraction <= 1:
                vertex_weight = distance / (distance + normalized_radius[neighbor_offset])
                weights[neighbor_offset] += vertex_weight / len(normalized_radius)
                weights[edge_start] += (1 - vertex_weight) * (1 - fraction) / len(normalized_radius)
                weights[edge_end] += (1 - vertex_weight) * fraction / len(normalized_radius)
                found = True
                break
        if not found:
            _fail(f"Opposite ray has no resolvable ring intersection at node {node_index}")
    weights /= weights.sum()
    residual = float(np.linalg.norm(weights @ ring))
    if np.any(weights <= 0) or not np.isfinite(weights).all() or residual > 4 * roundoff_tolerance:
        _fail(
            f"Invalid averaged-barycentric weights at node {node_index}", center_residual=residual
        )
    return weights


# Mean-value initialization


def _mean_value_initialize(
    source_nodes: np.ndarray,
    source_element: np.ndarray,
    source_topology: dict,
    constrained_node_indices: np.ndarray,
    constrained_uv: np.ndarray,
    options: dict,
    numerical_policy: dict,
) -> dict:
    """Initialize source UV coordinates from mean-value weights on the 3D mesh.

    Solve the free-vertex equations with boundary and landmark coordinates fixed.
    Return the common initialization result in the input computation dtype.
    Edge lengths and incident triangles must be resolvable. Full UV validation
    checks for flips and overlaps, including domains with holes or interior pins.

    References
    ----------
    Michael S. Floater (2003), Mean value coordinates.
    https://doi.org/10.1016/S0167-8396(03)00002-5
    """
    free_node_indices = np.setdiff1d(np.arange(len(source_nodes)), constrained_node_indices)
    weights = _mean_value_build_weight_matrix(
        source_nodes,
        source_element,
        source_topology,
        free_node_indices,
        numerical_policy,
    )
    source_uv, solver_records = _solve_initial_weighted_map(
        source_nodes,
        constrained_node_indices,
        constrained_uv,
        weights,
        options,
        numerical_policy,
    )
    result = _initialization_result(
        source_uv,
        weights,
        solver_records,
        {"method": "mean_value", "extension_to_original_initializer": False},
    )
    return _validate_initialization_result(result, source_element, numerical_policy)


def _mean_value_build_weight_matrix(
    source_nodes: np.ndarray,
    source_element: np.ndarray,
    source_topology: dict,
    free_node_indices: np.ndarray,
    numerical_policy: dict,
) -> sp.csr_matrix:
    """Build normalized mean-value weights from source geometry or a planar UV map.

    Supply coordinates with shape (node_count, 3). A planar map uses a zero
    third coordinate. The result has one row per free vertex and one column
    per source vertex. Weights on a valid planar map reproduce its coordinates.

    References
    ----------
    Michael S. Floater (2003), Mean value coordinates.
    https://doi.org/10.1016/S0167-8396(03)00002-5
    """
    compute_weights = partial(
        _mean_value_compute_vertex_weights,
        source_element=source_element,
        source_topology=source_topology,
    )
    return _build_vertex_weight_matrix(
        source_nodes,
        source_element,
        source_topology,
        free_node_indices,
        compute_weights,
        numerical_policy,
    )


def _mean_value_compute_vertex_weights(
    source_nodes, node_index, neighbors, numerical_policy, *, source_element, source_topology
):
    """Normalize mean-value edge weights around one free vertex.

    The actual incident faces supply the two angles for each edge.
    See Michael S. Floater (2003), Mean value coordinates,
    https://doi.org/10.1016/S0167-8396(03)00002-5.
    """
    weights = []
    for neighbor in neighbors:
        edge_index = source_topology["edge_lookup"][tuple(sorted((node_index, neighbor)))]
        opposite = _get_edge_opposite_vertices(
            source_element,
            source_topology["edge"],
            source_topology["edge_to_element_adj_table"],
            edge_index,
        )
        weights.append(
            _mean_value_compute_edge_weight(
                source_nodes,
                node_index,
                neighbor,
                opposite,
                numerical_policy,
            )
        )
    weights = np.asarray(weights, dtype=source_nodes.dtype)
    weights /= weights.sum()
    return weights


def _mean_value_compute_edge_weight(
    source_node_normalized, node_index, adjacent_node_index, opposite_node_index, numerical_policy
):
    """Compute a directed mean-value weight from an edge and its incident triangles.

    Use the two opposite vertices supplied by edge-to-face adjacency. The
    edge must be nonzero and both triangle angles must be resolvable.
    Coordinates have three components, including planar inputs with zero z.
    Return a finite positive scalar or raise ValueError.

    References
    ----------
    Michael S. Floater (2003), Mean value coordinates.
    https://doi.org/10.1016/S0167-8396(03)00002-5
    """
    if len(opposite_node_index) != 2:
        _fail("A free-node edge must have exactly two incident faces")
    edge_vector = source_node_normalized[adjacent_node_index] - source_node_normalized[node_index]
    opposite_vectors = (
        source_node_normalized[opposite_node_index] - source_node_normalized[node_index]
    )
    angle = np.arctan2(
        np.linalg.norm(np.cross(edge_vector, opposite_vectors), axis=1),
        opposite_vectors @ edge_vector,
    )
    weights = np.sum(np.tan(angle / 2)) / np.linalg.norm(edge_vector)
    if not np.isfinite(weights) or weights <= 0:
        _fail("Invalid mean-value edge weight", node=node_index, neighbor=adjacent_node_index)
    return weights


# Annular initialization


def _annular_initialize(
    source_nodes: np.ndarray,
    source_element: np.ndarray,
    source_topology: dict,
    constrained_node_indices: np.ndarray,
    constrained_uv: np.ndarray,
    options: dict,
    numerical_policy: dict,
) -> dict:
    """Initialize UV coordinates between two nested, star-shaped boundary loops.

    Solve direction and radial fields using averaged-barycentric source weights.
    Interpolate between the boundary polygons, validate the UV map, and compute
    planar mean-value weights for its equilibrium solve and shared refinement.
    Return the common initialization result with all supplied constraints fixed.

    Both loops must be strictly star-shaped about the inner polygon's centroid.
    The direction field must stay nonzero, and the UV triangles must pass full
    validation. Unresolved or unsupported geometry raises ValueError.

    References
    ----------
    The annular field construction combines the following published weights.
    Michael S. Floater (1997), Parametrization and smooth approximation of surface
    triangulations, section 6: https://www.cs.jhu.edu/~misha/Fall09/Floater97.pdf
    Michael S. Floater (2003), Mean value coordinates:
    https://doi.org/10.1016/S0167-8396(03)00002-5
    """
    source_uv, weights, details, field_record = _annular_construct_initial_map(
        source_nodes,
        source_element,
        source_topology,
        constrained_node_indices,
        constrained_uv,
        options,
        numerical_policy,
    )
    solver_records = [] if field_record is None else [field_record]
    free_node_indices = np.setdiff1d(np.arange(len(source_nodes)), constrained_node_indices)
    if len(free_node_indices):
        source_uv = _solve_parameterization_with_fixed_constraints(
            weights,
            free_node_indices,
            constrained_node_indices,
            constrained_uv,
            source_uv,
            options,
            numerical_policy,
            solver_records,
            stage="initialization",
        )
    details["seed_equilibrium_max_change"] = float(
        np.max(np.linalg.norm(source_uv - details["seed_uv_normalized"], axis=1))
    )
    result = _initialization_result(source_uv, weights, solver_records, details)
    return _validate_initialization_result(result, source_element, numerical_policy)


def _annular_construct_initial_map(
    source_node,
    source_element,
    source_topology,
    constrained_node_indices,
    constrained_uv,
    options,
    numerical_policy,
):
    """Construct annular UV coordinates from direction and radial fields.

    Fixed coordinates define unit directions and radial fractions between
    the prescribed polygons. Solve those fields, recover UV coordinates,
    validate the triangles, and compute planar mean-value weights. Return
    the UV map, weights, geometry details, and optional field-solver record.

    References
    ----------
    Field weights follow Michael S. Floater (1997), Parametrization and
    smooth approximation of surface triangulations, section 6:
    https://www.cs.jhu.edu/~misha/Fall09/Floater97.pdf
    UV weights follow Michael S. Floater (2003), Mean value coordinates:
    https://doi.org/10.1016/S0167-8396(03)00002-5
    """
    source_node_count = len(source_node)
    free_node_indices = np.setdiff1d(np.arange(source_node_count), constrained_node_indices)
    source_uv = np.zeros((source_node_count, 2), dtype=source_node.dtype)
    source_uv[constrained_node_indices] = constrained_uv
    center, polygons, inner, outer = _annular_prepare_boundary_loops(
        source_uv, source_topology, numerical_policy
    )
    fixed_direction = constrained_uv - center
    fixed_radius = np.linalg.norm(fixed_direction, axis=1)
    if np.any(fixed_radius <= numerical_policy["uv_distance_tolerance"]):
        _fail("An annular constraint coincides with the polar center")
    fixed_direction /= fixed_radius[:, None]
    inner_radius, outer_radius = [
        _annular_compute_boundary_radius(fixed_direction, poly, numerical_policy)
        for poly in polygons
    ]
    gap = outer_radius - inner_radius
    if np.any(gap <= numerical_policy["uv_distance_tolerance"]):
        _fail("Unresolved annular gap at a constraint")
    radial_fraction = (fixed_radius - inner_radius) / gap
    radial_tolerance = numerical_policy["uv_distance_tolerance"] / gap
    if np.any(radial_fraction < -radial_tolerance) or np.any(
        radial_fraction > 1 + radial_tolerance
    ):
        _fail("An annular constraint is outside the prescribed polygonal annulus")
    radial_fraction = np.clip(radial_fraction, 0, 1)
    radial_fraction[np.isin(constrained_node_indices, inner)] = 0
    radial_fraction[np.isin(constrained_node_indices, outer)] = 1
    fields = np.zeros((source_node_count, 3), dtype=source_node.dtype)
    fields[constrained_node_indices] = np.column_stack((fixed_direction, radial_fraction))
    fields, record = _annular_solve_direction_and_radial_fields(
        source_node,
        source_element,
        source_topology,
        constrained_node_indices,
        fields,
        options,
        numerical_policy,
    )
    direction_norm = np.linalg.norm(fields[:, :2], axis=1)
    if np.any(direction_norm <= numerical_policy["roundoff_factor"]):
        _fail("Annular harmonic direction vanishes or is unresolved; no valid angular seed")
    if np.any(fields[:, 2] < -numerical_policy["roundoff_factor"]) or np.any(
        fields[:, 2] > 1 + numerical_policy["roundoff_factor"]
    ):
        _fail("Annular radial field violates its boundary range")
    direction = fields[:, :2] / direction_norm[:, None]
    inner_radius, outer_radius = [
        _annular_compute_boundary_radius(direction, poly, numerical_policy) for poly in polygons
    ]
    if np.any(outer_radius - inner_radius <= numerical_policy["uv_distance_tolerance"]):
        _fail("Unresolved annular gap at an interior direction")
    radius = inner_radius + (outer_radius - inner_radius) * np.clip(fields[:, 2], 0, 1)
    source_uv = (center + radius[:, None] * direction).astype(source_node.dtype)
    source_uv[constrained_node_indices] = constrained_uv
    validity = _validate_uv_mesh(source_uv, source_element, None, numerical_policy)
    # Planar mean-value weights reproduce the validated UV coordinates and
    # provide the starting coefficients for cumulative stretch refinement.
    planar_nodes = np.column_stack((source_uv, np.zeros(source_node_count, dtype=source_uv.dtype)))
    weights = _mean_value_build_weight_matrix(
        planar_nodes, source_element, source_topology, free_node_indices, numerical_policy
    )
    diagnostics = {
        "method": "annular",
        "extension_to_original_initializer": True,
        "center_normalized": center,
        "inner_boundary_node_index": inner,
        "outer_boundary_node_index": outer,
        "field_weight_geometry": "source_averaged_barycentric",
        "refinement_initial_weight_geometry": "validated_seed_uv_mean_value",
        "radial_interpolation": "linear_between_prescribed_polygonal_boundaries",
        "minimum_direction_norm": float(direction_norm.min()),
        "radial_fraction_range": [float(fields[:, 2].min()), float(fields[:, 2].max())],
        "seed_uv_normalized": source_uv.copy(),
        "seed_uv_validation": validity,
    }
    return source_uv, weights, diagnostics, record


def _annular_solve_direction_and_radial_fields(
    source_nodes: np.ndarray,
    source_element: np.ndarray,
    source_topology: dict,
    constrained_node_indices: np.ndarray,
    fields: np.ndarray,
    options: dict,
    numerical_policy: dict,
) -> tuple[np.ndarray, dict | None]:
    """Solve two direction components and a radial fraction with shared source weights.

    ``fields`` has shape (source_node_count, 3), with prescribed values on the
    constrained rows and zero on the free rows. Return solved fields and a solver
    record, or None for the record when every vertex is constrained.

    References
    ----------
    The field equations use the local weights of Michael S. Floater (1997),
    Parametrization and smooth approximation of surface triangulations, section 6.
    https://www.cs.jhu.edu/~misha/Fall09/Floater97.pdf
    """
    free_node_indices = np.setdiff1d(np.arange(len(source_nodes)), constrained_node_indices)
    if not len(free_node_indices):
        return fields, None
    weights = _averaged_barycentric_build_weight_matrix(
        source_nodes,
        source_element,
        source_topology,
        free_node_indices,
        numerical_policy,
    )
    solver_records = []
    fields = _solve_parameterization_with_fixed_constraints(
        weights,
        free_node_indices,
        constrained_node_indices,
        fields[constrained_node_indices],
        fields,
        options,
        numerical_policy,
        solver_records,
        stage="annular_fields",
    )
    record = solver_records[0]
    record["rhs_labels"] = ["direction_x", "direction_y", "radial_fraction"]
    return fields, record


def _annular_prepare_boundary_loops(prescribed_uv, source_topology, numerical_policy):
    """Identify nested loops and verify their common star center.

    Use the inner polygon's area centroid as the center. Both polygonal loops
    must be strictly star-shaped about it and separated by a resolvable gap.
    Return the center, angle-sorted polygons, and inner/outer source indices.
    This geometric check supports the construction in ``_annular_initialize``.
    """
    if len(source_topology["loops"]) != 2:
        _fail("Annular initialization requires exactly two boundary loops")
    loops = [np.asarray(loop, dtype=np.int64) for loop in source_topology["loops"]]
    area2 = []
    for loop in loops:
        centered_boundary = prescribed_uv[loop]
        centered_inner = centered_boundary - centered_boundary.mean(axis=0)
        area2.append(
            float(
                np.sum(
                    _cross_2d(centered_inner, np.roll(centered_inner, -1, axis=0)), dtype=np.float64
                )
            )
        )
    order = np.argsort(np.abs(area2), kind="stable")
    inner_index, outer_index = loops[order[0]], loops[order[1]]
    inner = prescribed_uv[inner_index]
    base = inner.mean(axis=0)
    centered_inner = inner - base
    following = np.roll(centered_inner, -1, axis=0)
    cross = _cross_2d(centered_inner, following)
    total = float(np.sum(cross, dtype=np.float64))
    if abs(total) <= numerical_policy["roundoff_factor"] * float(
        np.max(np.sum(centered_inner * centered_inner, axis=1))
    ):
        _fail("Annular inner boundary has unresolved area")
    center = (
        base
        + np.sum((centered_inner + following) * cross[:, None], axis=0, dtype=np.float64)
        / (3 * total)
    ).astype(prescribed_uv.dtype)
    polygons = []
    for label, boundary_indices in (("inner", inner_index), ("outer", outer_index)):
        centered_boundary = prescribed_uv[boundary_indices] - center
        signed = _cross_2d(centered_boundary, np.roll(centered_boundary, -1, axis=0))
        if np.sum(signed, dtype=np.float64) < 0:
            centered_boundary = centered_boundary[::-1].copy()
            signed = _cross_2d(centered_boundary, np.roll(centered_boundary, -1, axis=0))
        edge = np.roll(centered_boundary, -1, axis=0) - centered_boundary
        length = np.linalg.norm(centered_boundary, axis=1)
        threshold = numerical_policy["roundoff_factor"] * length * np.roll(
            length, -1
        ) + numerical_policy["uv_distance_tolerance"] * np.linalg.norm(edge, axis=1)
        increments = np.arctan2(
            signed, np.sum(centered_boundary * np.roll(centered_boundary, -1, axis=0), axis=1)
        )
        if (
            np.any(length <= numerical_policy["uv_distance_tolerance"])
            or np.any(signed <= threshold)
            or abs(float(np.sum(increments, dtype=np.float64)) - 2 * np.pi)
            > len(centered_boundary) * numerical_policy["roundoff_factor"]
        ):
            _fail(
                f"Annular {label} boundary is not resolvably star-shaped about the inner boundary centroid",
                center=center,
                boundary=label,
            )
        angles = np.mod(np.arctan2(centered_boundary[:, 1], centered_boundary[:, 0]), 2 * np.pi)
        angular_order = np.argsort(angles, kind="stable")
        polygons.append(
            {"vertex": centered_boundary[angular_order], "angle": angles[angular_order]}
        )
    # Disjoint polygonal loops keep the same radial order between vertex rays.
    # Check the combined vertex directions to verify that the loops are nested.
    directions = np.concatenate([polygon["vertex"] for polygon in polygons])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    inner_radius, outer_radius = [
        _annular_compute_boundary_radius(directions, polygon, numerical_policy)
        for polygon in polygons
    ]
    if np.any(outer_radius - inner_radius <= numerical_policy["uv_distance_tolerance"]):
        _fail("Annular boundaries must be nested with a resolvable gap")
    return center, polygons, inner_index, outer_index


def _annular_compute_boundary_radius(direction, polygon, numerical_policy):
    """Intersect unit rays with an angle-sorted star-shaped boundary polygon.

    The polygon is centered at the origin. Return the positive distance
    along each ray, using its intersected edge. Unresolved intersections
    raise ValueError. This helper uses the supplied polygonal boundary.
    """
    angle = np.mod(np.arctan2(direction[:, 1], direction[:, 0]), 2 * np.pi)
    vertex = polygon["vertex"]
    edge_indices = (np.searchsorted(polygon["angle"], angle, side="right") - 1) % len(vertex)
    edge_start = vertex[edge_indices]
    edge = vertex[(edge_indices + 1) % len(vertex)] - edge_start
    denominator = _cross_2d(direction, edge)
    if np.any(
        abs(denominator) <= numerical_policy["roundoff_factor"] * np.linalg.norm(edge, axis=1)
    ):
        _fail("Unresolved annular ray/polygon intersection")
    radius = _cross_2d(edge_start, edge) / denominator
    if not np.isfinite(radius).all() or np.any(radius <= 0):
        _fail("Invalid annular boundary radius")
    return radius


# Untangle initialization


def _untangle_initialize(
    source_nodes: np.ndarray,
    source_element: np.ndarray,
    source_topology: dict,
    constrained_node_indices: np.ndarray,
    constrained_uv: np.ndarray,
    options: dict,
    numerical_policy: dict,
) -> dict:
    """Initialize a UV map and repair it when geometric validation fails.

    Start with averaged-barycentric source weights. A valid map keeps these
    weights. For an invalid map, optimize free UV coordinates with fixed boundary
    and landmark constraints, then compute planar mean-value weights and solve
    their equilibrium equations. Return the common validated initialization result.

    The source must be a manifold genus-zero patch with boundary. Fixed geometry
    must admit a valid embedding. Repair uses a bounded nonlinear optimization;
    failure within that budget raises ValueError. Arithmetic and linear-solver
    failures propagate directly.

    References
    ----------
    Vladimir Garanzha, Igor Kaporin, Liudmila Kudryavtseva, François Protais,
    Nicolas Ray, and Dmitry Sokolov (2021), Foldover-free maps in 50 lines of code,
    equations (2), (3), (5), and (6): https://arxiv.org/abs/2102.03069
    Michael S. Floater (1997), Parametrization and smooth approximation of surface
    triangulations, section 6: https://www.cs.jhu.edu/~misha/Fall09/Floater97.pdf
    Michael S. Floater (2003), Mean value coordinates:
    https://doi.org/10.1016/S0167-8396(03)00002-5
    """
    result = _averaged_barycentric_solve_initial_map(
        source_nodes,
        source_element,
        source_topology,
        constrained_node_indices,
        constrained_uv,
        options,
        numerical_policy,
    )
    source_uv = result["source_uv"]
    try:
        validity = _validate_uv_mesh(source_uv, source_element, None, numerical_policy)
    except ValueError as error:
        if getattr(error, "diagnostics", {}).get("arithmetic_failure"):
            error.diagnostics = {
                **error.diagnostics,
                "stage": "initialization",
                "source_uv_normalized": source_uv.copy(),
                "solver_records": result["solver_records"],
                "attempted_iterations": 0,
            }
            raise
        initial_error = {"message": str(error), "details": getattr(error, "diagnostics", {})}
        source_uv, details = _untangle_repair_parameterization(
            source_nodes,
            source_element,
            source_topology,
            source_uv,
            constrained_node_indices,
            numerical_policy,
            options["untangle_options"],
        )
        details["original_initialization_failure"] = initial_error
        free_node_indices = np.setdiff1d(np.arange(len(source_nodes)), constrained_node_indices)
        planar_nodes = np.column_stack(
            (source_uv, np.zeros(len(source_nodes), dtype=source_uv.dtype))
        )
        weights = _mean_value_build_weight_matrix(
            planar_nodes,
            source_element,
            source_topology,
            free_node_indices,
            numerical_policy,
        )
        source_uv = _solve_parameterization_with_fixed_constraints(
            weights,
            free_node_indices,
            constrained_node_indices,
            constrained_uv,
            source_uv,
            options,
            numerical_policy,
            result["solver_records"],
            stage="untangle_equilibrium",
        )
        details["seed_equilibrium_max_change"] = float(
            np.max(np.linalg.norm(source_uv - details["seed_uv_normalized"], axis=1))
        )
        validity = _validate_uv_mesh(
            source_uv, source_element, details["orientation"], numerical_policy
        )
        result.update(
            source_uv=source_uv, initial_weight_matrix=weights, initialization_details=details
        )
    else:
        result["initialization_details"].update(
            {
                "applied": False,
                "termination_reason": "original_initialization_valid",
                "nonlinear_iterations": 0,
                "function_evaluations": 0,
                "stages": [],
                "fixed_coordinate_max_change": 0.0,
                "refinement_initial_weight_geometry": "source_averaged_barycentric",
            }
        )
    result["uv_validation"] = validity
    return result


def _untangle_validate_options(options):
    """Validate nonlinear iteration budgets, history size, and positive area weight.

    Return a new dictionary with defaults of 80 outer iterations, 100 inner
    iterations, history size 20, and area weight 1.0. These settings control
    the repair solver described in ``_untangle_repair_parameterization``.
    """
    result = {
        "max_outer_iterations": 80,
        "max_inner_iterations": 100,
        "history_size": 20,
        "area_weight": 1.0,
    }
    if options is not None:
        if not isinstance(options, dict) or set(options) - set(result):
            _fail(
                "Unknown untangle_options; expected max_outer_iterations, max_inner_iterations, history_size, area_weight"
            )
        result.update(options)
    for name in ("max_outer_iterations", "max_inner_iterations", "history_size"):
        if (
            isinstance(result[name], bool)
            or not isinstance(result[name], Integral)
            or result[name] <= 0
        ):
            _fail(f"untangle_options[{name}] must be a positive integer")
    value = result["area_weight"]
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        _fail("untangle_options[area_weight] must be positive and finite")
    return result


def _untangle_repair_parameterization(
    source_node,
    source_element: np.ndarray,
    topology: dict,
    initial_uv: np.ndarray,
    constrained_node_indices: np.ndarray,
    numerical_policy: dict,
    settings: dict,
) -> tuple[np.ndarray, dict]:
    """Repair an invalid parameterization while keeping constrained vertices fixed.

    Optimize only free UV coordinates with bounded L-BFGS solves and an
    adaptive regularization parameter. Source geometry, UV coordinates, and
    arithmetic use the selected computation dtype. Return a validated UV map
    and optimization details. Fixed invalid triangles, failed descent, and
    exhaustion of the repair budget raise ValueError with diagnostics.

    The boundary must be simple and correctly nested, and the fixed constraints
    must permit a valid map. Validation of the resulting triangles determines
    success. A finite optimization budget can fail on a feasible problem.

    References
    ----------
    Vladimir Garanzha, Igor Kaporin, Liudmila Kudryavtseva, François Protais,
    Nicolas Ray, and Dmitry Sokolov (2021), Foldover-free maps in 50 lines of
    code, equation (6). https://arxiv.org/abs/2102.03069
    """
    free_node_indices = np.setdiff1d(np.arange(len(initial_uv)), constrained_node_indices)
    inverse_reference, area_fractions, orientation, domain_area = (
        _untangle_build_reference_triangles(
            source_node, source_element, topology, initial_uv, numerical_policy
        )
    )
    measure, bound = _triangle_measures(initial_uv[source_element], numerical_policy)
    immobile = np.all(np.isin(source_element, constrained_node_indices), axis=1)
    invalid_fixed = np.flatnonzero(immobile & (orientation * measure <= bound))
    if len(invalid_fixed):
        _fail(
            "Fixed constraints contain an invalid triangle that cannot move",
            stage="untangling",
            face_indices=invalid_fixed.tolist(),
            source_uv_normalized=initial_uv.copy(),
        )
    if not len(free_node_indices):
        _fail("No free UV coordinates are available for untangling", stage="untangling")
    reference = torch.from_numpy(inverse_reference.copy())
    fractions = torch.from_numpy(area_fractions.astype(source_node.dtype))
    face = torch.from_numpy(source_element)
    free_index = torch.from_numpy(free_node_indices)
    fixed_uv_tensor = torch.from_numpy(initial_uv.copy())
    free_uv = torch.tensor(
        initial_uv[free_node_indices], dtype=fixed_uv_tensor.dtype, requires_grad=True
    )
    details = {
        "method": "untangle",
        "extension_to_original_initializer": True,
        "applied": True,
        "reference": "Garanzha et al. 2021, equations (2), (3), (5), (6)",
        "optimizer": "torch_lbfgs_strong_wolfe",
        "options": dict(settings),
        "orientation": orientation,
        "reference_total_area": domain_area,
        "initial_uv_normalized": initial_uv.copy(),
        "stages": [],
        "fixed_coordinate_max_change": 0.0,
        "nonlinear_iterations": 0,
        "function_evaluations": 0,
        "refinement_initial_weight_geometry": "validated_seed_uv_mean_value",
    }
    epsilon = 1.0
    current_uv = initial_uv.copy()
    # Enable local gradients for repair and retain the chosen computation dtype.
    # The caller's dtype, gradient mode, and thread settings remain unchanged.
    with torch.enable_grad():
        for stage in range(settings["max_outer_iterations"]):
            optimizer = torch.optim.LBFGS(
                [free_uv],
                lr=1.0,
                max_iter=settings["max_inner_iterations"],
                max_eval=2 * settings["max_inner_iterations"],
                history_size=settings["history_size"],
                tolerance_grad=max(1e-10, 8 * numerical_policy["eps"]),
                tolerance_change=8 * numerical_policy["eps"],
                line_search_fn="strong_wolfe",
            )

            def evaluate():
                """Reconstruct the constrained tensor map and evaluate its repair energy."""
                source_uv = fixed_uv_tensor.index_copy(0, free_index, free_uv)
                return _untangle_compute_energy(
                    source_uv,
                    face,
                    reference,
                    fractions,
                    orientation,
                    epsilon,
                    settings["area_weight"],
                )

            def closure():
                """Evaluate repair energy and gradients for the L-BFGS line search."""
                optimizer.zero_grad()
                energy, _ = evaluate()
                if not torch.isfinite(energy):
                    _fail("Nonfinite untangling objective", stage="untangling", untangling=details)
                energy.backward()
                if free_uv.grad is None or not torch.isfinite(free_uv.grad).all():
                    _fail("Nonfinite untangling gradient", stage="untangling", untangling=details)
                return energy

            before = float(evaluate()[0].detach())
            try:
                optimizer.step(closure)
            except RuntimeError as error:
                _fail(
                    f"Untangling optimizer failed: {error}",
                    stage="untangling",
                    untangling=details,
                    source_uv_normalized=current_uv,
                )
            energy, determinant = evaluate()
            after, minimum = float(energy.detach()), float(determinant.min().detach())
            state = optimizer.state[free_uv]
            details["nonlinear_iterations"] += state.get("n_iter", 0)
            details["function_evaluations"] += state.get("func_evals", 0)
            current_uv = fixed_uv_tensor.index_copy(0, free_index, free_uv).detach().numpy().copy()
            if (
                not np.isfinite(current_uv).all()
                or not math.isfinite(after)
                or after > before * (1 + 64 * numerical_policy["eps"])
            ):
                _fail(
                    "Untangling solve did not produce a finite descent step",
                    stage="untangling",
                    untangling=details,
                    source_uv_normalized=current_uv,
                )
            assert np.array_equal(
                current_uv[constrained_node_indices], initial_uv[constrained_node_indices]
            )
            measure, bound = _triangle_measures(current_uv[source_element], numerical_policy)
            record = {
                "stage": stage + 1,
                "epsilon": epsilon,
                "energy_before": before,
                "energy_after": after,
                "minimum_determinant": minimum,
                "invalid_triangle_count": int(np.sum(orientation * measure <= bound)),
                "iterations": state.get("n_iter", 0),
                "function_evaluations": state.get("func_evals", 0),
            }
            details["stages"].append(record)
            try:
                validity = _validate_uv_mesh(
                    current_uv, source_element, orientation, numerical_policy
                )
            except ValueError as error:
                record["validity_error"] = str(error)
            else:
                details.update(
                    {
                        "termination_reason": "valid_uv",
                        "seed_uv_normalized": current_uv.copy(),
                        "seed_uv_validation": validity,
                    }
                )
                return current_uv, details
            # Reduce regularization with the progress factor in equation (6).
            # The next stage is accepted through full geometric validation.
            sigma = max(0.1, 1 - after / before)
            root = math.hypot(minimum, epsilon)
            factor = 1 - sigma * root / (abs(minimum) + root) if minimum < 0 else 1 - sigma
            epsilon *= factor
            if epsilon <= np.finfo(source_node.dtype).tiny ** 0.25:
                break
    details["termination_reason"] = "no_valid_initialization_found"
    _fail(
        "Untangling found no valid map within its budget. Feasibility remains undetermined.",
        stage="untangling",
        untangling=details,
        source_uv_normalized=current_uv,
    )


def _untangle_build_reference_triangles(
    source_node, source_element, topology, source_uv, numerical_policy
):
    """Build local triangle frames scaled to the prescribed UV domain area.

    Directed source boundary edges determine the target orientation. Return
    inverse reference matrices, area fractions, orientation, and domain area.
    The source triangles and fixed boundary must have resolvable area.

    References
    ----------
    Vladimir Garanzha et al. (2021), Foldover-free maps in 50 lines of code,
    sections 2 and 3.1. https://arxiv.org/abs/2102.03069
    """
    directed = [
        (int(first_edge), int(second_edge))
        for face in source_element
        for first_edge, second_edge in zip(face, np.roll(face, -1))
        if tuple(sorted((first_edge, second_edge))) in topology["boundary_edge_set"]
    ]
    boundary = np.asarray(directed)
    centered_uv = source_uv.astype(np.float64) - source_uv[topology["boundary_nodes"]].astype(
        np.float64
    ).mean(0)
    signed_area = float(
        np.sum(_cross_2d(centered_uv[boundary[:, 0]], centered_uv[boundary[:, 1]])) / 2
    )
    if (
        abs(signed_area)
        <= numerical_policy["uv_distance_tolerance"] * numerical_policy["uv_extent"]
    ):
        _fail("Prescribed domain has unresolved signed area")
    orientation = 1 if signed_area > 0 else -1
    source_triangles = source_node[source_element]
    first_edge, second_edge = (
        source_triangles[:, 1] - source_triangles[:, 0],
        source_triangles[:, 2] - source_triangles[:, 0],
    )
    length = np.linalg.norm(first_edge, axis=1)
    double_area = np.linalg.norm(np.cross(first_edge, second_edge), axis=1)
    area = double_area / 2
    scale = math.sqrt(abs(signed_area) / float(np.sum(area, dtype=np.float64)))
    reference = np.zeros((len(source_triangles), 2, 2), dtype=source_node.dtype)
    reference[:, 0, 0] = length
    reference[:, 0, 1] = np.sum(first_edge * second_edge, axis=1) / length
    reference[:, 1, 1] = double_area / length
    reference *= scale
    inverse = np.linalg.inv(reference)
    if not np.isfinite(inverse).all():
        _fail("Nonfinite untangling reference geometry")
    return inverse, area / np.sum(area, dtype=np.float64), orientation, abs(signed_area)


def _untangle_compute_energy(
    source_uv, face, inverse_reference, area_fraction, orientation, epsilon, area_weight
):
    """Evaluate the smooth shape-and-area energy and signed Jacobian determinants.

    Tensor UV coordinates have shape (node_count, 2). Reference inverses have
    shape (face_count, 2, 2), and area fractions sum to one. Return a scalar
    differentiable energy and one oriented determinant per triangle. The
    regularized determinant formula remains stable across inverted triangles.

    References
    ----------
    Vladimir Garanzha et al. (2021), Foldover-free maps in 50 lines of code,
    equations (2), (3), and (5). https://arxiv.org/abs/2102.03069
    """
    triangle = source_uv[face]
    edges = torch.stack((triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), dim=2)
    jacobian = edges @ inverse_reference
    determinant = orientation * (
        jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]
    )
    epsilon_tensor = torch.as_tensor(epsilon, dtype=source_uv.dtype, device=source_uv.device)
    root = torch.hypot(determinant, epsilon_tensor)
    chi = torch.where(
        determinant >= 0,
        (root + determinant) / 2,
        (epsilon_tensor / 2) * (epsilon_tensor / (root + torch.abs(determinant))),
    )
    density = (
        torch.sum(jacobian * jacobian, dim=(1, 2)) + area_weight * (determinant * determinant + 1)
    ) / chi
    return torch.sum(area_fraction * density), determinant


# Shared stretch computation and diffusion


def _refine_parameterization_by_stretch(
    source_node_normalized: np.ndarray,
    source_element: np.ndarray,
    constrained_node_indices: np.ndarray,
    constrained_uv: np.ndarray,
    initial_uv: np.ndarray,
    initial_diagnostics: dict,
    options: dict,
    numerical_policy: dict,
    inspect_iterate=None,
) -> tuple[np.ndarray, dict]:
    """Refine a parameterization with cumulative stretch-based weight updates.

    Copy the initial UV map and diagnostics before modifying them. Every
    candidate preserves fixed coordinates and must pass UV validation with
    non-increasing average stretch. Return the last accepted map and its
    diagnostics. A zero coefficient or iteration cap retains initialization.

    ``inspect_iterate``, when supplied, observes the seed and each accepted
    map. Automatic selection uses this observer to retain eligible outputs.
    Invalid geometry ends refinement; arithmetic and solver failures raise.

    References
    ----------
    Shin Yoshizawa, Alexander Belyaev, and Hans-Peter Seidel (2004), A fast
    and simple stretch-minimizing mesh parameterization, equations (1)-(6).
    https://doi.org/10.1109/SMI.2004.1314507
    """
    source_node_count = len(source_node_normalized)
    source_uv, diagnostics = initial_uv.copy(), deepcopy(initial_diagnostics)
    free_node_indices = diagnostics["free_node_index"]
    diagnostics["max_iterations"] = options["max_iteration"]
    weights, log_weights = (
        diagnostics["final_weight_matrix"],
        diagnostics["final_log_weight_matrix"],
    )
    validity, face_stretch = diagnostics["final_uv_validation"], diagnostics["face_stretch"]
    average_stretch = diagnostics["stretch_history"][-1]
    source_element_area = _compute_source_element_area(source_node_normalized, source_element)

    def observe():
        """Record the current numerical state and notify the optional iterate observer."""
        diagnostics.update(
            final_uv_validation=validity,
            face_stretch=face_stretch,
            final_weight_matrix=weights,
            final_log_weight_matrix=log_weights,
        )
        if inspect_iterate is not None:
            inspect_iterate(source_uv, diagnostics)

    observe()
    if not len(free_node_indices):
        diagnostics["termination_reason"] = "no_free_nodes"
    elif options["alpha"] > 0 and options["max_iteration"] > 0:
        diagnostics["termination_reason"] = "iteration_limit"
        for attempt in range(1, options["max_iteration"] + 1):
            try:
                node_stretch = _compute_local_stretch_node(
                    source_element, source_element_area, face_stretch, source_node_count
                )
                candidate_log, candidate_weight = _update_log_weight_by_stretch(
                    log_weights, node_stretch, options["alpha"]
                )
            except ValueError as error:
                error.diagnostics = {
                    **getattr(error, "diagnostics", {}),
                    "attempted_iterations": diagnostics["attempted_iterations"],
                }
                raise
            diagnostics["attempted_iterations"] = attempt
            candidate_uv = _solve_parameterization_with_fixed_constraints(
                candidate_weight,
                free_node_indices,
                constrained_node_indices,
                constrained_uv,
                source_uv,
                options,
                numerical_policy,
                diagnostics["solver_records"],
                stage="stretch_candidate",
                attempt=attempt,
            )
            if not np.isfinite(candidate_uv).all():
                _fail("Nonfinite candidate UV", attempted_iterations=attempt)
            try:
                candidate_validity = _validate_uv_mesh(
                    candidate_uv, source_element, validity["orientation"], numerical_policy
                )
            except ValueError as error:
                if getattr(error, "diagnostics", {}).get("arithmetic_failure"):
                    error.diagnostics["attempted_iterations"] = attempt
                    raise
                diagnostics["candidate_stretch_history"].append(None)
                diagnostics["candidate_records"].append(
                    {
                        "iteration": attempt,
                        "accepted": False,
                        "reason": "invalid_uv",
                        "error": str(error),
                        "details": getattr(error, "diagnostics", {}),
                    }
                )
                diagnostics["termination_reason"] = "invalid_uv"
                break
            try:
                candidate_face_stretch = _compute_local_stretch_face(
                    source_node_normalized, source_element, candidate_uv, numerical_policy
                )
                candidate_stretch = _compute_average_local_stretch(
                    source_element_area, candidate_face_stretch
                )
            except ValueError as error:
                error.diagnostics = {
                    **getattr(error, "diagnostics", {}),
                    "attempted_iterations": attempt,
                    "last_solver_record": diagnostics["solver_records"][-1],
                }
                raise
            diagnostics["candidate_stretch_history"].append(candidate_stretch)
            accepted = candidate_stretch <= average_stretch
            diagnostics["candidate_records"].append(
                {
                    "iteration": attempt,
                    "stretch": candidate_stretch,
                    "accepted": accepted,
                    "reason": "accepted" if accepted else "stretch_increase",
                }
            )
            if not accepted:
                diagnostics["termination_reason"] = "stretch_increase"
                break
            source_uv, weights, log_weights = candidate_uv, candidate_weight, candidate_log
            average_stretch, face_stretch, validity = (
                candidate_stretch,
                candidate_face_stretch,
                candidate_validity,
            )
            diagnostics["stretch_history"].append(average_stretch)
            diagnostics["accepted_iterations"] += 1
            observe()
    diagnostics["final_uv_validation"] = validity
    diagnostics["face_stretch"] = face_stretch
    diagnostics["final_weight_matrix"] = weights
    diagnostics["final_log_weight_matrix"] = log_weights
    return source_uv, diagnostics


def _parameterize_source_mesh(
    source_node_normalized,
    source_element,
    source_topology,
    constrained_node_indices,
    constrained_uv,
    options,
    numerical_policy,
):
    """Initialize a source map and refine it with one fixed diffusion coefficient.

    Return normalized UV coordinates and refinement diagnostics. Initializer
    references appear in their method docstrings. Shared refinement follows
    Yoshizawa, Belyaev, and Seidel (2004), A fast and simple stretch-minimizing
    mesh parameterization: https://doi.org/10.1109/SMI.2004.1314507
    """
    source_uv, diagnostics = _initialize_source_parameterization(
        source_node_normalized,
        source_element,
        source_topology,
        constrained_node_indices,
        constrained_uv,
        options,
        numerical_policy,
    )
    return _refine_parameterization_by_stretch(
        source_node_normalized,
        source_element,
        constrained_node_indices,
        constrained_uv,
        source_uv,
        diagnostics,
        options,
        numerical_policy,
    )


def _compute_source_element_area(source_node, source_element):
    """Return the unsigned 3D area of each source triangle."""
    triangles = source_node[source_element]
    return (
        np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        / 2
    )


def _compute_local_stretch_face(
    source_node_normalized, source_element, source_uv_normalized, numerical_policy
):
    """Compute the RMS singular-value stretch for each source triangle.

    Source coordinates are normalized 3D positions; UV coordinates are the
    normalized 2D parameterization. Return one positive stretch per face.

    References
    ----------
    Shin Yoshizawa, Alexander Belyaev, and Hans-Peter Seidel (2004), A fast
    and simple stretch-minimizing mesh parameterization, equations (1)-(3).
    https://doi.org/10.1109/SMI.2004.1314507
    """
    source_triangles, uv_triangles = (
        source_node_normalized[source_element],
        source_uv_normalized[source_element],
    )
    source_edges = np.stack(
        (
            source_triangles[:, 1] - source_triangles[:, 0],
            source_triangles[:, 2] - source_triangles[:, 0],
        ),
        axis=2,
    )
    uv_edges = np.stack(
        (uv_triangles[:, 1] - uv_triangles[:, 0], uv_triangles[:, 2] - uv_triangles[:, 0]), axis=2
    )
    jacobian = np.linalg.solve(
        uv_edges.transpose(0, 2, 1), source_edges.transpose(0, 2, 1)
    ).transpose(0, 2, 1)
    stretch = np.sqrt(np.sum(jacobian * jacobian, axis=(1, 2)) / 2)
    if np.any(stretch <= 0) or not np.isfinite(stretch).all():
        _fail("Nonfinite or nonpositive face stretch")
    return stretch


def _compute_local_stretch_node(source_element, source_element_area, face_stretch, n_source_node):
    """Compute area-weighted RMS stretch over each vertex's incident triangles.

    Accumulate areas and squared stretches in float64, then return one value
    per source vertex in the face-stretch dtype. Unused vertices are errors.

    References
    ----------
    Shin Yoshizawa, Alexander Belyaev, and Hans-Peter Seidel (2004), A fast
    and simple stretch-minimizing mesh parameterization.
    https://doi.org/10.1109/SMI.2004.1314507
    """
    area = source_element_area.astype(np.float64)
    weighted = area * face_stretch.astype(np.float64) ** 2
    denominator = np.bincount(
        source_element.ravel(), weights=np.repeat(area, 3), minlength=n_source_node
    )
    numerator = np.bincount(
        source_element.ravel(), weights=np.repeat(weighted, 3), minlength=n_source_node
    )
    if np.any(denominator <= 0):
        _fail("Undefined stretch at an unused or zero-area node")
    result = np.sqrt(numerator / denominator).astype(face_stretch.dtype)
    if not np.isfinite(result).all() or np.any(result <= 0):
        _fail("Nonfinite or nonpositive node stretch")
    return result


def _compute_average_local_stretch(source_element_area, face_stretch):
    """Return the source-area-weighted RMS of face stretch as a finite scalar.

    References
    ----------
    Shin Yoshizawa, Alexander Belyaev, and Hans-Peter Seidel (2004), A fast
    and simple stretch-minimizing mesh parameterization.
    https://doi.org/10.1109/SMI.2004.1314507
    """
    area, stretch = source_element_area.astype(np.float64), face_stretch.astype(np.float64)
    result = float(np.sqrt(np.sum(area * stretch**2) / np.sum(area)))
    if not math.isfinite(result):
        _fail("Nonfinite global stretch")
    return result


def _update_log_weight_by_stretch(log_weight_matrix, node_stretch, diffusion_coefficient):
    """Reweight each destination vertex by its stretch and normalize each row.

    Start from the current logarithmic CSR weights so updates accumulate.
    Return new logarithmic and ordinary weight matrices. Input weights remain
    unchanged. Unrepresentable or nonpositive weights raise ValueError.

    References
    ----------
    Shin Yoshizawa, Alexander Belyaev, and Hans-Peter Seidel (2004), A fast
    and simple stretch-minimizing mesh parameterization, equation (6).
    https://doi.org/10.1109/SMI.2004.1314507
    """
    candidate_log_weights = log_weight_matrix.copy()
    candidate_log_weights.data -= diffusion_coefficient * np.log(
        node_stretch[candidate_log_weights.indices]
    )
    for row_index in range(candidate_log_weights.shape[0]):
        start, end = candidate_log_weights.indptr[row_index : row_index + 2]
        candidate_log_weights.data[start:end] -= logsumexp(candidate_log_weights.data[start:end])
    weights = candidate_log_weights.copy()
    weights.data = np.exp(weights.data)
    if not np.isfinite(weights.data).all() or np.any(weights.data <= 0):
        _fail("Cumulative weights cannot be represented in computation dtype")
    return candidate_log_weights, weights


# Automatic diffusion selection and output checks


def _select_diffusion_coefficient(
    initial_uv: np.ndarray,
    initial_diagnostics: dict,
    inputs: dict,
    options: dict,
    numerical_policy: dict,
    settings: dict,
) -> tuple[np.ndarray, dict, np.ndarray, dict]:
    """Select a coefficient and accepted iterate using mapped output quality.

    Each trial starts from private copies of the supplied UV map and weights.
    The final accepted state at coefficient 0.5 is returned immediately if its
    corners and intersections pass. Otherwise, sample additional coefficients
    within ``max_trials`` and choose a passing state by stretch and corner margin.
    Return selected UV coordinates, diagnostics, output positions, and mapping
    diagnostics. Failure to find an acceptable sample raises ValueError with the
    trial records. Each coefficient uses the same per-trial iteration limit.

    References
    ----------
    The refinement in each trial follows Shin Yoshizawa, Alexander Belyaev,
    and Hans-Peter Seidel (2004), A fast and simple stretch-minimizing mesh
    parameterization: https://doi.org/10.1109/SMI.2004.1314507
    Coefficient sampling and output acceptance are defined by this search.
    """
    audit = {
        "requested_coefficient": "auto",
        "max_trials": settings["max_trials"],
        "initialization_runs": 1,
        "trials": [],
        "selected_coefficient": None,
        "selected_iteration": None,
        "total_refinement_attempts": 0,
    }
    candidates = []
    first_trial = _auto_diffusion_run_trial(
        0.5,
        initial_uv,
        initial_diagnostics,
        inputs,
        options,
        numerical_policy,
        audit,
        candidates,
    )
    final_candidate = next(
        (
            candidate
            for candidate in reversed(candidates)
            if candidate["iteration"] == first_trial["accepted_iterations"]
        ),
        None,
    )
    if final_candidate is not None and _auto_diffusion_validate_candidate(
        final_candidate, inputs, numerical_policy
    ):
        return _auto_diffusion_finish(final_candidate, audit, "default_coefficient_passed")

    can_refine = options["max_iteration"] > 0 and len(initial_diagnostics["free_node_index"]) > 0
    if can_refine:
        for coefficient in (0.0, 0.25, 0.75, 1.0):
            if len(audit["trials"]) >= settings["max_trials"]:
                break
            _auto_diffusion_run_trial(
                coefficient,
                initial_uv,
                initial_diagnostics,
                inputs,
                options,
                numerical_policy,
                audit,
                candidates,
            )
        while len(audit["trials"]) < settings["max_trials"]:
            coefficient = _auto_diffusion_next_coefficient(audit["trials"])
            if coefficient is None:
                break
            _auto_diffusion_run_trial(
                coefficient,
                initial_uv,
                initial_diagnostics,
                inputs,
                options,
                numerical_policy,
                audit,
                candidates,
            )

    for candidate in sorted(candidates, key=_auto_diffusion_candidate_priority):
        if _auto_diffusion_validate_candidate(candidate, inputs, numerical_policy):
            return _auto_diffusion_finish(candidate, audit, "best_passing_sample")
    audit.update(
        termination_reason="no_passing_coefficient_found", trial_count=len(audit["trials"])
    )
    _fail(
        "Automatic diffusion found no acceptable output within its search budget. "
        "Feasibility remains undetermined.",
        stage="auto_diffusion",
        auto_diffusion=audit,
    )


def _auto_diffusion_validate_options(options):
    """Validate the positive trial budget and return its settings, defaulting to nine trials."""
    settings = {"max_trials": 9}
    if options is not None:
        if not isinstance(options, dict) or set(options) - set(settings):
            _fail("auto_diffusion_options accepts only max_trials")
        settings.update(options)
    value = settings["max_trials"]
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        _fail("auto_diffusion_options.max_trials must be a positive integer")
    return {"max_trials": int(value)}


def _auto_diffusion_run_trial(
    coefficient,
    initial_uv,
    initial_diagnostics,
    inputs,
    options,
    numerical_policy,
    audit,
    candidates,
):
    """Refine one coefficient and retain eligible states even if the trial fails."""
    trial_options = {**options, "alpha": float(coefficient)}
    trial = {"coefficient": float(coefficient), "iterates": []}
    audit["trials"].append(trial)

    def inspect_iterate(source_uv, diagnostics):
        """Record output quality for the seed or an accepted refinement state."""
        _auto_diffusion_evaluate_iterate(
            source_uv,
            diagnostics,
            inputs,
            numerical_policy,
            trial,
            candidates,
        )

    constraints = inputs["constraints"]
    try:
        source_uv, diagnostics = _refine_parameterization_by_stretch(
            inputs["source"],
            inputs["data"]["source_element"],
            constraints["source_index"],
            constraints["source_uv"],
            initial_uv,
            initial_diagnostics,
            trial_options,
            numerical_policy,
            inspect_iterate,
        )
    except ValueError as error:
        details = getattr(error, "diagnostics", {})
        trial.update(
            attempted_iterations=details.get("attempted_iterations", 0),
            accepted_iterations=max(
                (record["iteration"] for record in trial["iterates"]), default=0
            ),
            termination_reason="trial_failed",
            error=str(error),
            failure_details=details,
        )
        audit["total_refinement_attempts"] += trial["attempted_iterations"]
        for candidate in candidates:
            if candidate["alpha"] == coefficient:
                candidate["diagnostics"][
                    "termination_reason"
                ] = "auto_selected_before_trial_failure"
        return trial
    trial.update(
        attempted_iterations=diagnostics["attempted_iterations"],
        accepted_iterations=diagnostics["accepted_iterations"],
        termination_reason=diagnostics["termination_reason"],
        candidate_records=deepcopy(diagnostics["candidate_records"]),
    )
    audit["total_refinement_attempts"] += diagnostics["attempted_iterations"]
    for candidate in candidates:
        if candidate["alpha"] == coefficient:
            if candidate["iteration"] == diagnostics["accepted_iterations"]:
                candidate["diagnostics"] = deepcopy(diagnostics)
            else:
                candidate["diagnostics"]["termination_reason"] = "auto_selected_earlier_iterate"
    return trial


def _auto_diffusion_evaluate_iterate(
    source_uv, diagnostics, inputs, numerical_policy, trial, candidates
):
    """Map an accepted UV state and save a private copy when its corners pass."""
    constraints = inputs["constraints"]
    iteration = diagnostics["accepted_iterations"]
    record = {
        "iteration": iteration,
        "stretch": diagnostics["stretch_history"][-1],
        "passed": False,
        "eligible": False,
        "intersection_status": "not_checked",
    }
    trial["iterates"].append(record)
    if not np.array_equal(source_uv[constraints["source_index"]], constraints["source_uv"]):
        _fail("Automatic diffusion changed a fixed source UV coordinate")
    try:
        output, transfer = _map_template_to_source_surface(
            inputs["data"]["source_node_raw"],
            inputs["data"]["source_element"],
            source_uv,
            inputs["template_uv"],
            constraints["template_index"],
            constraints["template_position_raw"],
            numerical_policy,
        )
        record["corner_quality"] = _output_corner_quality(
            output, inputs["data"]["template_element"], numerical_policy
        )
    except ValueError as error:
        record.update(passed=False, reason="transfer_or_geometry_failure", error=str(error))
        return
    record["eligible"] = record["corner_quality"]["passed"]
    if record["eligible"]:
        candidates.append(
            {
                "alpha": trial["coefficient"],
                "iteration": iteration,
                "record": record,
                "uv": source_uv.copy(),
                "diagnostics": deepcopy(diagnostics),
                "output": output,
                "transfer": transfer,
            }
        )


def _auto_diffusion_validate_candidate(candidate, inputs, numerical_policy):
    """Check output intersections once and cache the result on the candidate."""
    record = candidate["record"]
    if "intersection_check" not in record:
        try:
            record["intersection_check"] = _check_output_intersections(
                candidate["output"],
                inputs["data"]["template_element"],
                numerical_policy,
            )
        except ValueError as error:
            record["intersection_check"] = {"passed": False, "error": str(error)}
    record["passed"] = record["intersection_check"]["passed"]
    record["intersection_status"] = "passed" if record["passed"] else "failed"
    if not record["passed"]:
        record["eligible"] = False
        record["reason"] = "output_intersection"
    return record["passed"]


def _auto_diffusion_trial_priority(trial):
    """Rank sampled coefficients by eligible stretch or failed-corner quality."""
    eligible = [record for record in trial["iterates"] if record.get("eligible")]
    if eligible:
        return (0, min(record["stretch"] for record in eligible), 0.0)
    qualities = [
        record["corner_quality"] for record in trial["iterates"] if "corner_quality" in record
    ]
    return min(
        (
            (1, quality["failed_corner_count"], -quality["minimum_corner_margin"])
            for quality in qualities
        ),
        default=(2, math.inf, math.inf),
    )


def _auto_diffusion_next_coefficient(trials):
    """Choose the midpoint of the highest-priority remaining coefficient interval."""
    ordered = sorted(trials, key=lambda trial: trial["coefficient"])
    intervals = [
        (
            min(_auto_diffusion_trial_priority(left), _auto_diffusion_trial_priority(right)),
            -(right["coefficient"] - left["coefficient"]),
            (left["coefficient"] + right["coefficient"]) / 2,
        )
        for left, right in zip(ordered, ordered[1:])
        if right["coefficient"] - left["coefficient"] > 1e-12
    ]
    return min(intervals)[2] if intervals else None


def _auto_diffusion_candidate_priority(candidate):
    """Rank outputs by stretch, corner margin, coefficient, and iteration."""
    return (
        candidate["record"]["stretch"],
        -candidate["record"]["corner_quality"]["minimum_corner_margin"],
        candidate["alpha"],
        candidate["iteration"],
    )


def _auto_diffusion_finish(candidate, audit, reason):
    """Attach the completed search record and return the selected mapped output."""
    audit.update(
        selected_coefficient=candidate["alpha"],
        selected_iteration=candidate["iteration"],
        termination_reason=reason,
        trial_count=len(audit["trials"]),
        selected_corner_quality=candidate["record"]["corner_quality"],
        selected_intersection_check=candidate["record"]["intersection_check"],
    )
    candidate["diagnostics"]["auto_diffusion"] = audit
    return candidate["uv"], candidate["diagnostics"], candidate["output"], candidate["transfer"]


def _output_corner_quality(node, faces, numerical_policy):
    """Measure signed 3D corner sines relative to each element's mean normal.

    Round positions to output precision before evaluating triangle and quad
    corners. Return margins, failed corner indices, and a pass flag. Bounds
    account for coordinate subtraction, cross products, and normal uncertainty.
    An unresolved edge or mean normal fails the check. The test measures local
    corner quality; global contacts are checked by ``_check_output_intersections``.
    """
    output_positions = np.asarray(node, dtype=numerical_policy["dtype"]).astype(np.float64)
    scale = float(np.linalg.norm(np.ptp(output_positions, axis=0)))
    if not np.isfinite(output_positions).all() or not math.isfinite(scale) or scale <= 0:
        _fail("Nonfinite or collapsed output geometry")
    magnitude = max(1.0, float(np.max(abs(output_positions))) / scale)
    output_positions = (output_positions - output_positions[0]) / scale
    roundoff_factor = numerical_policy["roundoff_factor"]
    edge_error = 2 * math.sqrt(3) * roundoff_factor * magnitude
    records, minimum, minimum_margin = [], math.inf, math.inf
    nonpositive, unresolved = 0, []
    for size in sorted(set(map(len, faces))):
        if size not in (3, 4):
            _fail("Automatic diffusion supports triangle and quad templates")
        element_indices = np.array(
            [i for i, face in enumerate(faces) if len(face) == size], dtype=np.int64
        )
        element_positions = output_positions[np.asarray([faces[i] for i in element_indices])]
        edge = np.roll(element_positions, -1, axis=1) - element_positions
        length = np.linalg.norm(edge, axis=2)
        previous_length = np.roll(length, 1, axis=1)
        normal = np.cross(edge, -np.roll(edge, 1, axis=1))
        mean_normal = normal.sum(1)
        normal_length = np.linalg.norm(mean_normal, axis=1)
        denominator = length * previous_length
        error = (
            edge_error * (length + previous_length) + edge_error**2 + roundoff_factor * denominator
        )
        mean_error = error.sum(1) + roundoff_factor * np.linalg.norm(normal, axis=2).sum(1)
        degenerate = (length <= edge_error).any(1) | (normal_length <= mean_error)
        safe_norm = np.maximum(normal_length, np.finfo(float).tiny)
        safe_denominator = np.maximum(denominator, np.finfo(float).tiny)
        signed_corner_sine = (
            np.sum(normal * (mean_normal / safe_norm[:, None])[:, None], axis=2) / safe_denominator
        )
        tolerance = (
            error / safe_denominator
            + 2 * mean_error[:, None] / safe_norm[:, None]
            + roundoff_factor
        )
        failed = (signed_corner_sine <= tolerance) | degenerate[:, None]
        nonpositive += int(np.count_nonzero(signed_corner_sine <= 0))
        unresolved.extend(element_indices[degenerate].tolist())
        minimum = min(minimum, float(signed_corner_sine.min()))
        minimum_margin = min(minimum_margin, float((signed_corner_sine - tolerance).min()))
        for row, corner in np.argwhere(failed):
            records.append(
                {
                    "element_index": int(element_indices[row]),
                    "corner_index": int(corner),
                    "signed_corner_sine": float(signed_corner_sine[row, corner]),
                    "required_margin": float(tolerance[row, corner]),
                }
            )
    return {
        "passed": not records,
        "minimum_signed_corner_sine": minimum,
        "minimum_corner_margin": minimum_margin,
        "nonpositive_corner_count": nonpositive,
        "failed_corner_count": len(records),
        "failed_corners": records,
        "unresolved_element_indices": unresolved,
        "measure": "local signed corner sine relative to each element's mean normal",
    }


def _check_output_intersections(node, faces, numerical_policy):
    """Check triangle and quad outputs for contact beyond shared vertices or edges.

    Represent triangles directly and quads using both diagonals. Bounding boxes
    and plane separation filter candidate pairs before a barycentric feasibility
    solve. Return the pass flag, improper element pairs, and solver counts.
    Computation uses normalized float64 coordinates after rounding to output
    precision. The check applies to these straight-triangle representations.
    """
    output_positions = np.asarray(node, dtype=numerical_policy["dtype"]).astype(np.float64)
    output_positions = (output_positions - output_positions.mean(0)) / np.linalg.norm(
        np.ptp(output_positions, axis=0)
    )
    triangles, parents = [], []
    for i, face in enumerate(faces):
        choices = ((0, 1, 2),) if len(face) == 3 else ((0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 3))
        for choice in choices:
            triangles.append([face[k] for k in choice])
            parents.append(i)
    triangles, parents = np.asarray(triangles), np.asarray(parents)
    triangle_positions = output_positions[triangles]
    normals = np.cross(
        triangle_positions[:, 1] - triangle_positions[:, 0],
        triangle_positions[:, 2] - triangle_positions[:, 0],
    )
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    pairs = _find_overlapping_box_pairs(triangle_positions.min(1), triangle_positions.max(1), 1e-12)
    tolerance = max(1e-8, 8 * numerical_policy["roundoff_factor"])
    bad, checked, linear_programs = set(), 0, 0
    for i, j in pairs:
        if parents[i] == parents[j]:
            continue
        checked += 1
        first_triangle, second_triangle = triangle_positions[i], triangle_positions[j]
        shared = np.isin(triangles[i], triangles[j])
        shared_b = np.isin(triangles[j], triangles[i])
        # Strict plane separation outside the shared feature rules out an
        # improper intersection for this triangle pair.
        first_plane_distances, second_plane_distances = (
            first_triangle - second_triangle[0]
        ) @ normals[j], (second_triangle - first_triangle[0]) @ normals[i]

        def separated(distance, common):
            """Test whether all nonshared vertices lie strictly on one side of a plane."""
            other = distance[~common]
            return len(other) > 0 and (np.all(other > 1e-12) or np.all(other < -1e-12))

        if separated(first_plane_distances, shared) or separated(second_plane_distances, shared_b):
            continue
        if np.count_nonzero(shared) == 2:
            edge = first_triangle[shared][1] - first_triangle[shared][0]
            side_a = np.cross(edge, first_triangle[~shared][0] - first_triangle[shared][0])
            side_b = np.cross(edge, second_triangle[~shared_b][0] - first_triangle[shared][0])
            # Coplanar triangles on opposite sides meet only at their edge.
            if float(side_a @ side_b) < -1e-12 * np.linalg.norm(side_a) * np.linalg.norm(side_b):
                continue
        matrix = np.zeros((5, 6))
        matrix[:3, :3], matrix[:3, 3:] = first_triangle.T, -second_triangle.T
        matrix[3, :3], matrix[4, 3:] = 1, 1
        objective = np.r_[-(~shared).astype(float), np.zeros(3)]
        result = linprog(
            objective,
            A_eq=matrix,
            b_eq=[0, 0, 0, 1, 1],
            bounds=(0, None),
            method="highs",
            options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
        )
        linear_programs += 1
        if result.status == 2:
            continue
        if not result.success:
            _fail("Output intersection linear program failed", message_from_solver=result.message)
        if not shared.any() or shared.all() or -result.fun > tolerance:
            bad.add((int(parents[i]), int(parents[j])))
    return {
        "passed": not bad,
        "element_pairs_with_improper_contact": sorted(bad),
        "candidate_triangle_pairs_checked": checked,
        "linear_programs": linear_programs,
        "both_quad_diagonals_checked": True,
        "shared_feature_barycentric_tolerance": tolerance,
        "scope": "Straight-triangle representations of distinct output elements.",
    }


# Template mapping and output construction


def _map_template_to_source_surface(
    source_node_raw: np.ndarray,
    source_element: np.ndarray,
    source_uv_normalized: np.ndarray,
    template_uv_normalized: np.ndarray,
    template_constraint_node_index: np.ndarray,
    template_constraint_position_raw: np.ndarray,
    numerical_policy: dict,
) -> tuple[np.ndarray, dict]:
    """Map template vertices onto the source surface through its UV triangles.

    Source positions have shape (source_node_count, 3) in physical coordinates.
    Source and template UV arrays use the same normalized frame. Copy fixed
    template positions exactly, and locate each remaining vertex in the source
    UV triangulation. Barycentric interpolation uses a local affine form to
    limit cancellation under large coordinate translations.

    Return float64 output positions and per-vertex triangle indices, weights,
    projection coordinates, distances, and query status. A point farther than
    the UV distance tolerance from the domain raises ValueError. Unresolved
    triangles and failed barycentric reconstruction also raise ValueError.

    References
    ----------
    Liang Liang, Minliang Liu, Caitlin Martin, John A. Elefteriades, and Wei Sun
    (2017), A machine learning approach to investigate the relationship between
    shape features and numerically predicted risk of ascending aortic aneurysm,
    appendix, equation (10). https://doi.org/10.1007/s10237-017-0903-9
    """
    locator = _build_source_triangle_locator(source_uv_normalized, source_element, numerical_policy)
    template_node_count = len(template_uv_normalized)
    output = np.empty((template_node_count, 3), dtype=np.float64)
    output[template_constraint_node_index] = template_constraint_position_raw
    constrained_node_set = set(template_constraint_node_index.tolist())
    source_triangle_indices = np.full(template_node_count, -1, dtype=np.int64)
    vertex_barycentric_weights = np.full(
        (template_node_count, 3), np.nan, dtype=source_uv_normalized.dtype
    )
    projected_uv = np.full((template_node_count, 2), np.nan, dtype=source_uv_normalized.dtype)
    uv_distances = np.zeros(template_node_count, dtype=np.float64)
    query_statuses = ["constraint"] * template_node_count
    for template_node_index, query in enumerate(template_uv_normalized):
        if template_node_index in constrained_node_set:
            continue
        source_face_index, closest, barycentric_coordinates, distance2, query_status = (
            _find_source_triangle_for_template_point(query, locator, numerical_policy)
        )
        uv_distance = math.sqrt(distance2)
        if query_status == "degenerate":
            _fail(
                "Locator encountered a degenerate UV triangle",
                node_index=template_node_index,
                face_index=source_face_index,
            )
        if uv_distance > numerical_policy["uv_distance_tolerance"]:
            _fail(
                f"Template node {template_node_index} is outside the source UV domain",
                node_index=template_node_index,
                nearest_face=source_face_index,
                distance_normalized=uv_distance,
                allowed_distance=numerical_policy["uv_distance_tolerance"],
            )
        reconstruction = (
            barycentric_coordinates @ source_uv_normalized[source_element[source_face_index]]
        )
        if (
            np.any(barycentric_coordinates < 0)
            or abs(float(barycentric_coordinates.sum()) - 1) > numerical_policy["roundoff_factor"]
            or np.linalg.norm(reconstruction - closest) > numerical_policy["uv_distance_tolerance"]
        ):
            _fail(
                "Barycentric reconstruction check failed",
                node_index=template_node_index,
                face_index=source_face_index,
            )
        # Local affine form avoids cancellation under a large source translation.
        source_triangle_positions = source_node_raw[source_element[source_face_index]]
        output[template_node_index] = (
            source_triangle_positions[0]
            + barycentric_coordinates[1]
            * (source_triangle_positions[1] - source_triangle_positions[0])
            + barycentric_coordinates[2]
            * (source_triangle_positions[2] - source_triangle_positions[0])
        )
        (
            source_triangle_indices[template_node_index],
            vertex_barycentric_weights[template_node_index],
            projected_uv[template_node_index],
            uv_distances[template_node_index],
        ) = (source_face_index, barycentric_coordinates, closest, uv_distance)
        query_statuses[template_node_index] = (
            "inside" if uv_distance == 0 else "roundoff_projection"
        )
    return output, {
        "source_triangle_index": source_triangle_indices,
        "barycentric_weight": vertex_barycentric_weights,
        "projected_uv_normalized": projected_uv,
        "uv_distance_normalized": uv_distances,
        "query_status": query_statuses,
        "roundoff_projection_count": query_statuses.count("roundoff_projection"),
    }


def _build_source_triangle_locator(source_uv_normalized, source_element, numerical_policy):
    """Build a bounding-box locator for triangles of the normalized source UV map."""
    triangle = source_uv_normalized[source_element]
    return {
        "triangle": triangle,
        "tree": _build_box_tree(triangle.min(1), triangle.max(1)),
        "element": source_element,
    }


def _find_source_triangle_for_template_point(point_uv_normalized, locator, numerical_policy):
    """Find the closest UV triangle with a distance-ordered bounding-box search.

    Return its index, closest point, barycentric weights, squared distance,
    and projection status. Exact distance ties use the lowest face index.
    """
    tree, triangles = locator["tree"], locator["triangle"]
    queue, closest_candidate = [(0.0, 0)], None
    tolerance = numerical_policy["uv_distance_tolerance"]
    while queue:
        distance_lower_bound, box_index = heapq.heappop(queue)
        if (
            closest_candidate is not None
            and distance_lower_bound > (math.sqrt(closest_candidate[3]) + tolerance) ** 2
        ):
            continue
        box = tree[box_index]
        if "indices" in box:
            for face_index in box["indices"]:
                closest, weights, distance2, status = _closest_point_on_triangle(
                    point_uv_normalized, triangles[face_index], numerical_policy
                )
                # Exact distances are primary; the index resolves exact ties.
                if (
                    closest_candidate is None
                    or distance2 < closest_candidate[3]
                    or (distance2 == closest_candidate[3] and face_index < closest_candidate[0])
                ):
                    closest_candidate = (int(face_index), closest, weights, distance2, status)
        else:
            for child in box["children"]:
                heapq.heappush(
                    queue, (_squared_distance_to_box(point_uv_normalized, tree[child]), child)
                )
    return closest_candidate


def _compute_barycentric_coordinates(point_uv_normalized, triangle_uv_normalized, numerical_policy):
    """Compute three affine coordinates for a point in a nondegenerate UV triangle.

    The query has shape (2,) and the triangle has shape (3, 2). Return three
    weights in the triangle dtype. Degeneracy and poor conditioning raise
    ValueError. The caller checks containment and reconstruction accuracy.

    References
    ----------
    Liang Liang et al. (2017), A machine learning approach to investigate the
    relationship between shape features and numerically predicted risk of
    ascending aortic aneurysm, appendix, equation (10).
    https://doi.org/10.1007/s10237-017-0903-9
    """
    triangle_uv, query_uv = triangle_uv_normalized, point_uv_normalized
    triangle_edges = np.column_stack(
        (triangle_uv[1] - triangle_uv[0], triangle_uv[2] - triangle_uv[0])
    )
    singular_values = np.linalg.svd(triangle_edges, compute_uv=False)
    roundoff_factor = numerical_policy["roundoff_factor"]
    magnitude = max(float(np.max(abs(triangle_uv))), float(np.max(abs(query_uv))))
    error_bound = roundoff_factor * float(
        singular_values[0] / singular_values[1]
    ) + 4 * roundoff_factor * magnitude / float(singular_values[1])
    if (
        not math.isfinite(error_bound)
        or error_bound > numerical_policy["barycentric_condition_limit"]
    ):
        _fail(
            "Barycentric coordinates are ill-conditioned for this dtype; use float64 if needed",
            condition_bound=error_bound,
            triangle=triangle_uv,
        )
    edge_coordinates = np.linalg.solve(triangle_edges, query_uv - triangle_uv[0])
    return np.array(
        [1 - edge_coordinates[0] - edge_coordinates[1], edge_coordinates[0], edge_coordinates[1]],
        dtype=triangle_uv.dtype,
    )


def _closest_point_on_segment(point_uv_normalized, segment_uv_normalized, numerical_policy):
    """Project a UV query onto a segment, clamping its coordinate to [0, 1].

    Return the closest point, segment fraction, and squared distance.
    A collapsed segment uses its first endpoint.
    """
    a, b = segment_uv_normalized
    edge = b - a
    denominator = float(np.dot(edge, edge))
    fraction = (
        float(np.clip(np.dot(point_uv_normalized - a, edge) / denominator, 0, 1))
        if denominator > 0
        else 0.0
    )
    closest = a + fraction * edge
    difference = (point_uv_normalized - closest).astype(np.float64)
    return closest, fraction, float(np.dot(difference, difference))


def _closest_point_on_triangle(point_uv_normalized, triangle_uv_normalized, numerical_policy):
    """Find a query's closest point on a UV triangle or its boundary.

    Return the closest point, barycentric weights, squared distance, and
    status. Degenerate triangles use edge projections and are marked for
    rejection by the mapper.
    """
    triangle_uv, query_uv = triangle_uv_normalized, point_uv_normalized
    area, bound = _triangle_measures(triangle_uv[None], numerical_policy)
    degenerate = bool(abs(area[0]) <= bound[0])
    if not degenerate:
        weights = _compute_barycentric_coordinates(query_uv, triangle_uv, numerical_policy)
        if np.all(weights >= 0) and np.all(weights <= 1):
            return query_uv.copy(), weights, 0.0, "inside"
    best = None
    for i, j in ((0, 1), (1, 2), (2, 0)):
        closest, fraction, distance2 = _closest_point_on_segment(
            query_uv, triangle_uv[[i, j]], numerical_policy
        )
        weights = np.zeros(3, dtype=triangle_uv.dtype)
        weights[i], weights[j] = 1 - fraction, fraction
        if best is None or distance2 < best[2]:
            best = (closest, weights, distance2, "degenerate" if degenerate else "closest_boundary")
    return best


def _build_output_mesh(template_mesh, output_node, output_device, output_dtype, diagnostics):
    """Create the output mesh with template connectivity and copied metadata.

    Convert positions to the requested dtype and source device, clear derived
    geometry caches, and attach remeshing diagnostics to ``mesh_data``.
    """
    output = PolygonMesh(
        node=torch.as_tensor(output_node, dtype=output_dtype, device=output_device),
        element=deepcopy(template_mesh.element),
    )
    for name in (
        "name",
        "mapping_from_node_name_to_index",
        "node_set",
        "element_set",
        "node_data",
        "element_data",
        "mesh_data",
        "element_type",
    ):
        setattr(output, name, deepcopy(getattr(template_mesh, name)))
    output.clear_geometry_derived_data()
    output.mesh_data["template_barycentric_mapping_based_remesh"] = diagnostics
    return output


def _assemble_output_diagnostics(
    inputs: dict,
    source_uv: np.ndarray,
    diagnostics: dict,
    transfer: dict,
    source_mesh,
    template_mesh,
    computation_dtype,
) -> dict:
    """Add physical coordinates, mapping records, and input metadata to diagnostics."""
    template_uv = inputs["template_uv"]
    template_topology = inputs["template_topology"]
    transforms = inputs["transforms"]
    source_nodes_normalized = inputs["source"]
    numerical_policy = inputs["policy"]
    options = inputs["options"]
    constraints = inputs["constraints"]
    landmark_info = inputs["landmark_info"]
    source_area_tolerance = inputs["source_area_bound"]
    turns = []
    for loop in template_topology["loops"]:
        boundary_uv = template_uv[loop]
        edges = np.roll(boundary_uv, -1, axis=0) - boundary_uv
        turn = np.arctan2(
            _cross_2d(edges, np.roll(edges, -1, axis=0)),
            np.sum(edges * np.roll(edges, -1, axis=0), axis=1),
        )
        turns.append(turn)
    ratio = transforms["source_scale"] / transforms["uv_scale"]
    diagnostics.update(
        {
            **transforms,
            **transfer,
            "source_uv_normalized": source_uv.copy(),
            "source_uv": source_uv.astype(np.float64) * transforms["uv_scale"]
            + transforms["uv_center"],
            "source_node_normalized": source_nodes_normalized,
            "numerical_policy": numerical_policy,
            "options": options,
            "constraints": constraints,
            "landmark_correspondence": landmark_info,
            "input_source_dtype": str(source_mesh.node.dtype),
            "input_template_dtype": str(template_mesh.node.dtype),
            "computation_dtype": str(computation_dtype),
            "template_orientation_conflicts": template_topology["orientation_conflicts"],
            "template_boundary_turns": turns,
            "source_double_area_tolerances": source_area_tolerance,
            "stretch_history_raw": [v * ratio for v in diagnostics["stretch_history"]],
            "candidate_stretch_history_raw": [
                None if v is None else v * ratio for v in diagnostics["candidate_stretch_history"]
            ],
            "uv_distance": transfer["uv_distance_normalized"] * transforms["uv_scale"],
            "uv_distance_tolerance_raw": numerical_policy["uv_distance_tolerance"]
            * transforms["uv_scale"],
            "projected_uv": transfer["projected_uv_normalized"].astype(np.float64)
            * transforms["uv_scale"]
            + transforms["uv_center"],
        }
    )
    return diagnostics


# Shared linear algebra and geometry


def _solve_parameterization_with_fixed_constraints(
    weight_matrix,
    free_node_indices: np.ndarray,
    constrained_node_indices: np.ndarray,
    constrained_values: np.ndarray,
    previous_values: np.ndarray,
    options: dict,
    numerical_policy: dict,
    solver_records,
    *,
    stage,
    attempt=0,
) -> np.ndarray:
    """Solve for free rows and copy the fixed rows exactly into a new array.

    ``previous_values`` has shape (node_count, field_count); ``constrained_values``
    has shape (constraint_count, field_count). Two fields represent UV coordinates;
    annular initialization uses three fields. Append a solver record on success.
    A failed solve carries its refinement attempt in the exception diagnostics.
    """
    system_matrix, right_hand_side = _assemble_parameterization_system(
        weight_matrix,
        free_node_indices,
        constrained_node_indices,
        constrained_values,
    )
    try:
        solution, record = _solve_parameterization_system(
            system_matrix,
            right_hand_side,
            previous_values[free_node_indices],
            options["linear_solver"],
            options["solver_options"],
            numerical_policy,
        )
    except ValueError as error:
        error.diagnostics = {**getattr(error, "diagnostics", {}), "attempted_iterations": attempt}
        raise
    record["stage"] = stage
    solver_records.append(record)
    result = previous_values.copy()
    result[free_node_indices] = solution
    result[constrained_node_indices] = constrained_values
    return result


def _assemble_parameterization_system(
    weight_matrix, free_node_indices, constrained_node_indices, constrained_uv
):
    """Assemble the free-vertex system and contributions from fixed vertices.

    The weight matrix has one row per free vertex and one column per vertex.
    Return a square CSR matrix and a dense right-hand side with one column
    per coordinate or field. Constraint values use the listed index order.
    """
    row_sum = np.asarray(weight_matrix.sum(axis=1)).ravel()
    system_matrix = sp.diags(row_sum, format="csr") - weight_matrix[:, free_node_indices]
    right_hand_side = weight_matrix[:, constrained_node_indices] @ constrained_uv
    return system_matrix, right_hand_side


def _validate_solver_options(linear_solver, options, numerical_policy):
    """Validate solver-specific options and fill defaults for the numerical policy.

    Support sparse LU, GMRES, and BiCGSTAB. Return a new settings dictionary.
    Relative tolerance, fill factor, and iteration limits must be positive.
    Absolute tolerance and ILU drop tolerance may be zero. Unknown options or
    invalid values raise ValueError.

    Relative tolerance defaults to max(1e-10, 64 * machine_epsilon). When omitted,
    absolute tolerance is computed during the solve from the matrix norm,
    free-row count, UV extent, and machine epsilon. Sparse LU uses column ordering
    "COLAMD"; "NATURAL", "MMD_ATA", and "MMD_AT_PLUS_A" are also accepted.

    Iterative solvers default to 10000 iterations and ILU preconditioning with
    drop tolerance 1e-4 and fill factor 10. The preconditioner can be "none".
    GMRES restart defaults to the smaller of 50 and the number of free rows.
    """
    if linear_solver not in ("sparse_lu", "gmres", "bicgstab"):
        _fail("linear_solver must be sparse_lu, gmres, or bicgstab")
    opts = {} if options is None else dict(options)
    allowed = (
        {"rtol", "atol", "permc_spec"}
        if linear_solver == "sparse_lu"
        else {"rtol", "atol", "maxiter", "preconditioner", "drop_tol", "fill_factor"}
    )
    if linear_solver == "gmres":
        allowed.add("restart")
    if set(opts) - allowed:
        _fail("Unknown or inapplicable solver options", keys=sorted(set(opts) - allowed))
    opts.setdefault("rtol", max(1e-10, 64 * numerical_policy["eps"]))
    for key in ("rtol", "atol", "drop_tol", "fill_factor"):
        if key in opts:
            value = opts[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
                or (key in ("rtol", "fill_factor") and value == 0)
            ):
                _fail(f"Invalid solver option {key}")
    for key in ("maxiter", "restart"):
        if key in opts and (
            isinstance(opts[key], bool) or not isinstance(opts[key], Integral) or opts[key] <= 0
        ):
            _fail(f"{key} must be a positive integer")
    if linear_solver == "sparse_lu":
        opts.setdefault("permc_spec", "COLAMD")
        if opts["permc_spec"] not in ("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"):
            _fail("Invalid permc_spec")
    else:
        opts.setdefault("maxiter", 10000)
        opts.setdefault("preconditioner", "ilu")
        opts.setdefault("drop_tol", 1e-4)
        opts.setdefault("fill_factor", 10)
        if opts["preconditioner"] not in ("ilu", "none"):
            _fail("preconditioner must be ilu or none")
    return opts


def _solve_parameterization_system(
    coefficient_matrix,
    right_hand_side,
    initial_guess,
    linear_solver,
    solver_options,
    numerical_policy,
):
    """Solve a sparse system and verify its residual and backward error.

    The coefficient matrix is square; the right-hand side and initial guess
    have shape (free_node_count, field_count). Return a solution in the matrix
    dtype and per-field solver records. Iterative solvers may use ILU.
    Factorization failure, nonfinite output, or unmet tolerances raise ValueError.
    """
    system_matrix, rhs = coefficient_matrix, right_hand_side
    field_count = rhs.shape[1]
    settings = dict(solver_options)
    matrix_norm = float(np.max(np.asarray(abs(system_matrix).sum(axis=1))))
    atol = settings.get(
        "atol",
        64
        * numerical_policy["eps"]
        * matrix_norm
        * math.sqrt(len(rhs))
        * numerical_policy["uv_extent"],
    )
    rtol = settings["rtol"]
    record = {
        "solver": linear_solver,
        "matrix_dtype": str(system_matrix.dtype),
        "rtol": rtol,
        "atol": atol,
        "free_rows": len(rhs),
        "preconditioner": "none",
        "coordinates": [],
    }
    try:
        if linear_solver == "sparse_lu":
            lu = splu(system_matrix.tocsc(), permc_spec=settings["permc_spec"])
            solution = lu.solve(rhs)
            iterations, statuses, callback_counts = (
                [0] * field_count,
                [0] * field_count,
                [0] * field_count,
            )
        else:
            preconditioner = None
            if settings["preconditioner"] == "ilu":
                ilu = spilu(
                    system_matrix.tocsc(),
                    drop_tol=settings["drop_tol"],
                    fill_factor=settings["fill_factor"],
                )
                preconditioner = LinearOperator(
                    system_matrix.shape, matvec=ilu.solve, dtype=system_matrix.dtype
                )
                record["preconditioner"] = "ilu"
            solution = np.empty_like(rhs)
            iterations, statuses, callback_counts = [], [], []
            for dim in range(field_count):
                count = [0]

                def callback(_):
                    """Count completed iterative-solver callbacks."""
                    count[0] += 1

                kwargs = {
                    "x0": initial_guess[:, dim],
                    "rtol": rtol,
                    "atol": atol,
                    "maxiter": settings["maxiter"],
                    "M": preconditioner,
                    "callback": callback,
                }
                if linear_solver == "gmres":
                    coordinate_solution, status = gmres(
                        system_matrix,
                        rhs[:, dim],
                        restart=settings.get("restart", min(50, len(rhs))),
                        callback_type="legacy",
                        **kwargs,
                    )
                    inner_iterations = count[0]
                else:
                    # Count matrix products to include BiCGSTAB's final half-step,
                    # which can converge before the callback runs.
                    products = [0]

                    def matvec(vector):
                        """Multiply by the coefficient matrix and count the operation."""
                        products[0] += 1
                        return system_matrix @ vector

                    operator = LinearOperator(
                        system_matrix.shape, matvec=matvec, dtype=system_matrix.dtype
                    )
                    coordinate_solution, status = bicgstab(operator, rhs[:, dim], **kwargs)
                    initial_product = int(
                        np.any(initial_guess[:, dim]) and np.linalg.norm(rhs[:, dim]) > 0
                    )
                    inner_iterations = (max(0, products[0] - initial_product) + 1) // 2
                solution[:, dim] = coordinate_solution
                iterations.append(inner_iterations)
                statuses.append(int(status))
                callback_counts.append(count[0])
        if solution.dtype != system_matrix.dtype or not np.isfinite(solution).all():
            _fail("Solver returned nonfinite coordinates or changed dtype", solve=record)
        residual = rhs.astype(np.float64) - system_matrix.astype(np.float64) @ solution.astype(
            np.float64
        )
        for dim in range(field_count):
            residual_norm = float(np.linalg.norm(residual[:, dim]))
            rhs_norm = float(np.linalg.norm(rhs[:, dim].astype(np.float64)))
            denominator = matrix_norm * float(np.max(abs(solution[:, dim]))) + float(
                np.max(abs(rhs[:, dim]))
            )
            backward_error = (
                float(np.max(abs(residual[:, dim]))) / denominator
                if denominator
                else (0.0 if residual_norm == 0 else math.inf)
            )
            record["coordinates"].append(
                {
                    "iterations": iterations[dim],
                    "status": statuses[dim],
                    "callback_count": callback_counts[dim],
                    "residual_l2": residual_norm,
                    "residual_limit": max(rtol * rhs_norm, atol),
                    "backward_error": backward_error,
                }
            )
            if (
                statuses[dim] != 0
                or residual_norm > max(rtol * rhs_norm, atol)
                or backward_error > rtol
            ):
                _fail("Linear solve failed true-residual/backward-error checks", solve=record)
        return solution, record
    except (RuntimeError, np.linalg.LinAlgError) as error:
        _fail(f"Linear factorization/preconditioner failed: {error}", solve=record)


def _triangle_measures(triangle, numerical_policy):
    """Compute triangle double areas and their numerical resolution bounds.

    ``triangle`` has shape (face_count, 3, coordinate_count), with two or
    three coordinates. Planar measures are signed; spatial measures are
    magnitudes. Return one measure and one nonnegative bound per triangle.
    Nonfinite geometry raises ValueError with an arithmetic-failure flag.
    """
    a, b = triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
    longest = np.maximum.reduce(
        [np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1), np.linalg.norm(b - a, axis=1)]
    )
    f = numerical_policy["roundoff_factor"]
    e = 2 * f * np.max(np.abs(triangle), axis=(1, 2))
    if triangle.shape[2] == 2:
        measure = _cross_2d(a, b)
        bound = (
            f * (abs(a[:, 0] * b[:, 1]) + abs(a[:, 1] * b[:, 0]))
            + e * np.sum(abs(a) + abs(b), axis=1)
            + 2 * e**2
        )
    else:
        measure = np.linalg.norm(np.cross(a, b), axis=1)
        j, k = [1, 2, 0], [2, 0, 1]
        bounds = (
            f * (abs(a[:, j] * b[:, k]) + abs(a[:, k] * b[:, j]))
            + e[:, None] * (abs(a[:, j]) + abs(a[:, k]) + abs(b[:, j]) + abs(b[:, k]))
            + 2 * e[:, None] ** 2
        )
        bound = np.linalg.norm(bounds, axis=1)
    threshold = np.maximum(f * longest**2, bound)
    if not np.isfinite(measure).all() or not np.isfinite(threshold).all():
        _fail("Nonfinite triangle geometry arithmetic", arithmetic_failure=True)
    return measure, threshold


def _validate_uv_mesh(source_uv_normalized, source_element, expected_orientation, numerical_policy):
    """Validate orientation, triangle resolution, and global planar embedding.

    Source UV coordinates have shape (node_count, 2). Infer orientation when
    ``expected_orientation`` is None; otherwise enforce its sign. Check edge
    contacts and triangle containment, including faces sharing a vertex.
    Return orientation and geometric margins, or raise ValueError with the
    failing face or edge indices. Inputs remain unchanged.
    """
    source_uv, face = source_uv_normalized, source_element
    if not np.isfinite(source_uv).all():
        _fail("Nonfinite UV arithmetic")
    uv_triangles = source_uv[face]
    area2, bound = _triangle_measures(uv_triangles, numerical_policy)
    bad = np.flatnonzero(abs(area2) <= bound)
    if len(bad):
        _fail(
            "Degenerate or unresolved UV faces",
            face_indices=bad.tolist(),
            double_area=area2[bad],
            tolerance=bound[bad],
        )
    orientation = 1 if area2[0] > 0 else -1
    if expected_orientation is not None and orientation != expected_orientation:
        _fail("UV orientation changed")
    bad = np.flatnonzero(area2 * orientation <= 0)
    if len(bad):
        _fail("Flipped UV faces", face_indices=bad.tolist())
    tolerance = numerical_policy["uv_distance_tolerance"]
    pairs = _find_overlapping_box_pairs(uv_triangles.min(1), uv_triangles.max(1), tolerance)
    bad_pairs = []
    # Bound temporary memory by checking pairs in chunks. Shared features
    # permit contact only at their common vertex or edge.
    for start in range(0, len(pairs), 8192):
        pair = pairs[start : start + 8192]
        a, b = uv_triangles[pair[:, 0]], uv_triangles[pair[:, 1]]
        fa, fb = face[pair[:, 0]], face[pair[:, 1]]
        invalid = np.zeros(len(pair), dtype=bool)
        for own, other, ids, other_ids in ((a, b, fa, fb), (b, a, fb, fa)):
            e = np.roll(other, -1, axis=1) - other
            delta = own[:, :, None, :] - other[:, None, :, :]
            cross = _cross_2d(e[:, None], delta) * orientation
            edge_tol = tolerance * np.linalg.norm(e, axis=2)
            invalid |= np.any(np.all(cross > edge_tol[:, None], axis=2), axis=1)
            length2 = np.sum(e * e, axis=2)
            fraction = np.sum(delta * e[:, None], axis=3) / length2[:, None]
            projection = other[:, None] + np.clip(fraction, 0, 1)[..., None] * e[:, None]
            distance = np.linalg.norm(own[:, :, None] - projection, axis=3)
            shared = (ids[:, :, None] == other_ids[:, None, :]) | (
                ids[:, :, None] == np.roll(other_ids, -1, axis=1)[:, None, :]
            )
            invalid |= np.any((distance <= tolerance) & ~shared, axis=(1, 2))
        for i in range(3):
            aa, ab = a[:, i], a[:, (i + 1) % 3]
            ea = ab - aa
            for j in range(3):
                ba, bb = b[:, j], b[:, (j + 1) % 3]
                eb = bb - ba
                c0, c1 = _cross_2d(ea, ba - aa), _cross_2d(ea, bb - aa)
                d0, d1 = _cross_2d(eb, aa - ba), _cross_2d(eb, ab - ba)
                ta, tb = tolerance * np.linalg.norm(ea, axis=1), tolerance * np.linalg.norm(
                    eb, axis=1
                )
                invalid |= (((c0 > ta) & (c1 < -ta)) | ((c1 > ta) & (c0 < -ta))) & (
                    ((d0 > tb) & (d1 < -tb)) | ((d1 > tb) & (d0 < -tb))
                )
        if np.any(invalid):
            bad_pairs.extend(pair[invalid][:20].tolist())
            break
    if bad_pairs:
        _fail("UV triangles overlap or have nonshared/ambiguous contacts", face_pairs=bad_pairs)
    return {
        "orientation": orientation,
        "minimum_double_area": float(abs(area2).min()),
        "minimum_area_margin": float(np.min(abs(area2) / bound)),
        "overlap_candidate_pairs": len(pairs),
        "double_area_tolerances": bound,
    }


def _check_planar_edges(node, edges, tolerance, label):
    """Reject crossings and nonshared contacts between planar boundary edges.

    Edge endpoints index a two-coordinate node array. Shared vertices are
    permitted; other contacts within the distance tolerance raise ValueError.
    """
    xy = node[edges]
    pairs = _find_overlapping_box_pairs(xy.min(1), xy.max(1), tolerance)
    if not len(pairs):
        return
    a, b = xy[pairs[:, 0]], xy[pairs[:, 1]]
    ea, eb = a[:, 1] - a[:, 0], b[:, 1] - b[:, 0]
    ca0, ca1 = _cross_2d(ea, b[:, 0] - a[:, 0]), _cross_2d(ea, b[:, 1] - a[:, 0])
    cb0, cb1 = _cross_2d(eb, a[:, 0] - b[:, 0]), _cross_2d(eb, a[:, 1] - b[:, 0])
    ta, tb = tolerance * np.linalg.norm(ea, axis=1), tolerance * np.linalg.norm(eb, axis=1)
    bad = (((ca0 > ta) & (ca1 < -ta)) | ((ca1 > ta) & (ca0 < -ta))) & (
        ((cb0 > tb) & (cb1 < -tb)) | ((cb1 > tb) & (cb0 < -tb))
    )
    for own, other, pi, qi in ((a, b, 0, 1), (b, a, 1, 0)):
        e = other[:, 1] - other[:, 0]
        length2 = np.sum(e * e, axis=1)
        for k in range(2):
            delta = own[:, k] - other[:, 0]
            fraction = np.sum(delta * e, axis=1) / length2
            projection = other[:, 0] + np.clip(fraction, 0, 1)[:, None] * e
            contact = np.linalg.norm(own[:, k] - projection, axis=1) <= tolerance
            shared = np.any(edges[pairs[:, pi], k, None] == edges[pairs[:, qi]], axis=1)
            bad |= contact & ~shared
    if np.any(bad):
        _fail(
            f"{label}: crossing, overlapping, or numerically ambiguous nonshared boundary contact",
            edge_pairs=pairs[bad][:20].tolist(),
        )


def _check_domain_coverage(queries, query_indices, boundary_uv, boundary_edges, numerical_policy):
    """Check query coverage using even-odd parity over simple boundary loops.

    Holes are excluded by parity. Queries within the UV distance tolerance
    of a boundary are accepted. An uncovered query raises ValueError.
    """
    segment = boundary_uv[boundary_edges]
    a, b = segment[:, 0], segment[:, 1]
    edge = b - a
    length2 = np.sum(edge * edge, axis=1)
    tolerance = numerical_policy["uv_distance_tolerance"]
    for start in range(0, len(query_indices), 256):
        ids = query_indices[start : start + 256]
        q = queries[ids]
        crosses_y = (a[None, :, 1] > q[:, None, 1]) != (b[None, :, 1] > q[:, None, 1])
        denominator = np.where(edge[:, 1] == 0, 1, edge[:, 1])
        x = a[None, :, 0] + (q[:, None, 1] - a[None, :, 1]) * edge[None, :, 0] / denominator
        inside = np.sum(crosses_y & (q[:, None, 0] < x), axis=1) % 2 == 1
        delta = q[:, None] - a[None]
        fraction = np.clip(np.sum(delta * edge[None], axis=2) / length2, 0, 1)
        distance = np.linalg.norm(delta - fraction[..., None] * edge[None], axis=2).min(1)
        bad = np.flatnonzero(~inside & (distance > tolerance))
        if len(bad):
            k = bad[0]
            _fail(
                f"Template node {ids[k]} is outside the prescribed source boundary",
                node_index=int(ids[k]),
                distance_normalized=float(distance[k]),
                allowed_distance=tolerance,
            )


def _build_box_tree(lower, upper, leaf_size=8):
    """Build an axis-aligned bounding-box tree over indexed geometric primitives.

    Lower and upper bounds have shape (primitive_count, dimension). Return
    tree nodes with either child indices or sorted primitive indices.
    """
    nodes = []

    def build(ids):
        """Append a subtree and return its root index."""
        index = len(nodes)
        lo, hi = lower[ids].min(0), upper[ids].max(0)
        nodes.append({"lower": lo, "upper": hi, "count": len(ids)})
        if len(ids) <= leaf_size:
            nodes[index]["indices"] = np.sort(ids)
        else:
            centers = (lower[ids] + upper[ids]) / 2
            axis = np.argmax(np.ptp(centers, axis=0))
            ids = ids[np.argsort(centers[:, axis], kind="stable")]
            half = len(ids) // 2
            nodes[index]["children"] = (build(ids[:half]), build(ids[half:]))
        return index

    build(np.arange(len(lower)))
    return nodes


def _find_overlapping_box_pairs(lower, upper, tolerance):
    """Return distinct primitive-index pairs whose bounding boxes overlap within tolerance."""
    tree = _build_box_tree(lower, upper)
    stack, pairs = [(0, 0)], []
    while stack:
        ai, bi = stack.pop()
        a, b = tree[ai], tree[bi]
        if np.any(a["lower"] > b["upper"] + tolerance) or np.any(
            b["lower"] > a["upper"] + tolerance
        ):
            continue
        if "indices" in a and "indices" in b:
            ii, jj = np.meshgrid(a["indices"], b["indices"], indexing="ij")
            ii, jj = ii.ravel(), jj.ravel()
            use = (ii < jj) if ai == bi else np.ones(len(ii), dtype=bool)
            use &= np.all(lower[ii] <= upper[jj] + tolerance, axis=1) & np.all(
                lower[jj] <= upper[ii] + tolerance, axis=1
            )
            pairs.extend(zip(np.minimum(ii[use], jj[use]), np.maximum(ii[use], jj[use])))
        elif ai == bi:
            left, right = a["children"]
            stack.extend(((left, left), (left, right), (right, right)))
        elif "children" in a and ("indices" in b or a["count"] >= b["count"]):
            stack.extend((child, bi) for child in a["children"])
        else:
            stack.extend((ai, child) for child in b["children"])
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _squared_distance_to_box(point, box):
    """Return the squared distance from a point to an axis-aligned bounding box."""
    d = np.maximum(np.maximum(box["lower"] - point, point - box["upper"]), 0).astype(np.float64)
    return float(d @ d)


def _get_edge_opposite_vertices(source_element, edge, edge_to_element_adj_table, edge_index):
    """Return the third vertex of each triangle incident to the selected edge."""
    a, b = edge[edge_index]
    faces = edge_to_element_adj_table[edge_index]
    return np.array(
        [next(int(v) for v in source_element[k] if v != a and v != b) for k in faces],
        dtype=np.int64,
    )


def _order_vertex_one_ring(source_element, source_topology, node_index):
    """Order the neighbors of a free vertex around its closed manifold fan.

    Start at the smallest neighbor index and use a deterministic direction.
    Return an integer index array; an incomplete ring raises ValueError.
    """
    graph = source_topology["vertex_links"][node_index]
    start = min(graph)
    ordered, prev, cur = [start], None, start
    while True:
        nxt = min(v for v in graph[cur] if v != prev)
        if nxt == start:
            break
        ordered.append(nxt)
        prev, cur = cur, nxt
    if len(ordered) != len(graph):
        _fail(f"Incomplete one-ring at node {node_index}")
    return np.asarray(ordered, dtype=np.int64)


def _find_connected_vertices(graph, start):
    """Return the vertices reachable from a starting vertex in an adjacency graph."""
    seen, stack = {start}, [start]
    while stack:
        for j in graph[stack.pop()]:
            if j not in seen:
                seen.add(j)
                stack.append(j)
    return seen


def _copy_mesh_faces(mesh):
    """Copy mesh connectivity into Python lists without changing the mesh."""
    return (
        mesh.element.detach().cpu().tolist()
        if torch.is_tensor(mesh.element)
        else deepcopy(mesh.element)
    )


def _cross_2d(a, b):
    """Return the signed 2D cross product over the final coordinate axis."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _fail(message, **diagnostics):
    """Raise ValueError with structured diagnostics attached to the exception."""
    error = ValueError(message)
    error.diagnostics = diagnostics
    raise error
