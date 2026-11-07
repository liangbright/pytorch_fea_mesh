"""Isotropic remeshing of triangle surfaces, adapted from the author's C++ code on GitHub.

Each call works on its own copy of the mesh, so the source mesh and the feature lists
are never changed. The work is done in double precision on the CPU, and the output goes
back to the data type and device of the source. After the public function and its
report, the functions are grouped by stage: input checks, protected features, the working
topology, reference geometry, the iteration loop, local edits, collapsing and repair,
nested-triangle replacement, flipping and smoothing, and output checks.

The mesh library provides the mesh classes, the first adjacency tables and the geometry
calculations. This file keeps its own adjacency tables up to date, because they depend on
how edits are ordered, how faces are renumbered and how repair works on temporary copies.
_initialize_topology() describes the working topology, the connectivity and geometry tables
used during a call. The mesh tensors share memory with its arrays, and the library's own
adjacency tables are cleared during editing so that nothing reads an out-of-date copy.

A short edge can be blocked from collapsing by a link obstruction, usually because its end
points share extra neighbors. The remesher first tries a local repair that stays close to
the original shape. If that fails, the edge waits through the first ceil(max_iter/2)
iterations. In the remaining iterations, the nested triangles on that edge may be replaced
by one element. Protected features always take priority. See doc/remeshing.md for the rules
and the known limits.
"""

from __future__ import annotations

from collections import Counter, deque
from numbers import Integral, Real
import logging
import math
import time

import numpy as np
import torch
import vtk
from ..mesh.triangle_mesh import TriangleMesh
from ..mesh.mesh import Mesh

# Public API


def isotropic_triangle_surface_remesh(
    source_mesh: TriangleMesh,
    target_edge_length: float,
    flag_process_boundary: bool = False,
    *,
    feature_point_index_list: list[int] | None = None,
    feature_edge_index_list: list[int] | None = None,
    feature_curve_list: list[list[int]] | None = None,
    target_edge_length_ratio_max: float = 1.4,
    target_edge_length_ratio_min: float = 0.7,
    max_iter: int = 10,
    flag_project_to_source_mesh: bool = True,
    flag_use_coverage_then_tangential: bool = False,
) -> TriangleMesh:
    """Return a new remeshed surface. The source mesh is not changed.

    The first ten arguments match the ten inputs of the original C++ remesher, and
    flag_use_coverage_then_tangential is an addition. The first three can be passed by
    position. Arguments that aren't valid raise TypeError or ValueError. So does a source
    triangle with no area, see doc/remeshing.md.

    target_edge_length must be finite and at least 1e-5. For a smaller target, scale
    source_mesh.node and target_edge_length by the same factor. An edge counts as short at
    or below target_edge_length * target_edge_length_ratio_min, and as long above
    target_edge_length * target_edge_length_ratio_max.

    max_iter is the number of iterations. One last collapse pass runs after them, even when
    max_iter is 0.

    flag_process_boundary is False by default, which keeps every boundary point and
    edge exactly where it is. Set it to True to split, collapse and smooth the boundary.

    Feature points stay where they are. Feature edges stay in the mesh, and their end points
    stay where they are. Feature edges are numbered as in the edge table of the source mesh.
    A feature curve lists source points in order, with an existing edge between each
    consecutive pair, and those edges become feature edges. Repeat the first point to close
    the curve.

    flag_project_to_source_mesh moves smoothed vertices back onto the source surface, or
    onto their original boundary curve. As in the C++ remesher, splits and collapses
    don't project, and neither does repair.

    flag_use_coverage_then_tangential chooses how interior vertices are smoothed. By default
    (False), a vertex moves to the plain average of its neighbors, which pulls curved surfaces
    and thin parts in further with each iteration. With True, a vertex moves to the centre of the
    part of the source closest to it (coverage-based smoothing, also called Lloyd relaxation),
    and in the last two iterations it slides within its tangent plane instead (tangential
    smoothing), unless it is on a sharp crease or corner. The first iteration always uses
    coverage-based smoothing, so with max_iter=2 tangential smoothing runs only in the last
    iteration, and with max_iter=1 it doesn't run. It reads the source mesh even when
    flag_project_to_source_mesh is False.

    Diagnostics are in output.mesh_data['remeshing_report']. Its failures entry says what
    the run couldn't do: error is the error that stopped the run, or None. short_edges and
    long_edges count the edges still outside the length limits, and short_edge_reasons counts
    the short edges by why the last collapse tried on each one failed. area_ratio is the
    output area divided by the source area. If the run stops with an error once remeshing has
    started, the same report is in the error's diagnostics attribute.

    Fields of the source mesh are not copied to the output.
    """
    _validate_input(
        source_mesh,
        target_edge_length,
        flag_process_boundary,
        feature_point_index_list=feature_point_index_list,
        feature_edge_index_list=feature_edge_index_list,
        feature_curve_list=feature_curve_list,
        target_edge_length_ratio_max=target_edge_length_ratio_max,
        target_edge_length_ratio_min=target_edge_length_ratio_min,
        max_iter=max_iter,
        flag_project_to_source_mesh=flag_project_to_source_mesh,
        flag_use_coverage_then_tangential=flag_use_coverage_then_tangential,
    )
    started = time.perf_counter()
    mesh = _copy_source_mesh(source_mesh)
    parameters = _make_parameter(
        target_edge_length,
        target_edge_length_ratio_max,
        target_edge_length_ratio_min,
        max_iter,
        flag_process_boundary,
        flag_project_to_source_mesh,
        flag_use_coverage_then_tangential,
    )
    tolerances = _compute_tolerance(mesh, target_edge_length)
    topology = _initialize_topology(mesh, round_positions=source_mesh.node.dtype == torch.float32)
    features = _initialize_feature_constraint(
        source_mesh,
        mesh,
        topology,
        feature_point_index_list,
        feature_edge_index_list,
        feature_curve_list,
    )
    source_area = sum(topology['areas'].values())
    try:
        reference, boundary = _prepare_initial_mesh(mesh, topology, parameters, tolerances)
        _run_remeshing_iteration(
            mesh, topology, features, boundary, reference, parameters, tolerances
        )
        output = _finalize_output(
            mesh, topology, source_mesh, reference, parameters, features, boundary, tolerances
        )
    except Exception as error:
        error.diagnostics = _build_report(
            source_mesh, topology, parameters, source_area, started, error=error
        )
        raise
    output.mesh_data['remeshing_report'] = _build_report(
        source_mesh, topology, parameters, source_area, started, output=output
    )
    return output


def _build_report(
    source_mesh, topology, parameters, source_area, started, *, output=None, error=None
):
    """Return the report of a run. A run stopped by an error has no output sizes."""
    return {
        'parameters': dict(parameters),
        'operations': dict(topology['counts']),
        'failures': _report_failures(topology, parameters, source_area, error),
        'iterations': topology['iterations'],
        'nested_replacements': topology['replacements'],
        'input_nodes': len(source_mesh.node),
        'input_faces': len(source_mesh.element),
        'output_nodes': None if output is None else len(output.node),
        'output_faces': None if output is None else len(output.element),
        'elapsed_seconds': time.perf_counter() - started,
    }


def _report_failures(topology, parameters, source_area, error):
    """Summarize what the run couldn't do. See the public function for the entries.

    A short edge that no collapse tried counts as not_tried. It's usually on a boundary.
    """
    lengths = topology['lengths'].values()
    short_edges = [
        edge
        for edge, length in topology['lengths'].items()
        if length <= parameters['min_edge_length']
    ]
    return dict(
        error=None if error is None else f'{type(error).__name__}: {error}',
        short_edges=len(short_edges),
        short_edge_reasons=dict(
            Counter(topology['collapse_failures'].get(edge, 'not_tried') for edge in short_edges)
        ),
        long_edges=sum(length > parameters['max_edge_length'] for length in lengths),
        area_ratio=sum(topology['areas'].values()) / source_area if source_area else None,
    )


# Input validation and per-call configuration


def _as_numpy(value):
    """Return a CPU NumPy array. It may share memory with the input, so copy it before editing."""
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _new_mesh(node, element):
    """Build a working mesh. An empty mesh still gets arrays of the right shape."""
    device = node.device if torch.is_tensor(node) else 'cpu'
    node = torch.as_tensor(node, dtype=torch.float64, device=device).reshape(-1, 3)
    element = torch.as_tensor(element, dtype=torch.int64, device=device).reshape(-1, 3)
    if len(element):
        return TriangleMesh(node, element)
    mesh = TriangleMesh()
    mesh.node, mesh.element = node, element
    return mesh


def _edge(a, b):
    """Return the two end points in sorted order, which is how edges are identified here."""
    return (int(a), int(b)) if a < b else (int(b), int(a))


def _face_edge(face):
    """Return the three edges of a triangle, each with its end points sorted."""
    a, b, c = face
    return (_edge(a, b), _edge(b, c), _edge(c, a))


def _index_list(values, size, name):
    """Check a one-dimensional list of source indices and return it as ints, in the same order.

    None gives an empty list.
    """
    if values is None:
        return []
    if torch.is_tensor(values):
        if values.ndim != 1 or (
            values.dtype.is_floating_point or values.dtype.is_complex or values.dtype == torch.bool
        ):
            raise TypeError(f'{name} must contain integer indices')
        values = values.tolist()
    elif isinstance(values, np.ndarray):
        if values.ndim != 1 or values.dtype.kind not in 'iu':
            raise TypeError(f'{name} must contain integer indices')
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        raise TypeError(f'{name} must be an index sequence')
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in values):
        raise TypeError(f'{name} must contain integer indices')
    result = [int(v) for v in values]
    if any(v < 0 or v >= size for v in result):
        raise ValueError(f'{name} contains an out-of-range source index')
    return result


def _source_edge(source_mesh):
    """Return the edges of the source mesh in their original numbering.

    If the source has no edge table, build one on a private copy. If it has one, check
    that it matches the faces.
    """
    expected_edges = {
        _edge(a, b) for face in _as_numpy(source_mesh.element) for a, b in _face_edge(face)
    }
    if source_mesh.edge is None:
        private = _copy_source_mesh(source_mesh)
        if not len(private.element):
            return []
        private.build_edge()
        return [tuple(edge) for edge in private.edge.tolist()]
    source_edge_array = _as_numpy(source_mesh.edge)
    if (
        source_edge_array.ndim != 2
        or source_edge_array.shape[1] != 2
        or source_edge_array.dtype.kind not in 'iu'
    ):
        raise ValueError('source edge table must be an integer E x 2 array')
    edges = [_edge(*edge) for edge in source_edge_array]
    if len(set(edges)) != len(edges) or set(edges) != expected_edges:
        raise ValueError('source edge table does not match the faces')
    return edges


def _validate_input(
    source_mesh,
    target_edge_length,
    flag_process_boundary=False,
    *,
    feature_point_index_list=None,
    feature_edge_index_list=None,
    feature_curve_list=None,
    target_edge_length_ratio_max=1.4,
    target_edge_length_ratio_min=0.7,
    max_iter=10,
    flag_project_to_source_mesh=True,
    flag_use_coverage_then_tangential=False,
):
    """Check the public arguments and the source arrays, without changing the source."""
    if not isinstance(source_mesh, TriangleMesh):
        raise TypeError('source_mesh must be a TriangleMesh')
    if not torch.is_tensor(source_mesh.node) or source_mesh.node.dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise TypeError('source node coordinates must be float32 or float64 tensors')
    positions, faces = _as_numpy(source_mesh.node), _as_numpy(source_mesh.element)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError('source coordinates must be finite N x 3')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu':
        raise ValueError('source connectivity must be an integer M x 3 triangle array')
    if len(faces) and (
        faces.min() < 0
        or faces.max() >= len(positions)
        or np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0)
    ):
        raise ValueError('invalid triangle vertex indices')
    for name, value in [
        ('target_edge_length', target_edge_length),
        ('target_edge_length_ratio_min', target_edge_length_ratio_min),
        ('target_edge_length_ratio_max', target_edge_length_ratio_max),
    ]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f'{name} must be a real number')
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if target_edge_length < 1e-5:
        raise ValueError(
            'target_edge_length must be at least 1e-5; rescale '
            'source_mesh.node and target_edge_length by the same factor'
        )
    if target_edge_length_ratio_min >= target_edge_length_ratio_max:
        raise ValueError('minimum ratio must be smaller than maximum ratio')
    if not all(
        math.isfinite(target_edge_length * r)
        for r in (target_edge_length_ratio_min, target_edge_length_ratio_max)
    ):
        raise ValueError('edge length limits are too large to compute')
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, Integral) or max_iter < 0:
        raise ValueError('max_iter must be a nonnegative integer')
    if not isinstance(flag_process_boundary, bool) or not isinstance(
        flag_project_to_source_mesh, bool
    ):
        raise TypeError('flag_process_boundary and flag_project_to_source_mesh must be bool')
    if not isinstance(flag_use_coverage_then_tangential, bool):
        raise TypeError('flag_use_coverage_then_tangential must be bool')
    _index_list(feature_point_index_list, len(positions), 'feature_point_index_list')
    if feature_edge_index_list is not None:
        _index_list(
            feature_edge_index_list, len(_source_edge(source_mesh)), 'feature_edge_index_list'
        )
    if feature_curve_list is not None:
        if not isinstance(feature_curve_list, (list, tuple)):
            raise TypeError('feature_curve_list must be a sequence of curves')
        edges = set(_source_edge(source_mesh))
        for curve in feature_curve_list:
            ids = _index_list(curve, len(positions), 'feature_curve')
            if any(_edge(a, b) not in edges for a, b in zip(ids, ids[1:])):
                raise ValueError('feature curve contains a nonexistent source edge')


def _copy_source_mesh(source_mesh):
    """Copy the nodes and elements into separate CPU arrays, as float64 and int64."""
    node = source_mesh.make_node_copy('torch', dtype=torch.float64).detach().cpu().clone()
    element = torch.as_tensor(
        source_mesh.make_element_copy('numpy'), dtype=torch.int64, device=node.device
    ).clone()
    return _new_mesh(node, element)


def _make_parameter(
    target_edge_length,
    target_edge_length_ratio_max,
    target_edge_length_ratio_min,
    max_iter,
    flag_process_boundary,
    flag_project_to_source_mesh,
    flag_use_coverage_then_tangential=False,
):
    """Collect the settings for one call, including the edge length limits and repair budgets."""
    return dict(
        target_edge_length=float(target_edge_length),
        target_edge_length_ratio_max=float(target_edge_length_ratio_max),
        target_edge_length_ratio_min=float(target_edge_length_ratio_min),
        min_edge_length=float(target_edge_length * target_edge_length_ratio_min),
        max_edge_length=float(target_edge_length * target_edge_length_ratio_max),
        max_iter=int(max_iter),
        flag_process_boundary=flag_process_boundary,
        flag_project_to_source_mesh=flag_project_to_source_mesh,
        flag_use_coverage_then_tangential=flag_use_coverage_then_tangential,
        repair_rings=2,
        repair_max_faces=256,
        repair_max_attempts=8,
        repair_max_new_vertices=8,
        repair_distance=0.1 * float(target_edge_length),
    )


# Protected features and geometric tolerances


def _initialize_feature_constraint(
    source_mesh,
    mesh,
    topology,
    feature_point_index_list,
    feature_edge_index_list,
    feature_curve_list,
):
    """Collect the protected points and edges, and record where the protected points are.

    The end points of feature edges and feature curves are protected too. The returned
    positions dictionary gives the original coordinates of each protected vertex.
    """
    points = set(_index_list(feature_point_index_list, len(mesh.node), 'feature points'))
    edges = set()
    if feature_edge_index_list is not None:
        source_edges = _source_edge(source_mesh)
        edges.update(
            source_edges[i]
            for i in _index_list(feature_edge_index_list, len(source_edges), 'feature edges')
        )
    edges.update(_convert_feature_curve_to_edge(source_mesh, topology, feature_curve_list))
    points.update(v for e in edges for v in e)
    return {
        'points': points,
        'edges': edges,
        'positions': {v: topology['vertex_positions'][v].copy() for v in points},
    }


def _convert_feature_curve_to_edge(source_mesh, source_topology, feature_curve_list):
    """Return the edges between consecutive points of each curve; raise if one doesn't exist."""
    edges = set()
    for curve in feature_curve_list or []:
        ids = _index_list(curve, len(source_mesh.node), 'feature curve')
        for a, b in zip(ids, ids[1:]):
            e = _edge(a, b)
            if e not in source_topology['edge_to_faces']:
                raise ValueError('feature curve contains a nonexistent edge')
            edges.add(e)
    return edges


def _check_feature_constraint(mesh, topology, features, edit=None, *, old_to_new_vertex=None):
    """Return (valid, reason): whether protected points stay put and protected edges survive.

    With an edit, this checks what the edit would do, without applying it. It looks only at
    what the edit touches, so its cost doesn't grow with the number of protected items. Every
    other protected item is known to be fine, because every committed edit passed this check.
    The exception is a repair patch's absent_edges, see _extract_repair_patch, so those are
    always looked at. old_to_new_vertex is used only when checking the renumbered output
    against the original protected IDs.
    """
    if edit:
        protected_positions = features['positions']
        for v, q in edit.get('positions', {}).items():
            if v in protected_positions and not np.array_equal(q, protected_positions[v]):
                return False, 'modified_feature_point'
        if any(v in protected_positions for v in edit.get('retired', ())):
            return False, 'modified_feature_point'
        if features['edges']:
            removed_edges = {
                e for f in edit['remove_faces'] for e in _face_edge(topology['faces'][f])
            }
            touched_edges = [e for e in removed_edges if e in features['edges']]
            for key in touched_edges + list(features.get('absent_edges', ())):
                count = len(topology['edge_to_faces'].get(key, set()) - edit['remove_faces'])
                count += sum(key in _face_edge(f) for f in edit['add_faces'])
                if not count:
                    return False, 'lost_feature_edge'
        return True, 'ok'
    mapping = old_to_new_vertex
    for v, p in features['positions'].items():
        w = mapping.get(v, -1) if mapping is not None else v
        if w < 0 or w >= len(mesh.node):
            return False, 'lost_feature_point'
        if not np.array_equal(topology['vertex_positions'][w], p):
            return False, 'modified_feature_point'
    for e in features['edges']:
        key = _edge(*(mapping.get(v, -1) for v in e)) if mapping is not None else e
        if not topology['edge_to_faces'].get(key):
            return False, 'lost_feature_edge'
    return True, 'ok'


def _replacement_phase_allowed(iteration_index, max_iter):
    """Return whether this iteration, counted from zero, is in the late half.

    Only those iterations, the last floor(max_iter / 2), may use nested replacement.
    """
    return 0 <= iteration_index < max_iter and iteration_index >= (max_iter + 1) // 2


def _compute_tolerance(source_mesh, target_edge_length):
    """Work out the geometric tolerances from the size of the source and the target length."""
    positions = _as_numpy(source_mesh.node)
    extent = (
        float(np.linalg.norm(np.ptp(positions, axis=0))) if len(positions) else target_edge_length
    )
    scale = max(extent, float(target_edge_length), np.finfo(float).tiny ** 0.25)
    return {
        'length': scale * 1e-12,
        'area': scale * scale * 1e-14,
        'normal_dot': 1e-6,
    }


# Working topology and adjacency updates


def _initialize_topology(mesh, round_positions=False):
    """Build the working topology for one call from the mesh library's adjacency tables.

    Storage:
        vertex_positions / faces: the active N x 3 float64 and M x 3 int64 arrays.
        vertex_buffer / face_buffer: the arrays behind them, with spare room. After
            growing a buffer, the active arrays and the mesh tensors must be refreshed.
        round_positions: whether new positions are rounded to float32 when proposed, because
            the source is float32. See round_position.

    Connectivity:
        edge_to_faces: maps an edge (smaller vertex, larger vertex) to the IDs of its incident
            faces, the faces that use it.
        vertex_to_faces / vertex_neighbors: lists of sets, indexed by vertex ID.
        boundary_edges / boundary_neighbors: the edges with exactly one face, and the
            neighbors of each vertex along those edges.
        face_keys: maps sorted vertex triples to face IDs, to detect duplicate faces.

    Geometry and bookkeeping:
        lengths: edge lengths by edge. areas / normals / angles: values by face ID.
        dirty_edges / dirty_faces: the values that need recalculating.
        revision: goes up with each applied edit. A proposed edit must match it.
        counts / iterations: records of operations and timing.
        collapse_failures: maps an edge to the reason from the last collapse tried on it.
        replacements: records of accepted nested replacements.

    Vertex IDs stay the same until the output is renumbered. Face IDs can change, because
    a removed row is filled with the last row. Everything that refers to the moved face
    must be updated before another edit is considered.
    """
    vertex_count, face_count = len(mesh.node), len(mesh.element)
    if face_count:
        mesh.clear_adj_info()
        mesh.build_edge()
        mesh.build_node_to_element_adj_table()
        mesh.build_node_to_node_adj_table()
        mesh.build_edge_to_element_adj_table()
        edge_to_faces = {
            tuple(edge): set(ids)
            for edge, ids in zip(mesh.edge.tolist(), mesh.edge_to_element_adj_table)
        }
        vertex_to_faces = [set(ids) for ids in mesh.node_to_element_adj_table]
        vertex_neighbors = [set(ids) for ids in mesh.node_to_node_adj_table]
    else:
        edge_to_faces = {}
        vertex_to_faces = [set() for _ in range(vertex_count)]
        vertex_neighbors = [set() for _ in range(vertex_count)]
    vertex_buffer = np.empty((max(16, vertex_count * 2), 3))
    face_buffer = np.empty((max(16, face_count * 2), 3), dtype=np.int64)
    vertex_buffer[:vertex_count], face_buffer[:face_count] = _as_numpy(mesh.node), _as_numpy(
        mesh.element
    )
    topology = dict(
        vertex_positions=vertex_buffer[:vertex_count],
        faces=face_buffer[:face_count],
        vertex_buffer=vertex_buffer,
        face_buffer=face_buffer,
        edge_to_faces=edge_to_faces,
        vertex_to_faces=vertex_to_faces,
        vertex_neighbors=vertex_neighbors,
        face_keys={tuple(sorted(face)): i for i, face in enumerate(face_buffer[:face_count])},
        boundary_neighbors=[set() for _ in range(vertex_count)],
        boundary_edges=set(),
        lengths={},
        areas={},
        normals={},
        angles={},
        dirty_faces=set(range(face_count)),
        dirty_edges=set(edge_to_faces),
        revision=0,
        round_positions=round_positions,
        replacements=[],
        counts=Counter(),
        collapse_failures={},
        iterations=[],
    )
    for edge, incident_faces in edge_to_faces.items():
        if len(incident_faces) == 1:
            topology['boundary_edges'].add(edge)
            a, b = edge
            topology['boundary_neighbors'][a].add(b)
            topology['boundary_neighbors'][b].add(a)
    _set_mesh_view(mesh, topology)
    mesh.clear_adj_info()  # Drop the library's edge table so nothing reads an old copy.
    _refresh_geometry(mesh, topology, full=True)
    return topology


def _set_mesh_view(mesh, topology):
    """Point the mesh tensors at the active arrays, without copying."""
    mesh.node = torch.from_numpy(topology['vertex_positions'])
    mesh.element = torch.from_numpy(topology['faces'])


def _attach_face(topology, face_id, face, remove=False):
    """Add a face to the adjacency tables, or remove it, and mark its edges for recalculation."""
    for vertex_id in face:
        if remove:
            topology['vertex_to_faces'][vertex_id].discard(face_id)
        else:
            topology['vertex_to_faces'][vertex_id].add(face_id)
    for edge in _face_edge(face):
        topology['dirty_edges'].add(edge)
        a, b = edge
        if remove:
            topology['edge_to_faces'][edge].remove(face_id)
            if not topology['edge_to_faces'][edge]:
                del topology['edge_to_faces'][edge]
                topology['vertex_neighbors'][a].discard(b)
                topology['vertex_neighbors'][b].discard(a)
        else:
            topology['edge_to_faces'].setdefault(edge, set()).add(face_id)
            topology['vertex_neighbors'][a].add(b)
            topology['vertex_neighbors'][b].add(a)
    key = tuple(sorted(face))
    if remove:
        topology['face_keys'].pop(key, None)
    else:
        topology['face_keys'][key] = face_id


def _update_adjacency_after_edit(mesh, topology, changes):
    """Remove and add faces, updating only the adjacency entries they affect.

    Faces are removed from the highest ID down, so each requested ID still refers to the
    face it named. Each gap is filled with the last active face, and everything that
    referred to that face is updated. New faces go at the end, then the boundary tables
    are refreshed. _commit_edit() recalculates the geometry after this returns.
    """
    face_count = len(topology['faces'])
    for face_id in sorted(changes['remove_faces'], reverse=True):
        last_face_id = face_count - 1
        removed_face = tuple(topology['face_buffer'][face_id])
        _attach_face(topology, face_id, removed_face, remove=True)
        for measurement_name in ('areas', 'normals', 'angles'):
            topology[measurement_name].pop(face_id, None)
        topology['dirty_faces'].discard(face_id)
        if face_id != last_face_id:
            moved_face = tuple(topology['face_buffer'][last_face_id])
            _attach_face(topology, last_face_id, moved_face, remove=True)
            topology['face_buffer'][face_id] = moved_face
            _attach_face(topology, face_id, moved_face)
            for measurement_name in ('areas', 'normals', 'angles'):
                topology[measurement_name][face_id] = topology[measurement_name].pop(last_face_id)
            topology['dirty_faces'].add(face_id)
        topology['dirty_faces'].discard(last_face_id)
        face_count -= 1
    required_capacity = face_count + len(changes['add_faces'])
    if required_capacity > len(topology['face_buffer']):
        storage = np.empty(
            (max(required_capacity, len(topology['face_buffer']) * 2), 3), dtype=np.int64
        )
        storage[:face_count] = topology['face_buffer'][:face_count]
        topology['face_buffer'] = storage
    for face in changes['add_faces']:
        topology['face_buffer'][face_count] = face
        _attach_face(topology, face_count, face)
        topology['dirty_faces'].add(face_count)
        face_count += 1
    topology['faces'] = topology['face_buffer'][:face_count]
    for edge in topology['dirty_edges']:
        a, b = edge
        if len(topology['edge_to_faces'].get(edge, ())) == 1:
            topology['boundary_edges'].add(edge)
            topology['boundary_neighbors'][a].add(b)
            topology['boundary_neighbors'][b].add(a)
        else:
            topology['boundary_edges'].discard(edge)
            topology['boundary_neighbors'][a].discard(b)
            topology['boundary_neighbors'][b].discard(a)
    _set_mesh_view(mesh, topology)


def _refresh_geometry(mesh, topology, *, full=False):
    """Recalculate geometry for faces and edges marked as changed, or for all of them if full."""
    face_ids = sorted(range(len(topology['faces'])) if full else topology['dirty_faces'])
    if face_ids:
        connectivity = torch.from_numpy(topology['faces'][face_ids])
        areas, normals = TriangleMesh.cal_element_area_and_normal(mesh.node, connectivity)
        angles = TriangleMesh.cal_element_corner_angle(mesh.node, connectivity)
        # The library may limit small denominators when it normalizes, but the orientation
        # checks need a unit normal for every nonzero normal.
        normal_array = normals.numpy()
        normal_lengths = np.linalg.norm(normal_array, axis=1, keepdims=True)
        normal_array = np.divide(
            normal_array, normal_lengths, out=np.zeros_like(normal_array), where=normal_lengths > 0
        )
        for face_id, area, normal, corner_angles in zip(
            face_ids, areas.numpy().ravel(), normal_array, angles.numpy()
        ):
            topology['areas'][face_id] = float(area)
            topology['normals'][face_id] = normal.copy()
            topology['angles'][face_id] = corner_angles.copy()
    edges = sorted(topology['edge_to_faces'] if full else topology['dirty_edges'])
    active_edges = [edge for edge in edges if edge in topology['edge_to_faces']]
    if active_edges:
        edge_table = torch.tensor(active_edges, device=mesh.node.device)
        lengths = Mesh.cal_edge_length(mesh.node, edge_table).numpy().ravel()
        topology['lengths'].update(zip(active_edges, map(float, lengths)))
    for edge in edges:
        if edge not in topology['edge_to_faces']:
            topology['lengths'].pop(edge, None)
    topology['dirty_faces'].clear()
    topology['dirty_edges'].clear()


def _check_adjacency_consistency(mesh, topology):
    """Raise an error if the adjacency tables, edge lengths or face areas differ from a rebuild."""
    other = _copy_source_mesh(mesh)
    rebuilt = _initialize_topology(other)
    for name in (
        'edge_to_faces',
        'vertex_to_faces',
        'vertex_neighbors',
        'boundary_neighbors',
        'boundary_edges',
        'face_keys',
    ):
        if topology[name] != rebuilt[name]:
            raise RuntimeError(f'{name} differs from a fresh rebuild')
    if set(topology['lengths']) != set(rebuilt['lengths']):
        raise RuntimeError('the set of measured edges differs from a fresh rebuild')
    for e, length in rebuilt['lengths'].items():
        if not np.isclose(length, topology['lengths'][e], rtol=1e-12, atol=1e-12):
            raise RuntimeError('an edge length differs from a fresh rebuild')
    for f, area in rebuilt['areas'].items():
        if not np.isclose(area, topology['areas'][f], rtol=1e-12, atol=1e-12):
            raise RuntimeError('a face area differs from a fresh rebuild')


def _validate_mesh(topology, tolerances, *, strict_manifold=False, allow_empty=False):
    """Raise an error if the mesh is invalid; otherwise return a summary with its face count.

    strict_manifold also rejects nonmanifold edges, which have more than two faces.
    allow_empty accepts a mesh with no faces.
    """
    positions, faces = topology['vertex_positions'], topology['faces']
    if not len(faces) and not allow_empty:
        raise ValueError('empty mesh')
    if not np.isfinite(positions).all() or (
        len(faces) and (faces.min() < 0 or faces.max() >= len(positions))
    ):
        raise ValueError('mesh has non-finite coordinates or invalid vertex indices')
    if len(topology['face_keys']) != len(faces):
        raise ValueError('duplicate triangle faces')
    if any(
        area <= tolerances['area'] or not math.isfinite(area) for area in topology['areas'].values()
    ):
        raise ValueError('mesh has a degenerate triangle with no area')
    high_incidence_edge_count = sum(
        len(incident_faces) > 2 for incident_faces in topology['edge_to_faces'].values()
    )
    if strict_manifold and high_incidence_edge_count:
        raise ValueError('nonmanifold edge incidence: an edge has more than two faces')
    return dict(faces=len(faces))


# Original geometry, boundary constraints, and projection


def _prepare_initial_mesh(mesh, topology, parameters, tolerances):
    """Check the working mesh, record its geometry, and set up the boundary constraints."""
    summary = _validate_mesh(topology, tolerances, allow_empty=True)
    reference = _build_reference(mesh, topology, summary)
    return reference, _initialize_boundary_constraint(reference, parameters)


def _trace_boundary_component(topology, *, allow_unresolved=False):
    """Group the boundary edges into connected components, using this file's own tables.

    A component is one separate piece. Each is (vertices, edges, regular), where regular means
    every vertex has exactly two boundary neighbors. An irregular component raises an error
    unless allow_unresolved is set.
    """
    remaining = set(topology['boundary_edges'])
    components = []
    while remaining:
        seed = min(remaining)
        nodes, edges, queue = set(seed), set(), deque(seed)
        while queue:
            u = queue.popleft()
            for v in sorted(topology['boundary_neighbors'][u]):
                e = _edge(u, v)
                edges.add(e)
                if v not in nodes:
                    nodes.add(v)
                    queue.append(v)
        remaining.difference_update(edges)
        regular = all(len(topology['boundary_neighbors'][v]) == 2 for v in nodes)
        if not regular and not allow_unresolved:
            raise ValueError('a boundary vertex does not have exactly two boundary neighbors')
        components.append((tuple(sorted(nodes)), tuple(sorted(edges)), regular))
    return components


def _build_reference(mesh, topology, topology_summary):
    """Record the original geometry and boundary components for this call.

    The VTK locators for the surface and the boundary are built when first needed. labels
    maps each boundary vertex to its original boundary component, and boundary_positions keeps
    the original coordinates of those vertices.
    """
    original = _copy_source_mesh(mesh)
    components = _trace_boundary_component(topology, allow_unresolved=True)
    labels = {v: k for k, (nodes, _, _) in enumerate(components) for v in nodes}
    return dict(
        mesh=original,
        summary=topology_summary,
        surface_locator=None,
        boundary_components=components,
        boundary_locators={},
        labels=labels,
        boundary_positions={v: topology['vertex_positions'][v].copy() for v in labels},
        boundary_edges=set(topology['boundary_edges']),
    )


def _build_cell_locator(reference_mesh):
    """Build a VTK cell locator to reuse for many closest-point queries.

    Returns the locator together with the dataset it searches.
    """
    if not len(reference_mesh.element):
        raise ValueError('cannot build a locator for empty geometry')
    data = reference_mesh.convert_to_vtk()
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(data)
    locator.BuildLocator()
    return {'data': data, 'locator': locator}


def _initialize_boundary_constraint(reference, parameters):
    """Set up the boundary labels and, when boundary processing is off, the frozen boundary.

    labels gains a vertex when a boundary edge is split, and loses one when a vertex is
    removed. frozen has the same points, edges and positions entries as the feature
    constraints, and is empty when boundary processing is on.
    """
    frozen = not parameters['flag_process_boundary']
    return dict(
        enabled=not frozen,
        labels=dict(reference['labels']),
        frozen={
            'points': set(reference['boundary_positions']) if frozen else set(),
            'positions': dict(reference['boundary_positions']) if frozen else {},
            'edges': set(reference['boundary_edges']) if frozen else set(),
        },
    )


def _closest_point(point, locator):
    """Return the closest point on the reference geometry and the distance to it."""
    result = [0.0, 0.0, 0.0]
    cell_id, subcell_id, squared_distance = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
    locator['locator'].FindClosestPoint(
        np.asarray(point).tolist(), result, cell_id, subcell_id, squared_distance
    )
    if int(cell_id) < 0 or not np.isfinite(result).all():
        raise RuntimeError('VTK closest-point query failed')
    return np.asarray(result), math.sqrt(max(0.0, float(squared_distance)))


def _project_point(point, reference, parameters, *, boundary_component=None):
    """Return the closest point on the source surface or on one original boundary component.

    With projection off, return a copy of the point.
    """
    if not parameters['flag_project_to_source_mesh']:
        return np.asarray(point, dtype=float).copy()
    if boundary_component is None:
        if reference['surface_locator'] is None:
            reference['surface_locator'] = _build_cell_locator(reference['mesh'])
        locator = reference['surface_locator']
    else:
        if boundary_component not in reference['boundary_locators']:
            _, edges, _ = reference['boundary_components'][boundary_component]
            curve = Mesh(
                node=reference['mesh'].node,
                element=[list(e) for e in edges],
                element_type=None,
                mesh_type='polyline',
            )
            reference['boundary_locators'][boundary_component] = _build_cell_locator(curve)
        locator = reference['boundary_locators'][boundary_component]
    return _closest_point(point, locator)[0]


# Iteration scheduling


def _run_remeshing_iteration(mesh, topology, features, boundary, reference, parameters, tolerances):
    """Run max_iter iterations, then one last collapse pass over the interior.

    Each iteration processes the boundary (if enabled), flips edges to even out valence (the
    number of edges at each vertex), splits long edges, collapses short edges, flips edges to
    reduce large angles, and smooths interior vertices. All iterations run even if one changes
    nothing, so edges waiting for the late half get their turn. The loop only stops early if
    no faces are left.

    The last pass uses the index of the last iteration, so it can't start the late half
    early. It also tries the short edges that form during it, because nothing runs after it
    to pick them up. Those edges only get an ordinary collapse.
    """
    logger = logging.getLogger(__name__)
    for iteration_index in range(parameters['max_iter']):
        if not len(mesh.element):
            break
        iteration_started, previous_counts = time.perf_counter(), dict(topology['counts'])
        _process_boundary(
            mesh,
            topology,
            features,
            boundary,
            reference,
            parameters,
            tolerances,
            iteration_index=iteration_index,
        )
        _equalize_valence(mesh, topology, features, boundary, parameters, tolerances)
        _split_long_edge(
            mesh, topology, features, boundary, parameters, tolerances, boundary_only=False
        )
        _collapse_short_edge(
            mesh,
            topology,
            features,
            boundary,
            parameters,
            tolerances,
            boundary_only=False,
            iteration_index=iteration_index,
        )
        _improve_triangle_angle(mesh, topology, features, boundary, parameters, tolerances)
        _relax_vertex(
            mesh,
            topology,
            features,
            boundary,
            reference,
            parameters,
            tolerances,
            boundary_only=False,
            use_coverage=parameters['flag_use_coverage_then_tangential']
            and iteration_index < coverage_iteration_count(parameters['max_iter']),
        )
        iteration_report = dict(
            iteration=iteration_index + 1,
            faces=len(mesh.element),
            active_nodes=sum(bool(ids) for ids in topology['vertex_to_faces']),
            seconds=time.perf_counter() - iteration_started,
            operations={
                operation_name: operation_count - previous_counts.get(operation_name, 0)
                for operation_name, operation_count in topology['counts'].items()
            },
        )
        topology['iterations'].append(iteration_report)
        logger.info(
            'iteration %d/%d: %d faces, %.2fs, %s',
            iteration_index + 1,
            parameters['max_iter'],
            len(mesh.element),
            iteration_report['seconds'],
            iteration_report['operations'],
        )
        # Keep going even if nothing changed: deferred edges may still be replaced later.
    if len(mesh.element):
        _collapse_short_edge(
            mesh,
            topology,
            features,
            boundary,
            parameters,
            tolerances,
            boundary_only=False,
            iteration_index=parameters['max_iter'] - 1,
            include_new_edges=True,
        )


def _process_boundary(
    mesh,
    topology,
    features,
    boundary,
    reference,
    parameters,
    tolerances,
    *,
    iteration_index,
):
    """Split, collapse and smooth along the boundary, if boundary processing is on."""
    if not parameters['flag_process_boundary']:
        return
    _split_long_edge(mesh, topology, features, boundary, parameters, tolerances, boundary_only=True)
    _collapse_short_edge(
        mesh,
        topology,
        features,
        boundary,
        parameters,
        tolerances,
        boundary_only=True,
        iteration_index=iteration_index,
    )
    _relax_vertex(
        mesh, topology, features, boundary, reference, parameters, tolerances, boundary_only=True
    )


# Local edit proposals


def _collect_candidate_edge(topology, parameters, *, boundary_only, operation):
    """Return the edges an operation should try, in a repeatable order.

    boundary_only chooses boundary edges or all the others. A split takes long edges, a
    collapse takes short edges with the shortest first, and a flip takes edges with two faces.
    """
    edges = []
    for edge, incident_faces in topology['edge_to_faces'].items():
        if (len(incident_faces) == 1) != boundary_only:
            continue
        length = topology['lengths'][edge]
        if (
            (operation == 'split' and length > parameters['max_edge_length'])
            or (operation == 'collapse' and length <= parameters['min_edge_length'])
            or (operation == 'flip' and len(incident_faces) == 2)
        ):
            edges.append(edge)
    if operation == 'collapse':
        return sorted(edges, key=lambda edge: (topology['lengths'][edge], edge))
    return sorted(edges)


def round_position(topology, point):
    """Return a new position as it will be written, so the checks see the final coordinates.

    The working arrays are float64. For a float32 source, an edit that passes in float64 could
    otherwise merge two nodes or lose a triangle's area when the output is rounded.
    """
    point = np.asarray(point, dtype=float)
    return point.astype(np.float32).astype(float) if topology['round_positions'] else point


def _proposal(
    topology,
    kind,
    *,
    edge=None,
    remove=(),
    add=(),
    positions=None,
    new_points=(),
    retired=(),
    normal_refs=(),
):
    """Create a record of an edit, without changing the mesh.

    remove_faces lists current face IDs, and add_faces lists new triangles as vertex
    triples. positions gives new coordinates for existing vertices, and new_points adds
    vertices in order. retired vertices must end up with no faces. normal_refs are the
    normals the new faces must agree with. revision ties the edit to the current mesh.
    """
    return dict(
        kind=kind,
        revision=topology['revision'],
        edge=edge,
        remove_faces=set(remove),
        add_faces=[tuple(map(int, f)) for f in add],
        positions={}
        if positions is None
        else {v: round_position(topology, p) for v, p in positions.items()},
        new_points=[round_position(topology, p).copy() for p in new_points],
        retired=set(retired),
        normal_refs=list(normal_refs),
    )


def _propose_edge_split(topology, edge, features, boundary):
    """Propose midpoint subdivision: a new vertex at the middle of the edge splits its faces.

    Each incident face becomes two. Returns None if the edge no longer exists, is a feature
    edge or is on a frozen boundary. On a boundary edge, the new vertex joins the boundary
    component that both end points belong to.
    """
    if (
        edge not in topology['edge_to_faces']
        or edge in features['edges']
        or edge in boundary['frozen']['edges']
    ):
        return None
    a, b = edge
    new_vertex_id = len(topology['vertex_positions'])
    replacement_faces, normal_references = [], []
    for face_id in sorted(topology['edge_to_faces'][edge]):
        triangle = tuple(topology['faces'][face_id])
        for corner_index in range(3):
            u, v, w = (
                triangle[corner_index],
                triangle[(corner_index + 1) % 3],
                triangle[(corner_index + 2) % 3],
            )
            if _edge(u, v) == edge:
                replacement_faces.extend([(u, new_vertex_id, w), (new_vertex_id, v, w)])
                normal_references.extend(
                    [topology['normals'][face_id], topology['normals'][face_id]]
                )
                break
    edit = _proposal(
        topology,
        'split',
        edge=edge,
        remove=topology['edge_to_faces'][edge],
        add=replacement_faces,
        new_points=[(topology['vertex_positions'][a] + topology['vertex_positions'][b]) / 2],
        normal_refs=normal_references,
    )
    if edge in topology['boundary_edges']:
        label = boundary['labels'].get(a)
        if label is not None and label == boundary['labels'].get(b):
            edit['boundary_labels'] = {new_vertex_id: label}
    return edit


def _split_long_edge(mesh, topology, features, boundary, parameters, tolerances, *, boundary_only):
    """Split long edges that pass validation, and recheck the two halves in the same pass."""
    candidates = deque(
        _collect_candidate_edge(
            topology, parameters, boundary_only=boundary_only, operation='split'
        )
    )
    while candidates:
        edge = candidates.popleft()
        if (
            edge not in topology['edge_to_faces']
            or topology['lengths'][edge] <= parameters['max_edge_length']
        ):
            continue
        if (edge in topology['boundary_edges']) != boundary_only:
            continue
        edit = _propose_edge_split(topology, edge, features, boundary)
        if edit is None:
            continue
        valid, _ = _validate_edit(mesh, topology, edit, features, boundary, parameters, tolerances)
        if valid:
            new_vertex_id = len(mesh.node)
            _commit_edit(mesh, topology, edit, boundary)
            for v in edge:  # Recheck the two halves, not the new edges to the opposite vertices.
                child_edge = _edge(v, new_vertex_id)
                if topology['lengths'][child_edge] > parameters['max_edge_length']:
                    candidates.append(child_edge)


def _plan_collapse_face(topology, edge, survivor_vertex_id):
    """Find the faces that disappear, and rewrite the other faces at both ends for the survivor."""
    u, v = edge
    removed_vertex_id = v if survivor_vertex_id == u else u
    affected_faces = topology['vertex_to_faces'][u] | topology['vertex_to_faces'][v]
    collapsed_faces = topology['vertex_to_faces'][u] & topology['vertex_to_faces'][v]
    if collapsed_faces != topology['edge_to_faces'].get(edge, set()):
        raise RuntimeError('the faces of an edge differ from the faces shared by its end points')
    replacement_faces, original_face_ids = [], []
    for face_id in sorted(affected_faces - collapsed_faces):
        replacement_faces.append(
            tuple(
                survivor_vertex_id if int(w) == removed_vertex_id else int(w)
                for w in topology['faces'][face_id]
            )
        )
        original_face_ids.append(face_id)
    return dict(
        affected_face_ids=affected_faces,
        replacement_faces=replacement_faces,
        original_face_ids=original_face_ids,
    )


def _propose_edge_collapse(topology, edge, features, boundary):
    """Propose collapses in order of preference: to the midpoint, then to either end point.

    A protected end point stays where it is and survives. If both are protected, or the edge
    itself is protected or gone, there is no proposal. For an interior edge, an end point on
    the boundary counts as protected.
    """
    if (
        edge not in topology['edge_to_faces']
        or edge in features['edges']
        or edge in boundary['frozen']['edges']
    ):
        return []
    u, v = edge
    interior_edge = edge not in topology['boundary_edges']

    def is_protected(w):
        # Look up only the two end points: copying the protected sets would visit all of them.
        return (
            w in features['points']
            or w in boundary['frozen']['points']
            or (interior_edge and bool(topology['boundary_neighbors'][w]))
        )

    u_protected, v_protected = is_protected(u), is_protected(v)
    if u_protected and v_protected:
        return []
    if u_protected:
        placements = [(u, topology['vertex_positions'][u])]
    elif v_protected:
        placements = [(v, topology['vertex_positions'][v])]
    else:
        placements = [
            (u, (topology['vertex_positions'][u] + topology['vertex_positions'][v]) / 2),
            (v, topology['vertex_positions'][v]),
            (u, topology['vertex_positions'][u]),
        ]
    result = []
    for survivor, position in placements:
        plan = _plan_collapse_face(topology, edge, survivor)
        edit = _proposal(
            topology,
            'collapse',
            edge=edge,
            remove=plan['affected_face_ids'],
            add=plan['replacement_faces'],
            positions={survivor: np.array(position, copy=True)},
            retired={v if survivor == u else u},
            normal_refs=[topology['normals'][f] for f in plan['original_face_ids']],
        )
        edit['survivor'] = survivor
        result.append(edit)
    return result


def _passes_collapse_length_check(topology, edit, parameters, *, boundary_only):
    """Return False if a kept neighboring edge would newly become too short or too long.

    Also return False if the edge no longer exists or is no longer short. For a boundary
    collapse, only neighbors on the boundary are checked.
    """
    edge = edit['edge']
    if (
        edge not in topology['edge_to_faces']
        or topology['lengths'][edge] > parameters['min_edge_length']
    ):
        return False
    survivor = edit['survivor']
    position = edit['positions'][survivor]
    # Only check neighbors that stay connected to the surviving vertex.
    retained_neighbors = {
        neighbor
        for f in edit['add_faces']
        if survivor in f
        for neighbor in f
        if neighbor != survivor
    }
    for endpoint in edge:
        for neighbor in topology['vertex_neighbors'][endpoint] - set(edge):
            if neighbor not in retained_neighbors or (
                boundary_only and not topology['boundary_neighbors'][neighbor]
            ):
                continue
            old_length = topology['lengths'][_edge(endpoint, neighbor)]
            proposed_length = float(
                np.linalg.norm(position - topology['vertex_positions'][neighbor])
            )
            min_length, max_length = parameters['min_edge_length'], parameters['max_edge_length']
            if (old_length >= min_length and proposed_length < min_length) or (
                old_length <= max_length and proposed_length > max_length
            ):
                return False
    return True


def _propose_edge_flip(topology, edge, features):
    """Propose the other diagonal of the edge's two faces.

    Returns None if the edge doesn't have exactly two faces, is a feature edge, has faces that
    aren't coherently oriented (they must run along the edge in opposite directions), or if the
    other diagonal is already an edge.
    """
    incident_faces = topology['edge_to_faces'].get(edge, ())
    if len(incident_faces) != 2 or edge in features['edges']:
        return None
    first_face_id, second_face_id = sorted(incident_faces)
    first_face, second_face = tuple(topology['faces'][first_face_id]), tuple(
        topology['faces'][second_face_id]
    )
    for corner_index in range(3):
        u, v, a = (
            first_face[corner_index],
            first_face[(corner_index + 1) % 3],
            first_face[(corner_index + 2) % 3],
        )
        if _edge(u, v) == edge:
            break
    b = next(w for w in second_face if w not in edge)
    if len({u, v, a, b}) != 4 or _edge(a, b) in topology['edge_to_faces']:
        return None
    if not any(
        second_face[corner_index] == v and second_face[(corner_index + 1) % 3] == u
        for corner_index in range(3)
    ):
        return None
    reference_normal = topology['normals'][first_face_id] + topology['normals'][second_face_id]
    return _proposal(
        topology,
        'flip',
        edge=edge,
        remove=incident_faces,
        add=[(a, b, v), (b, a, u)],
        normal_refs=[reference_normal, reference_normal],
    )


def _propose_vertex_move(vertex_id, candidate_position, features, boundary, reference, parameters):
    """Return a vertex's new position, projected if enabled, or None if the vertex is fixed."""
    if vertex_id in features['points'] or vertex_id in boundary['frozen']['points']:
        return None
    point = np.asarray(candidate_position, dtype=float)
    if parameters['flag_project_to_source_mesh']:
        point = _project_point(
            point, reference, parameters, boundary_component=boundary['labels'].get(vertex_id)
        )
    return point


def _candidate_point(topology, edit, faces):
    """Return triangle coordinates as they would be after an edit, with moved and new vertices."""
    result = np.empty((len(faces), 3, 3))
    existing_vertex_count = len(topology['vertex_positions'])
    for face_index, face in enumerate(faces):
        for corner_index, vertex_id in enumerate(face):
            if vertex_id < existing_vertex_count:
                position = edit['positions'].get(vertex_id, topology['vertex_positions'][vertex_id])
            else:
                position = edit['new_points'][vertex_id - existing_vertex_count]
            result[face_index, corner_index] = position
    return result


def _geometry(points, with_angles=True):
    """Return the areas, unit normals and corner angles of an array of triangles.

    Without with_angles, the angles are None. Validation doesn't use them, and the triangles it
    checks can have two corners at the same place, which makes the angle calculation print.
    """
    if not len(points):
        return np.empty(0), np.empty((0, 3)), np.empty((0, 3))
    node = torch.from_numpy(np.asarray(points, dtype=float).reshape(-1, 3))
    element = torch.arange(len(node), device=node.device).reshape(-1, 3)
    area, normals = TriangleMesh.cal_element_area_and_normal(node, element)
    angles = TriangleMesh.cal_element_corner_angle(node, element).numpy() if with_angles else None
    normal_array = normals.numpy()
    normal_lengths = np.linalg.norm(normal_array, axis=1, keepdims=True)
    normal_array = np.divide(
        normal_array, normal_lengths, out=np.zeros_like(normal_array), where=normal_lengths > 0
    )
    return area.numpy().ravel(), normal_array, angles


# Edit validation and commit


def _validate_edit(mesh, topology, edit, features, boundary, parameters, tolerances):
    """Check an edit without changing anything, and return (accepted, reason).

    The checks run in this order: revision and protections, rules for the kind of edit,
    geometry, then edge connectivity. The first failure becomes the reported reason, so
    the order matters.
    """
    if edit['revision'] != topology['revision']:
        return False, 'stale_edit'
    for locks in (features, boundary['frozen']):
        valid, reason = _check_feature_constraint(mesh, topology, locks, edit)
        if not valid:
            return valid, reason
    kind = edit['kind']
    if kind == 'nested_replacement' and not (
        edit['repair_failure']
        and _replacement_phase_allowed(edit['iteration_index'], parameters['max_iter'])
    ):
        return False, 'replacement_not_permitted'
    if kind == 'repair_collapse' and not edit.get('repair_validated'):
        return False, 'unvalidated_repair'
    if kind == 'collapse':
        classification = _classify_collapse_neighborhood(topology, edit['edge'], features, boundary)
        if classification['problematic']:
            return False, 'link_obstruction'
        if not edit['add_faces']:
            return False, 'ordinary_component_erasure'
    valid, reason = _validate_edit_geometry(topology, edit, tolerances)
    if not valid:
        return valid, reason
    return _validate_edit_connectivity(topology, edit)


def _validate_edit_geometry(topology, edit, tolerances) -> tuple[bool, str]:
    """Check the triangles an edit would leave, and return (accepted, reason).

    It rejects positions that aren't finite, invalid vertex numbers, duplicate triangles,
    degenerate triangles (with no area), and inverted triangles (flipped against their
    reference normals).
    """
    kind = edit['kind']
    if any(not np.isfinite(p).all() for p in list(edit['positions'].values()) + edit['new_points']):
        return False, 'nonfinite_position'
    if kind == 'move':
        affected_faces = sorted(
            set().union(
                *(topology['vertex_to_faces'][vertex_id] for vertex_id in edit['positions'])
            )
        )
        faces = [tuple(topology['faces'][face_id]) for face_id in affected_faces]
        normal_references = [topology['normals'][face_id] for face_id in affected_faces]
    else:
        faces, normal_references = edit['add_faces'], edit['normal_refs']
    keys = set()
    for face in faces:
        if len(set(face)) != 3 or any(
            vertex_id < 0
            or vertex_id >= len(topology['vertex_positions']) + len(edit['new_points'])
            for vertex_id in face
        ):
            return False, 'invalid_triangle_indices'
        key = tuple(sorted(face))
        existing_face_id = topology['face_keys'].get(key)
        if kind != 'move' and (
            key in keys
            or (existing_face_id is not None and existing_face_id not in edit['remove_faces'])
        ):
            return False, 'duplicate_triangle'
        keys.add(key)
    areas, normals, _ = _geometry(_candidate_point(topology, edit, faces), with_angles=False)
    if not np.isfinite(areas).all() or np.any(areas <= tolerances['area']):
        return False, 'degenerate_triangle'
    if normal_references and np.any(
        np.einsum('ij,ij->i', normals, np.asarray(normal_references)) <= tolerances['normal_dot']
    ):
        return False, 'inverted_triangle'
    return True, 'ok'


def _validate_edit_connectivity(topology, edit) -> tuple[bool, str]:
    """Check the change in edge incidence (the number of faces on an edge) and retired vertices.

    An edit is rejected if it gives an edge too many faces or leaves a retired vertex in use.
    """
    kind = edit['kind']
    if kind == 'move':
        return True, 'ok'

    incidence_delta = Counter(edge for face in edit['add_faces'] for edge in _face_edge(face))
    incidence_delta.subtract(
        edge for face_id in edit['remove_faces'] for edge in _face_edge(topology['faces'][face_id])
    )
    # A split may pass high incidence (more than two faces) from its edge on to the two halves.
    # No other edit may raise an edge's incidence above two, or raise it further if it's higher.
    split_limit = len(topology['edge_to_faces'].get(edit['edge'], ())) if kind == 'split' else 2
    for edge, change in incidence_delta.items():
        old_incidence = len(topology['edge_to_faces'].get(edge, ()))
        if old_incidence + change > max(2, old_incidence, split_limit):
            return False, 'new_nonmanifold_edge'
    if any(
        topology['vertex_to_faces'][vertex_id] - edit['remove_faces']
        for vertex_id in edit['retired']
    ):
        return False, 'retired_referenced_vertex'
    if any(vertex_id in edit['retired'] for face in edit['add_faces'] for vertex_id in face):
        return False, 'retired_vertex_in_replacement'
    return True, 'ok'


def _commit_edit(mesh, topology, edit, boundary):
    """Apply an edit that has already passed validation, and record it.

    Each operation is counted, and a nested replacement also gets a record in the report. A
    stale edit, made for an earlier revision of the mesh, raises an error. Every other
    rejection happens before this point.
    """
    if edit['revision'] != topology['revision']:
        raise RuntimeError('cannot apply an edit made for an earlier version of the mesh')
    vertex_count = len(topology['vertex_positions'])
    required_capacity = vertex_count + len(edit['new_points'])
    if required_capacity > len(topology['vertex_buffer']):
        storage = np.empty((max(required_capacity, len(topology['vertex_buffer']) * 2), 3))
        storage[:vertex_count] = topology['vertex_positions']
        topology['vertex_buffer'] = storage
    for vertex_id, position in enumerate(edit['new_points'], vertex_count):
        topology['vertex_buffer'][vertex_id] = position
        topology['vertex_to_faces'].append(set())
        topology['vertex_neighbors'].append(set())
        topology['boundary_neighbors'].append(set())
    topology['vertex_positions'] = topology['vertex_buffer'][:required_capacity]
    for moved_vertex_id, position in edit['positions'].items():
        topology['vertex_positions'][moved_vertex_id] = position
    if edit['kind'] != 'move':
        _update_adjacency_after_edit(mesh, topology, edit)
    else:
        _set_mesh_view(mesh, topology)
    for moved_vertex_id in edit['positions']:
        topology['dirty_faces'].update(topology['vertex_to_faces'][moved_vertex_id])
        topology['dirty_edges'].update(
            _edge(moved_vertex_id, neighbor_id)
            for neighbor_id in topology['vertex_neighbors'][moved_vertex_id]
        )
    boundary['labels'].update(edit.get('boundary_labels', {}))
    for retired_vertex_id in edit['retired']:
        boundary['labels'].pop(retired_vertex_id, None)
    _refresh_geometry(mesh, topology)
    topology['revision'] += 1
    topology['counts'][edit['kind']] += 1
    if edit['kind'] == 'nested_replacement':
        topology['replacements'].append(
            dict(
                kind=edit['kind'],
                iteration=edit.get('iteration_index', -1) + 1,
                edge=list(edit['edge']),
                removed_faces=len(edit['remove_faces']),
                added_faces=len(edit['add_faces']),
                repair_failure=edit.get('repair_failure'),
                outer_triangle=[*map(int, edit['edge']), int(edit['outer_vertex'])],
            )
        )


# Collapsing short edges and local repair


def _classify_collapse_neighborhood(topology, edge, features, boundary):
    """Identify the protections and link obstructions that stand in the way of a collapse.

    The link of a vertex is the ring of vertices and edges opposite it in its faces. The link
    condition says an edge can collapse safely only if the links of its end points share
    nothing except the corners opposite the edge. Breaking that is a link obstruction.

    Returns a dictionary with these entries:
        incident: the faces of the edge.
        common: the neighbors shared by both end points.
        extra: the shared neighbors that aren't opposite corners of the edge's faces.
        problematic: the collapse is blocked, because the edge has more than two faces, there
            are extra neighbors, or the links of both end points share an edge.
        eligible: the edge exists, and isn't a feature edge or on a frozen boundary.
    """
    incident = topology['edge_to_faces'].get(edge, set())
    u, v = edge
    common_neighbors = topology['vertex_neighbors'][u] & topology['vertex_neighbors'][v]
    opposite_vertices = {
        int(w) for face_id in incident for w in topology['faces'][face_id] if w not in edge
    }
    obstructing_neighbors = common_neighbors - opposite_vertices
    # The link condition also has an edge part: if the links of both end points share an edge,
    # the collapse would create duplicate triangles, as on a tetrahedron.
    link_u = {
        _edge(*(int(w) for w in topology['faces'][face_id] if w != u))
        for face_id in topology['vertex_to_faces'][u]
    }
    link_v = {
        _edge(*(int(w) for w in topology['faces'][face_id] if w != v))
        for face_id in topology['vertex_to_faces'][v]
    }
    shared_link_edges = link_u & link_v
    return dict(
        incident=set(incident),
        common=set(common_neighbors),
        extra=set(obstructing_neighbors),
        problematic=len(incident) > 2 or bool(obstructing_neighbors) or bool(shared_link_edges),
        eligible=bool(incident)
        and edge not in features['edges']
        and edge not in boundary['frozen']['edges'],
    )


def _collapse_short_edge(
    mesh,
    topology,
    features,
    boundary,
    parameters,
    tolerances,
    *,
    boundary_only,
    iteration_index,
    include_new_edges=False,
):
    """Try to collapse each short edge, and keep the reason from the last try on each one.

    With include_new_edges, go round again until the edges that became short along the way
    have had one try too. Those edges only get an ordinary collapse, because a repair can
    replace a short edge with another one forever. Without repair no vertices are added, so
    this ends.
    """
    tried, allow_repair = set(), True
    while True:
        candidates = [
            edge
            for edge in _collect_candidate_edge(
                topology, parameters, boundary_only=boundary_only, operation='collapse'
            )
            if edge not in tried
        ]
        for edge in candidates:
            _, reason = _attempt_collapse(
                mesh,
                topology,
                edge,
                features,
                boundary,
                parameters,
                tolerances,
                boundary_only=boundary_only,
                iteration_index=iteration_index,
                allow_repair=allow_repair,
            )
            # An edge that changed while it waited in the queue wasn't really tried.
            if reason not in ('edge_removed', 'not_short', 'classification_changed'):
                topology['collapse_failures'][edge] = reason
        if not candidates or not include_new_edges:
            return
        tried.update(candidates)
        allow_repair = False


def _attempt_collapse(
    mesh,
    topology,
    edge,
    features,
    boundary,
    parameters,
    tolerances,
    *,
    boundary_only,
    iteration_index,
    allow_repair=True,
):
    """Try to collapse one short edge, using repair and nested replacement if it's blocked.

    An edge that isn't blocked only gets ordinary collapses: a rejection for numerical or
    geometric reasons never leads to repair or replacement. A blocked edge gets a repair
    attempt. If that fails and the iteration is in the late half, it gets a replacement
    attempt, and then the collapse is tried again. With allow_repair off, a blocked edge gets
    neither: repair turns it off on its private copy so it can't recurse, and the final pass
    turns it off for edges that form during it. The final pass otherwise works the same way,
    using the last iteration's index.

    Returns (changed, reason). A replacement counts as a change even if the collapse tried
    after it fails. If repair fails and replacement is refused, the reason names both.
    """
    # Earlier edits may have changed this edge since it was queued.
    if edge not in topology['edge_to_faces']:
        return False, 'edge_removed'
    if topology['lengths'][edge] > parameters['min_edge_length']:
        return False, 'not_short'
    if (edge in topology['boundary_edges']) != boundary_only:
        return False, 'classification_changed'
    classification = _classify_collapse_neighborhood(topology, edge, features, boundary)
    if not classification['eligible']:
        return False, 'protected_edge'
    ordinary = _propose_edge_collapse(topology, edge, features, boundary)
    if not ordinary:
        return False, 'protected_endpoints'

    # An edge that isn't blocked only gets ordinary collapses.
    reason = 'link_obstruction'
    if not classification['problematic']:
        for edit in ordinary:
            if not _passes_collapse_length_check(
                topology, edit, parameters, boundary_only=boundary_only
            ):
                reason = 'neighbor_length'
                continue
            valid, reason = _validate_edit(
                mesh, topology, edit, features, boundary, parameters, tolerances
            )
            if valid:
                edit['iteration_index'] = iteration_index
                _commit_edit(mesh, topology, edit, boundary)
                return True, 'ordinary_collapsed'
        # A rejection here doesn't allow repair or replacement.
        return False, reason

    # Repair a private copy of the patch, and apply the result only if it passes validation.
    if not allow_repair:
        return False, 'link_obstruction'
    edit, failure = _try_repair_collapse_patch(
        mesh,
        topology,
        edge,
        features,
        boundary,
        parameters,
        tolerances,
        boundary_only=boundary_only,
        iteration_index=iteration_index,
    )
    if edit is not None:
        valid, failure = _validate_edit(
            mesh, topology, edit, features, boundary, parameters, tolerances
        )
        if valid:
            edit['iteration_index'] = iteration_index
            _commit_edit(mesh, topology, edit, boundary)
            return True, 'repaired_collapsed'

    if not _replacement_phase_allowed(iteration_index, parameters['max_iter']):
        return False, 'deferred_after_repair_failure'

    # Late fallback: replace nested triangles, then try this edge again.
    replaced, refusal = _replace_nested_triangles(
        mesh,
        topology,
        edge,
        features,
        boundary,
        parameters,
        tolerances,
        iteration_index=iteration_index,
        repair_failure=failure or 'repair_failed',
    )
    if not replaced:
        return False, f"{failure or 'repair_failed'}, {refusal}"
    accepted, reason = _attempt_collapse(
        mesh,
        topology,
        edge,
        features,
        boundary,
        parameters,
        tolerances,
        boundary_only=boundary_only,
        iteration_index=iteration_index,
    )
    return True, reason if accepted else 'nested_triangles_replaced'


def _extract_repair_patch(mesh, topology, edge, features, boundary, parameters, *, ring_count):
    """Copy the faces around an edge into a patch with its own vertex numbering.

    ring_count is how many rings of faces around the edge's end points to include. Features,
    frozen boundaries and the patch's interface (the edges and vertices it shares with the
    rest of the mesh) are protected, so putting the patch back can't change any other face.
    A protected edge can join two patch vertices without lying on a patch face. Those are its
    absent_edges, and an edit that doesn't add them is refused.
    Returns (patch, 'ok'), or
    (None, 'repair_patch_budget') if the patch would have too many faces.
    """
    faces = set().union(*(topology['vertex_to_faces'][v] for v in edge))
    for _ in range(max(0, ring_count - 1)):
        vertices = {int(v) for f in faces for v in topology['faces'][f]}
        faces.update(f for v in vertices for f in topology['vertex_to_faces'][v])
    if len(faces) > parameters['repair_max_faces']:
        return None, 'repair_patch_budget'
    ids = sorted(faces)
    submesh, source_ids = mesh.get_sub_mesh(ids, return_node_idx_list=True)
    source_ids = [int(v) for v in source_ids]
    global_to_local = {v: i for i, v in enumerate(source_ids)}
    fixed = {v for v in source_ids if topology['vertex_to_faces'][v] - faces}
    fixed.update(features['points'] & set(source_ids))
    fixed.update(boundary['frozen']['points'] & set(source_ids))
    # Protect interface edges, not every edge that joins two fixed vertices.
    interface_edges = {
        e for e, fs in topology['edge_to_faces'].items() if fs & faces and fs - faces
    }
    protected_edges = features['edges'] | boundary['frozen']['edges'] | interface_edges
    patch_edges = {e for f in faces for e in _face_edge(topology['faces'][f])}
    local_edges, absent_edges = set(), set()
    for a, b in protected_edges:
        if a in global_to_local and b in global_to_local:
            local_edge = _edge(global_to_local[a], global_to_local[b])
            local_edges.add(local_edge)
            if (a, b) not in patch_edges:
                absent_edges.add(local_edge)
    local_points = {global_to_local[v] for v in fixed}
    local_points.update(v for e in local_edges for v in e)
    local_features = dict(
        points=local_points,
        edges=local_edges,
        positions={v: _as_numpy(submesh.node)[v].copy() for v in local_points},
        absent_edges=absent_edges,
    )
    local_boundary = dict(
        enabled=boundary['enabled'],
        frozen=dict(points=set(), edges=set(), positions={}),
        labels={
            global_to_local[v]: label
            for v, label in boundary['labels'].items()
            if v in global_to_local
        },
    )
    return (
        dict(
            mesh=submesh,
            original=_copy_source_mesh(submesh),
            face_ids=faces,
            source_ids=source_ids,
            edge=_edge(*(global_to_local[v] for v in edge)),
            original_edge=edge,
            round_positions=topology['round_positions'],
            features=local_features,
            boundary=local_boundary,
        ),
        'ok',
    )


def _repair_patch_geometry(patch, parameters, tolerances):
    """Try to clear a link obstruction by subdividing the edges to the extra neighbors.

    This works on a private copy. Each split adds a midpoint on an existing edge, so the shape
    doesn't change. Splitting stops when the edge is no longer blocked, no split is allowed, or
    a budget runs out. The caller then tries the collapse and checks the result. A junction,
    an edge where more than two sheets of surface meet, can't become a manifold edge (one with
    at most two faces) without separating or erasing a sheet, so it's a repair failure.
    Returns (repaired, reason).
    """
    mesh = _copy_source_mesh(patch['mesh'])
    topology = _initialize_topology(mesh, round_positions=patch['round_positions'])
    edge, features, boundary = patch['edge'], patch['features'], patch['boundary']
    if len(topology['edge_to_faces'][edge]) > 2:
        return None, 'repair_requires_sheet_erasure_or_disconnection'
    for attempt in range(parameters['repair_max_attempts'] + 1):
        classification = _classify_collapse_neighborhood(topology, edge, features, boundary)
        if not classification['problematic']:
            return dict(mesh=mesh, topology=topology, features=features, boundary=boundary), 'ok'
        if (
            attempt >= parameters['repair_max_new_vertices']
            or attempt == parameters['repair_max_attempts']
        ):
            break
        candidates = sorted(
            {_edge(v, w) for w in classification['extra'] for v in edge},
            key=lambda e: (len(topology['edge_to_faces'].get(e, ())), e),
        )
        changed = False
        for candidate in candidates:
            edit = _propose_edge_split(topology, candidate, features, boundary)
            if edit is None:
                continue
            valid, _ = _validate_edit(
                mesh, topology, edit, features, boundary, parameters, tolerances
            )
            if valid:
                _commit_edit(mesh, topology, edit, boundary)
                changed = True
                break
        if not changed:
            return None, 'repair_interface_or_feature_obstruction'
    return None, 'repair_attempt_budget'


def _sample_triangle(mesh):
    """Return the corners, centroids and edge midpoints of all triangles, for distance checks."""
    xyz = _as_numpy(mesh.node)[_as_numpy(mesh.element)]
    if not len(xyz):
        return np.empty((0, 3))
    return np.concatenate(
        (
            xyz.reshape(-1, 3),
            xyz.mean(axis=1),
            (xyz[:, 0] + xyz[:, 1]) / 2,
            (xyz[:, 1] + xyz[:, 2]) / 2,
            (xyz[:, 2] + xyz[:, 0]) / 2,
        )
    )


def _validate_repair_collapse(patch, repaired, parameters, tolerances):
    """Check a repaired patch after its collapse, and return (valid, reason).

    The patch must still have faces, keep its protections, be a valid manifold mesh (no edge
    with more than two faces), stay within repair_distance of the original at the sampled
    points, and keep the same number of connected components.
    """
    mesh, topology = repaired['mesh'], repaired['topology']
    if not len(mesh.element):
        return False, 'repair_erases_component'
    valid, reason = _check_feature_constraint(mesh, topology, patch['features'])
    if not valid:
        return valid, reason
    try:
        _validate_mesh(topology, tolerances, strict_manifold=True)
    except ValueError:
        return False, 'invalid_repaired_patch'
    # Check sampled distances in both directions, including points inside triangles. This
    # measures the distance approximately. It doesn't prove a bound on the Hausdorff distance
    # (the largest distance from a point on either surface to the other surface).
    for source, target in ((patch['original'], mesh), (mesh, patch['original'])):
        locator = _build_cell_locator(target)
        if any(
            _closest_point(p, locator)[1] > parameters['repair_distance']
            for p in _sample_triangle(source)
        ):
            return False, 'repair_shape_error'
    # The repair must not change the number of connected components.
    if _component_count(patch['original']) != _component_count(mesh):
        return False, 'repair_changes_components'
    return True, 'ok'


def _component_count(mesh):
    """Count the connected components: the groups of faces joined through shared vertices."""
    faces = _as_numpy(mesh.element)
    vertex_to_faces = {}
    for face_id, face in enumerate(faces):
        for vertex_id in face:
            vertex_to_faces.setdefault(int(vertex_id), []).append(face_id)
    unseen, count = set(range(len(faces))), 0
    while unseen:
        queue = [unseen.pop()]
        count += 1
        while queue:
            face_id = queue.pop()
            for vertex_id in faces[face_id]:
                for neighbor_face_id in vertex_to_faces[int(vertex_id)]:
                    if neighbor_face_id in unseen:
                        unseen.remove(neighbor_face_id)
                        queue.append(neighbor_face_id)
    return count


def _compose_repair_collapse_edit(mesh, topology, patch, repaired, features):
    """Turn an accepted repaired patch into one edit of the live mesh.

    The patch's faces replace the faces it was copied from, and its new vertices are added at
    the end. A patch vertex that no face uses any more is retired, unless it's a feature point
    or still has faces outside the patch.
    """
    local_mesh = repaired['mesh']
    original_vertex_count = len(patch['source_ids'])
    vertex_mapping = dict(enumerate(patch['source_ids']))
    new_points = _as_numpy(local_mesh.node)[original_vertex_count:]
    vertex_mapping.update(
        {original_vertex_count + i: len(mesh.node) + i for i in range(len(new_points))}
    )
    replacement_faces = [
        tuple(vertex_mapping[int(v)] for v in face) for face in _as_numpy(local_mesh.element)
    ]
    referenced = {v for face in replacement_faces for v in face}
    retired = {
        v
        for v in patch['source_ids']
        if v not in referenced
        and not (topology['vertex_to_faces'][v] - patch['face_ids'])
        and v not in features['points']
    }
    positions = {
        global_id: _as_numpy(local_mesh.node)[local_id].copy()
        for local_id, global_id in enumerate(patch['source_ids'])
        if global_id in referenced
    }
    edit = _proposal(
        topology,
        'repair_collapse',
        edge=patch['original_edge'],
        remove=patch['face_ids'],
        add=replacement_faces,
        positions=positions,
        new_points=new_points,
        retired=retired,
    )
    edit['repair_validated'] = True
    edit['boundary_labels'] = {
        vertex_mapping[v]: label for v, label in repaired['boundary']['labels'].items()
    }
    return edit


def _try_repair_collapse_patch(
    mesh,
    topology,
    edge,
    features,
    boundary,
    parameters,
    tolerances,
    *,
    boundary_only,
    iteration_index,
):
    """Try repair on a small patch, then a larger one. Return (edit, 'ok') or (None, reason).

    Each try copies a patch, splits edges to unblock the edge, collapses it and checks the
    result. reason comes from the last step that failed.
    """
    reason = 'repair_failed'
    for rings in range(1, parameters['repair_rings'] + 1):
        patch, reason = _extract_repair_patch(
            mesh, topology, edge, features, boundary, parameters, ring_count=rings
        )
        if patch is None:
            continue
        repaired, reason = _repair_patch_geometry(patch, parameters, tolerances)
        if repaired is None:
            continue
        accepted, reason = _attempt_collapse(
            repaired['mesh'],
            repaired['topology'],
            patch['edge'],
            repaired['features'],
            repaired['boundary'],
            parameters,
            tolerances,
            boundary_only=boundary_only,
            iteration_index=iteration_index,
            allow_repair=False,
        )
        if not accepted:
            continue
        valid, reason = _validate_repair_collapse(patch, repaired, parameters, tolerances)
        if valid:
            return _compose_repair_collapse_edit(mesh, topology, patch, repaired, features), 'ok'
    return None, reason


# Nested-triangle replacement
#
# A short edge (u, v) is blocked when u and v share extra neighbors w. Each one forms a
# triangle of edges (u, v, w) that is not an element, and merging u and v would flatten it.
# The nest on one side of (u, v) is the surface inside the outermost such triangle. It must
# be a disk: a patch with one boundary and no holes. Replacing the nest with the one element
# (u, v, w) clears that side of (u, v), which can leave (u, v) as an ordinary short edge. The
# boundary of the nest doesn't change, so the surface keeps its Euler value (vertices minus
# edges plus faces, which changes only if holes, boundaries or components change), boundary
# loops and connected components, and can't be emptied.


def _edge_direction(face, edge):
    """Return 1 if the triangle runs along the edge from edge[0] to edge[1], otherwise -1."""
    face = tuple(face)
    i = face.index(edge[0])
    return 1 if face[(i + 1) % 3] == edge[1] else -1


def _certify_disk(topology, selected_faces, barriers):
    """Check that the selected faces form a disk bounded exactly by the barrier edges.

    The faces must be consistently oriented, have Euler value 1 (vertices minus edges plus
    faces), and have a valid link at every vertex. The link of a vertex joins the other two
    corners of each of its faces. At an interior vertex the link must be a closed loop, and at
    a boundary vertex it must be a path between its two boundary neighbors. This is a
    topological certificate: it checks connectivity only, not self-intersection (whether the
    surface passes through itself). Returns (disk, reason).
    """
    edge_to_faces, vertex_links = {}, {}
    for face_id in selected_faces:
        face = tuple(topology['faces'][face_id])
        for edge in _face_edge(face):
            edge_to_faces.setdefault(edge, []).append(face_id)
        for i, vertex in enumerate(face):
            a, b = face[(i + 1) % 3], face[(i + 2) % 3]
            link = vertex_links.setdefault(vertex, {})
            link.setdefault(a, set()).add(b)
            link.setdefault(b, set()).add(a)
    disk_boundary = {edge for edge, ids in edge_to_faces.items() if len(ids) == 1}
    if disk_boundary != barriers or any(len(ids) > 2 for ids in edge_to_faces.values()):
        return None, 'disk_boundary_mismatch'
    if len(vertex_links) - len(edge_to_faces) + len(selected_faces) != 1:
        return None, 'region_is_not_a_disk'
    for edge, ids in edge_to_faces.items():
        if len(ids) == 2 and (
            _edge_direction(topology['faces'][ids[0]], edge)
            == _edge_direction(topology['faces'][ids[1]], edge)
        ):
            return None, 'inconsistent_disk_orientation'
    boundary_links = {}
    for a, b in barriers:
        boundary_links.setdefault(a, set()).add(b)
        boundary_links.setdefault(b, set()).add(a)
    for vertex, link in vertex_links.items():
        unseen = set(link)
        pending = [unseen.pop()]
        while pending:
            for neighbor in link[pending.pop()]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    pending.append(neighbor)
        if unseen:
            return None, 'disconnected_vertex_link'
        link_endpoints = {v for v, neighbors in link.items() if len(neighbors) == 1}
        if any(
            len(neighbors) not in (1, 2) for neighbors in link.values()
        ) or link_endpoints != boundary_links.get(vertex, set()):
            return None, 'invalid_vertex_link'
    return dict(faces=selected_faces, vertices=set(vertex_links), edges=set(edge_to_faces)), 'ok'


def _distance_to_triangle(point, a, b, c):
    """Return the distance from a point to a triangle. A degenerate triangle uses its edges."""
    normal = np.cross(b - a, c - a)
    squared = float(normal @ normal)
    if squared > 0:
        s = float(np.cross(b - a, point - a) @ normal) / squared
        t = float(np.cross(point - a, c - a) @ normal) / squared
        if s >= 0 and t >= 0 and s + t <= 1:
            return abs(float((point - a) @ normal)) / math.sqrt(squared)
    best = math.inf
    for start, end in ((a, b), (b, c), (c, a)):
        direction = end - start
        length_squared = max(float(direction @ direction), 1e-300)
        fraction = float(np.clip((point - start) @ direction / length_squared, 0, 1))
        best = min(best, float(np.linalg.norm(point - (start + fraction * direction))))
    return best


def _search_nest_side(topology, edge, anchor_face_id, w, max_faces):
    """Search the side of the triangle of edges (u, v, w) that holds this element.

    The search starts at anchor_face_id and spreads across every edge except the triangle's
    three. Returns (disk, status). status is 'ok' if that side is a disk, 'not_disk' if it
    isn't, or 'over_budget' if the search stopped at max_faces and the answer is unknown.
    """
    barriers = set(_face_edge((*edge, w)))
    selected, queue = {anchor_face_id}, deque([anchor_face_id])
    while queue:
        for face_edge in _face_edge(topology['faces'][queue.popleft()]):
            if face_edge in barriers:
                continue
            for other in topology['edge_to_faces'][face_edge]:
                if other not in selected:
                    selected.add(other)
                    if len(selected) > max_faces:
                        return None, 'over_budget'
                    queue.append(other)
    disk = _certify_disk(topology, selected, barriers)[0]
    return disk, 'ok' if disk is not None else 'not_disk'


def _largest_nest(topology, edge, anchor, common, incident, max_faces):
    """Return (w, disk, other_side_disk) for the largest nest on the anchor's side.

    Returns None if there is no nest, or 'over_budget' if a search hit the face budget and nothing
    on this side can be used.

    On a closed surface, a triangle of edges has a disk on both sides, so a side counts as
    the nest only if the other side isn't known to be a disk or is at least as large. A side
    with one element counts as size 1, and has nothing inside it to replace.

    If a search hits the face budget, the answer is unknown. Unless the other side is a
    known smaller nest, nothing on this side is used, because a smaller disk could hide a
    vertex of a larger one.
    """

    def opposite(face_id):
        return next(int(x) for x in topology['faces'][face_id] if x not in edge)

    best = None
    for w in sorted(common - {opposite(anchor)}):
        disk, status = _search_nest_side(topology, edge, anchor, w, max_faces)
        other_disk, other_size = None, None
        for other in incident - {anchor}:
            if opposite(other) == w:
                other_size = 1
            else:
                other_disk, _ = _search_nest_side(topology, edge, other, w, max_faces)
                other_size = None if other_disk is None else len(other_disk['faces'])
        if status == 'over_budget':
            if other_size is None:
                return 'over_budget'
            continue
        if disk is None or (other_size is not None and len(disk['faces']) > other_size):
            continue
        if best is None or len(disk['faces']) > len(best[1]['faces']):
            best = (w, disk, other_disk)
    return best


def _element_crosses_surface(topology, element, removed_faces):
    """Return whether a new element would pass through a remaining face it shares no vertex with.

    This doesn't detect overlap within one plane, or faces that share a vertex with the element.
    """
    positions = topology['vertex_positions']
    faces = np.asarray(topology['faces'])
    triangle = positions[list(element)]
    corners = positions[faces]
    near = np.all(corners.min(axis=1) <= triangle.max(axis=0), axis=1) & np.all(
        corners.max(axis=1) >= triangle.min(axis=0), axis=1
    )

    def segment_hits(start, end, tri):
        direction, e1, e2 = end - start, tri[1] - tri[0], tri[2] - tri[0]
        h = np.cross(direction, e2)
        det = float(e1 @ h)
        if abs(det) <= 1e-14 * max(float(e1 @ e1), float(e2 @ e2), 1e-300):
            return False
        from_corner, q = start - tri[0], np.cross(start - tri[0], e1)
        a, b, t = float(from_corner @ h) / det, float(direction @ q) / det, float(e2 @ q) / det
        return a >= 0 and b >= 0 and a + b <= 1 and 0 < t < 1

    for face_id in np.flatnonzero(near):
        if int(face_id) in removed_faces or set(map(int, faces[face_id])) & set(element):
            continue
        other = corners[face_id]
        for first, second in ((triangle, other), (other, triangle)):
            for i in range(3):
                if segment_hits(first[i], first[(i + 1) % 3], second):
                    return True
    return False


def _propose_nested_replacement(
    topology, edge, anchor, w, disk, *, iteration_index, repair_failure
):
    """Propose the one element (u, v, w), oriented like the element it replaces on (u, v)."""
    edit = _proposal(
        topology,
        'nested_replacement',
        edge=edge,
        remove=disk['faces'],
        add=[tuple(int(x) if x in edge else w for x in topology['faces'][anchor])],
        retired={int(x) for x in disk['vertices']} - {*edge, w},
        normal_refs=[topology['normals'][anchor]],
    )
    edit.update(outer_vertex=w, iteration_index=iteration_index, repair_failure=repair_failure)
    return edit


def _replace_nested_triangles(
    mesh,
    topology,
    edge,
    features,
    boundary,
    parameters,
    tolerances,
    *,
    iteration_index,
    repair_failure,
):
    """Replace the nest on each side of (u, v) with one element, where that is allowed.

    Each nest is replaced whole or not at all, and this stops once the edge isn't blocked.
    If every inner vertex is within repair_distance of (u, v, w), the whole nest is too,
    because the distance to a triangle is largest at a corner of a flat face. The element is
    also that close to the nest, because the nest covers it when projected onto its plane.
    This limits each replacement, not the total over several edits.

    Only the largest nest is tried, because replacing a smaller one could remove the vertex
    that keeps the larger one over the limit. If the other side is also within the limit,
    the two sides would lie on top of each other, so a closed shape that thin is left alone.
    An edge with more than two faces is left alone too. Returns the number of replacements
    and the reason for the last refusal, which is 'no_nest' if no side had a nest to try, or
    'nest_over_budget' if a search for one went over the face budget.
    """
    replaced, limit, refusal = 0, parameters['repair_distance'], 'no_nest'
    while True:
        current = _classify_collapse_neighborhood(topology, edge, features, boundary)
        if not current['problematic']:
            return replaced, refusal
        if len(current['incident']) > 2:
            return replaced, 'edge_has_more_than_two_faces'
        progress = False
        positions = topology['vertex_positions']
        for anchor in sorted(current['incident']):
            nest = _largest_nest(
                topology,
                edge,
                anchor,
                current['common'],
                current['incident'],
                parameters['repair_max_faces'],
            )
            if nest == 'over_budget':
                refusal = 'nest_over_budget'
                continue
            if nest is None:
                continue
            w, disk, other_disk = nest
            triangle = positions[edge[0]], positions[edge[1]], positions[w]

            def within_limit(vertices):
                return all(
                    _distance_to_triangle(positions[x], *triangle) <= limit for x in vertices
                )

            edit = _propose_nested_replacement(
                topology,
                edge,
                anchor,
                w,
                disk,
                iteration_index=iteration_index,
                repair_failure=repair_failure,
            )
            if not within_limit(edit['retired']):
                refusal = 'nest_over_limit'
                continue
            if other_disk is not None and within_limit(other_disk['vertices']):
                refusal = 'other_side_within_limit'
                continue
            if _element_crosses_surface(topology, (*edge, w), disk['faces']):
                refusal = 'element_crosses_surface'
                continue
            valid, reason = _validate_edit(
                mesh, topology, edit, features, boundary, parameters, tolerances
            )
            if valid:
                _commit_edit(mesh, topology, edit, boundary)
                replaced += 1
                progress = True
                break
            refusal = f'replacement_{reason}'
        if not progress:
            return replaced, refusal


# Mesh quality and relaxation stages


def _equalize_valence(mesh, topology, features, boundary, parameters, tolerances):
    """Flip edges to bring each vertex closer to its target valence.

    The valence of a vertex is the number of edges at it. The target is 6 inside the surface
    and 4 on the boundary. A flip is kept only if it lowers the total deviation from the target
    for the four vertices involved and passes validation.
    """
    for edge in _collect_candidate_edge(
        topology, parameters, boundary_only=False, operation='flip'
    ):
        edit = _propose_edge_flip(topology, edge, features)
        if edit is None:
            continue
        opposite = sorted({v for f in edit['add_faces'] for v in f} - set(edge))
        vertices = list(edge) + opposite
        vertex_degrees = np.array([len(topology['vertex_neighbors'][v]) for v in vertices])
        target_degrees = np.array([4 if topology['boundary_neighbors'][v] else 6 for v in vertices])
        if (
            np.abs(vertex_degrees + [-1, -1, 1, 1] - target_degrees).sum()
            >= np.abs(vertex_degrees - target_degrees).sum()
        ):
            continue
        valid, _ = _validate_edit(mesh, topology, edit, features, boundary, parameters, tolerances)
        if valid:
            _commit_edit(mesh, topology, edit, boundary)


def _improve_triangle_angle(mesh, topology, features, boundary, parameters, tolerances):
    """Flip the edge opposite any angle over 120 degrees, if that reduces the largest angle."""
    # Queue edges, not face IDs, because face IDs can change as edits are applied.
    candidates = set()
    for face_id, angles in topology['angles'].items():
        if angles.max() > 2 * math.pi / 3:
            corner_index = int(angles.argmax())
            triangle = topology['faces'][face_id]
            candidates.add(
                _edge(triangle[(corner_index + 1) % 3], triangle[(corner_index + 2) % 3])
            )
    for edge in sorted(candidates):
        edit = _propose_edge_flip(topology, edge, features)
        if edit is None:
            continue
        old_max_angle = max(
            float(topology['angles'][face_id].max()) for face_id in edit['remove_faces']
        )
        _, _, angles = _geometry(_candidate_point(topology, edit, edit['add_faces']))
        if angles.max() >= old_max_angle - 1e-12:
            continue
        valid, _ = _validate_edit(mesh, topology, edit, features, boundary, parameters, tolerances)
        if valid:
            _commit_edit(mesh, topology, edit, boundary)


COVERAGE_SAMPLES_PER_TARGET_AREA = 100
TANGENTIAL_FINAL_ITERATIONS = 2
TANGENTIAL_MIN_NORMAL_AGREEMENT = 0.8


def coverage_iteration_count(max_iter):
    """Return how many iterations use coverage-based smoothing, at least one."""
    return max(1, max_iter - TANGENTIAL_FINAL_ITERATIONS)


def coverage_samples(reference, parameters):
    """Sample the source surface once, evenly by area."""
    if 'coverage_samples' not in reference:
        node = _as_numpy(reference['mesh'].node).reshape(-1, 3)
        element = _as_numpy(reference['mesh'].element).reshape(-1, 3)
        cross = np.cross(
            node[element[:, 1]] - node[element[:, 0]], node[element[:, 2]] - node[element[:, 0]]
        )
        double_area = np.linalg.norm(cross, axis=1)
        target = parameters['target_edge_length']
        count = COVERAGE_SAMPLES_PER_TARGET_AREA * double_area.sum() / 2 / target**2
        count = int(np.clip(count, 2000, 200000))
        generator = np.random.default_rng(0)
        chosen = generator.choice(len(element), count, p=double_area / double_area.sum())
        random = generator.random((count, 2))
        root = np.sqrt(random[:, 0])
        reference['coverage_samples'] = (
            (1 - root)[:, None] * node[element[chosen, 0]]
            + (root * (1 - random[:, 1]))[:, None] * node[element[chosen, 1]]
            + (root * random[:, 1])[:, None] * node[element[chosen, 2]]
        )
    return reference['coverage_samples']


def coverage_smoothing_positions(mesh, topology, reference, parameters):
    """Return where coverage-based smoothing (Lloyd relaxation) moves each vertex that owns samples.

    The source surface is divided among the vertices by giving each sample to its closest vertex,
    and each vertex moves to the centre of its samples. Unlike an average of neighbors, this
    measures how much of the source each vertex represents, so vertices can't cluster or retreat
    from a tip without another vertex's share growing and pulling them back.
    """
    from scipy.spatial import cKDTree

    points = coverage_samples(reference, parameters)
    active = np.array([v for v in range(len(mesh.node)) if topology['vertex_to_faces'][v]])
    _, owner = cKDTree(topology['vertex_positions'][active]).query(points)
    counts = np.bincount(owner, minlength=len(active))
    centers = np.stack(
        [np.bincount(owner, weights=points[:, axis], minlength=len(active)) for axis in range(3)],
        axis=1,
    )
    return {int(active[i]): centers[i] / counts[i] for i in range(len(active)) if counts[i]}


def tangential_smoothing_position(topology, vertex_id, neighbors):
    """Return an interior vertex's smoothed position, or None if it shouldn't slide.

    Tangential smoothing starts from the average of the neighbors and removes the part of the
    move along the vertex normal, so the vertex slides within its tangent plane. The vertex normal
    is the sum of the area-weighted normals of the vertex's faces. If they agree less than
    TANGENTIAL_MIN_NORMAL_AGREEMENT, the vertex is on a sharp crease or corner, or has no tangent
    plane because they cancel, and it stays where it is.
    """
    position = topology['vertex_positions'][vertex_id]
    average = topology['vertex_positions'][neighbors].mean(axis=0)
    face_ids = topology['vertex_to_faces'][vertex_id]
    area_normal = sum(topology['areas'][face_id] * topology['normals'][face_id] for face_id in face_ids)
    length = float(np.linalg.norm(area_normal))
    if length < TANGENTIAL_MIN_NORMAL_AGREEMENT * sum(topology['areas'][f] for f in face_ids):
        return None
    normal = area_normal / length
    return average - normal * float((average - position) @ normal)


def _relax_vertex(
    mesh,
    topology,
    features,
    boundary,
    reference,
    parameters,
    tolerances,
    *,
    boundary_only,
    use_coverage=False,
):
    """Smooth the interior vertices, or the boundary vertices if boundary_only is set.

    By default, an interior vertex moves to the average of its neighbors. With
    flag_use_coverage_then_tangential, it moves by coverage-based smoothing if use_coverage is
    set and it owns samples, and otherwise by tangential smoothing. A boundary vertex moves to the
    average of itself and its two boundary neighbors. The new position is then projected, if
    projection is on. Protected vertices don't move, and each move must pass validation.
    """
    if boundary_only and not parameters['flag_process_boundary']:
        return
    centers = {}
    if use_coverage and not boundary_only:
        centers = coverage_smoothing_positions(mesh, topology, reference, parameters)
    for vertex_id in range(len(mesh.node)):
        if (
            vertex_id in features['points']
            or not topology['vertex_to_faces'][vertex_id]
            or bool(topology['boundary_neighbors'][vertex_id]) != boundary_only
        ):
            continue
        if boundary_only:
            neighbors = sorted(topology['boundary_neighbors'][vertex_id])
            if len(neighbors) != 2:
                continue
            # A vertex with no boundary label has no original boundary curve to project onto, so
            # don't move it onto an unrelated one. Without projection, plain smoothing is fine.
            if parameters['flag_project_to_source_mesh'] and vertex_id not in boundary['labels']:
                continue
            candidate = (
                topology['vertex_positions'][vertex_id]
                + topology['vertex_positions'][neighbors].sum(axis=0)
            ) / 3
        else:
            neighbors = sorted(topology['vertex_neighbors'][vertex_id])
            if len(neighbors) < 3:
                continue
            # Leave a vertex alone if any of its edges has more than two faces.
            if any(
                len(topology['edge_to_faces'][_edge(vertex_id, neighbor_id)]) > 2
                for neighbor_id in neighbors
            ):
                continue
            if parameters['flag_use_coverage_then_tangential']:
                candidate = centers.get(vertex_id)
                if candidate is None:
                    candidate = tangential_smoothing_position(topology, vertex_id, neighbors)
                if candidate is None:
                    continue
            else:
                candidate = topology['vertex_positions'][neighbors].mean(axis=0)
        point = _propose_vertex_move(
            vertex_id, candidate, features, boundary, reference, parameters
        )
        if (
            point is None
            or np.linalg.norm(point - topology['vertex_positions'][vertex_id])
            <= tolerances['length']
        ):
            continue
        edit = _proposal(topology, 'move', positions={vertex_id: point})
        valid, _ = _validate_edit(mesh, topology, edit, features, boundary, parameters, tolerances)
        if valid:
            _commit_edit(mesh, topology, edit, boundary)


# Output validation and compaction


def _validate_result(
    mesh, topology, reference, parameters, features, boundary, tolerances, *, old_to_new_vertex=None
):
    """Check the finished mesh, and raise an error if it fails.

    It must be a valid mesh, keep its protected features and boundaries, have no replacement
    outside the rules, and still have faces if the source had any.
    """
    _validate_mesh(topology, tolerances, allow_empty=True)
    for locks in (features, boundary['frozen']):
        valid, reason = _check_feature_constraint(
            mesh, topology, locks, old_to_new_vertex=old_to_new_vertex
        )
        if not valid:
            raise RuntimeError(f'output breaks a protected feature or boundary: {reason}')
    for record in topology['replacements']:
        if not record['repair_failure'] or not _replacement_phase_allowed(
            record['iteration'] - 1, parameters['max_iter']
        ):
            raise RuntimeError(
                'a recorded replacement has no failed repair or is outside the late iterations'
            )
    if not len(mesh.element) and reference['summary']['faces']:
        raise RuntimeError('all source faces disappeared')


def _compact_mesh(mesh, features):
    """Remove unused vertices and renumber the rest, keeping protected points that have no faces.

    Returns (output, old_to_new_vertex).
    """
    if len(mesh.element):
        output, retained_vertex_ids = mesh.get_sub_mesh(
            list(range(len(mesh.element))), return_node_idx_list=True
        )
        retained_vertex_ids = [int(vertex_id) for vertex_id in retained_vertex_ids]
        output.node = output.node.clone()
        output.element = output.element.clone()
    else:
        output, retained_vertex_ids = _new_mesh([], []), []
    old_to_new_vertex = {
        vertex_id: output_vertex_id
        for output_vertex_id, vertex_id in enumerate(retained_vertex_ids)
    }
    isolated_protected_vertices = sorted(features['points'] - set(retained_vertex_ids))
    if isolated_protected_vertices:
        output.node = torch.cat((output.node, mesh.node[isolated_protected_vertices].clone()))
        old_to_new_vertex.update(
            {
                vertex_id: len(retained_vertex_ids) + output_vertex_id
                for output_vertex_id, vertex_id in enumerate(isolated_protected_vertices)
            }
        )
    output.clear_adj_info()
    return output, old_to_new_vertex


def _finalize_output(
    mesh, topology, source_mesh, reference, parameters, features, boundary, tolerances
):
    """Check the result, renumber it, restore the source data type and device, and check again."""
    _validate_result(mesh, topology, reference, parameters, features, boundary, tolerances)
    protected = dict(features)
    protected['points'] = features['points'] | boundary['frozen']['points']
    output, mapping = _compact_mesh(mesh, protected)
    output.node = output.node.to(
        dtype=source_mesh.node.dtype, device=source_mesh.node.device
    ).clone()
    output.element = output.element.to(dtype=torch.int64, device=source_mesh.node.device).clone()
    # Check again after the conversion: rounding to float32 can create degenerate triangles.
    check = _copy_source_mesh(output)
    checked = _initialize_topology(check)
    checked['replacements'] = topology['replacements']
    _validate_result(
        check,
        checked,
        reference,
        parameters,
        features,
        boundary,
        tolerances,
        old_to_new_vertex=mapping,
    )
    output.clear_adj_info()
    return output
