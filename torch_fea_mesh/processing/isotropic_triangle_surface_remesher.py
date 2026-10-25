"""Isotropic triangle remeshing using torch_fea_mesh, NumPy, PyTorch and VTK.

All mutable state belongs to a call. Source geometry and feature lists are never
modified. Working coordinates are CPU float64; output restores the source dtype
and device. Functions are grouped in execution order: input preparation, working
topology, reference geometry, iteration stages, local edits, collapse repair,
enclosure certification, and output validation.

The mesh library supplies containers, initial adjacency, geometry primitives,
and I/O. This remesher owns dynamic adjacency updates because its edit ordering,
stable vertex IDs, face-row swaps, and temporary repair patches are specific to
this algorithm. The working ``topology`` dictionary is documented at its factory,
_initialize_topology(). Mesh tensors share its active NumPy arrays; library
adjacency fields are cleared rather than left stale during local edits.

Problematic collapses try bounded geometry-preserving local repair first. Failed
attempts are deferred for ceil(max_iter/2) iterations. Only in the remaining
iterations may a verified enclosed region be erased. Features always take priority.
See doc/remeshing.md for policies and geometric limitations.
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
    flag_process_boundary: bool = True,
    *,
    feature_point_index_list: list[int] | None = None,
    feature_edge_index_list: list[int] | None = None,
    feature_curve_list: list[list[int]] | None = None,
    target_edge_length_ratio_max: float = 1.4,
    target_edge_length_ratio_min: float = 0.7,
    max_iter: int = 10,
    flag_project_to_source_mesh: bool = True,
) -> TriangleMesh:
    """Return a fresh remesh, honoring all ten C++ input fields.

    target_edge_length must be finite and at least 1e-5. For smaller targets,
    rescale source_mesh.node and target_edge_length by the same factor.

    Feature edges use ORIGINAL source edge-table numbering. Curves specify
    consecutive source point pairs; repeat the first point to close a curve.
    False boundary processing freezes boundary geometry. False projection skips
    closest-point repositioning, including in repairs. As in C++, projection is
    applied to relaxation, not ordinary split/collapse placement.

    The original three positional arguments remain supported. Diagnostics are
    returned in output.mesh_data['remeshing_report']; no source fields are copied.
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
    )
    tolerances = _compute_tolerance(mesh, target_edge_length)
    topology = _initialize_topology(mesh)
    features = _initialize_feature_constraint(
        source_mesh,
        mesh,
        topology,
        feature_point_index_list,
        feature_edge_index_list,
        feature_curve_list,
    )
    reference, boundary = _prepare_initial_mesh(mesh, topology, features, parameters, tolerances)
    collapse_history = {}
    _run_remeshing_iteration(
        mesh, topology, features, boundary, reference, parameters, tolerances, collapse_history
    )
    output = _finalize_output(
        mesh, topology, source_mesh, reference, parameters, features, boundary, tolerances
    )
    output.mesh_data['remeshing_report'] = {
        'parameters': dict(parameters),
        'operations': dict(topology['counts']),
        'collapse_outcomes': dict(topology['outcomes']),
        'iterations': topology['iterations'],
        'region_deletions': topology['deletions'],
        'input_nodes': len(source_mesh.node),
        'input_faces': len(source_mesh.element),
        'output_nodes': len(output.node),
        'output_faces': len(output.element),
        'elapsed_seconds': time.perf_counter() - started,
    }
    return output


# Input validation and per-call configuration


def _as_numpy(value):
    """Expose a CPU NumPy array; callers must copy it before independent mutation."""
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _new_mesh(node, element):
    """Construct a working mesh, preserving shaped arrays for empty connectivity."""
    node = torch.as_tensor(node, dtype=torch.float64).reshape(-1, 3)
    element = torch.as_tensor(element, dtype=torch.int64).reshape(-1, 3)
    if len(element):
        return TriangleMesh(node, element)
    mesh = TriangleMesh()
    mesh.node, mesh.element = node, element
    return mesh


def _edge(a, b):
    """Return the canonical endpoint pair used as a stable undirected edge key."""
    return (int(a), int(b)) if a < b else (int(b), int(a))


def _face_edge(face):
    """Return the three canonical edges of an oriented triangle."""
    a, b, c = face
    return (_edge(a, b), _edge(b, c), _edge(c, a))


def _index_list(values, size, name):
    """Validate a one-dimensional source-index sequence while preserving its order."""
    if values is None:
        return []
    if torch.is_tensor(values):
        if values.ndim != 1 or values.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
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
    """Read original edge numbering, building a private table only when absent."""
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
        raise ValueError('source edge table is stale or inconsistent with its faces')
    return edges


def _validate_input(
    source_mesh,
    target_edge_length,
    flag_process_boundary=True,
    *,
    feature_point_index_list=None,
    feature_edge_index_list=None,
    feature_curve_list=None,
    target_edge_length_ratio_max=1.4,
    target_edge_length_ratio_min=0.7,
    max_iter=10,
    flag_project_to_source_mesh=True,
):
    """Validate public arguments and source arrays without modifying the source."""
    if not isinstance(source_mesh, TriangleMesh):
        raise TypeError('source_mesh must be your mesh library TriangleMesh')
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
        raise ValueError('edge-length thresholds overflow')
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, Integral) or max_iter < 0:
        raise ValueError('max_iter must be a nonnegative integer')
    if not isinstance(flag_process_boundary, bool) or not isinstance(
        flag_project_to_source_mesh, bool
    ):
        raise TypeError('boundary and projection flags must be bool')
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
    """Copy geometry/connectivity into independent CPU float64/int64 storage."""
    node = source_mesh.make_node_copy('torch', dtype=torch.float64).detach().cpu().clone()
    element = torch.as_tensor(source_mesh.make_element_copy('numpy'), dtype=torch.int64).clone()
    return _new_mesh(node, element)


def _make_parameter(
    target_edge_length,
    target_edge_length_ratio_max,
    target_edge_length_ratio_min,
    max_iter,
    flag_process_boundary,
    flag_project_to_source_mesh,
):
    """Normalize scalar inputs and derive edge thresholds and local repair budgets."""
    return dict(
        target_edge_length=float(target_edge_length),
        target_edge_length_ratio_max=float(target_edge_length_ratio_max),
        target_edge_length_ratio_min=float(target_edge_length_ratio_min),
        min_edge_length=float(target_edge_length * target_edge_length_ratio_min),
        max_edge_length=float(target_edge_length * target_edge_length_ratio_max),
        max_iter=int(max_iter),
        flag_process_boundary=flag_process_boundary,
        flag_project_to_source_mesh=flag_project_to_source_mesh,
        early_iteration_count=(int(max_iter) + 1) // 2,
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
    """Build protected point/edge sets and snapshots of fixed coordinates.

    Explicit feature edges and curve edges also protect their endpoints. The
    returned positions dictionary stores original coordinates by vertex ID.
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
    """Collect existing edges between consecutive source IDs in each curve."""
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
    """Return (valid, reason) for fixed points and surviving protected edges.

    An edit is checked prospectively, without mutation. old_to_new_vertex is used
    only when checking the compacted output against original protected IDs.
    """
    mapping = old_to_new_vertex
    for v, p in features['positions'].items():
        w = mapping.get(v, -1) if mapping is not None else v
        if w < 0 or w >= len(mesh.node):
            return False, 'lost_feature_point'
        q = (
            edit.get('positions', {}).get(w, topology['vertex_positions'][w])
            if edit
            else topology['vertex_positions'][w]
        )
        if not np.array_equal(q, p) or (edit and w in edit.get('retired', set())):
            return False, 'modified_feature_point'
    for e in features['edges']:
        key = _edge(*(mapping.get(v, -1) for v in e)) if mapping is not None else e
        incidence = topology['edge_to_faces'].get(key, set())
        if edit:
            count = len(incidence - edit['remove_faces'])
            count += sum(key in _face_edge(f) for f in edit['add_faces'])
        else:
            count = len(incidence)
        if not count:
            return False, 'lost_feature_edge'
    return True, 'ok'


def _deletion_phase_allowed(iteration_index, max_iter):
    """Allow deletion only in the late half of actual, zero-based iterations."""
    return 0 <= iteration_index < max_iter and iteration_index >= (max_iter + 1) // 2


def _compute_tolerance(source_mesh, target_edge_length):
    """Derive geometric acceptance tolerances from source extent and target length."""
    positions = _as_numpy(source_mesh.node)
    extent = (
        float(np.linalg.norm(np.ptp(positions, axis=0))) if len(positions) else target_edge_length
    )
    scale = max(extent, float(target_edge_length), np.finfo(float).tiny ** 0.25)
    return {
        'length': scale * 1e-12,
        'area': scale * scale * 1e-14,
        'plane': scale * 1e-8,
        'barycentric': 1e-9,
        'normal_dot': 1e-6,
    }


# Working topology and algorithm-specific adjacency updates


def _initialize_topology(mesh):
    """Build the per-call working topology from the mesh library's adjacency.

    Storage:
        vertex_positions / faces: active N x 3 float64 / M x 3 int64 arrays.
        vertex_buffer / face_buffer: backing arrays with spare capacity. Growing
            a buffer requires refreshing the active view and mesh tensor views.

    Connectivity:
        edge_to_faces: canonical (min_vertex, max_vertex) -> incident face IDs.
        vertex_to_faces / vertex_neighbors: lists of sets indexed by vertex ID.
        boundary_edges / boundary_neighbors: edges with exactly one incident
            face and their vertex-neighbor sets.
        face_keys: sorted vertex triples -> face IDs, for duplicate detection.

    Geometry and bookkeeping:
        lengths: edge-keyed lengths; areas / normals / angles: face-ID-keyed
            measurements. dirty_edges / dirty_faces identify values to refresh.
        revision: incremented after each committed edit; proposals must match it.
        counts / outcomes / iterations: operation, rejection, and timing records.
        ledger / deletions: accepted repair and enclosed-region deletion records.

    Vertex IDs remain stable until final compaction. Face IDs can change when a
    deleted row is replaced by the last row. All incidences and measurements for
    that moved face must be remapped before another edit is considered.
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
        ledger=[],
        deletions=[],
        counts=Counter(),
        outcomes=Counter(),
        iterations=[],
    )
    for edge, incident_faces in edge_to_faces.items():
        if len(incident_faces) == 1:
            topology['boundary_edges'].add(edge)
            a, b = edge
            topology['boundary_neighbors'][a].add(b)
            topology['boundary_neighbors'][b].add(a)
    _set_mesh_view(mesh, topology)
    mesh.clear_adj_info()  # Never expose stale dense library edge rows.
    _refresh_geometry(mesh, topology, full=True)
    return topology


def _set_mesh_view(mesh, topology):
    """Refresh mesh tensors sharing the active topology arrays, without copying."""
    mesh.node = torch.from_numpy(topology['vertex_positions'])
    mesh.element = torch.from_numpy(topology['faces'])


def _attach_face(topology, face_id, face, remove=False):
    """Attach or detach a face's incidences; mark its edge geometry as dirty."""
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
    """Apply face removals/additions and repair only affected adjacency entries.

    Remove in descending ID order so each requested ID still names its original
    face. Fill each hole with the last active face, updating all references to its
    former row. Append replacement faces, then refresh affected boundary links.
    Geometry is refreshed by _commit_edit() after this function returns.
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
    """Reuse library geometry primitives on only dirty gathered faces and edges."""
    face_ids = sorted(range(len(topology['faces'])) if full else topology['dirty_faces'])
    if face_ids:
        connectivity = torch.from_numpy(topology['faces'][face_ids])
        areas, normals = TriangleMesh.cal_element_area_and_normal(mesh.node, connectivity)
        angles = TriangleMesh.cal_element_corner_angle(mesh.node, connectivity)
        # Library normalization may clamp small denominators. The orientation
        # tests below require unit directions for every nonzero normal.
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
        lengths = Mesh.cal_edge_length(mesh.node, torch.tensor(active_edges)).numpy().ravel()
        topology['lengths'].update(zip(active_edges, map(float, lengths)))
    for edge in edges:
        if edge not in topology['edge_to_faces']:
            topology['lengths'].pop(edge, None)
    topology['dirty_faces'].clear()
    topology['dirty_edges'].clear()


def _check_adjacency_consistency(mesh, topology):
    """Compare local adjacency and measurements against an independent rebuild."""
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
            raise RuntimeError(f'incremental {name} differs from independent library rebuild')
    if set(topology['lengths']) != set(rebuilt['lengths']):
        raise RuntimeError('stale edge measurements')
    for e, length in rebuilt['lengths'].items():
        if not np.isclose(length, topology['lengths'][e], rtol=1e-12, atol=1e-12):
            raise RuntimeError('stale edge length')
    for f, area in rebuilt['areas'].items():
        if not np.isclose(area, topology['areas'][f], rtol=1e-12, atol=1e-12):
            raise RuntimeError('stale face geometry')


def _validate_mesh(mesh, topology, tolerances, *, strict_manifold=False, allow_empty=False):
    """Reject invalid retained geometry and return a topology-size summary."""
    positions, faces = topology['vertex_positions'], topology['faces']
    if not len(faces) and not allow_empty:
        raise ValueError('empty mesh')
    if not np.isfinite(positions).all() or (
        len(faces) and (faces.min() < 0 or faces.max() >= len(positions))
    ):
        raise ValueError('invalid geometry/connectivity')
    if len(topology['face_keys']) != len(faces):
        raise ValueError('duplicate triangle faces')
    if any(
        area <= tolerances['area'] or not math.isfinite(area) for area in topology['areas'].values()
    ):
        raise ValueError('degenerate source or retained triangle')
    high_incidence_edge_count = sum(
        len(incident_faces) > 2 for incident_faces in topology['edge_to_faces'].values()
    )
    if strict_manifold and high_incidence_edge_count:
        raise ValueError('nonmanifold edge incidence')
    return dict(
        nodes=len(positions),
        faces=len(faces),
        edges=len(topology['edge_to_faces']),
        high_incidence_edges=high_incidence_edge_count,
        boundary_edges=len(topology['boundary_edges']),
    )


# Original geometry, boundary constraints, and projection


def _prepare_initial_mesh(mesh, topology, features, parameters, tolerances):
    """Validate the working mesh and snapshot its projection and boundary reference."""
    summary = _validate_mesh(mesh, topology, tolerances, allow_empty=True)
    reference = _build_reference(mesh, topology, summary, features, parameters)
    return reference, _initialize_boundary_constraint(reference, features, parameters)


def _trace_boundary_component(mesh, topology, *, allow_unresolved=False):
    """Trace current endpoint maps; library traversal would rebuild stale edge rows."""
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
            raise ValueError('unresolved branched/open boundary')
        components.append((tuple(sorted(nodes)), tuple(sorted(edges)), regular))
    return components


def _build_reference(mesh, topology, topology_summary, features, parameters):
    """Snapshot original geometry and boundary components for this remeshing call.

    Surface/boundary VTK locators are built lazily. Boundary labels map vertex IDs
    to original component IDs; boundary_positions retains original coordinates.
    """
    original = _copy_source_mesh(mesh)
    components = _trace_boundary_component(mesh, topology, allow_unresolved=True)
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
    """The library wrapper rebuilds a locator per call; retain one here instead."""
    if not len(reference_mesh.element):
        raise ValueError('cannot construct a locator on empty geometry')
    data = reference_mesh.convert_to_vtk()
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(data)
    locator.BuildLocator()
    return {'data': data, 'locator': locator}


def _initialize_boundary_constraint(reference, features, parameters):
    """Track original boundary labels and optional fixed-point/edge constraints.

    labels can grow when boundary edges are split. frozen uses the same points,
    edges, and positions schema as feature constraints.
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
    """Return the closest reference point and Euclidean distance from a VTK query."""
    result = [0.0, 0.0, 0.0]
    cell_id, subcell_id, squared_distance = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
    locator['locator'].FindClosestPoint(
        np.asarray(point).tolist(), result, cell_id, subcell_id, squared_distance
    )
    if int(cell_id) < 0 or not np.isfinite(result).all():
        raise RuntimeError('VTK closest-point query failed')
    return np.asarray(result), math.sqrt(max(0.0, float(squared_distance)))


def _project_point(point, reference, parameters, *, boundary_component=None):
    """Project onto the source surface or one original boundary component, if enabled."""
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


def _run_remeshing_iteration(
    mesh, topology, features, boundary, reference, parameters, tolerances, collapse_history
):
    """Run stages in the prescribed order, then the final interior collapse pass.

    Keep deferred problematic edges alive through the early half, even if an
    iteration accepts no edits. The final pass cannot advance the deletion phase.
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
            collapse_history,
            iteration_index=iteration_index,
        )
        _equalize_valence(mesh, topology, features, boundary, reference, parameters, tolerances)
        _split_long_edge(
            mesh,
            topology,
            features,
            boundary,
            reference,
            parameters,
            tolerances,
            boundary_only=False,
        )
        _collapse_short_edge(
            mesh,
            topology,
            features,
            boundary,
            reference,
            parameters,
            tolerances,
            collapse_history,
            boundary_only=False,
            iteration_index=iteration_index,
        )
        _improve_triangle_angle(
            mesh, topology, features, boundary, reference, parameters, tolerances
        )
        _relax_vertex(
            mesh,
            topology,
            features,
            boundary,
            reference,
            parameters,
            tolerances,
            boundary_only=False,
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
        # Do not terminate on zero accepted edits while deferred edges await phase 2.
    if len(mesh.element):
        _collapse_short_edge(
            mesh,
            topology,
            features,
            boundary,
            reference,
            parameters,
            tolerances,
            collapse_history,
            boundary_only=False,
            iteration_index=parameters['max_iter'] - 1,
            is_final_pass=True,
        )


def _process_boundary(
    mesh,
    topology,
    features,
    boundary,
    reference,
    parameters,
    tolerances,
    collapse_history,
    *,
    iteration_index,
):
    """Split, collapse, and relax current boundary edges/vertices when enabled."""
    if not parameters['flag_process_boundary']:
        return
    _split_long_edge(
        mesh, topology, features, boundary, reference, parameters, tolerances, boundary_only=True
    )
    _collapse_short_edge(
        mesh,
        topology,
        features,
        boundary,
        reference,
        parameters,
        tolerances,
        collapse_history,
        boundary_only=True,
        iteration_index=iteration_index,
    )
    _relax_vertex(
        mesh, topology, features, boundary, reference, parameters, tolerances, boundary_only=True
    )


# Local edit proposals


def _collect_candidate_edge(mesh, topology, parameters, *, boundary_only, operation):
    """Snapshot eligible edge keys in deterministic order; collapses try shortest first."""
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
    """Create an edit record without changing topology or geometry.

    remove_faces names current rows; add_faces contains replacement vertex triples.
    positions changes existing vertices; new_points appends vertices in order.
    retired vertices must lose all references. normal_refs constrains replacement
    orientation. revision binds the proposal to the topology that produced it.
    """
    return dict(
        kind=kind,
        revision=topology['revision'],
        edge=edge,
        remove_faces=set(remove),
        add_faces=[tuple(map(int, f)) for f in add],
        positions={} if positions is None else positions,
        new_points=[np.asarray(p, dtype=float).copy() for p in new_points],
        retired=set(retired),
        normal_refs=list(normal_refs),
    )


def _propose_edge_split(mesh, topology, edge, features, boundary, reference, parameters):
    """Propose midpoint subdivision of every incident triangle, preserving orientation."""
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


def _split_long_edge(
    mesh, topology, features, boundary, reference, parameters, tolerances, *, boundary_only
):
    """Commit valid splits and revisit long child segments in the same pass."""
    candidates = deque(
        _collect_candidate_edge(
            mesh, topology, parameters, boundary_only=boundary_only, operation='split'
        )
    )
    accepted_count = 0
    while candidates:
        edge = candidates.popleft()
        if (
            edge not in topology['edge_to_faces']
            or topology['lengths'][edge] <= parameters['max_edge_length']
        ):
            continue
        if (edge in topology['boundary_edges']) != boundary_only:
            continue
        edit = _propose_edge_split(mesh, topology, edge, features, boundary, reference, parameters)
        if edit is None:
            continue
        valid, _ = _validate_edit(
            mesh, topology, edit, features, boundary, reference, parameters, tolerances
        )
        if valid:
            new_vertex_id = len(mesh.node)
            _commit_edit(mesh, topology, edit, features, boundary)
            accepted_count += 1
            for v in edge:  # Source repeats child segments, not every new spoke.
                child_edge = _edge(v, new_vertex_id)
                if topology['lengths'][child_edge] > parameters['max_edge_length']:
                    candidates.append(child_edge)
    return accepted_count


def _plan_collapse_face(mesh, topology, edge, survivor_vertex_id):
    """Identify disappearing faces and rewire the other endpoint faces to the survivor."""
    u, v = edge
    removed_vertex_id = v if survivor_vertex_id == u else u
    affected_faces = topology['vertex_to_faces'][u] | topology['vertex_to_faces'][v]
    collapsed_faces = topology['vertex_to_faces'][u] & topology['vertex_to_faces'][v]
    if collapsed_faces != topology['edge_to_faces'].get(edge, set()):
        raise RuntimeError('edge and endpoint face incidences disagree')
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
        collapsed_face_ids=collapsed_faces,
        replacement_faces=replacement_faces,
        original_face_ids=original_face_ids,
    )


def _propose_edge_collapse(mesh, topology, edge, features, boundary, reference, parameters):
    """Propose midpoint/endpoint placements in priority order, respecting fixed vertices."""
    if (
        edge not in topology['edge_to_faces']
        or edge in features['edges']
        or edge in boundary['frozen']['edges']
    ):
        return []
    u, v = edge
    protected = set(features['points']) | set(boundary['frozen']['points'])
    if edge not in topology['boundary_edges']:
        protected.update(w for w in edge if topology['boundary_neighbors'][w])
    if u in protected and v in protected:
        return []
    if u in protected:
        placements = [(u, topology['vertex_positions'][u])]
    elif v in protected:
        placements = [(v, topology['vertex_positions'][v])]
    else:
        placements = [
            (u, (topology['vertex_positions'][u] + topology['vertex_positions'][v]) / 2),
            (v, topology['vertex_positions'][v]),
            (u, topology['vertex_positions'][u]),
        ]
    result = []
    for survivor, position in placements:
        plan = _plan_collapse_face(mesh, topology, edge, survivor)
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


def _passes_collapse_length_check(mesh, topology, edit, parameters, *, boundary_only):
    """Prevent retained neighboring edges from newly crossing the length thresholds."""
    edge = edit['edge']
    if (
        edge not in topology['edge_to_faces']
        or topology['lengths'][edge] > parameters['min_edge_length']
    ):
        return False
    survivor, retired_vertex_id = edit['survivor'], next(iter(edit['retired']))
    position = edit['positions'][survivor]
    # Only retained connections matter for the explicit region-erasure extension.
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


def _propose_edge_flip(mesh, topology, edge, features):
    """Propose the opposite diagonal of two coherently oriented incident triangles."""
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


def _propose_vertex_move(
    mesh, vertex_id, candidate_position, features, boundary, reference, parameters
):
    """Return an optionally projected candidate position, or None for a fixed vertex."""
    if vertex_id in features['points'] or vertex_id in boundary['frozen']['points']:
        return None
    point = np.asarray(candidate_position, dtype=float)
    if parameters['flag_project_to_source_mesh']:
        point = _project_point(
            point, reference, parameters, boundary_component=boundary['labels'].get(vertex_id)
        )
    return point


def _candidate_point(topology, edit, faces):
    """Gather proposed triangle coordinates, including moved and newly appended vertices."""
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


def _geometry(points):
    """Compute areas, unit normals, and corner angles for an array of triangle coordinates."""
    if not len(points):
        return np.empty(0), np.empty((0, 3)), np.empty((0, 3))
    node = torch.from_numpy(np.asarray(points, dtype=float).reshape(-1, 3))
    element = torch.arange(len(node)).reshape(-1, 3)
    area, normals = TriangleMesh.cal_element_area_and_normal(node, element)
    angles = TriangleMesh.cal_element_corner_angle(node, element)
    normal_array = normals.numpy()
    normal_lengths = np.linalg.norm(normal_array, axis=1, keepdims=True)
    normal_array = np.divide(
        normal_array, normal_lengths, out=np.zeros_like(normal_array), where=normal_lengths > 0
    )
    return area.numpy().ravel(), normal_array, angles.numpy()


# Edit validation and commit


def _validate_edit(mesh, topology, edit, features, boundary, reference, parameters, tolerances):
    """Validate an edit without mutation, returning (accepted, reason).

    Check revision and protections first, then operation-specific permission,
    candidate geometry, and finally changed incidences. The first rejection is
    retained in diagnostics, so the order of these checks is significant.
    """
    if edit['revision'] != topology['revision']:
        return False, 'stale_edit'
    for locks in (features, boundary['frozen']):
        valid, reason = _check_feature_constraint(mesh, topology, locks, edit)
        if not valid:
            return valid, reason
    kind = edit['kind']
    if kind == 'enclosed_region_collapse':
        valid, reason = _validate_enclosed_region_collapse(
            mesh, topology, edit, features, boundary, reference, parameters, tolerances
        )
        if not valid:
            return valid, reason
    if kind == 'repair_collapse' and not edit.get('repair_validated'):
        return False, 'unvalidated_repair'
    if kind == 'collapse':
        classification = _classify_collapse_neighborhood(
            mesh, topology, edit['edge'], features, boundary
        )
        if classification['problematic']:
            return False, 'link_obstruction'
        if not edit['add_faces']:
            return False, 'ordinary_component_erasure'
    valid, reason = _validate_edit_geometry(topology, edit, tolerances)
    if not valid:
        return valid, reason
    return _validate_edit_connectivity(topology, edit)


def _validate_edit_geometry(topology, edit, tolerances) -> tuple[bool, str]:
    """Reject invalid positions, duplicate triangles, degeneracy, or inversion."""
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
    areas, normals, _ = _geometry(_candidate_point(topology, edit, faces))
    if not np.isfinite(areas).all() or np.any(areas <= tolerances['area']):
        return False, 'degenerate_triangle'
    if normal_references and np.any(
        np.einsum('ij,ij->i', normals, np.asarray(normal_references)) <= tolerances['normal_dot']
    ):
        return False, 'inverted_triangle'
    return True, 'ok'


def _validate_edit_connectivity(topology, edit) -> tuple[bool, str]:
    """Check edge incidence changes and ensure retired vertices lose all faces."""
    kind = edit['kind']
    if kind == 'move':
        return True, 'ok'

    incidence_delta = Counter(edge for face in edit['add_faces'] for edge in _face_edge(face))
    incidence_delta.subtract(
        edge for face_id in edit['remove_faces'] for edge in _face_edge(topology['faces'][face_id])
    )
    # A split may pass its source edge's high incidence to child edges. Otherwise
    # an edit may neither create incidence above two nor worsen high incidence.
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


def _commit_edit(mesh, topology, edit, features, boundary):
    """Commit prevalidated records; all expected rejection happens before mutation."""
    if edit['revision'] != topology['revision']:
        raise RuntimeError('attempt to commit a stale edit')
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
    if edit['kind'] in ('repair_collapse', 'enclosed_region_collapse'):
        accepted_edit = dict(
            kind=edit['kind'],
            iteration=edit.get('iteration_index', -1) + 1,
            edge=list(edit['edge']),
            removed_faces=len(edit['remove_faces']),
            added_faces=len(edit['add_faces']),
            repair_failure=edit.get('deletion_evidence', {}).get('repair_failure'),
        )
        topology['ledger'].append(accepted_edit)
        if edit['kind'] == 'enclosed_region_collapse':
            accepted_edit['enclosed_face_count'] = len(edit['enclosed_faces'])
            accepted_edit['enclosures'] = [
                dict(
                    method=region['method'],
                    cycle=list(map(int, region['cycle']['vertices'])),
                    enclosed_face_count=len(region['faces']),
                    support_face_count=len(region.get('support_faces', ())),
                    nested_chord_face_count=len(region.get('nested_chord_faces', ())),
                    cap_count=region.get('cap_count', 1),
                    cap_face_counts=[len(cap['faces']) for cap in region.get('caps', ())],
                )
                for region in edit['regions']
            ]
            topology['deletions'].append(accepted_edit.copy())


# Collapse policy and bounded local repair


def _classify_collapse_neighborhood(mesh, topology, edge, features, boundary):
    """Identify protection and vertex/edge link obstructions to ordinary collapse."""
    incident = topology['edge_to_faces'].get(edge, set())
    u, v = edge
    common_neighbors = topology['vertex_neighbors'][u] & topology['vertex_neighbors'][v]
    opposite_vertices = {
        int(w) for face_id in incident for w in topology['faces'][face_id] if w not in edge
    }
    obstructing_neighbors = common_neighbors - opposite_vertices
    # The link condition also has an edge part: shared opposite edges would
    # create duplicate triangles (e.g. a tetrahedral pair of caps) on collapse.
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
        shared_link_edges=shared_link_edges,
        problematic=len(incident) > 2 or bool(obstructing_neighbors) or bool(shared_link_edges),
        revision=topology['revision'],
        eligible=bool(incident)
        and edge not in features['edges']
        and edge not in boundary['frozen']['edges'],
    )


def _collapse_short_edge(
    mesh,
    topology,
    features,
    boundary,
    reference,
    parameters,
    tolerances,
    collapse_history,
    *,
    boundary_only,
    iteration_index,
    is_final_pass=False,
):
    """Attempt each queued collapse and accumulate its accepted/rejected outcome."""
    accepted_count = 0
    candidates = _collect_candidate_edge(
        mesh, topology, parameters, boundary_only=boundary_only, operation='collapse'
    )
    for edge in candidates:
        accepted, reason = _attempt_collapse(
            mesh,
            topology,
            edge,
            features,
            boundary,
            reference,
            parameters,
            tolerances,
            collapse_history,
            boundary_only=boundary_only,
            iteration_index=iteration_index,
            is_final_pass=is_final_pass,
        )
        accepted_count += accepted
        topology['outcomes'][reason] += 1
    for edge in list(collapse_history):
        if edge not in topology['edge_to_faces']:
            del collapse_history[edge]
    return accepted_count


def _attempt_collapse(
    mesh,
    topology,
    edge,
    features,
    boundary,
    reference,
    parameters,
    tolerances,
    collapse_history,
    *,
    boundary_only,
    iteration_index,
    is_final_pass=False,
    allow_repair=True,
):
    """Try ordinary collapse, local repair, then phase-gated region deletion.

    Failed proposals leave the live mesh unchanged. Ordinary geometry failures
    never authorize deletion: the edge must have a link/incidence obstruction,
    fail a current repair attempt, and pass the iteration gate. Scratch repair
    patches disable recursive repair. A final pass keeps its supplied iteration.
    """
    # Recheck queued candidates against the current topology.
    if edge not in topology['edge_to_faces']:
        return False, 'edge_removed'
    if topology['lengths'][edge] > parameters['min_edge_length']:
        return False, 'not_short'
    if (edge in topology['boundary_edges']) != boundary_only:
        return False, 'classification_changed'
    classification = _classify_collapse_neighborhood(mesh, topology, edge, features, boundary)
    if not classification['eligible']:
        return False, 'protected_edge'
    ordinary = _propose_edge_collapse(
        mesh, topology, edge, features, boundary, reference, parameters
    )
    if not ordinary:
        return False, 'protected_endpoints'

    # An ordinary neighborhood gets only ordinary collapse proposals.
    reason = 'link_obstruction'
    if not classification['problematic']:
        for edit in ordinary:
            if not _passes_collapse_length_check(
                mesh, topology, edit, parameters, boundary_only=boundary_only
            ):
                reason = 'neighbor_length'
                continue
            valid, reason = _validate_edit(
                mesh, topology, edit, features, boundary, reference, parameters, tolerances
            )
            if valid:
                edit['iteration_index'] = iteration_index
                _commit_edit(mesh, topology, edit, features, boundary)
                return True, 'ordinary_collapsed'
        # Numerical/geometry rejections are not permission for destructive fallback.
        return False, reason

    # Repair on a private patch; commit the combined edit only after validation.
    if not allow_repair:
        return False, 'scratch_link_obstruction'
    edit, failure = _try_repair_collapse_patch(
        mesh,
        topology,
        edge,
        features,
        boundary,
        reference,
        parameters,
        tolerances,
        boundary_only=boundary_only,
        iteration_index=iteration_index,
    )
    if edit is not None:
        valid, failure = _validate_edit(
            mesh, topology, edit, features, boundary, reference, parameters, tolerances
        )
        if valid:
            edit['iteration_index'] = iteration_index
            _commit_edit(mesh, topology, edit, features, boundary)
            return True, 'repaired_collapsed'

    # A failed repair is recorded even during the early, protected iterations.
    collapse_history[edge] = dict(
        iteration_index=iteration_index, revision=topology['revision'], repair_failure=failure
    )
    if not _deletion_phase_allowed(iteration_index, parameters['max_iter']):
        return False, 'deferred_after_repair_failure'
    current = _classify_collapse_neighborhood(mesh, topology, edge, features, boundary)
    if not current['problematic']:
        return False, 'no_longer_problematic'

    # Late fallback requires complete, unambiguous enclosure certificates.
    regions = []
    for cycle in _find_enclosing_cycle(mesh, topology, edge, tolerances):
        region, diagnostic = _collect_enclosed_face(mesh, topology, cycle, tolerances)
        if region is not None:
            regions.append(region)
    regions, diagnostic = _select_outermost_enclosed_region(regions, edge)
    if not regions:
        return False, diagnostic
    evidence = dict(
        iteration_index=iteration_index,
        revision=topology['revision'],
        repair_failure=failure,
        problematic=True,
    )
    for edit in _propose_enclosed_region_collapse(
        mesh, topology, edge, regions, features, boundary, reference, parameters, evidence
    ):
        if not _passes_collapse_length_check(
            mesh, topology, edit, parameters, boundary_only=boundary_only
        ):
            reason = 'neighbor_length'
            continue
        valid, reason = _validate_edit(
            mesh, topology, edit, features, boundary, reference, parameters, tolerances
        )
        if valid:
            _commit_edit(mesh, topology, edit, features, boundary)
            return True, 'enclosed_region_collapsed'
    return False, reason


def _extract_repair_patch(mesh, topology, edge, features, boundary, parameters, *, ring_count):
    """Copy a bounded face neighborhood and map source IDs into local patch IDs.

    Protect feature geometry, frozen boundaries, and patch interfaces so replacing
    the patch cannot alter unrelated faces. Return (patch, reason).
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
    # Protect interface edges, not arbitrary chords between two fixed vertices.
    interface_edges = {
        e for e, fs in topology['edge_to_faces'].items() if fs & faces and fs - faces
    }
    protected_edges = features['edges'] | boundary['frozen']['edges'] | interface_edges
    local_edges = {
        _edge(global_to_local[a], global_to_local[b])
        for a, b in protected_edges
        if a in global_to_local and b in global_to_local
    }
    local_points = {global_to_local[v] for v in fixed}
    local_points.update(v for e in local_edges for v in e)
    local_features = dict(
        points=local_points,
        edges=local_edges,
        positions={v: _as_numpy(submesh.node)[v].copy() for v in local_points},
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
            global_to_local=global_to_local,
            edge=_edge(*(global_to_local[v] for v in edge)),
            original_edge=edge,
            features=local_features,
            boundary=local_boundary,
            revision=topology['revision'],
        ),
        'ok',
    )


def _repair_patch_geometry(patch, reference, parameters, tolerances):
    """Resolve nonfacial link obstructions by conforming local edge subdivision.

    This preserves triangle support exactly before retrying collapse. Repair is
    specific to this remesher's link checks and interface constraints. Genuine
    junctions with >2 incident sheets cannot be turned into a manifold edge without
    separating or erasing sheets, so they return an explicit repair failure.
    """
    mesh = _copy_source_mesh(patch['mesh'])
    topology = _initialize_topology(mesh)
    edge, features, boundary = patch['edge'], patch['features'], patch['boundary']
    if len(topology['edge_to_faces'][edge]) > 2:
        return None, 'repair_requires_sheet_erasure_or_disconnection'
    for attempt in range(parameters['repair_max_attempts'] + 1):
        classification = _classify_collapse_neighborhood(mesh, topology, edge, features, boundary)
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
            edit = _propose_edge_split(
                mesh, topology, candidate, features, boundary, reference, parameters
            )
            if edit is None:
                continue
            valid, _ = _validate_edit(
                mesh, topology, edit, features, boundary, reference, parameters, tolerances
            )
            if valid:
                _commit_edit(mesh, topology, edit, features, boundary)
                changed = True
                break
        if not changed:
            return None, 'repair_interface_or_feature_obstruction'
    return None, 'repair_attempt_budget'


def _sample_triangle(mesh):
    """Sample triangle vertices, centroids, and edge midpoints for repair distance checks."""
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


def _validate_repair_collapse(patch, repaired, reference, parameters, tolerances):
    """Check repaired-patch constraints, manifoldness, sampled distance, and components."""
    mesh, topology = repaired['mesh'], repaired['topology']
    if not len(mesh.element):
        return False, 'repair_erases_component'
    valid, reason = _check_feature_constraint(mesh, topology, patch['features'])
    if not valid:
        return valid, reason
    try:
        _validate_mesh(mesh, topology, tolerances, strict_manifold=True)
    except ValueError:
        return False, 'invalid_repaired_patch'
    # Check sampled distance BOTH ways, including triangle interiors. This is a
    # measured approximation bound, not a proof of exact Hausdorff distance.
    for source, target in ((patch['original'], mesh), (mesh, patch['original'])):
        locator = _build_cell_locator(target)
        if any(
            _closest_point(p, locator)[1] > parameters['repair_distance']
            for p in _sample_triangle(source)
        ):
            return False, 'repair_shape_error'
    # No component/sheet may disappear merely because another one lies nearby.
    if _component_count(patch['original']) != _component_count(mesh):
        return False, 'repair_changes_components'
    return True, 'ok'


def _component_count(mesh):
    """Count face components connected through shared vertices."""
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


def _compose_repair_collapse_edit(mesh, topology, patch, repaired, features, boundary):
    """Map an accepted scratch-patch result into one atomic edit of the live mesh."""
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
    reference,
    parameters,
    tolerances,
    *,
    boundary_only,
    iteration_index,
):
    """Try bounded patch sizes and return a validated compound edit or failure reason."""
    reason = 'repair_failed'
    for rings in range(1, parameters['repair_rings'] + 1):
        patch, reason = _extract_repair_patch(
            mesh, topology, edge, features, boundary, parameters, ring_count=rings
        )
        if patch is None:
            continue
        repaired, reason = _repair_patch_geometry(patch, reference, parameters, tolerances)
        if repaired is None:
            continue
        accepted, reason = _attempt_collapse(
            repaired['mesh'],
            repaired['topology'],
            patch['edge'],
            repaired['features'],
            repaired['boundary'],
            reference,
            parameters,
            tolerances,
            {},
            boundary_only=boundary_only,
            iteration_index=iteration_index,
            allow_repair=False,
        )
        if not accepted:
            continue
        valid, reason = _validate_repair_collapse(
            patch, repaired, reference, parameters, tolerances
        )
        if valid:
            return (
                _compose_repair_collapse_edit(mesh, topology, patch, repaired, features, boundary),
                'ok',
            )
    return None, reason


# Enclosure discovery and certification


def _find_enclosing_cycle(mesh, topology, edge, tolerances):
    """Enumerate three-edge cycles through the candidate edge and its common neighbors."""
    u, v = edge
    return [
        dict(vertices=(u, v, w), edge=edge, revision=topology['revision'])
        for w in sorted(topology['vertex_neighbors'][u] & topology['vertex_neighbors'][v])
    ]


def _inside_cycle(points, triangle, tolerances):
    """Verified coplanar barycentric containment; never a screen projection test."""
    a, b, c = triangle
    e0, e1 = b - a, c - a
    normal = np.cross(e0, e1)
    normal_length = np.linalg.norm(normal)
    if normal_length <= 2 * tolerances['area']:
        return np.zeros(np.shape(points)[:-1], dtype=bool)
    displacement = points - a
    d00, d01, d11 = np.dot(e0, e0), np.dot(e0, e1), np.dot(e1, e1)
    determinant = d00 * d11 - d01 * d01
    if determinant <= 0:
        return np.zeros(np.shape(points)[:-1], dtype=bool)
    v = (d11 * (displacement @ e0) - d01 * (displacement @ e1)) / determinant
    w = (d00 * (displacement @ e1) - d01 * (displacement @ e0)) / determinant
    barycentric_tolerance = tolerances['barycentric']
    return (
        (np.abs(displacement @ normal) / normal_length <= tolerances['plane'])
        & (v >= -barycentric_tolerance)
        & (w >= -barycentric_tolerance)
        & (v + w <= 1 + barycentric_tolerance)
    )


def _collect_planar_enclosed_face(mesh, topology, cycle, tolerances):
    """Collect all connected coplanar interior faces, including nonincident faces.

    Curved/ambiguous interiors are conservatively declined. All boundary-edge seeds
    are considered, so nested triangles sharing (u,v) are never truncated to two.
    """
    vertices = cycle['vertices']
    barriers = set(_face_edge(vertices))
    triangle = topology['vertex_positions'][list(vertices)]
    inside_cache = {}

    def inside(face_id):
        if face_id not in inside_cache:
            inside_cache[face_id] = bool(
                _inside_cycle(
                    topology['vertex_positions'][topology['faces'][face_id]], triangle, tolerances
                ).all()
            )
        return inside_cache[face_id]

    seeds = {
        face_id
        for edge in barriers
        for face_id in topology['edge_to_faces'].get(edge, ())
        if inside(face_id)
    }
    if not seeds:
        return None, 'no_verified_planar_interior'
    selected, queue = set(seeds), deque(sorted(seeds))
    while queue:
        face_id = queue.popleft()
        for edge in _face_edge(topology['faces'][face_id]):
            if edge in barriers:
                continue
            for other in sorted(topology['edge_to_faces'].get(edge, ())):
                if other not in selected and inside(other):
                    selected.add(other)
                    queue.append(other)
    # Every cycle side must be supported on its inner side. Additional exposed
    # interior edges may belong to nested overlaid triangles in the user's example.
    if any(not (topology['edge_to_faces'][edge] & selected) for edge in barriers):
        return None, 'incomplete_enclosing_cycle'
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    directions = np.array([np.dot(topology['normals'][face_id], normal) for face_id in selected])
    if np.any(directions > 0) and np.any(directions < 0):
        return None, 'ambiguous_sheet_orientation'
    # A traversal interface off the cycle is evidence of a folded/partial region.
    for face_id in selected:
        for edge in _face_edge(topology['faces'][face_id]):
            if edge not in barriers and any(
                other not in selected for other in topology['edge_to_faces'][edge]
            ):
                return None, 'ambiguous_interior_interface'
    return (
        dict(
            cycle=cycle,
            faces=selected,
            vertices={int(v) for face_id in selected for v in topology['faces'][face_id]},
            triangle=triangle.copy(),
            revision=topology['revision'],
            method='coplanar_containment',
        ),
        'ok',
    )


def _edge_direction(face, edge):
    """Orientation of an existing edge in an oriented triangle, independent of XYZ."""
    face = tuple(face)
    i = face.index(edge[0])
    return 1 if face[(i + 1) % 3] == edge[1] else -1


def _collect_cut_disk(topology, barriers, anchor, excluded):
    """Find and certify one surface disk without crossing cycle edges or sheets.

    Other faces incident to the collapsing edge are provisionally excluded so
    redundant (u,v,w_i) chords do not masquerade as sheet continuations. A valid
    disk has exactly the requested boundary, coherent orientation, Euler value 1,
    and a connected path/cycle link at every boundary/interior vertex.
    """
    selected_faces, queue = {anchor}, deque([anchor])
    while queue:
        face_id = queue.popleft()
        for edge in _face_edge(topology['faces'][face_id]):
            if edge in barriers:
                continue
            adjacent = topology['edge_to_faces'][edge] - excluded
            if len(adjacent) > 2:
                return None, 'unresolved_nonmanifold_disk_edge'
            for other in sorted(adjacent):
                if other not in selected_faces:
                    selected_faces.add(other)
                    queue.append(other)
    return _certify_disk(topology, selected_faces, barriers)


def _certify_disk(topology, selected_faces, barriers):
    """Certify a traversed region's boundary, Euler value, orientation, and links.

    A vertex link connects each incident triangle's two other vertices. Interior
    links must be connected cycles; boundary links must be connected paths ending
    at that vertex's two boundary neighbors. This is a topological certificate,
    not a geometric self-intersection check. Return (certificate, reason).
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


def _collect_curved_enclosed_face(mesh, topology, cycle, tolerances):
    """Resolve a disk or a certified pair of caps, including nested UV chords.

    No projected containment, normal-angle threshold, or smallest-side heuristic
    is used. Two disks meeting only on their common boundary may be erased together;
    intersecting candidates and junctions of more than two caps remain unresolved.
    Arbitrarily curved/folded embedded disks are supported; as elsewhere in this
    remesher this is not a general geometric self-intersection certification.
    """
    edge = cycle['edge']
    cycle_edges = set(_face_edge(cycle['vertices']))
    if edge not in cycle_edges or not all(e in topology['edge_to_faces'] for e in cycle_edges):
        return None, 'invalid_enclosing_cycle'
    incident_faces = topology['edge_to_faces'][edge]
    candidates, diagnostics = [], []
    for anchor_face_id in sorted(incident_faces):
        certified_disk, reason = _collect_cut_disk(
            topology, cycle_edges, anchor_face_id, incident_faces - {anchor_face_id}
        )
        if certified_disk is None:
            diagnostics.append(reason)
            continue
        edge_direction = _edge_direction(topology['faces'][anchor_face_id], edge)
        enclosed_faces = set(certified_disk['faces'])
        unresolved_attachment = False
        for face_id in incident_faces - {anchor_face_id}:
            face = topology['faces'][face_id]
            face_edges = set(_face_edge(face))
            belongs_to_disk = (
                set(face) <= certified_disk['vertices']
                and face_edges <= certified_disk['edges']
                and _edge_direction(face, edge) == edge_direction
            )
            if belongs_to_disk:
                enclosed_faces.add(face_id)
            elif (face_edges & certified_disk['edges']) - cycle_edges:
                # An opposite/unknown sheet attached inside the disk cannot be
                # silently omitted or absorbed into the selected interior.
                unresolved_attachment = True
                break
        if unresolved_attachment:
            diagnostics.append('ambiguous_internal_sheet_attachment')
            continue
        candidates.append(
            dict(
                cycle=cycle,
                faces=enclosed_faces,
                vertices=certified_disk['vertices'],
                support_faces=certified_disk['faces'],
                nested_chord_faces=enclosed_faces - certified_disk['faces'],
                revision=topology['revision'],
                method='curved_disk',
                side_direction=edge_direction,
                boundary_edges=cycle_edges,
                euler_characteristic=1,
            )
        )
    unique = {frozenset(region['faces']): region for region in candidates}
    maximal = [
        region
        for region in unique.values()
        if not any(region['faces'] < other['faces'] for other in unique.values())
    ]
    if len(maximal) == 1:
        if len(maximal[0]['support_faces']) == 1:
            # A lone triangle does not certify the enclosure of other, curved
            # sheets sharing its edge. The legacy planar proof may still apply.
            return None, 'no_curved_disk_interior'
        return maximal[0], 'ok'
    if len(maximal) == 2:
        return _combine_cap_pair(topology, cycle, maximal)
    if len(maximal) > 2:
        return None, 'ambiguous_cycle_sides'
    return None, diagnostics[0] if diagnostics else 'no_enclosed_disk'


def _combine_cap_pair(topology, cycle, caps):
    """Certify two disjoint disk interiors and return their complete union.

    Both disks were individually certified. Their support meshes must meet exactly
    on the three cycle edges/vertices, with opposite boundary orientations. This
    authorizes removing both sides, not an arbitrary union of ambiguous sheets.
    Feature, boundary, repair and iteration checks still occur in the edit path.
    """
    first, second = caps
    boundary_vertices = set(cycle['vertices'])
    barriers = set(_face_edge(cycle['vertices']))
    if (
        first['faces'] & second['faces']
        or first['vertices'] & second['vertices'] != boundary_vertices
    ):
        return None, 'ambiguous_intersecting_caps'
    support_edges = [
        {e for f in cap['support_faces'] for e in _face_edge(topology['faces'][f])} for cap in caps
    ]
    if support_edges[0] & support_edges[1] != barriers:
        return None, 'ambiguous_intersecting_caps'
    for edge in barriers:
        sides = [
            next(f for f in cap['support_faces'] if edge in _face_edge(topology['faces'][f]))
            for cap in caps
        ]
        if _edge_direction(topology['faces'][sides[0]], edge) == _edge_direction(
            topology['faces'][sides[1]], edge
        ):
            return None, 'ambiguous_cap_orientation'
    return (
        dict(
            cycle=cycle,
            faces=first['faces'] | second['faces'],
            vertices=first['vertices'] | second['vertices'],
            support_faces=first['support_faces'] | second['support_faces'],
            nested_chord_faces=first['nested_chord_faces'] | second['nested_chord_faces'],
            revision=topology['revision'],
            method='curved_two_caps',
            caps=caps,
            cap_count=2,
            cycle_edges=barriers,
            boundary_edges=set(),
            euler_characteristic=2,
        ),
        'ok',
    )


def _collect_enclosed_face(mesh, topology, cycle, tolerances):
    """Collect a planar enclosure, an intrinsic disk, or both certified cap sides.

    Keep the existing geometric rule for coplanar overlaid triangles whose free
    vertices need not belong to a connected support disk. Only use that rule when
    all UV-incident faces are coplanar with the cycle; it must not silently ignore
    another curved sheet. Otherwise establish the enclosure from surface topology.
    """
    triangle = topology['vertex_positions'][list(cycle['vertices'])]
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    magnitude = np.linalg.norm(normal)
    if magnitude <= 2 * tolerances['area']:
        return None, 'degenerate_enclosing_cycle'
    curved, reason = _collect_curved_enclosed_face(mesh, topology, cycle, tolerances)
    if reason.startswith('ambiguous_'):
        return None, reason
    vertices = {
        int(v)
        for f in topology['edge_to_faces'].get(cycle['edge'], ())
        for v in topology['faces'][f]
    }
    distance = (
        np.abs((topology['vertex_positions'][sorted(vertices)] - triangle[0]) @ normal) / magnitude
    )
    planar = None
    if np.all(distance <= tolerances['plane']):
        planar, _ = _collect_planar_enclosed_face(mesh, topology, cycle, tolerances)
    if curved is not None and planar is not None:
        if curved['faces'] <= planar['faces']:
            return planar, 'ok'
        if planar['faces'] <= curved['faces']:
            return curved, 'ok'
        return None, 'ambiguous_enclosure_certificates'
    if curved is not None:
        return curved, 'ok'
    if planar is not None:
        return planar, 'ok'
    return None, reason


def _select_outermost_enclosed_region(regions, edge):
    """Keep maximal face sets by containment and reject overlapping incomparable regions."""
    unique = {}
    for region in regions:
        unique[frozenset(region['faces'])] = region
    regions = list(unique.values())
    maximal = [
        region
        for region in regions
        if not any(region['faces'] < other['faces'] for other in regions)
    ]
    for i, region in enumerate(maximal):
        for other in maximal[i + 1 :]:
            if region['faces'] & other['faces']:
                return [], 'crossing_enclosed_regions'
    return maximal, 'ok' if maximal else 'no_unambiguous_enclosing_region'


def _propose_enclosed_region_collapse(
    mesh, topology, edge, regions, features, boundary, reference, parameters, deletion_evidence
):
    """Combine complete certified-region removal with each permitted endpoint collapse."""
    if not _deletion_phase_allowed(deletion_evidence['iteration_index'], parameters['max_iter']):
        return []
    enclosed = set().union(*(r['faces'] for r in regions))
    results = []
    for ordinary in _propose_edge_collapse(
        mesh, topology, edge, features, boundary, reference, parameters
    ):
        survivor = ordinary['survivor']
        plan = _plan_collapse_face(mesh, topology, edge, survivor)
        replacement_faces, normal_references = [], []
        for face_id, triangle in zip(plan['original_face_ids'], plan['replacement_faces']):
            if face_id not in enclosed:
                replacement_faces.append(triangle)
                normal_references.append(topology['normals'][face_id])
        edit = _proposal(
            topology,
            'enclosed_region_collapse',
            edge=edge,
            remove=enclosed | plan['affected_face_ids'],
            add=replacement_faces,
            positions=ordinary['positions'],
            retired=ordinary['retired'],
            normal_refs=normal_references,
        )
        edit.update(
            survivor=survivor,
            enclosed_faces=enclosed,
            regions=regions,
            deletion_evidence=dict(deletion_evidence),
            iteration_index=deletion_evidence['iteration_index'],
        )
        results.append(edit)
    return results


def _validate_enclosed_region_collapse(
    mesh, topology, edit, features, boundary, reference, parameters, tolerances
):
    """Recheck current repair evidence, phase permission, certificates, and deletion scope."""
    evidence = edit.get('deletion_evidence', {})
    if (
        evidence.get('revision') != topology['revision']
        or not evidence.get('repair_failure')
        or not _deletion_phase_allowed(evidence.get('iteration_index', -1), parameters['max_iter'])
    ):
        return False, 'deletion_not_permitted'
    if not _classify_collapse_neighborhood(mesh, topology, edit['edge'], features, boundary)[
        'problematic'
    ]:
        return False, 'no_longer_problematic'
    for region in edit['regions']:
        if region['revision'] != topology['revision']:
            return False, 'stale_enclosure'
        current, reason = _collect_enclosed_face(mesh, topology, region['cycle'], tolerances)
        if current is None or current['faces'] != region['faces']:
            return False, 'enclosure_certificate_changed'
    expected = set().union(*(r['faces'] for r in edit['regions']))
    if expected != edit['enclosed_faces'] or not expected <= edit['remove_faces']:
        return False, 'incorrect_deletion_scope'
    endpoint_faces = set().union(*(topology['vertex_to_faces'][v] for v in edit['edge']))
    if edit['remove_faces'] != expected | endpoint_faces:
        return False, 'unrelated_face_deletion'
    return True, 'ok'


# Mesh quality and relaxation stages


def _equalize_valence(mesh, topology, features, boundary, reference, parameters, tolerances):
    """Accept flips that reduce deviation from boundary/interior target valences."""
    accepted_count = 0
    for edge in _collect_candidate_edge(
        mesh, topology, parameters, boundary_only=False, operation='flip'
    ):
        edit = _propose_edge_flip(mesh, topology, edge, features)
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
        valid, _ = _validate_edit(
            mesh, topology, edit, features, boundary, reference, parameters, tolerances
        )
        if valid:
            _commit_edit(mesh, topology, edit, features, boundary)
            accepted_count += 1
    return accepted_count


def _improve_triangle_angle(mesh, topology, features, boundary, reference, parameters, tolerances):
    """Try flips opposite angles over 120 degrees and retain only valid improvements."""
    accepted_count = 0
    # Edge keys survive unrelated face-row swaps; never queue volatile face indices.
    candidates = set()
    for face_id, angles in topology['angles'].items():
        if angles.max() > 2 * math.pi / 3:
            corner_index = int(angles.argmax())
            triangle = topology['faces'][face_id]
            candidates.add(
                _edge(triangle[(corner_index + 1) % 3], triangle[(corner_index + 2) % 3])
            )
    for edge in sorted(candidates):
        edit = _propose_edge_flip(mesh, topology, edge, features)
        if edit is None:
            continue
        old_max_angle = max(
            float(topology['angles'][face_id].max()) for face_id in edit['remove_faces']
        )
        _, _, angles = _geometry(_candidate_point(topology, edit, edit['add_faces']))
        if angles.max() >= old_max_angle - 1e-12:
            continue
        valid, _ = _validate_edit(
            mesh, topology, edit, features, boundary, reference, parameters, tolerances
        )
        if valid:
            _commit_edit(mesh, topology, edit, features, boundary)
            accepted_count += 1
    return accepted_count


def _relax_vertex(
    mesh, topology, features, boundary, reference, parameters, tolerances, *, boundary_only
):
    """Move eligible vertices toward neighborhood averages, with optional source projection."""
    if boundary_only and not parameters['flag_process_boundary']:
        return 0
    accepted_count = 0
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
            # New boundaries have no original boundary-curve support. Do not snap
            # them onto an unrelated loop; without projection raw smoothing is legal.
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
            # Keep unresolved junction geometry until an explicit repair can handle it.
            if any(
                len(topology['edge_to_faces'][_edge(vertex_id, neighbor_id)]) > 2
                for neighbor_id in neighbors
            ):
                continue
            candidate = topology['vertex_positions'][neighbors].mean(axis=0)
        point = _propose_vertex_move(
            mesh, vertex_id, candidate, features, boundary, reference, parameters
        )
        if (
            point is None
            or np.linalg.norm(point - topology['vertex_positions'][vertex_id])
            <= tolerances['length']
        ):
            continue
        edit = _proposal(topology, 'move', positions={vertex_id: point})
        valid, _ = _validate_edit(
            mesh, topology, edit, features, boundary, reference, parameters, tolerances
        )
        if valid:
            _commit_edit(mesh, topology, edit, features, boundary)
            accepted_count += 1
    return accepted_count


# Output validation and compaction


def _validate_result(
    mesh, topology, reference, parameters, features, boundary, tolerances, *, old_to_new_vertex=None
):
    """Check output geometry, protected features/boundaries, and authorized deletion history."""
    _validate_mesh(mesh, topology, tolerances, allow_empty=True)
    for locks in (features, boundary['frozen']):
        valid, reason = _check_feature_constraint(
            mesh, topology, locks, old_to_new_vertex=old_to_new_vertex
        )
        if not valid:
            raise RuntimeError(f'output constraint violation: {reason}')
    for record in topology['deletions']:
        if not record['repair_failure'] or not _deletion_phase_allowed(
            record['iteration'] - 1, parameters['max_iter']
        ):
            raise RuntimeError('invalid deletion phase in accepted-edit ledger')
    if not len(mesh.element) and reference['summary']['faces'] and not topology['deletions']:
        raise RuntimeError('all source faces disappeared without authorized region deletion')


def _compact_mesh(mesh, features):
    """Use library submesh extraction; retain protected isolated points explicitly."""
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
    """Compact, restore source dtype/device, and revalidate after coordinate casting."""
    _validate_result(mesh, topology, reference, parameters, features, boundary, tolerances)
    protected = dict(features)
    protected['points'] = features['points'] | boundary['frozen']['points']
    output, mapping = _compact_mesh(mesh, protected)
    output.node = output.node.to(
        dtype=source_mesh.node.dtype, device=source_mesh.node.device
    ).clone()
    device = (
        source_mesh.element.device
        if torch.is_tensor(source_mesh.element)
        else source_mesh.node.device
    )
    output.element = output.element.to(dtype=torch.int64, device=device).clone()
    # Recheck after casting: float32 output can otherwise hide new degeneracy.
    check = _copy_source_mesh(output)
    checked = _initialize_topology(check)
    checked['deletions'] = topology['deletions']
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
