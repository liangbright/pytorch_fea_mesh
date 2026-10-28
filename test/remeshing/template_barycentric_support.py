"""Fixture loading/export and independent checks; no additional mesh library."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from torch_fea_mesh.mesh.polygon_mesh import PolygonMesh

PROJECT = Path(__file__).resolve().parent / "fixture/template_barycentric"
DIAGNOSTIC_KEY = "template_barycentric_mapping_based_remesh"


def load_mesh(path):
    path = Path(path)
    meta = json.loads(path.read_text())
    vtk = path.parent / meta["PointAndFace"]
    lines = vtk.read_text().splitlines()
    if lines[2].strip() != "ASCII" or lines[3].strip() != "DATASET POLYDATA":
        raise ValueError("Fixture reader accepts legacy ASCII POLYDATA only")
    tokens = " ".join(lines[4:]).split()
    assert tokens[0] == "POINTS"
    count = int(tokens[1]); offset = 3+3*count
    node = np.asarray(tokens[3:offset], dtype=np.float64).reshape(count, 3)
    assert tokens[offset] == "POLYGONS"
    count = int(tokens[offset+1]); end = offset+3+int(tokens[offset+2]); offset += 3
    faces = []
    for _ in range(count):
        size = int(tokens[offset])
        faces.append(list(map(int, tokens[offset+1:offset+1+size])))
        offset += size+1
    assert offset == end
    mesh = PolygonMesh(node=node, element=faces)
    mesh.name = meta.get("Name", "")
    mesh.node_set = {key: value for row in meta.get("PointSetList", []) for key, value in row.items()}
    mesh.element_set = {key: value for row in meta.get("FaceSetList", []) for key, value in row.items()}
    mesh.mapping_from_node_name_to_index = meta.get("NamedPointList", {})
    # Read the generated fixtures' scalar/vector FIELD metadata without VTK.
    while offset < len(tokens):
        section = tokens[offset]
        assert section in ("POINT_DATA", "CELL_DATA"), section
        tuples = int(tokens[offset+1]); offset += 2
        assert tokens[offset] == "FIELD"
        fields = int(tokens[offset+2]); offset += 3
        table = mesh.node_data if section == "POINT_DATA" else mesh.element_data
        for _ in range(fields):
            name, dim, rows, scalar = tokens[offset:offset+4]
            dim, rows = int(dim), int(rows); offset += 4
            assert rows == tuples
            table[name] = torch.tensor(np.asarray(tokens[offset:offset+dim*rows], dtype=float).reshape(rows, dim))
            offset += dim*rows
    assert len(mesh.node) == meta["PointCount"] and len(mesh.element) == meta["FaceCount"]
    return mesh


def triangulate_source(mesh):
    output = PolygonMesh(node=mesh.node.clone(), element=None)
    node, faces = mesh.node.numpy(), mesh.element.tolist() if torch.is_tensor(mesh.element) else mesh.element
    triangles, parent = [], []
    for k, f in enumerate(faces):
        if len(f) == 3:
            triangles.append(f); parent.append(k)
        else:
            assert len(f) == 4
            a, b, c, d = f
            pair = [[a, b, c], [a, c, d]] if np.linalg.norm(node[c]-node[a]) < np.linalg.norm(node[d]-node[b]) else [[a, b, d], [b, c, d]]
            triangles.extend(pair); parent.extend([k, k])
    output.element = torch.tensor(triangles, dtype=torch.int64)
    for key in ("name", "node_set", "node_data", "mapping_from_node_name_to_index", "mesh_data"):
        setattr(output, key, deepcopy(getattr(mesh, key)))
    output.element_data["parent_element"] = torch.tensor(parent)[:, None]
    return output


def load_cases():
    root = PROJECT / "TestData2"
    source = triangulate_source(load_mesh(root/"OutputLeaflet_1_2017_5_18_17_0.json"))
    template = load_mesh(root/"TemplateMesh_9.json")
    cases = [{"case_id": "testdata2", "source": source, "template": template,
              "source_segments": prepare_segments(source.node_set, source.mapping_from_node_name_to_index),
              "template_segments": prepare_segments(template.node_set, template.mapping_from_node_name_to_index)}]
    manifest = json.loads((PROJECT/"TestDataGenerated/manifest.json").read_text())
    for fixture in manifest["fixtures"]:
        corr = json.loads((PROJECT/fixture["files"]["correspondence"]).read_text())
        cases.append({"case_id": fixture["case_id"], "source": load_mesh(PROJECT/fixture["files"]["source_json"]),
                      "template": load_mesh(PROJECT/fixture["files"]["template_json"]),
                      "source_segments": corr["boundary_segment_list_of_source_mesh"],
                      "template_segments": corr["boundary_segment_list_of_template_mesh"]})
    return cases








def mesh_snapshot(mesh):
    return deepcopy(mesh.__dict__)


def assert_same(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b


def verify_result(case, output):
    """Independent physical-space constraints and SVD metric checks."""
    source, template = case["source"], case["template"]
    p, result = source.node.double().cpu().numpy(), output.node.double().cpu().numpy()
    original_template = template.node.double().cpu().numpy()
    d = output.mesh_data[DIAGNOSTIC_KEY]
    faces = source.element.numpy()
    eps = np.finfo(np.float32 if output.node.dtype == torch.float32 else np.float64).eps
    scale = np.linalg.norm(np.ptp(p, axis=0))
    rounding = 64*eps*(scale+np.max(abs(p)))
    assert np.isfinite(result).all() and result.shape == original_template.shape
    assert_same(template.element, output.element)
    for key in ("node_set", "element_set", "mapping_from_node_name_to_index", "node_data", "element_data"):
        assert_same(getattr(template, key), getattr(output, key))
    membership_error, interpolation_error = 0.0, 0.0
    for s, t in zip(case["source_segments"], case["template_segments"]):
        if len(s) == 1:
            np.testing.assert_array_equal(output.node[t].cpu().numpy(), source.node[s].to(output.node.dtype).cpu().numpy())
            continue
        a, b = p[s[:-1]], p[s[1:]]
        edge = b-a
        query = result[t]
        v = query[:, None]-a[None]
        fraction = np.clip(np.sum(v*edge[None], axis=2)/np.sum(edge**2, axis=1), 0, 1)
        projection = a[None]+fraction[..., None]*edge[None]
        error = np.min(np.linalg.norm(query[:, None]-projection, axis=2), axis=1)
        membership_error = max(membership_error, float(error.max()))
        sf = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p[s], axis=0), axis=1))]; sf /= sf[-1]
        tf = np.r_[0, np.cumsum(np.linalg.norm(np.diff(original_template[t], axis=0), axis=1))]; tf /= tf[-1]
        expected = np.column_stack([np.interp(tf, sf, p[s, k]) for k in range(3)])
        interpolation_error = max(interpolation_error, float(np.linalg.norm(query-expected, axis=1).max()))
        np.testing.assert_array_equal(output.node[[t[0], t[-1]]].cpu().numpy(), source.node[[s[0], s[-1]]].to(output.node.dtype).cpu().numpy())
    assert membership_error <= rounding, (membership_error, rounding)
    assert interpolation_error <= rounding, (interpolation_error, rounding)
    uv = d["source_uv_normalized"].astype(np.float64)
    normalized_source = d["source_node_normalized"].astype(np.float64)
    pp = np.stack((normalized_source[faces[:, 1]]-normalized_source[faces[:, 0]], normalized_source[faces[:, 2]]-normalized_source[faces[:, 0]]), axis=2)
    qq = np.stack((uv[faces[:, 1]]-uv[faces[:, 0]], uv[faces[:, 2]]-uv[faces[:, 0]]), axis=2)
    jacobian = pp @ np.linalg.inv(qq)
    sigma = np.linalg.svd(jacobian, compute_uv=False)
    face_stretch = np.sqrt(np.mean(sigma**2, axis=1))
    area = np.linalg.norm(np.cross(pp[:, :, 0], pp[:, :, 1]), axis=1)/2
    metric = np.sqrt(np.sum(area*face_stretch**2)/area.sum())
    np.testing.assert_allclose(metric, d["stretch_history"][-1], rtol=256*eps, atol=0)
    raw_uv = np.asarray(d["source_uv"], dtype=np.float64)
    raw_p = np.stack((p[faces[:, 1]]-p[faces[:, 0]], p[faces[:, 2]]-p[faces[:, 0]]), axis=2)
    raw_q = np.stack((raw_uv[faces[:, 1]]-raw_uv[faces[:, 0]], raw_uv[faces[:, 2]]-raw_uv[faces[:, 0]]), axis=2)
    raw_sigma = np.linalg.svd(raw_p @ np.linalg.inv(raw_q), compute_uv=False)
    raw_area = np.linalg.norm(np.cross(raw_p[:, :, 0], raw_p[:, :, 1]), axis=1)/2
    raw_metric = np.sqrt(np.sum(raw_area*np.mean(raw_sigma**2, axis=1))/raw_area.sum())
    # Restoring a large origin to a small UV domain loses edge precision even
    # in float64. Bound that cancellation using coordinate magnitudes and the
    # smallest triangle singular values; keep the normalized check above tight.
    raw_frame_condition = (np.linalg.norm(np.max(abs(raw_uv), axis=0))/np.linalg.svd(raw_q, compute_uv=False)[:, -1].min()
                           + np.linalg.norm(np.max(abs(p), axis=0))/np.linalg.svd(raw_p, compute_uv=False)[:, -1].min())
    raw_metric_rtol = 256*eps + 16*np.finfo(np.float64).eps*raw_frame_condition
    np.testing.assert_allclose(raw_metric, d["stretch_history_raw"][-1], rtol=raw_metric_rtol, atol=0)
    assert np.all(np.diff(d["stretch_history"]) <= 0)
    assert d["attempted_iterations"] <= d["max_iterations"]
    assert len(d["stretch_history"]) == d["accepted_iterations"]+1
    assert len(d["candidate_stretch_history"]) == d["attempted_iterations"]
    constraints = d["constraints"]
    np.testing.assert_array_equal(d["source_uv_normalized"][constraints["source_index"]], constraints["source_uv"])
    free = np.flatnonzero(d["source_triangle_index"] >= 0)
    fi, w = d["source_triangle_index"][free], d["barycentric_weight"][free]
    xyz = p[faces[fi]]
    reconstructed = xyz[:, 0]+w[:, 1, None]*(xyz[:, 1]-xyz[:, 0])+w[:, 2, None]*(xyz[:, 2]-xyz[:, 0])
    surface_error = float(np.linalg.norm(result[free]-reconstructed, axis=1).max(initial=0))
    assert surface_error <= rounding
    assert set(d["query_status"]) <= {"inside", "roundoff_projection", "constraint"}
    if case["case_id"] == "testdata2":
        assert len(d["free_node_index"]) == 837 and len(free) == 1404
        assert len(output.node) == 1526 and len(output.element) == 1464
        assert len(output.node_set) == 13 and len(d["template_orientation_conflicts"]) == 64
    return {"boundary_membership_max_error": membership_error, "boundary_interpolation_max_error": interpolation_error,
            "surface_reconstruction_max_error": surface_error, "coordinate_tolerance": rounding,
            "svd_global_stretch": metric, "svd_global_stretch_raw": raw_metric, "raw_stretch_relative_tolerance": raw_metric_rtol,
            "source_nodes": len(source.node), "source_triangles": len(source.element),
            "output_nodes": len(output.node), "output_elements": len(output.element),
            "initial_stretch": d["stretch_history"][0], "final_stretch": d["stretch_history"][-1],
            "stretch_history": d["stretch_history"], "attempted_iterations": d["attempted_iterations"],
            "accepted_iterations": d["accepted_iterations"], "termination_reason": d["termination_reason"],
            "roundoff_projection_count": d["roundoff_projection_count"], "query_count": len(free),
            "template_orientation_conflicts": len(d["template_orientation_conflicts"])}






def prepare_segments(point_sets, named_points):
    free_edge = point_sets["LC_to_T_to_RC"]
    assert free_edge[0] == named_points["LC"]
    assert free_edge[-1] == named_points["RC"]
    assert free_edge.count(named_points["Tip"]) == 1
    tip_offset = free_edge.index(named_points["Tip"])
    assert 0 < tip_offset < len(free_edge) - 1
    right = point_sets["RC_to_H"]
    left = point_sets["LC_to_H"]
    assert right[0] == named_points["RC"] and right[-1] == named_points["Hinge"]
    assert left[0] == named_points["LC"] and left[-1] == named_points["Hinge"]
    return [free_edge[:tip_offset + 1], free_edge[tip_offset:], right, left]
