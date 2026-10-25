"""Save inspectable source/output pairs using the project's mesh library."""
from pathlib import Path
from numbers import Integral

import torch
import vtk
from torch_fea_mesh.mesh.mesh import Mesh
from torch_fea_mesh.mesh.polyline_mesh import PolylineMesh
from torch_fea_mesh.io.save_point_as_vtk_file import save_point_as_vtk
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh


def save_mesh_vtk(mesh, path):
    """Write VTK and verify geometry/connectivity, including an empty result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save_as_vtk(str(path))
    if len(mesh.node) == 0:
        # The library's high-level reader rejects zero points; its underlying
        # VTK reader supports a legitimately empty remeshing result.
        restored = mesh.load_vtk_file(str(path), mesh.mesh_type)
        if restored.GetNumberOfPoints() != 0 or restored.GetNumberOfCells() != len(mesh.element):
            raise AssertionError(f'Empty VTK round trip failed: {path}')
    else:
        restored = TriangleMesh()
        restored.load_from_vtk(str(path), dtype=torch.float64)
        faces = torch.as_tensor(restored.element, dtype=torch.int64).reshape(-1, 3)
        expected_faces = torch.as_tensor(mesh.element).detach().cpu().reshape(-1, 3)
        expected_nodes = mesh.node.detach().cpu().to(torch.float64)
        if (restored.node.shape != expected_nodes.shape
                or not torch.equal(faces, expected_faces)
                or not torch.allclose(restored.node, expected_nodes, rtol=1e-10, atol=1e-10)):
            raise AssertionError(f'VTK geometry/connectivity round trip failed: {path}')
    return str(path.resolve())


def _feature_index_list(values, size):
    if hasattr(values, 'tolist'):
        values = values.tolist()
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in values):
        raise ValueError('Feature indices must be integers')
    if any(v < 0 or v >= size for v in values):
        raise ValueError('Feature index outside the source mesh')
    return list(dict.fromkeys(int(v) for v in values))


def _check_feature_vtk(path, nodes, elements, cell_type, point_ids, edge_ids=None):
    """Check coordinates, connectivity, cell types and original index labels."""
    data = Mesh.load_vtk_file(str(path), 'polyline')
    restored = torch.tensor([data.GetPoint(i) for i in range(data.GetNumberOfPoints())],
                            dtype=torch.float64).reshape(-1, 3)
    expected = nodes.to(dtype=torch.float64, device='cpu')
    if restored.shape != expected.shape or not torch.allclose(restored, expected, atol=1e-10, rtol=1e-10):
        raise AssertionError(f'Feature coordinates changed in {path}')
    if data.GetNumberOfCells() != len(elements):
        raise AssertionError(f'Feature cell count changed in {path}')
    for i, ids in enumerate(elements):
        cell = data.GetCell(i)
        if data.GetCellType(i) != cell_type or list(ids) != [
                cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]:
            raise AssertionError(f'Feature connectivity changed in {path}')
    for attributes, name, ids in [(data.GetPointData(), 'source_point_index', point_ids),
                                  (data.GetCellData(), 'source_edge_index', edge_ids)]:
        if ids is not None and len(ids):
            array = attributes.GetArray(name)
            if array is None or [int(array.GetTuple1(i)) for i in range(len(ids))] != list(ids):
                raise AssertionError(f'Feature index labels changed in {path}')


def save_feature_vtk(source_mesh, directory, *, feature_point_index_list=None,
                     feature_edge_index_list=None):
    """Export explicit source features; duplicate selections appear only once.

    Edge numbering comes from the original source table, before remeshing.
    Building a missing table uses a private mesh and never changes the source.
    None skips an export; an explicitly empty list writes an empty VTK file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    nodes = source_mesh.node.detach().cpu()
    paths = {}
    if feature_edge_index_list is not None:
        table = source_mesh.edge
        if table is None:
            if len(source_mesh.element):
                private = TriangleMesh(nodes.clone(), source_mesh.make_element_copy('numpy'))
                private.build_edge()
                table = private.edge
            else:
                table = torch.empty((0, 2), dtype=torch.int64)
        table = torch.as_tensor(table).detach().cpu()
        ids = _feature_index_list(feature_edge_index_list, len(table))
        if ids:
            lines, point_ids = PolylineMesh(nodes, table).get_sub_mesh(ids, return_node_idx_list=True)
            point_ids = point_ids.tolist()
        else:
            lines = PolylineMesh(nodes[:0].clone(), torch.empty((0, 2), dtype=torch.int64))
            point_ids = []
        lines.node_data['source_point_index'] = torch.tensor(point_ids).reshape(-1, 1)
        lines.element_data['source_edge_index'] = torch.tensor(ids).reshape(-1, 1)
        path = directory / 'feature_edges.vtk'
        lines.save_as_vtk(str(path))
        _check_feature_vtk(path, lines.node, lines.element.tolist(), vtk.VTK_LINE, point_ids, ids)
        paths['feature_edges_file'] = str(path.resolve())
    if feature_point_index_list is not None:
        ids = _feature_index_list(feature_point_index_list, len(nodes))
        points = nodes[ids]
        path = directory / 'feature_points.vtk'
        save_point_as_vtk(points, str(path))
        # The library writes coordinates only. Add VTK vertex cells so the
        # points display directly in ParaView, plus their original source IDs.
        data = Mesh.load_vtk_file(str(path), 'polyline')
        vertices = vtk.vtkCellArray()
        labels = vtk.vtkIdTypeArray()
        labels.SetName('source_point_index')
        for i, source_id in enumerate(ids):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(i)
            labels.InsertNextValue(source_id)
        data.SetVerts(vertices)
        data.GetPointData().AddArray(labels)
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(str(path))
        writer.SetInputData(data)
        writer.SetFileVersion(42)
        writer.SetFileTypeToBinary()
        if writer.Write() != 1:
            raise OSError(f'Could not save feature points: {path}')
        _check_feature_vtk(path, points, [[i] for i in range(len(ids))], vtk.VTK_VERTEX, ids)
        paths['feature_points_file'] = str(path.resolve())
    return paths


def save_mesh_pair(source_mesh, output_mesh, directory, *, feature_point_index_list=None,
                   feature_edge_index_list=None):
    """Save the triangle input/output and optional feature selections together."""
    directory = Path(directory)
    return dict(
        source_mesh_file=save_mesh_vtk(source_mesh, directory / 'source_mesh.vtk'),
        output_mesh_file=save_mesh_vtk(output_mesh, directory / 'output_mesh.vtk'),
        **save_feature_vtk(source_mesh, directory,
            feature_point_index_list=feature_point_index_list,
            feature_edge_index_list=feature_edge_index_list),
    )
