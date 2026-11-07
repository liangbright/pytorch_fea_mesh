# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 22:54:53 2022

@author: liang
"""

import os
import numpy as np
import torch
#%%
# These writers are used without VTK, or with VTK 9.0. They behave as Mesh.convert_to_vtk and
# Mesh.run_vtk_writer do: the elements and the errors come from Mesh.make_vtk_cells, which both use.
#%%
# VTK cell type of each cell name that Mesh.make_vtk_cells gives. The numbers are written
# out because this file has to work without VTK.
# VTK_POLYHEDRON (42) is left out: it needs a face stream, and VTK reads a plain list of
# nodes as one, which can give a wrong cell or crash the reader.
VTK_CELL_TYPE_BY_NAME={
    'VTK_LINE':3,
    'VTK_POLY_LINE':4,
    'VTK_TRIANGLE':5,
    'VTK_POLYGON':7,
    'VTK_QUAD':9,
    'VTK_TETRA':10,
    'VTK_HEXAHEDRON':12,
    'VTK_WEDGE':13,
    'VTK_PYRAMID':14,
    'VTK_QUADRATIC_TRIANGLE':22,
    'VTK_QUADRATIC_QUAD':23,
    'VTK_QUADRATIC_TETRA':24,
    'VTK_QUADRATIC_HEXAHEDRON':25,
    'VTK_QUADRATIC_WEDGE':26,
}
#%%
def to_numpy(data, dtype=np.float64):
    #used by Mesh.convert_to_vtk too, so both ways of saving accept the same input
    if isinstance(data, torch.Tensor):
        data=data.detach().to('cpu')
        if data.dtype == torch.bfloat16:
            data=data.float() #NumPy has no native bfloat16 dtype
        data=data.numpy()
    return np.asarray(data, dtype=dtype)
#%%
def encode_vtk_name(name):
    #the same encoding as vtkDataWriter: the reader splits a line at white space and decodes %XX
    #the same names as vtkAbstractArray.SetName: None is no name, and bytes are the name as it is
    if name is None:
        name=''
    if isinstance(name, str):
        name=name.encode('utf-8')
    if not isinstance(name, bytes):
        raise TypeError('the name of a field must be a string, not '+type(name).__name__)
    if len(name) == 0:
        return 'unknown'
    text=''
    for byte in name:
        if byte < 33 or byte > 126 or byte == ord('"') or byte == ord('%'):
            text=text+'%'+format(byte, '02X')
        else:
            text=text+chr(byte)
    return text
#%%
def append_node(out, node):
    #return the number of nodes
    node=to_numpy(node)
    if node.size == 0:
        node=node.reshape(0, 3)
    if node.ndim != 2 or node.shape[1] != 3:
        raise ValueError('node must have shape (N, 3)')
    out.append('POINTS '+str(node.shape[0])+' double'+'\n')
    for x, y, z in node.tolist():
        out.append(str(x)+' '+str(y)+' '+str(z)+'\n')
    return node.shape[0]
#%%
def append_cell(out, keyword, counts, connectivity):
    #keyword is LINES, POLYGONS or CELLS
    out.append(keyword+' '+str(len(counts))+' '+str(len(counts)+len(connectivity))+'\n')
    connectivity=connectivity.tolist()
    start=0
    for count in counts.tolist():
        out.append(' '.join(str(id) for id in [count]+connectivity[start:start+count])+'\n')
        start=start+count
#%%
def append_field(out, section, field, count):
    #section is POINT_DATA or CELL_DATA, and count is the number of nodes or elements
    if len(field) == 0:
        return
    out.append(section+' '+str(count)+'\n')
    out.append('FIELD FieldData '+str(len(field))+'\n')
    for name, data in field.items():
        data=to_numpy(data)
        if data.ndim != 2 or data.shape[0] != count or data.shape[1] < 1:
            raise ValueError('invalid data shape, name:'+name+
                             '; expected ('+str(count)+', number_of_components)')
        out.append(encode_vtk_name(name)+' '+str(data.shape[1])+' '+str(data.shape[0])+' double'+'\n')
        for row in data.tolist():
            out.append(' '.join(str(value) for value in row)+' \n')
#%%
def write_file(out, filename):
    #the same errors as Mesh.run_vtk_writer
    #open() would take a number as an open file descriptor, and write the mesh there
    if not isinstance(filename, (str, bytes, os.PathLike)):
        raise TypeError('filename must be a path, not '+type(filename).__name__)
    try:
        file=open(filename, 'w', encoding = 'utf-8')
    except OSError:
        raise RuntimeError('cannot write '+filename+': CannotOpenFileError') from None
    try:
        with file:
            file.writelines(out)
    except OSError:
        #a VTK writer also deletes the part it has written
        try:
            os.remove(filename)
        except OSError:
            pass
        raise RuntimeError('cannot write '+filename+': OutOfDiskSpaceError') from None
#%%
def save_poly_data_as_vtk(mesh, filename, keyword):
    #keyword is LINES or POLYGONS
    out=[]
    out.append('# vtk DataFile Version 4.2'+'\n')
    out.append('vtk output'+'\n')
    out.append('ASCII'+'\n')
    out.append('DATASET POLYDATA'+'\n')
    #------------------------------------------------------------------
    n_node=append_node(out, mesh.node)
    #------------------------------------------------------------------
    counts, connectivity, _, _=mesh.make_vtk_cells()
    if len(counts) > 0:
        append_cell(out, keyword, counts, connectivity)
    #------------------------------------------------------------------
    append_field(out, 'POINT_DATA', mesh.node_data, n_node)
    append_field(out, 'CELL_DATA', mesh.element_data, len(counts))
    #------------------------------------------------------------------
    write_file(out, filename)
#%%
def save_unstructured_grid_as_vtk(mesh, filename):
    out=[]
    out.append('# vtk DataFile Version 4.2'+'\n')
    out.append('vtk output'+'\n')
    out.append('ASCII'+'\n')
    out.append('DATASET UNSTRUCTURED_GRID'+'\n')
    #------------------------------------------------------------------
    n_node=append_node(out, mesh.node)
    #------------------------------------------------------------------
    counts, connectivity, cell_names, inverse=mesh.make_vtk_cells()
    append_cell(out, 'CELLS', counts, connectivity)
    #------------------------------------------------------------------
    out.append('CELL_TYPES '+str(len(counts))+'\n')
    for m in range(0, len(counts)):
        out.append(str(VTK_CELL_TYPE_BY_NAME[cell_names[inverse[m]]])+'\n')
    #------------------------------------------------------------------
    append_field(out, 'POINT_DATA', mesh.node_data, n_node)
    append_field(out, 'CELL_DATA', mesh.element_data, len(counts))
    #------------------------------------------------------------------
    write_file(out, filename)
#%%
def save_polyline_mesh_as_vtk(mesh, filename):
    save_poly_data_as_vtk(mesh, filename, 'LINES')
#%%
def save_polygon_mesh_as_vtk(mesh, filename):
    save_poly_data_as_vtk(mesh, filename, 'POLYGONS')
#%%
def save_quadratic_polygon_mesh_as_vtk(mesh, filename):
    #POLYDATA has no quadratic cells
    save_unstructured_grid_as_vtk(mesh, filename)
#%%
def save_polyhedron_mesh_as_vtk(mesh, filename):
    save_unstructured_grid_as_vtk(mesh, filename)
#%%
