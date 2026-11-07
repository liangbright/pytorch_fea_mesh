import itertools
import operator
import torch
from torch.linalg import vector_norm as norm
import numpy as np
from copy import deepcopy, copy as shallowcopy
from ..io.save_mesh_as_vtk_file import (save_polyline_mesh_as_vtk, save_polygon_mesh_as_vtk, save_quadratic_polygon_mesh_as_vtk,
                                        save_polyhedron_mesh_as_vtk, to_numpy)
from ..io.check_legacy_vtk_file import find_cut_in_legacy_vtk_file
import os
import re
_Flag_VTK_IMPORT_=False
try:
    import vtk
    _Flag_VTK_IMPORT_=True
except Exception as e:
    # vtk is optional, so a failed import only turns off the vtk-based functions
    print('cannot import vtk: '+str(e))
#%%
# Nodes per element of the classes that allow only one element shape. An empty
# element array needs this width, because geometry code indexes element[:,k].
ELEMENT_WIDTH_BY_CLASS={'TriangleMesh':3, 'QuadMesh':4, 'Tri6Mesh':6,
                         'TetrahedronMesh':4, 'Tet10Mesh':10, 'HexahedronMesh':8}

# element_type label: (name of the VTK cell type, nodes per element)
VTK_CELL_BY_ELEMENT_TYPE={
    # Abaqus solid
    'C3D4':('VTK_TETRA',4), 'C3D5':('VTK_PYRAMID',5), 'C3D6':('VTK_WEDGE',6),
    'C3D8':('VTK_HEXAHEDRON',8), 'C3D10':('VTK_QUADRATIC_TETRA',10),
    'C3D15':('VTK_QUADRATIC_WEDGE',15), 'C3D20':('VTK_QUADRATIC_HEXAHEDRON',20),
    # Abaqus shell, membrane and rigid surface
    'S3':('VTK_TRIANGLE',3), 'S4':('VTK_QUAD',4), 'S8':('VTK_QUADRATIC_QUAD',8),
    'STRI3':('VTK_TRIANGLE',3), 'STRI65':('VTK_QUADRATIC_TRIANGLE',6),
    'M3D3':('VTK_TRIANGLE',3), 'M3D4':('VTK_QUAD',4),
    'M3D6':('VTK_QUADRATIC_TRIANGLE',6), 'M3D8':('VTK_QUADRATIC_QUAD',8),
    'R3D3':('VTK_TRIANGLE',3), 'R3D4':('VTK_QUAD',4),
    # Abaqus truss and beam
    'T3D2':('VTK_LINE',2), 'B31':('VTK_LINE',2),
    # FEBio, and the names used in mesh_type
    'TET4':('VTK_TETRA',4), 'PYRA5':('VTK_PYRAMID',5), 'PENTA6':('VTK_WEDGE',6),
    'WEDGE6':('VTK_WEDGE',6), 'HEX8':('VTK_HEXAHEDRON',8),
    'TET10':('VTK_QUADRATIC_TETRA',10), 'HEX20':('VTK_QUADRATIC_HEXAHEDRON',20),
    'WEDGE15':('VTK_QUADRATIC_WEDGE',15),
    'TRI3':('VTK_TRIANGLE',3), 'QUAD4':('VTK_QUAD',4),
    'TRI6':('VTK_QUADRATIC_TRIANGLE',6), 'QUAD8':('VTK_QUADRATIC_QUAD',8),
    'LINE2':('VTK_LINE',2),
}
# The solid cells a polyhedron mesh holds, and the label a reader gives each. A VTK file
# stores the shape of a cell but not its formulation, so the label names only the shape:
# a VTK_HEXAHEDRON may have been a C3D8 or a C3D8R.
ELEMENT_TYPE_BY_VTK_CELL={
    'VTK_TETRA':'TET4', 'VTK_PYRAMID':'PYRA5', 'VTK_WEDGE':'WEDGE6', 'VTK_HEXAHEDRON':'HEX8',
    'VTK_QUADRATIC_TETRA':'TET10', 'VTK_QUADRATIC_WEDGE':'WEDGE15',
    'VTK_QUADRATIC_HEXAHEDRON':'HEX20',
    # A voxel is a hex once select_vtk_cells has put its nodes in hex order.
    'VTK_VOXEL':'HEX8',
}
# A voxel or pixel, which VTK uses for image data, lists its corners in grid order: x, then y,
# then z. A hex or quad goes around each face. This gives the hex or quad order of its nodes.
NODE_ORDER_OF_GRID_CELL={'VTK_PIXEL':[0,1,3,2], 'VTK_VOXEL':[0,1,3,2,4,5,7,6]}
# The 2D VTK cells whose nodes go around the boundary, as in a linear polygon mesh. Any other
# 2D cell, such as VTK_QUADRATIC_TRIANGLE, lists its corners first.
LINEAR_POLYGON_VTK_CELLS=('VTK_TRIANGLE', 'VTK_TRIANGLE_STRIP', 'VTK_PIXEL', 'VTK_QUAD', 'VTK_POLYGON')
# An Abaqus name adds letters that do not change the cell: C3D8R is C3D8, S4R5 is S4.
ABAQUS_NAME_WITH_SUFFIX=re.compile(r'((?:C3D|M3D|R3D|T3D|S|B)\d+)[A-Z]+\d*')

def find_vtk_cell_by_element_type(element_type):
    """Return (VTK cell type name, nodes per element), or None for an unknown label."""
    label=str(element_type).strip().upper()
    match=ABAQUS_NAME_WITH_SUFFIX.fullmatch(label)
    if match is not None:
        label=match.group(1)
    return VTK_CELL_BY_ELEMENT_TYPE.get(label)

def get_dimension_of_vtk_cell_name(name):
    # Read from the tables, not from vtk, because a mesh can be built without vtk.
    if name in ELEMENT_TYPE_BY_VTK_CELL:
        return 3
    return 1 if name == 'VTK_LINE' else 2

def get_dimension_of_mesh_type(mesh_type):
    """Return the dimension of a mesh's elements: polyline 1, polygon 2, polyhedron 3.

    A mesh never mixes dimensions. Shapes of one dimension may mix, such as hex8 and tet4.
    Every save and load function in Mesh follows this rule, except save_as_torch and
    load_from_torch, which store and restore a mesh exactly as it is.
    """
    if 'polyline' in mesh_type:
        return 1
    if 'polygon' in mesh_type:
        return 2
    return 3
#%%
class Mesh:
    def __init__(self, node, element, element_type, mesh_type):
        if (('polyhedron' not in mesh_type)
            and ('polygon' not in mesh_type)
            and ('polyline' not in mesh_type)):
            raise ValueError('unknown mesh_type: '+mesh_type)
        #--------------------------------------------------------------------
        self.mesh_type=mesh_type
        self.clear()
        if node is not None:
            self.node=Mesh.convert_node(node)
        #--------------------------------------------------------------------
        if element is not None:
            self.element=self.convert_element(element, None if self.node is None else self.node.device)
        #--------------------------------------------------------------------
        if element_type is not None:
            if isinstance(element_type, np.ndarray):
                pass
            elif isinstance(element_type, list):
                element_type=np.asarray(element_type)
            elif isinstance(element_type, torch.Tensor):
                element_type=element_type.detach().cpu().numpy()
            else:
                raise ValueError('unsupported python-object type of element_type')
            self.element_type=element_type
            self.check_element_type(element_type)
        #--------------------------------------------------------------------

    def check_element_type(self, element_type):
        """Raise if a label has another dimension than this mesh, or would mix polygon orders.

        An unknown label is only metadata, so it is allowed. Without this check, a 4-node
        shell in a polyhedron mesh would become a tet on any path that loses the label,
        such as copy, and a C3D6 label on a Tri6Mesh would be saved as a wedge.
        A polygon mesh never mixes linear and quadratic elements. A polygon's nodes go around
        its boundary, and a TRI6 element lists its corners first, so a TRI6 element in a
        linear polygon mesh would get wrong edges, boundary and area.
        """
        if element_type is None:
            return
        dimension=get_dimension_of_mesh_type(self.mesh_type)
        is_quadratic_mesh=('tri6' in self.mesh_type) or ('quad8' in self.mesh_type)
        for label in np.unique(np.asarray(element_type).astype(str)):
            found=find_vtk_cell_by_element_type(label)
            if found is None:
                continue
            if get_dimension_of_vtk_cell_name(found[0]) != dimension:
                raise ValueError('a '+self.mesh_type+' mesh holds only '+str(dimension)
                                 +'D elements, but element_type has '+str(label))
            if dimension == 2 and ('QUADRATIC' in found[0]) != is_quadratic_mesh:
                raise ValueError('a '+self.mesh_type+' mesh holds only '
                                 +('quadratic' if is_quadratic_mesh else 'linear')
                                 +' elements, because a polygon mesh never mixes linear and quadratic'
                                 +' elements, but element_type has '+str(label))

    def check_element_dimension(self, element, element_type):
        """Raise if element or element_type would give this mesh another dimension.

        An element needs more nodes than its dimension, so 2 nodes in a polygon mesh
        make a line. A 4-node element with no label in a polyhedron mesh is a tet.
        """
        self.check_element_type(element_type)
        if element is None or len(element) == 0:
            return
        if torch.is_tensor(element) or (isinstance(element, np.ndarray) and element.dtype != object):
            if len(element.shape) != 2:
                raise ValueError('invalid element.shape')
            smallest=element.shape[1]
        else:
            smallest=min(len(e) for e in element)
        dimension=get_dimension_of_mesh_type(self.mesh_type)
        if smallest <= dimension:
            raise ValueError('a '+self.mesh_type+' mesh holds only '+str(dimension)
                             +'D elements, but an element has '+str(smallest)+' nodes')

    @staticmethod
    def convert_node(node, dtype=None):
        """Return node as a 2D tensor, for the constructor and copy. A tensor is returned as it is.

        A list becomes float64. A numpy array becomes float64 if it is float64, otherwise float32.
        dtype, if it is given, replaces these for a list or a numpy array.
        """
        if torch.is_tensor(node):
            pass
        elif isinstance(node, (list, tuple)):
            node=torch.tensor(node, dtype=torch.float64 if dtype is None else dtype)
        elif isinstance(node, np.ndarray):
            if dtype is None:
                dtype=torch.float64 if node.dtype == np.float64 else torch.float32
            node=torch.tensor(node, dtype=dtype)
        else:
            raise ValueError('unsupported python-object type of node')
        node=Mesh.reshape_empty_node(node)
        if len(node.shape) != 2:
            raise ValueError('invalid self.node.shape')
        return node

    @staticmethod
    def check_element_array_is_integer(array):
        """Raise if a tensor or a numpy array holds float or bool. With no values, any dtype is fine."""
        if torch.is_tensor(array):
            is_integer=not (array.is_floating_point() or array.is_complex() or array.dtype == torch.bool)
        else:
            is_integer=array.dtype.kind in 'iu'
        if not is_integer and len(array.reshape(-1)) > 0:
            raise ValueError('element must hold integers, but its dtype is '+str(array.dtype))

    @staticmethod
    def make_python_int(value):
        """Return an integer of any type as a python int. A float or a bool raises ValueError."""
        if isinstance(value, bool) or (torch.is_tensor(value) and value.dtype == torch.bool):
            raise ValueError('element must hold integers, but found '+repr(value)[:40])
        try:
            return operator.index(value)
        except TypeError:
            raise ValueError('element must hold integers, but found '+repr(value)[:40]) from None

    @staticmethod
    def make_nested_list_of_int(element):
        """Return rows as a nested list of python int. If element already is one, it is returned as it is."""
        #this comes first because it is the usual case, and these loops run in C
        if set(map(type, element)) <= {list} and set(map(type, itertools.chain.from_iterable(element))) <= {int}:
            return element if isinstance(element, list) else list(element)
        rows=[]
        for row in element:
            if torch.is_tensor(row) or isinstance(row, np.ndarray):
                if len(row.shape) != 1:
                    raise ValueError('invalid self.element.shape')
                Mesh.check_element_array_is_integer(row)
                rows.append(row.tolist())
            elif isinstance(row, (list, tuple)):
                rows.append([Mesh.make_python_int(value) for value in row])
            else:
                raise ValueError('invalid self.element.shape')
        return rows

    def convert_element(self, element, device=None):
        """Return element as self.element must hold it, for the constructor and copy.

        self.element is an int64 tensor, or a nested list of python int. Any integer type is converted,
        and a float or a bool raises ValueError. So this rule covers what the class sets: the constructor,
        copy, the readers and load_from_torch. It can't cover a later mesh.element=... or mesh.element.append(...).
        Elements with different numbers of nodes stay a nested list.
        A list or a numpy array becomes a tensor on device, which should be the device of the node.
        A tensor stays on its device.
        """
        if isinstance(element, np.ndarray) and element.dtype == object:
            element=element.tolist()
        if torch.is_tensor(element):
            Mesh.check_element_array_is_integer(element)
            element=element.to(torch.int64)
        elif isinstance(element, np.ndarray):
            Mesh.check_element_array_is_integer(element)
            element=torch.tensor(element, dtype=torch.int64, device=device)
        elif isinstance(element, (list, tuple)):
            element=Mesh.make_nested_list_of_int(element)
            try:
                element=torch.tensor(element, dtype=torch.int64, device=device)
            except (ValueError, TypeError, RuntimeError):
                pass
        else:
            raise ValueError('unsupported python-object type of element')
        if torch.is_tensor(element):
            element=self.reshape_empty_element(element)
            if len(element.shape) != 2:
                raise ValueError('invalid self.element.shape')
        self.check_element_width(element)
        return element

    def get_element_width_of_class(self):
        """Return the nodes per element of a class that allows only one shape, otherwise None."""
        for cls in type(self).__mro__:
            if cls.__name__ in ELEMENT_WIDTH_BY_CLASS:
                return ELEMENT_WIDTH_BY_CLASS[cls.__name__]
        return None

    @staticmethod
    def find_element_width(element):
        """Return the set of the numbers of nodes in the elements."""
        if torch.is_tensor(element) or (isinstance(element, np.ndarray) and element.dtype != object):
            return {element.shape[1]} if len(element.shape) == 2 else {None}
        return {len(e) for e in element}

    def check_element_width(self, element):
        """Raise if element does not fit a class that allows only one element shape.

        Geometry code indexes element[:,k], so a quad in a TriangleMesh would silently lose
        a corner and a hex in a TetrahedronMesh would give the volume of a wrong tet.
        """
        width=self.get_element_width_of_class()
        if width is None or element is None:
            return
        found=Mesh.find_element_width(element)
        if len(found-{width}) > 0:
            raise ValueError('a '+type(self).__name__+' element must have '+str(width)
                             +' nodes, but found '+str(sorted(found-{width}, key=str)))

    def make_empty_element(self):
        """Return an empty int64 element tensor of shape (0, nodes per element)."""
        width=self.get_element_width_of_class()
        if width is not None:
            return torch.empty((0, width), dtype=torch.int64)
        # No fixed shape: use the simplest element of this dimension.
        if 'polyline' in self.mesh_type:
            width=2
        elif 'polygon' in self.mesh_type:
            width=3
        else:
            width=4
        return torch.empty((0, width), dtype=torch.int64)

    def reshape_empty_element(self, element):
        # torch.tensor([]) has shape (0,), but element[:,k] needs shape (0,n).
        if torch.is_tensor(element) and element.numel() == 0 and len(element.shape) == 1:
            return self.make_empty_element().to(element.device)
        return element

    @staticmethod
    def reshape_empty_node(node):
        # torch.tensor([]) has shape (0,), but node[:,k] needs shape (0,3).
        if torch.is_tensor(node) and node.numel() == 0 and len(node.shape) == 1:
            return node.reshape(0,3)
        return node

    def clear(self):
        """Reset to an empty mesh, preserving mesh_type."""
        self.name=''
        self.node=None
        self.element=None
        self.element_type=None
        self.mapping_from_node_name_to_index={}
        self.node_set={}
        self.element_set={}
        self.node_data={}
        self.element_data={}
        self.mesh_data={}
        self.clear_geometry_derived_data()

    def clear_geometry_derived_data(self, clear_adjacency_data=True):
        """Clear derived values and, by default, adjacency; retain geometry and user data.

        Pass clear_adjacency_data=False after moving nodes without changing elements:
        the adjacency tables stay valid, and they are slow to build again.
        """
        if clear_adjacency_data == True:
            self.clear_adj_info()
        self.edge_length=None

    def clear_adj_info(self):
        #if a new node or a new element is added/deleted after a mesh is created,
        #then the adj info will become invalid and therefore need to be cleared
        self.edge=None #only one undirected edge between two adj nodes
        self.map_node_pair_to_edge=None#sparse matrix: self.map_node_pair_to_edge[i,j] - 1 is index (>=0) in self.edge
        self.node_to_node_adj_link=None#similar to edge_index in pytorch geometric
        self.node_to_node_adj_table=None
        self.node_to_edge_adj_table=None
        self.node_to_element_adj_table=None
        self.edge_to_edge_adj_table=None
        self.edge_to_element_adj_table=None# an edge of an element
        self.element_to_edge_adj_table=None# an edge of an element
        if 'polyline' in self.mesh_type:
            self.element_to_element_adj_link={'node':None, 'edge':None}
            self.element_to_element_adj_table={'node':None, 'edge':None}
        elif 'polygon' in self.mesh_type:
            self.element_to_element_adj_link={'node':None, 'edge':None}
            self.element_to_element_adj_table={'node':None, 'edge':None}
        elif 'polyhedron' in self.mesh_type:
            self.face=None
            self.face_to_element_adj_table=None
            self.element_to_face_adj_table=None
            self.element_to_element_adj_link={'node':None, 'edge':None, 'face':None}
            self.element_to_element_adj_table={'node':None, 'edge':None, 'face':None}

    def load_from_stl(self, filename, dtype):
        if _Flag_VTK_IMPORT_ == False:
            raise ValueError('vtk is not imported')
        if not os.path.isfile(filename):
            raise ValueError('not exist: '+filename)
        if isinstance(dtype, str):
            if dtype == 'float32':
                dtype=torch.float32
            elif dtype == 'float64':
                dtype=torch.float64
            else:
                raise ValueError('unsupported dtype:'+str(dtype))
        if ('polygon' in self.mesh_type) and ('tri6' not in self.mesh_type) and  ('quad8' not in self.mesh_type):
            reader = vtk.vtkSTLReader()
        else:
            raise ValueError('unsupported mesh_type:'+self.mesh_type)
        reader.SetFileName(filename)
        reader.Update()
        mesh_vtk = reader.GetOutput()
        self.read_mesh_vtk(mesh_vtk, dtype)
    
    def load_from_vtp(self, filename, dtype):
        if _Flag_VTK_IMPORT_ == False:
            raise ValueError('vtk is not imported')
        if not os.path.isfile(filename):
            raise ValueError('not exist: '+filename)
        if isinstance(dtype, str):
            if dtype == 'float32':
                dtype=torch.float32
            elif dtype == 'float64':
                dtype=torch.float64
            else:
                raise ValueError('unsupported dtype:'+str(dtype))
        if ('polygon' in self.mesh_type) and ('tri6' not in self.mesh_type) and  ('quad8' not in self.mesh_type):
            reader = vtk.vtkXMLPolyDataReader()
        else:
            raise ValueError('unsupported mesh_type:'+self.mesh_type)
        reader.SetFileName(filename)
        reader.Update()
        mesh_vtk = reader.GetOutput()
        # A file that VTK cannot parse has no vtkPoints, so it still fails below.
        if mesh_vtk.GetPoints() is not None and mesh_vtk.GetNumberOfPoints() == 0:
            self.copy(torch.empty((0,3), dtype=dtype), [])
            return
        try:
            self.read_mesh_vtk(mesh_vtk, dtype)
        except Exception as e:
            raise ValueError('cannot convert vtk data to mesh: '+str(e)) from e

    @staticmethod
    def is_empty_legacy_vtk_file(filename, mesh_vtk):
        """Tell a legacy VTK file that declares zero points from an unreadable one.

        VTK returns an empty dataset in both cases, without an error code that
        separates them, so read what the file declares.
        """
        if mesh_vtk.GetNumberOfPoints() != 0:
            return False
        with open(filename, 'rb') as file:
            text=file.read()
        dataset=b'UNSTRUCTURED_GRID' if mesh_vtk.IsA('vtkUnstructuredGrid') else b'POLYDATA'
        return (text.startswith(b'# vtk DataFile')
                and re.search(rb'^DATASET\s+'+dataset+rb'\s', text, re.M|re.I) is not None
                and re.search(rb'^POINTS\s+0\s', text, re.M|re.I) is not None)

    @staticmethod
    def load_vtk_file(filename, mesh_type):
        if _Flag_VTK_IMPORT_ == False:
            raise ValueError('vtk is not imported')
        if not os.path.isfile(filename):
            raise ValueError('not exist: '+filename)        
        if 'polyhedron' in mesh_type:
            reader = vtk.vtkUnstructuredGridReader()
        elif 'polygon' in mesh_type:
            if ('tri6' in mesh_type) or ('quad8' in mesh_type):
                reader = vtk.vtkUnstructuredGridReader()
            else:
                reader = vtk.vtkPolyDataReader()
        elif 'polyline' in mesh_type:
            reader = vtk.vtkPolyDataReader()
        else:
            raise ValueError('unsupported mesh_type:'+mesh_type)
        cut=find_cut_in_legacy_vtk_file(filename)
        if cut is not None:
            raise ValueError('the file ends too early: '+cut+': '+filename)
        reader.SetFileName(filename)
        reader.Update()
        mesh_vtk = reader.GetOutput()
        return mesh_vtk
    
    def load_from_vtk(self, filename, dtype, return_mesh_vtk=False):
        if not os.path.isfile(filename):
            raise ValueError('not exist: '+filename)        
        if isinstance(dtype, str):
            if dtype == 'float32':
                dtype=torch.float32
            elif dtype == 'float64':
                dtype=torch.float64
            else:
                raise ValueError('unsupported dtype:'+str(dtype))
        mesh_vtk=Mesh.load_vtk_file(filename, self.mesh_type)
        if Mesh.is_empty_legacy_vtk_file(filename, mesh_vtk):
            self.copy(torch.empty((0,3), dtype=dtype), [])
        else:
            try:
                self.read_mesh_vtk(mesh_vtk, dtype)
            except Exception as e:
                raise ValueError('cannot convert vtk data to mesh: '+str(e)) from e
        if return_mesh_vtk == True:
            return mesh_vtk

    @staticmethod
    def get_vtk_cell_dimension(cell_type):
        cell=vtk.vtkGenericCell()
        cell.SetCellType(int(cell_type))
        return cell.GetCellDimension()

    @staticmethod
    def reorder_voxel_and_pixel_node(counts, connectivity, cell_types):
        """Return connectivity with voxels and pixels in the node order of a hex and a quad.

        In grid order, the faces of an element would cross and its volume or area would be 0.
        The result is a copy if anything changes, because connectivity can share memory with
        the VTK data. A cell that has the wrong number of nodes for its type is left alone.
        """
        for name, order in NODE_ORDER_OF_GRID_CELL.items():
            is_cell=(cell_types == getattr(vtk, name)) & (counts == len(order))
            if is_cell.any():
                start=np.concatenate(([0], np.cumsum(counts)[:-1]))[is_cell].reshape(-1,1)
                old=connectivity
                connectivity=old.copy()
                connectivity[start+np.arange(len(order))]=old[start+np.array(order)]
        return connectivity

    def select_vtk_cells(self, counts, connectivity, cell_types):
        """Keep the VTK cells this mesh can hold, split triangle strips, and reorder voxels and pixels.

        A mesh holds cells of one dimension: polyline 1, polygon 2, polyhedron 3.
        A cell with too few nodes for that dimension, such as a 1-point line, is dropped.
        counts[k] and cell_types[k] are the node count and VTK type of cell k.
        Returns (counts, connectivity, source), where source[m] is the VTK cell
        that element m came from. This keeps CellData aligned with the elements.
        source is None when every cell was kept unchanged.
        """
        connectivity=Mesh.reorder_voxel_and_pixel_node(counts, connectivity, cell_types)
        dimension=get_dimension_of_mesh_type(self.mesh_type)
        # A lookup table per distinct cell type, because this runs for every mesh that is read.
        table=np.array([count > 0 and Mesh.get_vtk_cell_dimension(cell_type) == dimension
                        for cell_type, count in enumerate(np.bincount(cell_types))], dtype=bool)
        keep=table[cell_types] & (counts > dimension)
        is_strip=keep & (cell_types == vtk.VTK_TRIANGLE_STRIP)
        if keep.all() and not is_strip.any():
            return counts, connectivity, None
        if not keep.all():
            print('Mesh read_mesh_vtk: dropped '+str(int((~keep).sum()))+' cells that are not '
                  +str(dimension)+'D')
        n_output=np.where(is_strip, counts-2, keep.astype(np.int64))
        source=np.repeat(np.arange(len(counts), dtype=np.int64), n_output)
        new_counts=np.where(is_strip[source], 3, counts[source])
        if is_strip.any():
            offsets=np.concatenate(([0], np.cumsum(counts)))
            pieces=[]
            for k in np.flatnonzero(keep):
                cell=connectivity[offsets[k]:offsets[k+1]]
                if is_strip[k]:
                    triangle=np.stack((cell[:-2], cell[1:-1], cell[2:]), axis=1)
                    # Every second triangle is flipped to keep one orientation,
                    # the same order as vtkTriangleStrip::DecomposeStrip.
                    triangle[1::2,[0,1]]=triangle[1::2,[1,0]]
                    cell=triangle.reshape(-1)
                pieces.append(cell)
            connectivity=np.concatenate(pieces) if pieces else connectivity[:0]
        else:
            connectivity=connectivity[np.repeat(keep, counts)]
        return new_counts, connectivity, source

    @staticmethod
    def get_element_type_of_vtk_cell(cell_type):
        """Return the label of a solid VTK cell type, or None if it has no label."""
        for name, label in ELEMENT_TYPE_BY_VTK_CELL.items():
            if getattr(vtk, name) == cell_type:
                return label
        return None

    def check_vtk_cell_order(self, cell_types, source):
        """Raise if a polygon mesh would read the nodes of a VTK cell in the wrong order.

        VTK data has cell types and no labels, so check_element_type can't see this. The rule
        is the same: a polygon mesh never mixes linear and quadratic elements. Without this
        check, a 6-node VTK_POLYGON in a Tri6Mesh gets a wrong area, and a VTK_QUADRATIC_TRIANGLE
        in a linear polygon mesh gets wrong edges, both with no error.
        source is from select_vtk_cells, so a cell that was dropped is not checked.
        """
        if get_dimension_of_mesh_type(self.mesh_type) != 2:
            return
        if source is not None:
            cell_types=cell_types[source]
        if ('tri6' in self.mesh_type) or ('quad8' in self.mesh_type):
            names=[]
            if 'tri6' in self.mesh_type:
                names.append('VTK_QUADRATIC_TRIANGLE')
            if 'quad8' in self.mesh_type:
                names.append('VTK_QUADRATIC_QUAD')
        else:
            names=LINEAR_POLYGON_VTK_CELLS
        allowed=[getattr(vtk, name) for name in names]
        for cell_type in np.flatnonzero(np.bincount(cell_types)).tolist():
            if cell_type not in allowed:
                # The number is the one in the CELL_TYPES section of a legacy VTK file.
                raise ValueError('a '+self.mesh_type+' mesh cannot read VTK cell type '+str(cell_type)
                                 +': its node order is not the order of this mesh,'
                                 +' and a polygon mesh never mixes linear and quadratic elements')

    def make_element_type_from_vtk_cells(self, cell_types, source):
        """Return a label for each element read from VTK cells, or None if no label is needed.

        Saving a mesh with no labels takes each cell type from the node count. So labels
        are set only if the node count gives no cell type for a cell in this file, such as
        a hex in a mesh_type with no 'hex8'. All other meshes keep element_type=None, as
        merge_mesh and others expect after a read. source is from select_vtk_cells.
        """
        if 'polyhedron' not in self.mesh_type:
            return None
        if source is not None:
            cell_types=cell_types[source]
        if len(cell_types) == 0:
            return None
        present=np.flatnonzero(np.bincount(cell_types))
        labels=np.full(int(present[-1])+1, None, dtype=object)
        needed=False
        for cell_type in present.tolist():
            label=Mesh.get_element_type_of_vtk_cell(cell_type)
            if label is None:
                continue
            labels[cell_type]=label
            try:
                Mesh.get_vtk_cell_type(self.mesh_type, VTK_CELL_BY_ELEMENT_TYPE[label][1])
            except ValueError:
                needed=True
        return labels[cell_types] if needed else None

    def make_element_from_vtk_cells(self, counts, connectivity):
        if len(counts) == 0:
            return self.make_empty_element()
        if np.all(counts == counts[0]):
            return torch.tensor(connectivity.reshape(len(counts), int(counts[0])),
                                dtype=torch.int64)
        offsets=np.empty(len(counts)+1, dtype=np.int64)
        offsets[0]=0
        np.cumsum(counts, out=offsets[1:])
        indices=connectivity.tolist()
        return [indices[start:end] for start, end in zip(offsets[:-1], offsets[1:])]

    def read_mesh_vtk(self, mesh_vtk, dtype):
        """Bulk-read polydata/unstructured geometry and numeric data arrays.

        Uniform elements become an int64 tensor; ragged elements remain lists.
        Cells of another dimension are dropped, triangle strips are split into
        triangles, and voxels and pixels get the node order of a hex and a quad;
        see select_vtk_cells. Nodes are never removed.
        A polygon mesh refuses a cell with another node order; see check_vtk_cell_order.
        element_type is None unless saving a solid needs it;
        see make_element_type_from_vtk_cells.
        Non-numeric arrays, such as vtkStringArray, are skipped.
        Coordinates and fields use the requested dtype and own their storage.
        Reading replaces existing geometry, metadata and caches.
        Other VTK dataset classes, such as vtkImageData, raise ValueError.
        """
        if not (mesh_vtk.IsA('vtkPolyData') or mesh_vtk.IsA('vtkUnstructuredGrid')):
            raise ValueError('read_mesh_vtk supports vtkPolyData and vtkUnstructuredGrid, not '+mesh_vtk.GetClassName())
        from vtk.util.numpy_support import vtk_to_numpy

        def copy_numeric_array(vtk_array):
            if vtk_array.GetDataType() == vtk.VTK_BIT:
                # vtk_to_numpy mishandles multi-component packed bit arrays
                # in some VTK versions. Expand them in VTK, without scalar calls.
                expanded=vtk.vtkUnsignedCharArray()
                expanded.DeepCopy(vtk_array)
                vtk_array=expanded
            data=vtk_to_numpy(vtk_array)
            # 64-bit integer coordinates/fields go through float64, as the
            # scalar VTK getters do, because they return doubles.
            if data.dtype.kind in 'iu' and data.dtype.itemsize == 8:
                data=data.astype(np.float64)
            data=data.reshape(vtk_array.GetNumberOfTuples(),
                              vtk_array.GetNumberOfComponents())
            return torch.tensor(data, dtype=dtype)

        def read_cell(cells):
            if hasattr(cells, 'GetOffsetsArray'):
                offsets=vtk_to_numpy(cells.GetOffsetsArray())
                connectivity=vtk_to_numpy(cells.GetConnectivityArray())
                return np.diff(offsets), connectivity
            # VTK 8 exposes packed [count, ids..., count, ids...] connectivity.
            packed=vtk_to_numpy(cells.GetData())
            counts=np.empty(cells.GetNumberOfCells(), dtype=np.int64)
            is_node=np.ones(len(packed), dtype=bool)
            position=0
            for i in range(len(counts)):
                counts[i]=packed[position]
                is_node[position]=False
                position+=int(counts[i])+1
            return counts, packed[is_node]

        if mesh_vtk.GetNumberOfPoints() == 0:
            raise ValueError('cannot load node')
        node=copy_numeric_array(mesh_vtk.GetPoints().GetData())

        if mesh_vtk.IsA('vtkPolyData'):
            # This is VTK's global cell order, also used by CellData tuples.
            cell_arrays=((mesh_vtk.GetVerts(), vtk.VTK_VERTEX),
                         (mesh_vtk.GetLines(), vtk.VTK_LINE),
                         (mesh_vtk.GetPolys(), vtk.VTK_POLYGON),
                         (mesh_vtk.GetStrips(), vtk.VTK_TRIANGLE_STRIP))
        else:
            cell_arrays=((mesh_vtk.GetCells(), None),)
        blocks=[read_cell(cells)+(cell_type,) for cells, cell_type in cell_arrays
                if cells is not None and cells.GetNumberOfCells()]
        source=None
        element_type=None
        if not blocks:
            element=self.make_empty_element()
        else:
            if len(blocks) == 1:
                counts, connectivity=blocks[0][:2]
            else:
                counts=np.concatenate([block[0] for block in blocks])
                connectivity=np.concatenate([block[1] for block in blocks])
            if mesh_vtk.IsA('vtkPolyData'):
                cell_types=np.repeat([block[2] for block in blocks],
                                     [len(block[0]) for block in blocks])
            elif Mesh.get_vtk_version() >= (9,6):
                cell_types=vtk_to_numpy(mesh_vtk.GetCellTypes())
            else:
                # Deprecated in VTK 9.6, where GetCellTypes() replaces it.
                cell_types=vtk_to_numpy(mesh_vtk.GetCellTypesArray())
            counts, connectivity, source=self.select_vtk_cells(counts, connectivity, cell_types)
            self.check_vtk_cell_order(cell_types, source)
            element=self.make_element_from_vtk_cells(counts, connectivity)
            element_type=self.make_element_type_from_vtk_cells(cell_types, source)
        self.check_element_width(element)

        node_data={}
        element_data={}
        for attributes, fields, index in ((mesh_vtk.GetPointData(), node_data, None),
                                          (mesh_vtk.GetCellData(), element_data, source)):
            for i in range(attributes.GetNumberOfArrays()):
                vtk_array=attributes.GetArray(i)
                if vtk_array is None:
                    # GetArray returns None for vtkStringArray and vtkVariantArray.
                    print('Mesh read_mesh_vtk: skipped non-numeric array '
                          +str(attributes.GetAbstractArray(i).GetName()))
                    continue
                data=copy_numeric_array(vtk_array)
                if index is not None:
                    data=data[torch.as_tensor(index, dtype=torch.int64)]
                fields[vtk_array.GetName()]=data
        self.clear()
        self.node=node
        self.element=element
        self.element_type=element_type
        self.node_data=node_data
        self.element_data=element_data

    @staticmethod
    def get_vtk_cell_name(mesh_type, n_node):
        #the name of the VTK cell, such as 'VTK_TETRA'. It needs no VTK, so both ways of saving use this rule
        #n_nodes: the number of nodes in an element
        #assume all elements have the same type
        if 'polyhedron' in mesh_type:
            if n_node < 4:
                raise ValueError('wrong number of nodes')
            elif n_node == 4:
                cell_name='VTK_TETRA'
            elif (n_node == 6) and ('wedge6' in mesh_type): 
                cell_name='VTK_WEDGE'
            elif (n_node == 8) and ('hex8' in mesh_type):
                cell_name='VTK_HEXAHEDRON'
            elif (n_node == 10) and ('tet10' in mesh_type):
                cell_name='VTK_QUADRATIC_TETRA'
            else:
                #cell_type=vtk.VTK_POLYHEDRON
                #print('warning: vtk may get stuck when cell_type=vtk.VTK_POLYHEDRON')
                raise ValueError('not supported: '+str(mesh_type)+' '+str(n_node))
        elif 'polygon' in mesh_type:
            if n_node < 3:
                raise ValueError('wrong number of nodes')
            elif n_node == 3:
                cell_name='VTK_TRIANGLE'
            elif n_node == 4:
                cell_name='VTK_QUAD'
            elif n_node == 6:
                if 'tri6' in mesh_type:
                    cell_name='VTK_QUADRATIC_TRIANGLE'
                else:
                    cell_name='VTK_POLYGON'
            elif n_node == 8:
                if 'quad8' in mesh_type:
                    cell_name='VTK_QUADRATIC_QUAD'
                else:
                    cell_name='VTK_POLYGON'
            else:
                cell_name='VTK_POLYGON'
        elif 'polyline' in mesh_type:
            if n_node < 2:
                raise ValueError('wrong number of nodes')
            elif n_node == 2:
                cell_name='VTK_LINE'
            else:
                cell_name='VTK_POLY_LINE'
        else:
            raise ValueError('unsupported mesh_type:'+mesh_type)
        return cell_name

    @staticmethod
    def get_vtk_cell_type(mesh_type, n_node):
        return getattr(vtk, Mesh.get_vtk_cell_name(mesh_type, n_node))

    @staticmethod
    def get_vtk_cell_type_from_element_type(element_type):
        #element_type: tri3, quad4, hex8, etc in Febio
        #element_type: C3D8, C3D4, etc in Abaqus
        found=find_vtk_cell_by_element_type(element_type)
        if found is None:
            raise ValueError('unknown element_type: '+str(element_type))
        return getattr(vtk, found[0])

    def get_vtk_cell_name_of_labeled_element(self, element_type, n_node):
        found=find_vtk_cell_by_element_type(element_type)
        if found is None:
            # The label is only metadata here, such as a user-defined element.
            # Treat the element like one that has no label.
            return Mesh.get_vtk_cell_name(self.mesh_type, n_node)
        if found[1] != n_node:
            raise ValueError('element_type '+str(element_type)+' needs '+str(found[1])
                             +' nodes, but the element has '+str(n_node))
        return found[0]

    def get_vtk_cell_type_of_labeled_element(self, element_type, n_node):
        return getattr(vtk, self.get_vtk_cell_name_of_labeled_element(element_type, n_node))

    def make_vtk_cells(self, id_dtype=np.int64):
        """Return the cells to save: counts, connectivity, cell_names and inverse.

        counts holds the number of nodes of each element, and connectivity holds their nodes in one array.
        The VTK cell of element m is cell_names[inverse[m]].
        It needs no VTK: convert_to_vtk and the writers in save_mesh_as_vtk_file.py both use it,
        so they accept the same elements and raise the same errors.
        """
        n_element=0 if self.element is None else len(self.element)
        if n_element == 0:
            counts=np.empty(0, dtype=id_dtype)
            connectivity=np.empty(0, dtype=id_dtype)
        elif torch.is_tensor(self.element) or (isinstance(self.element, np.ndarray)
                                               and self.element.dtype != object):
            element=to_numpy(self.element, id_dtype)
            if element.ndim != 2:
                raise ValueError('element must be a 2D array or a list of 1D arrays')
            counts=np.full(n_element, element.shape[1], dtype=id_dtype)
            connectivity=element.reshape(-1)
        else:
            # Mixed polygons/polylines/solids may have different node counts.
            rows=[to_numpy(e, id_dtype) for e in self.element]
            if any(e.ndim != 1 for e in rows):
                raise ValueError('each element must be a 1D array')
            counts=np.array([len(e) for e in rows], dtype=id_dtype)
            connectivity=np.concatenate(rows)
        # Resolve cells per distinct node count or element label, not per cell.
        if self.element_type is None:
            sizes, inverse=np.unique(counts, return_inverse=True)
            cell_names=[Mesh.get_vtk_cell_name(self.mesh_type, int(size)) for size in sizes]
        else:
            self.check_element_type(self.element_type)
            if len(self.element_type) != n_element:
                raise ValueError('len(element_type) != len(element)')
            # str() also handles a missing label, which is None in an object array.
            labels, label_index=np.unique(np.asarray(self.element_type).astype(str).reshape(-1),
                                          return_inverse=True)
            sizes, size_index=np.unique(counts, return_inverse=True)
            keys, inverse=np.unique(label_index*len(sizes)+size_index, return_inverse=True)
            cell_names=[self.get_vtk_cell_name_of_labeled_element(labels[key//len(sizes)], int(sizes[key%len(sizes)]))
                        for key in keys]
        return counts, connectivity, cell_names, inverse

    def convert_to_vtk(self):
        """Convert geometry and 2D numeric fields to independent VTK arrays.

        Coordinates and fields remain doubles, as in convert_to_vtk_slow.
        Connectivity uses VTK's native ID width. Ragged elements are flattened
        once; no individual points, cells, or field components are sent to VTK.
        """
        if _Flag_VTK_IMPORT_ == False:
            raise ValueError('vtk is not imported')
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

        node=to_numpy(self.node if self.node is not None else [])
        if node.size == 0:
            node=node.reshape(0, 3)
        if node.ndim != 2 or node.shape[1] != 3:
            raise ValueError('node must have shape (N, 3)')
        point_vtk=vtk.vtkPoints()
        point_vtk.SetData(numpy_to_vtk(node, deep=True))

        id_dtype=np.int64 if vtk.vtkIdTypeArray().GetDataTypeSize() == 8 else np.int32
        counts, connectivity, cell_names, inverse=self.make_vtk_cells(id_dtype)
        n_element=len(counts)
        offsets=np.empty(n_element+1, dtype=id_dtype)
        offsets[0]=0
        np.cumsum(counts, out=offsets[1:])
        types=np.array([getattr(vtk, name) for name in cell_names], dtype=np.uint8)[inverse]

        cells=vtk.vtkCellArray()
        if hasattr(cells, 'SetData'):
            cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                          numpy_to_vtkIdTypeArray(connectivity, deep=True))
        else:
            # VTK 8 uses packed [count, ids..., count, ids...] connectivity.
            packed=np.empty(n_element+len(connectivity), dtype=id_dtype)
            count_positions=offsets[:-1]+np.arange(n_element, dtype=id_dtype)
            packed[count_positions]=counts
            is_node=np.ones(len(packed), dtype=bool)
            is_node[count_positions]=False
            packed[is_node]=connectivity
            cells.SetCells(n_element, numpy_to_vtkIdTypeArray(packed, deep=True))

        if ('polyhedron' in self.mesh_type or
                ('polygon' in self.mesh_type and
                 ('tri6' in self.mesh_type or 'quad8' in self.mesh_type))):
            mesh_vtk=vtk.vtkUnstructuredGrid()
            mesh_vtk.SetCells(numpy_to_vtk(types, deep=True), cells)
        elif 'polygon' in self.mesh_type:
            mesh_vtk=vtk.vtkPolyData()
            mesh_vtk.SetPolys(cells)
        elif 'polyline' in self.mesh_type:
            mesh_vtk=vtk.vtkPolyData()
            mesh_vtk.SetLines(cells)
        else:
            raise ValueError('unsupported mesh_type:'+self.mesh_type)
        mesh_vtk.SetPoints(point_vtk)

        for fields, target, count in ((self.node_data, mesh_vtk.GetPointData(), len(node)),
                                     (self.element_data, mesh_vtk.GetCellData(), n_element)):
            for name, data in fields.items():
                data=to_numpy(data)
                if data.ndim != 2 or data.shape[0] != count or data.shape[1] < 1:
                    raise ValueError('invalid data shape, name:'+name+
                                     '; expected ('+str(count)+', number_of_components)')
                vtk_array=numpy_to_vtk(data, deep=True)
                vtk_array.SetName(name)
                target.AddArray(vtk_array)
        return mesh_vtk

    def convert_to_vtk_slow(self):
        if self.node is None or self.element is None:
            return self.convert_to_vtk()
        if _Flag_VTK_IMPORT_ == False:
            raise ValueError('vtk is not imported')
        point_vtk = vtk.vtkPoints()
        point_vtk.SetDataTypeToDouble()
        point_vtk.SetNumberOfPoints(len(self.node))
        for n in range(0, len(self.node)):
            point_vtk.SetPoint(n, float(self.node[n,0]), float(self.node[n,1]), float(self.node[n,2]))
        if 'polyhedron' in self.mesh_type:
            mesh_vtk = vtk.vtkUnstructuredGrid()
        elif 'polygon' in self.mesh_type:
            if ('tri6' in self.mesh_type) or ('quad8' in self.mesh_type):
                mesh_vtk = vtk.vtkUnstructuredGrid()
            else:               
                mesh_vtk = vtk.vtkPolyData()
        elif 'polyline' in self.mesh_type:                
            mesh_vtk = vtk.vtkPolyData()
        else:
            raise ValueError('unsupported mesh_type:'+self.mesh_type)
        mesh_vtk.SetPoints(point_vtk)
        mesh_vtk.Allocate(len(self.element))
        self.check_element_type(self.element_type)
        if self.element_type is not None and len(self.element_type) != len(self.element):
            raise ValueError('len(element_type) != len(element)')
        for n in range(0, len(self.element)):
            e=[int(id) for id in self.element[n]]
            if self.element_type is not None:
                cell_type=self.get_vtk_cell_type_of_labeled_element(self.element_type[n], len(e))
            # vtkPolyData has no quadratic cells, so its cell type comes from the node count.
            if self.element_type is None or mesh_vtk.IsA('vtkPolyData'):
                cell_type=Mesh.get_vtk_cell_type(self.mesh_type, len(e))
            mesh_vtk.InsertNextCell(cell_type, len(e), e)
        #--------- convert node_data to PointData --------#
        for name, data in self.node_data.items():
            #data should be a 2D array (self.node.shape[0], ?)
            if self.node.shape[0] != data.shape[0]:
                raise ValueError('self.node.shape[0] != data.shape[0], name:'+name)
            vtk_array=vtk.vtkDoubleArray()
            #run SetNumberOfComponents before SetNumberOfTuples
            vtk_array.SetNumberOfComponents(data.shape[1])
            vtk_array.SetNumberOfTuples(data.shape[0])
            vtk_array.SetName(name)
            for i in range(0, data.shape[0]):
                for j in range(0, data.shape[1]):
                    vtk_array.SetComponent(i,j,float(data[i,j]))
            mesh_vtk.GetPointData().AddArray(vtk_array)
        #--------- convert element_data to CellData --------#
        for name, data in self.element_data.items():
            #data should be a 2D array (len(self.element), ?)
            if len(self.element) != data.shape[0]:
                raise ValueError('len(self.element) != data.shape[0], name:'+name)
            vtk_array=vtk.vtkDoubleArray()
            vtk_array.SetNumberOfComponents(data.shape[1])
            vtk_array.SetNumberOfTuples(data.shape[0])
            vtk_array.SetName(name)
            for i in range(0, data.shape[0]):
                for j in range(0, data.shape[1]):
                    vtk_array.SetComponent(i,j,float(data[i,j]))
            mesh_vtk.GetCellData().AddArray(vtk_array)
        return mesh_vtk

    @staticmethod
    def get_vtk_version():
        # Compare (major, minor) as a tuple: VTK 10.0 is newer than 9.1, but its minor is 0.
        return (vtk.vtkVersion.GetVTKMajorVersion(), vtk.vtkVersion.GetVTKMinorVersion())

    def save_as_vtk(self, filename, vtk42=True):
        self.check_element_dimension(self.element, self.element_type)
        use_vtk=True
        if _Flag_VTK_IMPORT_ == True:
            if vtk42 == True:
                version=Mesh.get_vtk_version()
                if version[0]<=8:
                    pass
                elif version>=(9,1):
                    pass
                else:
                    use_vtk=False
        else:
            use_vtk=False
        if use_vtk == False:
            mesh=self
            if self.node is None or self.element is None:
                mesh=shallowcopy(self)
                if mesh.node is None:
                    mesh.node=torch.empty((0,3), dtype=torch.float64)
                if mesh.element is None:
                    mesh.element=[]
            if 'polyhedron' in self.mesh_type:
                save_polyhedron_mesh_as_vtk(mesh, filename)
            elif 'polygon' in self.mesh_type:
                if ('tri6' in self.mesh_type) or ('quad8' in self.mesh_type):
                    save_quadratic_polygon_mesh_as_vtk(mesh, filename)
                else:
                    save_polygon_mesh_as_vtk(mesh, filename)
            elif 'polyline' in self.mesh_type:
                save_polyline_mesh_as_vtk(mesh, filename)
            else:
                raise ValueError('unsupported mesh_type: '+self.mesh_type)
            if vtk42 == False:
                print('Mesh save_as_vtk: can only save to 4.2 version vtk, although vtk42=False')
            return
        #-----------------------------
        mesh_vtk=self.convert_to_vtk()
        if mesh_vtk is None:
            return
        if 'polyhedron' in self.mesh_type:
            writer=vtk.vtkUnstructuredGridWriter()
        elif 'polygon' in self.mesh_type:
            if ('tri6' in self.mesh_type) or ('quad8' in self.mesh_type):
                writer=vtk.vtkUnstructuredGridWriter()
            else:
                writer=vtk.vtkPolyDataWriter()
        elif 'polyline' in self.mesh_type:
            writer=vtk.vtkPolyDataWriter()
        else:
            raise ValueError('unsupported mesh_type: '+self.mesh_type)
        if vtk42 == True:
            version=Mesh.get_vtk_version()
            if version[0]<=8:
                pass
            elif version>=(9,1):
                writer.SetFileVersion(42)
            else:
                print('Mesh save_as_vtk: cannot save to 4.2 version vtk')
        writer.SetFileTypeToASCII()
        writer.SetPrecision(17)#17 is enough for double/fp64
        writer.SetInputData(mesh_vtk)
        writer.SetFileName(filename)
        Mesh.run_vtk_writer(writer, filename)

    @staticmethod
    def run_vtk_writer(writer, filename):
        # A VTK writer only logs a failure, such as a missing folder or a full disk,
        # so a long run could finish with its result unsaved.
        if writer.Write() == 0 or writer.GetErrorCode() != 0:
            raise RuntimeError('cannot write '+filename+': '
                               +vtk.vtkErrorCode.GetStringFromErrorCode(writer.GetErrorCode()))
    
    def save_as_vtp(self, filename):
        if _Flag_VTK_IMPORT_ == False:            
            raise ValueError('vtk is not imported')
        self.check_element_dimension(self.element, self.element_type)
        #-----------------------------
        mesh_vtk=self.convert_to_vtk()
        if mesh_vtk is None:
            return
        if ('polygon' in self.mesh_type) and ('tri6' not in self.mesh_type) and  ('quad8' not in self.mesh_type):
            writer=vtk.vtkXMLPolyDataWriter()
        else:
            raise ValueError('unsupported mesh_type: '+self.mesh_type)
        writer.SetInputData(mesh_vtk)
        writer.SetFileName(filename)
        Mesh.run_vtk_writer(writer, filename)

    def save_as_torch(self, filename, save_adj_info=True):
        data={'name':self.name,
              'mesh_type':self.mesh_type,
              'node':self.node,
              'element':self.element,
              'element_type':self.element_type,
              'node_set':self.node_set,
              'element_set':self.element_set,
              'node_data':self.node_data,
              'element_data':self.element_data,
              'mesh_data':self.mesh_data,
              'mapping_from_node_name_to_index':self.mapping_from_node_name_to_index}
        if save_adj_info == True:
            data['edge']=self.edge
            data['node_to_node_adj_link']=self.node_to_node_adj_link
            data['node_to_node_adj_table']=self.node_to_node_adj_table
            data['node_to_edge_adj_table']=self.node_to_edge_adj_table
            data['node_to_element_adj_table']=self.node_to_element_adj_table
            data['edge_to_edge_adj_table']=self.edge_to_edge_adj_table
            data['edge_to_element_adj_table']=self.edge_to_element_adj_table            
            data['element_to_element_adj_link']=self.element_to_element_adj_link
            data['element_to_element_adj_table']=self.element_to_element_adj_table
            data['element_to_edge_adj_table']=self.element_to_edge_adj_table
            if 'polyhedron' in self.mesh_type:
                data['face']=self.face
                data['face_to_element_adj_table']=self.face_to_element_adj_table
                data['element_to_face_adj_table']=self.element_to_face_adj_table
        torch.save(data,  filename)

    def load_from_torch(self, filename, dtype=None):
        """Load a mesh saved by save_as_torch.

        dtype: the type of node and of the floating-point tensors in node_data and element_data,
        such as torch.float64 or 'float64'. None keeps the types in the file.
        Everything else, such as element, mesh_data and integer fields, stays as it was saved.

        WARNING: this uses torch.load(weights_only=False), which can run code
        stored in the file. Load only files that you trust.
        """
        if not os.path.isfile(filename):
            raise ValueError('not exist: '+filename)
        if isinstance(dtype, str):
            if dtype == 'float32':
                dtype=torch.float32
            elif dtype == 'float64':
                dtype=torch.float64
            else:
                raise ValueError('unsupported dtype:'+str(dtype))

        def convert_tensor(value):
            # A new leaf, because .to(dtype) alone would make a node that needs grad a non-leaf.
            return value.detach().to(dtype).requires_grad_(value.requires_grad)

        def convert_fields(fields):
            if dtype is None or not isinstance(fields, dict):
                return fields
            return {name:(convert_tensor(value) if torch.is_tensor(value) and value.is_floating_point() else value)
                    for name, value in fields.items()}

        data=torch.load(filename, map_location='cpu', weights_only=False)
        if 'node' not in data:
            raise ValueError('node is not in data.keys()')
        if 'element' not in data:
            raise ValueError('element is not in data.keys()')
        node=data['node']
        if node is not None and not isinstance(node, torch.Tensor):
            node=torch.tensor(node)
        if node is not None and dtype is not None:
            node=convert_tensor(node)
        node_data=convert_fields(data.get('node_data'))
        element_data=convert_fields(data.get('element_data'))
        #the shape is restored as it was saved, with no dimension or width check. The integer rule still applies
        element=data['element']
        if isinstance(element, (list, tuple)):
            element=Mesh.make_nested_list_of_int(element)
        elif element is not None:
            if isinstance(element, np.ndarray) and element.dtype == object:
                element=Mesh.make_nested_list_of_int(element.tolist())
            else:
                Mesh.check_element_array_is_integer(element)
                element=element.to(torch.int64) if torch.is_tensor(element) else torch.tensor(element, dtype=torch.int64)
        self.clear()
        self.node=node
        self.element=element
        self.element_type=data.get('element_type')
        if 'name' in data.keys():
            self.name=data['name']
        #the saved mesh_type replaces this mesh's mesh_type only if it says more: polygon becomes polygon_tri6.
        #polyhedron_hex8 stays as it is for a file saved as polyhedron, because a plain polyhedron can't save a hex
        saved_mesh_type=data.get('mesh_type')
        if isinstance(saved_mesh_type, str) and saved_mesh_type.startswith(self.mesh_type+'_'):
            self.mesh_type=saved_mesh_type
        if 'node_set' in data.keys():
            self.node_set=data['node_set']
        if 'element_set' in data.keys():
            self.element_set=data['element_set']
        if 'node_data' in data.keys():
            self.node_data=node_data
        if 'element_data' in data.keys():
            self.element_data=element_data
        if 'mesh_data' in data.keys():
            self.mesh_data=data['mesh_data']
        if 'mapping_from_node_name_to_index' in data.keys():
            self.mapping_from_node_name_to_index=data['mapping_from_node_name_to_index']
        if 'edge' in data.keys():
            self.edge=data['edge']
        if 'node_to_node_adj_link' in data.keys():
            self.node_to_node_adj_link=data['node_to_node_adj_link']
        if 'node_to_node_adj_table' in data.keys():
            self.node_to_node_adj_table=data['node_to_node_adj_table']
        if 'node_to_edge_adj_table' in data.keys():
            self.node_to_edge_adj_table=data['node_to_edge_adj_table']
        if 'node_to_element_adj_table' in data.keys():
            self.node_to_element_adj_table=data['node_to_element_adj_table']
        if 'edge_to_edge_adj_table' in data.keys():
            self.edge_to_edge_adj_table=data['edge_to_edge_adj_table']
        if 'edge_to_element_adj_table' in data.keys():
            self.edge_to_element_adj_table=data['edge_to_element_adj_table']
        if 'element_to_element_adj_link' in data.keys():
            self.element_to_element_adj_link=data['element_to_element_adj_link']
        if 'element_to_element_adj_table' in data.keys():
            self.element_to_element_adj_table=data['element_to_element_adj_table']
        if 'element_to_edge_adj_table' in data.keys():
            self.element_to_edge_adj_table=data['element_to_edge_adj_table']
        if 'polyhedron' in self.mesh_type:
            if 'face' in data.keys():
                self.face=data['face']
            if 'face_to_element_adj_table' in data.keys():
                self.face_to_element_adj_table=data['face_to_element_adj_table']
            if 'element_to_face_adj_table' in data.keys():
                self.element_to_face_adj_table=data['element_to_face_adj_table']
    
    def save(self, filename):
        filenamebase, file_type = os.path.splitext(filename)
        if file_type == '.vtk':
            self.save_as_vtk(filename)
        elif  file_type == '.vtp':
            self.save_as_vtp(filename)
        elif file_type == '.pt' or file_type == '.pth':
            self.save_as_torch(filename, True)
        else:
            raise ValueError('unknown file_type:'+file_type)
    
    def load(self, filename, dtype):
        """Load by file extension. WARNING: .pt and .pth files can run code; see load_from_torch."""
        if not os.path.isfile(filename):
            raise ValueError('not exist: '+filename)        
        filenamebase, file_type = os.path.splitext(filename)
        if file_type == '.vtk':
            self.load_from_vtk(filename, dtype)
        elif  file_type == '.vtp':
            self.load_from_vtp(filename, dtype)
        elif  file_type == '.stl':
            self.load_from_stl(filename, dtype)
        elif file_type == '.pt' or file_type == '.pth':
            self.load_from_torch(filename, dtype)
        else:
            raise ValueError('unknown file_type:'+file_type)
            
    def copy(self, node, element, dtype=None, detach=True):
        """Replace geometry with independent copies and reset other mesh state.

        node and element are converted as in the constructor. dtype and detach apply to node.
        """
        node_copy=None
        if node is not None:
            node_copy=Mesh.convert_node(node, dtype)
            if torch.is_tensor(node):
                node_copy=node_copy.clone()
                if dtype is not None:
                    node_copy=node_copy.to(dtype)
                if detach==True:
                    node_copy=node_copy.detach()
        element_copy=None
        if element is not None:
            element_copy=self.convert_element(element, None if node_copy is None else node_copy.device)
            if torch.is_tensor(element_copy):
                element_copy=element_copy.clone().detach()
            else:
                element_copy=deepcopy(element_copy)
        self.clear()
        self.node=node_copy
        self.element=element_copy

    def copy_to_list(self, x):
        if isinstance(x, list):
            y=deepcopy(x)
        elif isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
            y=x.tolist()
        elif isinstance(x, tuple) or isinstance(x, set):
            y=deepcopy(list(x))
        else:
            raise ValueError('unsupported python-object type')
        return y

    def make_element_copy(self, object_type=None):
        element=self.element
        if object_type is None:
            object_type=('torch' if torch.is_tensor(element) else
                         'numpy' if isinstance(element, np.ndarray) else 'list')
        if element is None:
            element=np.empty((0,0), dtype=np.int64)
        if object_type == 'list' or object_type == list:
            return self.copy_to_list(element)
        elif object_type == 'numpy' or object_type == np.ndarray:
            if torch.is_tensor(element):
                element=element.detach().to(device='cpu', dtype=torch.int64)
                element=element.numpy()
            return np.array(element, dtype=np.int64, copy=True)
        elif object_type == 'torch' or object_type == torch.Tensor:
            if torch.is_tensor(element):
                return element.detach().to(device='cpu', dtype=torch.int64, copy=True)
            if isinstance(element, np.ndarray):
                if element.dtype.kind == 'O':
                    element=element.tolist()
                elif not element.dtype.isnative or any(s < 0 for s in element.strides):
                    element=np.array(element, dtype=element.dtype.newbyteorder('='),
                                     copy=True, order='C')
            return torch.tensor(element, dtype=torch.int64, device='cpu')
        raise ValueError('object_type must be list, numpy, or torch')

    def make_node_copy(self, object_type=None, dtype=None):
        node=self.node
        if object_type is None:
            object_type=('torch' if torch.is_tensor(node) else
                         'numpy' if isinstance(node, np.ndarray) else 'list')
        if node is None:
            node=np.empty((0,3), dtype=np.float32)
        if object_type == 'list' or object_type == list:
            return self.copy_to_list(node)
        elif object_type == 'numpy' or object_type == np.ndarray:
            if dtype is None:
                dtype=getattr(node, 'dtype', np.float32)
            if isinstance(dtype, torch.dtype):
                # Resolve the dtype without converting any coordinate values.
                dtype=torch.empty(0, dtype=dtype, device='cpu').numpy().dtype
            if torch.is_tensor(node):
                node=node.detach().cpu().resolve_conj().resolve_neg()
                if node.dtype == torch.bfloat16:
                    node=node.float()
                node=node.numpy()
            return np.array(node, dtype=dtype, copy=True)
        elif object_type == 'torch' or object_type == torch.Tensor:
            if dtype is None:
                dtype=getattr(node, 'dtype', torch.float32)
            if not isinstance(dtype, torch.dtype):
                dtype=torch.from_numpy(np.empty(0, dtype=np.dtype(dtype).newbyteorder('='))).dtype
            if torch.is_tensor(node):
                return node.detach().to(device='cpu', dtype=dtype, copy=True)
            if isinstance(node, np.ndarray):
                if node.dtype.kind == 'O':
                    node=node.tolist()
                elif not node.dtype.isnative or any(s < 0 for s in node.strides):
                    node=np.array(node, dtype=node.dtype.newbyteorder('='),
                                  copy=True, order='C')
            return torch.tensor(node, dtype=dtype, device='cpu')
        raise ValueError('object_type must be list, numpy, or torch')
    
    def get_node_idx_by_name(self, name):
        if name in self.mapping_from_node_name_to_index.keys():
            return self.mapping_from_node_name_to_index[name]
        else:
            return None

    def set_node_name(self, idx, name, replace_old_name=False):
        if replace_old_name == True:
            self.mapping_from_node_name_to_index[name]=idx
        else:
            if name in self.mapping_from_node_name_to_index.keys():
                raise ValueError('node '+str(idx)+' already has a name:'+str(name))
            else:
                self.mapping_from_node_name_to_index[name]=idx
    
    def build_edge(self):
        #undirected edge represents a connection between two nodes
        #self.edge[k] is [node_idx_a, node_idx_b] and node_idx_a < node_idx_b: self.edge[k,0] < self.edge[k,1]
        #edge is determined by element_type
        #implement this function in a derived class (e.g. PolygonMesh)
        raise NotImplementedError

    def get_edge_idx_from_node_pair(self, nodeA_idx, nodeB_idx):
        # nodeA---an edge---nodeB
        if self.node_to_edge_adj_table is None:
            self.build_node_to_edge_adj_table()
        if nodeA_idx == nodeB_idx:
            return None
        edge_idx_listA=self.node_to_edge_adj_table[nodeA_idx]
        edge_idx_listB=self.node_to_edge_adj_table[nodeB_idx]
        temp=np.intersect1d(edge_idx_listA, edge_idx_listB, assume_unique=True)
        if len(temp) == 0:
            edge_idx=None
        elif len(temp) == 1:
            edge_idx=int(temp.item())
        else:
            raise ValueError('more than one edge between node '+str(nodeA_idx)+' and node '+str(nodeB_idx))
        return edge_idx

    def update_edge_length(self):
        if self.edge is None:
            self.build_edge()
        self.edge_length=Mesh.cal_edge_length(self.node, self.edge)

    @staticmethod
    def cal_edge_length(node, edge):
        x_j=node[edge[:,0]]
        x_i=node[edge[:,1]]
        edge_length=norm(x_i-x_j, ord=2, dim=1, keepdim=True)
        return edge_length #(K,1)

    def build_node_to_node_adj_link(self):
        #no self link
        #this is useful for GNN
        if self.edge is None:
            self.build_edge()
        adj_link=[]
        for k in range(0, len(self.edge)):
            idx0=int(self.edge[k,0])
            idx1=int(self.edge[k,1])
            if idx0 != idx1:
                adj_link.append([idx0, idx1])
                adj_link.append([idx1, idx0])
        adj_link=torch.tensor(adj_link, dtype=torch.int64).reshape(-1,2)#(0,2) if there is no link
        adj_link=torch.unique(adj_link, dim=0, sorted=True)
        self.node_to_node_adj_link=adj_link

    def build_node_to_node_adj_table(self):
        #no self link
        if self.edge is None:
            self.build_edge()
        adj_table=[[] for _ in range(self.node.shape[0])]
        for k in range(0, len(self.edge)):
            idx0=int(self.edge[k,0])
            idx1=int(self.edge[k,1])
            if idx0 != idx1:
                adj_table[idx0].append(idx1)
                adj_table[idx1].append(idx0)
        self.node_to_node_adj_table=adj_table

    def build_node_to_edge_adj_table(self):
        if self.edge is None:
            self.build_edge()
        adj_table=[[] for _ in range(self.node.shape[0])]
        for k in range(0, len(self.edge)):
            idx0=int(self.edge[k,0])
            idx1=int(self.edge[k,1])
            adj_table[idx0].append(k)
            adj_table[idx1].append(k)
        self.node_to_edge_adj_table=adj_table

    def build_node_to_element_adj_table(self):
        #do not do this: node_to_element_adj_table=[[]]*self.node.shape[0]
        # a=[[]]*2=[[],[]], and a[0] and a[1] are the same object
        # a=[[],[]], and a[0] and a[1] are two different objects
        adj_table=[[] for _ in range(self.node.shape[0])]
        element=self.element
        if isinstance(element, torch.Tensor):
            element=element.detach().cpu().numpy()
        for m in range(0, len(element)):
            e_m=element[m]
            for k in range(0, len(e_m)):
                adj_table[e_m[k]].append(m)
        self.node_to_element_adj_table=adj_table

    def build_edge_to_edge_adj_table(self):
        if self.node_to_edge_adj_table is None:
            self.build_node_to_edge_adj_table()
        adj_table=[[] for _ in range(self.edge.shape[0])]
        for k in range(0, len(self.edge)):
            idx0=int(self.edge[k,0])
            idx1=int(self.edge[k,1])
            adj_table[k].extend(self.node_to_edge_adj_table[idx0])
            adj_table[k].extend(self.node_to_edge_adj_table[idx1])
            adj_table[k]=list(set(adj_table[k])-set([k]))
        self.edge_to_edge_adj_table=adj_table
        
    def build_edge_to_element_adj_table(self):
        if self.element_to_edge_adj_table is None:
            self.build_element_to_edge_adj_table()
        adj_table=[[] for _ in range(self.edge.shape[0])]
        for m in range(0, len(self.element)):
            adj_edge_idx_list=self.element_to_edge_adj_table[m]
            for edge_idx in adj_edge_idx_list:
                adj_table[edge_idx].append(m)
        self.edge_to_element_adj_table=adj_table

    def build_element_to_edge_adj_table(self):
        #this table is determined by element_type
        #implement this function in a derived class (e.g., PolygonMesh)
        raise NotImplementedError

    def build_element_to_element_adj_link_node(self):
        if self.node_to_element_adj_table is None:
            self.build_node_to_element_adj_table()
        adj_link=[]
        for n in range(0, len(self.node_to_element_adj_table)):
            e_set=self.node_to_element_adj_table[n]
            for m1 in range(0, len(e_set)):
                for m2 in range(m1+1, len(e_set)):
                    eid1=e_set[m1]; eid2=e_set[m2]
                    #an element with a repeated node or edge is listed twice: no self link
                    if eid1 != eid2:
                        adj_link.append([eid1, eid2])
                        adj_link.append([eid2, eid1])
        adj_link=torch.tensor(adj_link, dtype=torch.int64).reshape(-1,2)#(0,2) if there is no link
        adj_link=torch.unique(adj_link, dim=0, sorted=True)
        self.element_to_element_adj_link['node']=adj_link

    def build_element_to_element_adj_link_edge(self):
        if self.edge_to_element_adj_table is None:
            self.build_edge_to_element_adj_table()
        adj_link=[]
        for n in range(0, len(self.edge_to_element_adj_table)):
            e_set=self.edge_to_element_adj_table[n]
            for m1 in range(0, len(e_set)):
                for m2 in range(m1+1, len(e_set)):
                    eid1=e_set[m1]; eid2=e_set[m2]
                    #an element with a repeated node or edge is listed twice: no self link
                    if eid1 != eid2:
                        adj_link.append([eid1, eid2])
                        adj_link.append([eid2, eid1])
        adj_link=torch.tensor(adj_link, dtype=torch.int64).reshape(-1,2)#(0,2) if there is no link
        adj_link=torch.unique(adj_link, dim=0, sorted=True)
        self.element_to_element_adj_link['edge']=adj_link

    def build_element_to_element_adj_link_face(self):
        if self.face_to_element_adj_table is None:
            self.build_face_to_element_adj_table()
        adj_link=[]
        for n in range(0, len(self.face_to_element_adj_table)):
            e_set=self.face_to_element_adj_table[n]
            for m1 in range(0, len(e_set)):
                for m2 in range(m1+1, len(e_set)):
                    eid1=e_set[m1]; eid2=e_set[m2]
                    #an element with a repeated node or edge is listed twice: no self link
                    if eid1 != eid2:
                        adj_link.append([eid1, eid2])
                        adj_link.append([eid2, eid1])
        adj_link=torch.tensor(adj_link, dtype=torch.int64).reshape(-1,2)#(0,2) if there is no link
        adj_link=torch.unique(adj_link, dim=0, sorted=True)
        self.element_to_element_adj_link['face']=adj_link

    def build_element_to_element_adj_link(self, adj):
        #no self link
        if adj not in ['node', 'edge', 'face']:
            raise ValueError('adj should be node, edge, or face')
        if adj == 'node':
            return self.build_element_to_element_adj_link_node()
        elif adj == 'edge':
            return self.build_element_to_element_adj_link_edge()
        elif adj == 'face':
            return self.build_element_to_element_adj_link_face()
        else:
            raise NotImplementedError

    def build_element_to_element_adj_table(self, adj):
        #no self link
        if self.element_to_element_adj_link[adj] is None:
            self.build_element_to_element_adj_link(adj=adj)
        adj_link=self.element_to_element_adj_link[adj]
        adj_table=[[] for _ in range(len(self.element))]
        for k in range(0, len(adj_link)):
            link=adj_link[k]
            idx0=int(link[0])
            idx1=int(link[1])
            if idx0 != idx1:
                adj_table[idx0].append(idx1)
        self.element_to_element_adj_table[adj]=adj_table

    @staticmethod
    def make_element_idx_list(element_idx_list):
        """Return element indices as a list of python int. A boolean mask is rejected."""
        if torch.is_tensor(element_idx_list) or isinstance(element_idx_list, np.ndarray):
            if len(element_idx_list.shape) != 1:
                raise ValueError('element_idx_list must be 1D, but its shape is '+str(tuple(element_idx_list.shape)))
            element_idx_list=element_idx_list.tolist()
        idx_list=[]
        for idx in element_idx_list:
            if isinstance(idx, (bool, np.bool_)) or (torch.is_tensor(idx) and idx.dtype == torch.bool):
                raise ValueError('get_sub_mesh does not accept a boolean mask: pass the indices of the elements')
            idx_list.append(operator.index(idx))
        return idx_list

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        """Return a new mesh made of the elements in element_idx_list, and the nodes they use.

        element_idx_list: element indices as a list, a numpy array or a tensor on any device.
        A boolean mask is not accepted.
        return_node_idx_list=True also returns node_idx_list: new_mesh.node[k] is self.node[node_idx_list[k]].

        The new mesh has node and element, and element_type on a plain Mesh, PolygonMesh or PolyhedronMesh.
        It does not keep node_data, element_data, node_set, element_set, mesh_data or name, so copy
        what you need from this mesh.
        """
        if element_idx_list is not None:
            element_idx_list=Mesh.make_element_idx_list(element_idx_list)
            if len(element_idx_list) == 0:
                element_idx_list=None
        new_node=None
        new_element=None
        node_idx_list=None
        #----------------------------
        use_slow_method=True        
        if torch.is_tensor(self.element) and element_idx_list is not None:
            if len(self.element.shape) == 2:
                old_element=self.element[element_idx_list]
                node_idx_list, new_element=torch.unique(old_element.reshape(-1), return_inverse=True)
                #node_idx_list[new_element] is old_element.reshape(-1)
                new_node=self.node[node_idx_list]
                new_element=new_element.reshape(len(element_idx_list),-1)
                use_slow_method=False
        #----------------------------
        if use_slow_method==True and element_idx_list is not None:
            new_element=[]
            node_idx_list=[]
            for m in range(0, len(element_idx_list)):
                elm=self.element[element_idx_list[m]]
                elm=self.copy_to_list(elm)
                new_element.append(elm)
                node_idx_list.extend(elm)
            node_idx_list=np.unique(node_idx_list)
            new_node=self.node[node_idx_list]
            #map old node idx to new node idx
            map={}
            for n in range(0, len(node_idx_list)):
                old_idx=node_idx_list[n]
                map[old_idx]=n #n is new_idx
            for m in range(0, len(new_element)):
                for n in range(0, len(new_element[m])):
                    old_idx=new_element[m][n]
                    new_element[m][n]=map[old_idx]
        #------------------------------------------------
        new_element_type=None
        if self.element_type is not None:
            # element_type can be a list: assigned by hand, or restored by load_from_torch
            element_type=np.asarray(self.element_type) if isinstance(self.element_type, list) else self.element_type
            new_element_type=(element_type[element_idx_list] if element_idx_list is not None
                              else element_type[:0])
        new_mesh=Mesh(new_node, new_element, new_element_type, self.mesh_type)
        if return_node_idx_list == False:
            return new_mesh
        else:
            return new_mesh, node_idx_list
#%%
