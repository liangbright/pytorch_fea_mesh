import torch
from torch.linalg import vector_norm as norm
import numpy as np
from copy import deepcopy, copy as shallowcopy
from ..io.save_mesh_as_vtk_file import save_polyline_mesh_as_vtk, save_polygon_mesh_as_vtk, save_polyhedron_mesh_as_vtk
import os
_Flag_VTK_IMPORT_=False
try:
    import vtk
    _Flag_VTK_IMPORT_=True
except:
    print('cannot import vtk')
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
            if torch.is_tensor(node):
                self.node=node
            elif isinstance(node, list):
                self.node=torch.tensor(node, dtype=torch.float64)
            elif isinstance(node, np.ndarray):
                if node.dtype == np.float64:
                    self.node=torch.tensor(node, dtype=torch.float64)
                else:
                    self.node=torch.tensor(node, dtype=torch.float32)
            else:
                raise ValueError('unsupported python-object type of node')
            
            if len(self.node.shape) != 2:
                raise ValueError('invalid self.node.shape')
        #--------------------------------------------------------------------
        if element is not None:
            if torch.is_tensor(element):
                self.element=element.to(torch.int64)
            elif isinstance(element, np.ndarray):
                try:
                    self.element=torch.tensor(element, dtype=torch.int64)
                except:
                    self.element=element.tolist()
            elif isinstance(element, list):
                try:
                    self.element=torch.tensor(element, dtype=torch.int64)
                except:
                    self.element=element
            else:
                raise ValueError('unsupported python-object type of element')
            
            if torch.is_tensor(self.element):
                if len(self.element.shape) != 2:
                    raise ValueError('invalid self.element.shape')
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
        #--------------------------------------------------------------------

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

    def clear_geometry_derived_data(self):
        """Clear adjacency and derived values; retain geometry and user data."""
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
            self.element_to_element_adj_link={'node':None}
            self.element_to_element_adj_table={'node':None}
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
        try:
            self.read_mesh_vtk(mesh_vtk, dtype)
        except:
            raise ValueError('cannot convert vtk data to mesh')

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
        try:
            self.read_mesh_vtk(mesh_vtk, dtype)
        except:
            raise ValueError('cannot convert vtk data to mesh')
        if return_mesh_vtk == True:
            return mesh_vtk
        
    def read_mesh_vtk(self, mesh_vtk, dtype):
        """Bulk-read polydata/unstructured geometry and numeric data arrays.

        Uniform elements become an int64 tensor; ragged elements remain lists.
        Coordinates and fields use the requested dtype and own their storage.
        Reading replaces existing geometry, metadata and caches.
        Other VTK dataset classes retain the original reader as a fallback.
        """
        if not (mesh_vtk.IsA('vtkPolyData') or mesh_vtk.IsA('vtkUnstructuredGrid')):
            return self.read_mesh_vtk_slow(mesh_vtk, dtype)
        from vtk.util.numpy_support import vtk_to_numpy

        def copy_numeric_array(vtk_array):
            if vtk_array.GetDataType() == vtk.VTK_BIT:
                # vtk_to_numpy mishandles multi-component packed bit arrays
                # in some VTK versions. Expand them in VTK, without scalar calls.
                expanded=vtk.vtkUnsignedCharArray()
                expanded.DeepCopy(vtk_array)
                vtk_array=expanded
            data=vtk_to_numpy(vtk_array)
            # Scalar VTK getters used by the old reader return doubles. Match
            # that rounding for 64-bit integer coordinates/fields as well.
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
            cell_arrays=(mesh_vtk.GetVerts(), mesh_vtk.GetLines(),
                         mesh_vtk.GetPolys(), mesh_vtk.GetStrips())
        else:
            cell_arrays=(mesh_vtk.GetCells(),)
        blocks=[read_cell(cells) for cells in cell_arrays
                if cells is not None and cells.GetNumberOfCells()]
        if not blocks:
            element=[]
        else:
            if len(blocks) == 1:
                counts, connectivity=blocks[0]
            else:
                counts=np.concatenate([block[0] for block in blocks])
                connectivity=np.concatenate([block[1] for block in blocks])
            if np.all(counts == counts[0]):
                element=torch.tensor(connectivity.reshape(len(counts), int(counts[0])),
                                     dtype=torch.int64)
            else:
                offsets=np.empty(len(counts)+1, dtype=np.int64)
                offsets[0]=0
                np.cumsum(counts, out=offsets[1:])
                indices=connectivity.tolist()
                element=[indices[start:end] for start, end in zip(offsets[:-1], offsets[1:])]

        node_data={}
        element_data={}
        for attributes, fields in ((mesh_vtk.GetPointData(), node_data),
                                   (mesh_vtk.GetCellData(), element_data)):
            for i in range(attributes.GetNumberOfArrays()):
                vtk_array=attributes.GetArray(i)
                fields[vtk_array.GetName()]=copy_numeric_array(vtk_array)
        self.clear()
        self.node=node
        self.element=element
        self.node_data=node_data
        self.element_data=element_data

    def read_mesh_vtk_slow(self, mesh_vtk, dtype):
        #----------- load point/node --------------------#
        node=np.zeros((mesh_vtk.GetNumberOfPoints(), 3))
        for n in range(mesh_vtk.GetNumberOfPoints()):
            node[n]=mesh_vtk.GetPoint(n)
        if len(node) == 0:
            raise ValueError('cannot load node')
        node=torch.tensor(node, dtype=dtype)
        #----------- load element -----------------------#
        element=[]
        m_list=[]
        for n in range(mesh_vtk.GetNumberOfCells()):
            cell_n=mesh_vtk.GetCell(n)
            m_list.append(cell_n.GetNumberOfPoints())
            temp=[]
            for k in range(cell_n.GetNumberOfPoints()):
                temp.append(cell_n.GetPointId(k))
            element.append(temp)
        if len(element) == 0:
            #print('warning: cannot load element @ read_mesh_vtk')
            pass
        if len(element) > 0:
            if min(m_list) == max(m_list):
                element=torch.tensor(element, dtype=torch.int64)
        node_data={}
        element_data={}
        #---------- load PointData -----------#
        PointDataSetCount = mesh_vtk.GetPointData().GetNumberOfArrays()
        for n in range(0, PointDataSetCount):
            vtk_array=mesh_vtk.GetPointData().GetArray(n)
            name=vtk_array.GetName()
            data=np.zeros((vtk_array.GetNumberOfTuples(), vtk_array.GetNumberOfComponents()))
            for i in range(0, data.shape[0]):
                for j in range(0, data.shape[1]):
                    data[i,j]=vtk_array.GetComponent(i,j)
            node_data[name]=torch.tensor(data, dtype=dtype)
        #---------- load CellData -----------#
        CellDataSetCount = mesh_vtk.GetCellData().GetNumberOfArrays()
        for n in range(0, CellDataSetCount):
            vtk_array=mesh_vtk.GetCellData().GetArray(n)
            name=vtk_array.GetName()
            data=np.zeros((vtk_array.GetNumberOfTuples(), vtk_array.GetNumberOfComponents()))
            for i in range(0, data.shape[0]):
                for j in range(0, data.shape[1]):
                    data[i,j]=vtk_array.GetComponent(i,j)
            element_data[name]=torch.tensor(data, dtype=dtype)
        self.clear()
        self.node=node
        self.element=element
        self.node_data=node_data
        self.element_data=element_data

    @staticmethod
    def get_vtk_cell_type(mesh_type, n_node):
        #n_nodes: the number of nodes in an element
        #assume all elements have the same type
        if 'polyhedron' in mesh_type:
            if n_node < 4:
                raise ValueError('wrong number of nodes')
            elif n_node == 4:
                cell_type=vtk.VTK_TETRA
            elif (n_node == 6) and ('wedge6' in mesh_type): 
                cell_type=vtk.VTK_WEDGE
            elif (n_node == 8) and ('hex8' in mesh_type):
                cell_type=vtk.VTK_HEXAHEDRON
            elif (n_node == 10) and ('tet10' in mesh_type):
                cell_type=vtk.VTK_QUADRATIC_TETRA
            else:
                #cell_type=vtk.VTK_POLYHEDRON
                #print('warning: vtk may get stuck when cell_type=vtk.VTK_POLYHEDRON')
                raise ValueError('not supported: '+str(mesh_type)+' '+str(n_node))
        elif 'polygon' in mesh_type:
            if n_node < 3:
                raise ValueError('wrong number of nodes')
            elif n_node == 3:
                cell_type=vtk.VTK_TRIANGLE
            elif n_node == 4:
                cell_type=vtk.VTK_QUAD
            elif n_node == 6:
                if 'tri6' in mesh_type:
                    cell_type=vtk.VTK_QUADRATIC_TRIANGLE
                else:
                    cell_type=vtk.VTK_POLYGON
            elif n_node == 8:
                if 'quad8' in mesh_type:
                    cell_type=vtk.VTK_QUADRATIC_QUAD
                else:
                    cell_type=vtk.VTK_POLYGON
            else:
                cell_type=vtk.VTK_POLYGON
        elif 'polyline' in mesh_type:
            if n_node < 2:
                raise ValueError('wrong number of nodes')
            elif n_node == 2:
                cell_type=vtk.VTK_LINE 
            else:
                cell_type=vtk.VTK_POLY_LINE
        else:
            raise ValueError('unsupported mesh_type:'+mesh_type)
        return cell_type

    @staticmethod
    def get_vtk_cell_type_from_element_type(element_type):
        #element_type: tri3, quad4, hex8, etc in Febio
        #element_type: C3D8, C3D4, etc in Abaqus 
        raise NotImplementedError

    def convert_to_vtk(self):
        """Convert geometry and 2D numeric fields to independent VTK arrays.

        Coordinates and fields remain doubles, as in convert_to_vtk_slow.
        Connectivity uses VTK's native ID width. Ragged elements are flattened
        once; no individual points, cells, or field components are sent to VTK.
        """
        if _Flag_VTK_IMPORT_ == False:
            raise ValueError('vtk is not imported')
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

        def as_numpy(value, dtype):
            if torch.is_tensor(value):
                value=value.detach().cpu()
                # NumPy has no native bfloat16 dtype.
                if value.dtype == torch.bfloat16:
                    value=value.float()
                value=value.numpy()
            # numpy_to_vtk handles noncontiguous arrays when necessary.
            return np.asarray(value, dtype=dtype)

        node=as_numpy(self.node if self.node is not None else [], np.float64)
        if node.size == 0:
            node=node.reshape(0, 3)
        if node.ndim != 2 or node.shape[1] != 3:
            raise ValueError('node must have shape (N, 3)')
        point_vtk=vtk.vtkPoints()
        point_vtk.SetData(numpy_to_vtk(node, deep=True))

        id_dtype=np.int64 if vtk.vtkIdTypeArray().GetDataTypeSize() == 8 else np.int32
        n_element=0 if self.element is None else len(self.element)
        if n_element == 0:
            counts=np.empty(0, dtype=id_dtype)
            connectivity=np.empty(0, dtype=id_dtype)
        elif torch.is_tensor(self.element) or (isinstance(self.element, np.ndarray)
                                               and self.element.dtype != object):
            element=as_numpy(self.element, id_dtype)
            if element.ndim != 2:
                raise ValueError('element must be a 2D array or a list of 1D arrays')
            counts=np.full(n_element, element.shape[1], dtype=id_dtype)
            connectivity=element.reshape(-1)
        else:
            # Mixed polygons/polylines/solids may have different node counts.
            rows=[as_numpy(e, id_dtype) for e in self.element]
            if any(e.ndim != 1 for e in rows):
                raise ValueError('each element must be a 1D array')
            counts=np.array([len(e) for e in rows], dtype=id_dtype)
            connectivity=np.concatenate(rows)
        offsets=np.empty(n_element+1, dtype=id_dtype)
        offsets[0]=0
        np.cumsum(counts, out=offsets[1:])

        # Resolve types per distinct node count or element label, not per cell.
        if self.element_type is None:
            sizes, inverse=np.unique(counts, return_inverse=True)
            types=np.array([Mesh.get_vtk_cell_type(self.mesh_type, int(size))
                            for size in sizes], dtype=np.uint8)[inverse]
        else:
            if len(self.element_type) != n_element:
                raise ValueError('len(element_type) != len(element)')
            labels, inverse=np.unique(self.element_type, return_inverse=True)
            types=np.array([Mesh.get_vtk_cell_type_from_element_type(label)
                            for label in labels], dtype=np.uint8)[inverse]

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
                data=as_numpy(data, np.float64)
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
        for n in range(0, len(self.element)):
            e=[int(id) for id in self.element[n]]
            if self.element_type is None:
                cell_type=Mesh.get_vtk_cell_type(self.mesh_type, len(e))
            else:
                cell_type=Mesh.get_vtk_cell_type_from_element_type(self.element_type[n])
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

    def save_as_vtk(self, filename, vtk42=True):
        use_vtk=True
        if _Flag_VTK_IMPORT_ == True:
            if vtk42 == True:
                version=vtk.vtkVersion.GetVTKVersion()
                version=[int(a) for a in version.split('.')]
                if version[0]<=8:
                    pass
                elif version[0]>=9 and version[1]>=1:
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
                if vtk42 == False:
                    print('Mesh save_as_vtk: can only save to 4.2 version vtk, although vtk42=False')
            elif 'polygon' in self.mesh_type:
                if ('tri6' in self.mesh_type) or ('quad8' in self.mesh_type):
                    raise ValueError('unsupported mesh_type: '+self.mesh_type)
                else:
                    save_polygon_mesh_as_vtk(mesh, filename)
                    if vtk42 == False:
                        print('Mesh save_as_vtk: can only save to 4.2 version vtk, although vtk42=False')
            elif 'polyline' in self.mesh_type:
                save_polyline_mesh_as_vtk(mesh, filename)
            else:
                raise ValueError('unsupported mesh_type: '+self.mesh_type)
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
            version=vtk.vtkVersion.GetVTKVersion()
            version=[int(a) for a in version.split('.')]
            if version[0]<=8:
                pass
            elif version[0]>=9 and version[1]>=1:
                writer.SetFileVersion(42)
            else:
                print('Mesh save_as_vtk: cannot save to 4.2 version vtk')
        writer.SetFileTypeToASCII()
        writer.SetInputData(mesh_vtk)
        writer.SetFileName(filename)
        writer.Write()
    
    def save_as_vtp(self, filename):
        if _Flag_VTK_IMPORT_ == False:            
            raise ValueError('vtk is not imported')
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
        writer.Write()        

    def save_as_torch(self, filename, save_adj_info=True):
        data={'mesh_type':self.mesh_type,
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

    def load_from_torch(self, filename):
        if not os.path.isfile(filename):
            raise ValueError('not exist: '+filename)
        data=torch.load(filename, map_location='cpu', weights_only=False)
        if 'node' not in data:
            raise ValueError('node is not in data.keys()')
        if 'element' not in data:
            raise ValueError('element is not in data.keys()')
        node=data['node']
        if node is not None and not isinstance(node, torch.Tensor):
            node=torch.tensor(node)
        self.clear()
        self.node=node
        self.element=data['element']
        self.element_type=data.get('element_type')
        if 'node_set' in data.keys():
            self.node_set=data['node_set']
        if 'element_set' in data.keys():
            self.element_set=data['element_set']
        if 'node_data' in data.keys():
            self.node_data=data['node_data']
        if 'element_data' in data.keys():
            self.element_data=data['element_data']
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
    
    def load(self, filename, dtype=None):
        if not os.path.isfile(filename):
            raise ValueError('not exist: '+filename)        
        if dtype is None:
            dtype='float32'
        filenamebase, file_type = os.path.splitext(filename)
        if file_type == '.vtk':
            self.load_from_vtk(filename, dtype)
        elif  file_type == '.vtp':
            self.load_from_vtp(filename, dtype)
        elif  file_type == '.stl':
            self.load_from_stl(filename, dtype)
        elif file_type == '.pt' or file_type == '.pth':
            self.load_from_torch(filename)
            if dtype is not None:
                print('warning: dtype is not None and it is ignored @ mesh.load')
        else:
            raise ValueError('unknown file_type:'+file_type)
            
    def copy(self, node, element, dtype=None, detach=True):
        """Replace geometry with independent copies and reset other mesh state."""
        if node is None:
            node_copy=None
        elif isinstance(node, torch.Tensor):
            if dtype is None:
                node_copy=node.clone()
            else:
                node_copy=node.clone().to(dtype)
            if detach==True:
                node_copy=node_copy.detach()
        elif isinstance(node, np.ndarray):
            if dtype is None:
                node_copy=torch.tensor(node.copy())
            else:
                node_copy=torch.tensor(node.copy(), dtype=dtype)
        elif isinstance(node, tuple) or isinstance(node, list):
            if dtype is None:
                node_copy=torch.tensor(node, dtype=torch.float64)
            else:
                node_copy=torch.tensor(node, dtype=dtype)
        else:
            raise NotImplementedError
        if element is None:
            element_copy=None
        elif isinstance(element, torch.Tensor):
            element_copy=element.clone()
            if detach==True:
                element_copy=element_copy.detach()
        elif isinstance(element, np.ndarray):
            element_copy=torch.tensor(element.copy(), dtype=torch.int64)
        elif isinstance(element, tuple) or isinstance(element, list):
            try:
                element_copy=torch.tensor(element, dtype=torch.int64)
            except:
                element_copy=deepcopy(element)
        else:
            raise NotImplementedError
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
        adj_link=torch.tensor(adj_link, dtype=torch.int64)
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
                    adj_link.append([eid1, eid2])
                    adj_link.append([eid2, eid1])
        adj_link=torch.tensor(adj_link, dtype=torch.int64)
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
                    adj_link.append([eid1, eid2])
                    adj_link.append([eid2, eid1])
        adj_link=torch.tensor(adj_link, dtype=torch.int64)
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
                    adj_link.append([eid1, eid2])
                    adj_link.append([eid2, eid1])
        adj_link=torch.tensor(adj_link, dtype=torch.int64)
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

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        if element_idx_list is not None:
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
            new_element_type=(self.element_type[element_idx_list] if element_idx_list is not None
                              else self.element_type[:0])
        new_mesh=Mesh(new_node, new_element, new_element_type, self.mesh_type)
        if return_node_idx_list == False:
            return new_mesh
        else:
            return new_mesh, node_idx_list
#%%
