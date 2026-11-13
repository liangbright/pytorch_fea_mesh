import torch
import numpy as np
from copy import deepcopy
from torch_fea_mesh.mesh.mesh import Mesh
import json
#%%
class PolygonMesh(Mesh):

    def __init__(self, node=None, element=None):
        super().__init__(node=node, element=element, element_type=None, mesh_type='polygon')

    def clear(self):
        super().clear()

    def clear_geometry_derived_data(self, clear_adjacency_data=True):
        super().clear_geometry_derived_data(clear_adjacency_data)

    def save_as_mdk_json(self, filename):
        #to be compatible with MDK SavePolygonMeshFromJsonDataFile
        #use json to store node_set, element_set, etc
        #use vtk to store node, element, node_data, element_data, etc
        data={}
        data["Name"]=self.name
        data["ObjectType"]="PolygonMesh"
        data["ScalarType"]="double"
        data["IndexType"]="int_max"
        data["PointCount"]=self.node.shape[0]
        data["FaceCount"]=len(self.element)
        data["PointSetCount"]=len(self.node_set)
        data["FaceSetCount"]=len(self.element_set)
        data["PointSetList"]=[]
        #a set may be a tensor or a numpy array
        for i, (k, v) in enumerate(self.node_set.items()):
            data["PointSetList"].append({k:np.asarray(self.copy_to_list(v)).tolist()})
        data["FaceSetList"]=[]
        for i, (k, v) in enumerate(self.element_set.items()):
            data["FaceSetList"].append({k:np.asarray(self.copy_to_list(v)).tolist()})
        data["PointAndFace"]=filename.split("/")[-1]+".vtk"
        #dumps comes first, so that a failure does not leave a truncated file
        text=json.dumps(data, indent=4)
        with open(filename, "w") as outfile:
            outfile.write(text)
        self.save_as_vtk(filename+".vtk", vtk42=True)

    def load_from_mdk_json(self, filename, dtype):
        self.load_from_vtk(filename+".vtk", dtype)
        with open(filename) as f:
            data=json.load(f)
        for k in range(0, data["PointSetCount"]):
            temp=data["PointSetList"][k]
            name=list(temp.keys())[0]
            value=list(temp.values())[0]
            self.node_set[name]=value
        for k in range(0, data["FaceSetCount"]):
            temp=data["FaceSetList"][k]
            name=list(temp.keys())[0]
            value=list(temp.values())[0]
            self.element_set[name]=value
        self.name=data["Name"]

    def make_element_tensor(self):
        #for a class with a fixed number of nodes in an element, such as TriangleMesh
        #self.element can be a python list: assigned by hand, or restored by load_from_torch, which has no width check
        #return a tensor (M, number of nodes in an element). self.element is not changed
        if self.node is None or self.element is None:
            raise ValueError('this '+type(self).__name__+' has no node or no element')
        return self.convert_element(self.element, self.node.device)

    def build_edge(self):
        self.build_element_to_edge_adj_table()
        
    def build_element_to_edge_adj_table(self):
        self.check_element_type(self.element_type)
        flag_quadratic_element=False
        if ('tri6' in self.mesh_type) or ('quad8' in self.mesh_type):
            flag_quadratic_element=True
        element=self.element
        if not isinstance(element, list):
            element=element.tolist()
        edge=[]
        for m in range(0, len(element)):
            elm=element[m]
            if flag_quadratic_element==True:
                #re-order the nodes in elm
                elmA=elm[0:len(elm)//2]
                elmB=elm[len(elm)//2:]
                elm=[]
                for n in range(0, len(elmA)):
                   elm.append(elmA[n]) 
                   elm.append(elmB[n])               
            for k in range(0, len(elm)):
                if k < len(elm)-1:
                    a=elm[k]; b=elm[k+1]
                else:
                    a=elm[k]; b=elm[0]
                if a < b:
                    edge.append([a, b])
                else:
                    edge.append([b, a])
        #reshape: np.array([]) has shape (0,), but self.edge must have shape (K,2)
        edge=np.array(edge, dtype=np.int64).reshape(-1,2)
        edge_unique, inverse=np.unique(edge, return_inverse=True, axis=0)
        self.edge=torch.tensor(edge_unique, dtype=torch.int64)
        use_slow_method=True
        if torch.is_tensor(self.element):
            if len(self.element.shape) == 2:
                self.element_to_edge_adj_table=inverse.reshape(self.element.shape).tolist()
                use_slow_method=False
        if use_slow_method==True:
            adj_table=[]
            idx=0
            for m in range(0, len(element)):
                adj_table.append(inverse[idx:(idx+len(element[m]))].tolist())
                idx=idx+len(element[m])        
            self.element_to_edge_adj_table=adj_table

    def find_boundary_node(self):
        #return index list of nodes on boundary
        if self.edge is None:
            self.build_edge()
        if self.edge_to_element_adj_table is None:
            self.build_edge_to_element_adj_table()
        edge=self.edge.detach().cpu().numpy()
        boundary=[]
        for k in range(0, len(edge)):
            adj_elm_idx=self.edge_to_element_adj_table[k]
            if len(adj_elm_idx) == 1:
                boundary.append(edge[k,0])
                boundary.append(edge[k,1])
        boundary=np.unique(boundary).tolist()
        return boundary
    
    def find_boundary_edge(self):
        if self.edge is None:
            self.build_edge()
        if self.edge_to_element_adj_table is None:
            self.build_edge_to_element_adj_table()
        edge=self.edge.detach().cpu().numpy()
        boundary=[]
        for k in range(0, len(edge)):
            adj_elm_idx=self.edge_to_element_adj_table[k]
            if len(adj_elm_idx) == 1:
                boundary.append(k)
        return boundary
    
    def find_boundary_node_and_edge(self):
        if self.edge is None:
            self.build_edge()
        if self.edge_to_element_adj_table is None:
            self.build_edge_to_element_adj_table()
        edge=self.edge.detach().cpu().numpy()
        boundary_node=[]
        boundary_edge=[]
        for k in range(0, len(edge)):
            adj_elm_idx=self.edge_to_element_adj_table[k]
            if len(adj_elm_idx) == 1:
                boundary_node.append(edge[k,0])
                boundary_node.append(edge[k,1])
                boundary_edge.append(k)
        boundary_node=np.unique(boundary_node).tolist()
        return boundary_node, boundary_edge
    
    def find_boundary_element(self, adj):
        #return index list of elements on boundary
        if adj == "edge":
            if self.edge is None:
                self.build_edge()
            if self.edge_to_element_adj_table is None:
                self.build_edge_to_element_adj_table()
            edge=self.edge.detach().cpu().numpy()
            boundary=[]
            for k in range(0, len(edge)):
                adj_elm_idx=self.edge_to_element_adj_table[k]
                if len(adj_elm_idx) == 1:
                    boundary.append(adj_elm_idx[0])
            boundary=np.unique(boundary).tolist()
            return boundary
        elif adj == "node":
            boundary_node=self.find_boundary_node()
            if self.node_to_element_adj_table is None:
                self.build_node_to_element_adj_table()
            boundary=[]
            for node_idx in boundary_node:
                adj_elm_idx=self.node_to_element_adj_table[node_idx]
                boundary.extend(adj_elm_idx)
            boundary=np.unique(boundary).tolist()
            return boundary
        else:
            raise ValueError
        
    def is_quad(self):
        if isinstance(self.element, torch.Tensor):
            if self.element.shape[1] == 4:
                #this is QuadMesh
                return True
            else:
                return False
        if len(self.element) == 0:
            return False
        m_min=len(self.element[0])
        m_max=len(self.element[0])
        for m in range(1, len(self.element)):
            m_min=min(m_min, len(self.element[m]))
            m_max=max(m_max, len(self.element[m]))
        if m_min == m_max == 4:
            return True
        else:
            return False

    def is_tri(self):
        if isinstance(self.element, torch.Tensor):
            if self.element.shape[1] == 3:
                #this is TriangleMesh
                return True
            else:
                return False
        if len(self.element) == 0:
            return False
        m_min=len(self.element[0])
        m_max=len(self.element[0])
        for m in range(1, len(self.element)):
            m_min=min(m_min, len(self.element[m]))
            m_max=max(m_max, len(self.element[m]))
        if m_min == m_max == 3:
            return True
        else:
            return False

    def is_quad_tri(self):
        if isinstance(self.element, torch.Tensor):
            if self.element.shape[1] == 3 or self.element.shape[1] == 4:
                return True
        if len(self.element) == 0:
            return False
        m_min=len(self.element[0])
        m_max=len(self.element[0])
        for m in range(1, len(self.element)):
            m_min=min(m_min, len(self.element[m]))
            m_max=max(m_max, len(self.element[m]))
        if (m_min >= 3) and (m_max <= 4):
            return True
        else:
            return False

    def _is_quad_cut_along_x0x2(self):
        #choose the diagonal for quad_to_tri with mode='auto'
        #return a list with one bool per quad, in element order
        #True: cut along x0-x2; False: cut along x1-x3
        #a cut is valid if both of its triangles face the same way as the whole quad:
        #   a cut outside a concave quad gives one triangle that faces the other way
        #   a cut through a straight corner gives a triangle with zero area
        #if only one cut is valid, use it
        #otherwise use the shorter cut: both are valid in a convex quad, and neither in a bowtie
        #the normal of the whole quad is used, so that a warped quad that is convex when seen along
        #its normal keeps the shorter cut, even if the normals of its two triangles are far apart
        quad=[self.copy_to_list(elm) for elm in self.element]
        quad=[elm for elm in quad if len(elm) == 4]
        quad=torch.tensor(quad, dtype=torch.int64).reshape(-1,4)
        x=self.node.detach()[quad]
        x0, x1, x2, x3=x.unbind(dim=1)
        is_x0x2_shorter=((x0-x2)**2).sum(dim=1) < ((x1-x3)**2).sum(dim=1)
        #float64: so that the sign of the test is right for a thin quad in a float32 mesh
        x=x.to(torch.float64)
        if x.shape[2] == 2:
            x=torch.nn.functional.pad(x, (0,1))
        x0, x1, x2, x3=x.unbind(dim=1)
        normal=torch.linalg.cross(x2-x0, x3-x1, dim=1)
        def is_cut_valid(a, b, c, d):
            #cut along a-c: triangles (a, b, c) and (a, c, d)
            normal_abc=torch.linalg.cross(b-a, c-a, dim=1)
            normal_acd=torch.linalg.cross(c-a, d-a, dim=1)
            return ((normal_abc*normal).sum(dim=1) > 0) & ((normal_acd*normal).sum(dim=1) > 0)
        is_x0x2_valid=is_cut_valid(x0, x1, x2, x3)
        is_x1x3_valid=is_cut_valid(x1, x2, x3, x0)
        return torch.where(is_x0x2_valid == is_x1x3_valid, is_x0x2_shorter, is_x0x2_valid).tolist()

    def quad_to_tri(self, mode=0, clone_node=True):
        #divide every quad element to two triangle elements        
        if isinstance(self.element, torch.Tensor):
            if self.element.shape[1] == 3:
                node=self.node
                if clone_node == True:
                    node=node.clone()
                new_mesh=PolygonMesh(node, self.element.clone())
                return new_mesh
        #-------------------------------------------------------    
        if mode == 'auto':
            is_cut_along_x0x2=iter(self._is_quad_cut_along_x0x2())
        element_new=[]
        for m in range(0, len(self.element)):
            elm=self.copy_to_list(self.element[m])
            if len(elm) == 4:
                #-----------
                # x3------x2
                # |       |
                # |       |
                # x0------x1
                # mode=0: cut along x0-x2
                # mode=1: cut along x1-x3
                # mode='auto': cut along the shorter diagonal that is inside the quad
                #-----------
                id0, id1, id2, id3=elm
                if mode == 0:
                    element_new.append([id0, id2, id3])
                    element_new.append([id0, id1, id2])
                elif mode == 1:
                    element_new.append([id0, id1, id3])
                    element_new.append([id1, id2, id3])
                elif mode == 'auto':
                    if next(is_cut_along_x0x2):
                        element_new.append([id0, id2, id3])
                        element_new.append([id0, id1, id2])
                    else:
                        element_new.append([id0, id1, id3])
                        element_new.append([id1, id2, id3])
                else:
                    raise ValueError("mode is invalid")
            else:
                element_new.append(elm)
        if len(element_new) == 0:
            #a mesh with no elements: torch.tensor([]) has shape (0,), which is not a valid element array
            element_new=torch.empty((0,3), dtype=torch.int64)
        node=self.node
        if clone_node == True:
            node=node.clone()
        new_mesh=PolygonMesh(node, element_new)   
        return new_mesh   
    
    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        sub_mesh, node_idx_list=super().get_sub_mesh(element_idx_list, return_node_idx_list=True)
        element_type=sub_mesh.element_type
        sub_mesh=PolygonMesh(sub_mesh.node, sub_mesh.element)
        sub_mesh.element_type=element_type
        #mesh_type can say more than polygon: load_from_torch can set polygon_tri6
        sub_mesh.mesh_type=self.mesh_type
        if return_node_idx_list == False:
            return sub_mesh
        else:
            return sub_mesh, node_idx_list
#%%
