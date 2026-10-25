import torch
import numpy as np
from copy import deepcopy
from .mesh import Mesh
import json
#%%
class PolygonMesh(Mesh):

    def __init__(self, node=None, element=None):
        super().__init__(node=node, element=element, element_type=None, mesh_type='polygon')

    def clear(self):
        super().clear()

    def clear_geometry_derived_data(self):
        super().clear_geometry_derived_data()

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
        for i, (k, v) in enumerate(self.node_set.items()):
            data["PointSetList"].append({k:v})
        data["FaceSetList"]=[]
        for i, (k, v) in enumerate(self.element_set.items()):
            data["FaceSetList"].append({k:v})
        data["PointAndFace"]=filename.split("/")[-1]+".vtk"
        with open(filename, "w") as outfile:
            json.dump(data, outfile, indent=4)
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

    def build_edge(self):
        self.build_element_to_edge_adj_table()
        
    def build_element_to_edge_adj_table(self):
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
        edge=np.array(edge, dtype=np.int64)
        edge_unique, inverse=np.unique(edge, return_inverse=True, axis=0)
        self.edge=torch.tensor(edge_unique, dtype=torch.int64)
        use_slow_method=True
        if torch.is_tensor(self.element):
            if len(self.element.shape) == 2:
                self.element_to_edge_adj_table=inverse.reshape(-1,self.element.shape[1]).tolist()
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
        m_min=len(self.element[0])
        m_max=len(self.element[0])
        for m in range(1, len(self.element)):
            m_min=min(m_min, len(self.element[m]))
            m_max=max(m_max, len(self.element[m]))
        if (m_min >= 3) and (m_max <= 4):
            return True
        else:
            return False

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
                #-----------
                id0, id1, id2, id3=elm
                if mode == 0:
                    element_new.append([id0, id2, id3])
                    element_new.append([id0, id1, id2])
                elif mode == 1:
                    element_new.append([id0, id1, id3])
                    element_new.append([id1, id2, id3])
                elif mode == 'auto':
                    L02=((self.node[id0]-self.node[id2])**2).sum()
                    L13=((self.node[id1]-self.node[id3])**2).sum()
                    if L02 < L13:
                        element_new.append([id0, id2, id3])
                        element_new.append([id0, id1, id2])
                    else:
                        element_new.append([id0, id1, id3])
                        element_new.append([id1, id2, id3])
                else:
                    raise ValueError("mode is invalid")
            else:
                element_new.append(elm)
        node=self.node
        if clone_node == True:
            node=node.clone()
        new_mesh=PolygonMesh(node, element_new)   
        return new_mesh   
    
    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        sub_mesh, node_idx_list=super().get_sub_mesh(element_idx_list, return_node_idx_list=True)
        sub_mesh=PolygonMesh(sub_mesh.node, sub_mesh.element)
        if return_node_idx_list == False:
            return sub_mesh
        else:
            return sub_mesh, node_idx_list
#%%
