import torch
import torch_scatter
import numpy as np
from torch.linalg import det, cross
from .polyhedron_mesh import PolyhedronMesh
#%%
class TetrahedronMesh(PolyhedronMesh):
    #4-node C3D4/TET4 mesh
    def __init__(self, node=None, element=None):
        super().__init__(node=node, element=element)
        self.mesh_type='polyhedron_tet4'

    def clear(self):
        super().clear()

    def clear_geometry_derived_data(self):
        super().clear_geometry_derived_data()
        self.element_volume=None

    def build_edge(self):
        self.build_element_to_edge_adj_table()
        
    def build_element_to_edge_adj_table(self):
        element=self.element
        if not isinstance(element, list):
            element=element.tolist()
        edge=[]
        for m in range(0, len(element)):
            id0, id1, id2, id3=element[m]
            edge.append([id0, id1])
            edge.append([id0, id2])
            edge.append([id0, id3])
            edge.append([id1, id2])
            edge.append([id1, id3])
            edge.append([id2, id3])
        edge=np.array(edge, dtype=np.int64)
        edge=np.sort(edge, axis=1)
        edge_unique, inverse=np.unique(edge, return_inverse=True, axis=0)
        self.edge=torch.tensor(edge_unique, dtype=torch.int64)
        self.element_to_edge_adj_table=inverse.reshape(-1,6).tolist()
    
    def build_face(self):
        #self.face[k] is a triangle: [node_idx0, node_idx1, node_idx2]
        self.build_element_to_face_adj_table()

    def build_element_to_face_adj_table(self):
        element=self.element
        if not isinstance(element, list):
            element=element.tolist()
        face=[]
        for m in range(0, len(element)):
            id0, id1, id2, id3=element[m]
            #face normal is from inside to outside of element[m]
            face.append([id0, id2, id1])
            face.append([id0, id1, id3])
            face.append([id0, id3, id2])
            face.append([id1, id2, id3])
        face=np.array(face, dtype=np.int64)
        face_sorted=np.sort(face, axis=1)
        face_sorted_unique, index, inverse=np.unique(face_sorted, return_index=True, return_inverse=True, axis=0)
        self.face=torch.tensor(face[index], dtype=torch.int64)
        self.element_to_face_adj_table=inverse.reshape(-1,4).tolist()

    def update_element_volume(self):
        self.element_volume=TetrahedronMesh.cal_element_volume(self.node, self.element)
    
    @staticmethod
    def cal_element_volume(node, element):
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        a=x1-x0
        b=x2-x0
        c=x3-x0
        volume=(1/6)*(cross(a,b)*c).sum(dim=-1).abs() #shape (M,)
        return volume

    def subdivide(self):
        #draw a 3D figure and code this...
        pass

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        sub_mesh, node_idx_list=super().get_sub_mesh(element_idx_list, return_node_idx_list=True)
        sub_mesh=TetrahedronMesh(sub_mesh.node, sub_mesh.element)
        if return_node_idx_list == False:
            return sub_mesh
        else:
            return sub_mesh, node_idx_list        
#%%
