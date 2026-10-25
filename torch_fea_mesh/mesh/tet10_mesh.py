import torch
#import torch_scatter
import numpy as np
from torch.linalg import det, cross
from .tetrahedron_mesh import TetrahedronMesh
from .polyhedron_mesh import PolyhedronMesh
#%%
class Tet10Mesh(PolyhedronMesh):
    #10-node C3D10/TET10 mesh
    def __init__(self, node=None, element=None):
        super().__init__(node=node, element=element)
        self.mesh_type='polyhedron_tet10'

    def clear(self):
        super().clear()

    def clear_geometry_derived_data(self):
        super().clear_geometry_derived_data()
        self.element_volume=None

    def create_from_tet4(self, tet4_mesh):
        #tet4_mesh: TetrahedronMesh
        #add a node in the middle of each edge
        if tet4_mesh.edge is None:
            tet4_mesh.build_edge()
        x_j=tet4_mesh.node[tet4_mesh.edge[:,0]]
        x_i=tet4_mesh.node[tet4_mesh.edge[:,1]]
        nodeA=(x_j+x_i)/2
        #create new mesh
        node_new=torch.cat([tet4_mesh.node, nodeA], dim=0)
        element=tet4_mesh.element.tolist()
        element_new=[]
        for m in range(0, len(element)):
            #see element.pptx in pytorch-fea document
            id0=element[m][0]
            id1=element[m][1]
            id2=element[m][2]
            id3=element[m][3]
            id4=tet4_mesh.node.shape[0]+tet4_mesh.get_edge_idx_from_node_pair(id0, id1)
            id5=tet4_mesh.node.shape[0]+tet4_mesh.get_edge_idx_from_node_pair(id1, id2)
            id6=tet4_mesh.node.shape[0]+tet4_mesh.get_edge_idx_from_node_pair(id0, id2)
            id7=tet4_mesh.node.shape[0]+tet4_mesh.get_edge_idx_from_node_pair(id0, id3)
            id8=tet4_mesh.node.shape[0]+tet4_mesh.get_edge_idx_from_node_pair(id1, id3)
            id9=tet4_mesh.node.shape[0]+tet4_mesh.get_edge_idx_from_node_pair(id2, id3)
            element_new.append([id0, id1, id2, id3, id4, id5, id6, id7, id8, id9])
        self.__init__(node_new, element_new)
    
    def simplify_to_tet4(self, return_node_idx_list=False):        
        element_out=self.element[:,0:4].clone()
        node_idx_list, element_out=torch.unique(element_out.reshape(-1), return_inverse=True)
        node_out=self.node[node_idx_list]
        element_out=element_out.reshape(-1,4)
        mesh_new=TetrahedronMesh(node_out, element_out)
        if return_node_idx_list == False:
            return mesh_new
        else:
            return mesh_new, node_idx_list 
        
    def build_edge(self):
        self.build_element_to_edge_adj_table()
        
    def build_element_to_edge_adj_table(self):
        element=self.element
        if not isinstance(element, list):
            element=element.tolist()
        edge=[]
        for m in range(0, len(element)):
            id0, id1, id2, id3, id4, id5, id6, id7, id8, id9=element[m]
            edge.append([id0, id4])
            edge.append([id0, id6])
            edge.append([id0, id7])
            edge.append([id1, id4])
            edge.append([id1, id5])
            edge.append([id1, id8])
            edge.append([id2, id5])
            edge.append([id2, id6])
            edge.append([id2, id9])
            edge.append([id3, id7])
            edge.append([id3, id8])
            edge.append([id3, id9])
        edge=np.array(edge, dtype=np.int64)
        edge=np.sort(edge, axis=1)
        edge_unique, inverse=np.unique(edge, return_inverse=True, axis=0)
        self.edge=torch.tensor(edge_unique, dtype=torch.int64)
        self.element_to_edge_adj_table=inverse.reshape(-1,12).tolist()
    
    def build_face(self):
        #self.face[k] is a tri6 element
        self.build_element_to_face_adj_table()

    def build_element_to_face_adj_table(self):
        element=self.element
        if not isinstance(element, list):
            element=element.tolist()
        face=[]
        for m in range(0, len(element)):
            id0, id1, id2, id3, id4, id5, id6, id7, id8, id9=element[m]
            #face normal is from inside to outside of element[m]
            face.append([id0, id2, id1, id6, id5, id4])
            face.append([id0, id1, id3, id4, id8, id7])
            face.append([id0, id3, id2, id7, id9, id6])
            face.append([id1, id2, id3, id5, id9, id8])
        face=np.array(face, dtype=np.int64)
        face_sorted=np.sort(face, axis=1)
        face_sorted_unique, index, inverse=np.unique(face_sorted, return_index=True, return_inverse=True, axis=0)
        self.face=torch.tensor(face[index], dtype=torch.int64)
        self.element_to_face_adj_table=inverse.reshape(-1,4).tolist()
    
    def update_element_volume(self):
        self.element_volume=Tet10Mesh.cal_element_volume(self.node, self.element)
    
    @staticmethod
    def cal_element_volume(node, element):
        #tet4 formula
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
        sub_mesh=Tet10Mesh(sub_mesh.node, sub_mesh.element)
        if return_node_idx_list == False:
            return sub_mesh
        else:
            return sub_mesh, node_idx_list              
#%%
