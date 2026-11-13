import torch
#import torch_scatter
import numpy as np
from torch.linalg import det, cross
from torch_fea_mesh.mesh.tetrahedron_mesh import TetrahedronMesh
from torch_fea_mesh.mesh.polyhedron_mesh import PolyhedronMesh
#%%
class Tet10Mesh(PolyhedronMesh):
    #10-node C3D10/TET10 mesh
    def __init__(self, node=None, element=None):
        super().__init__(node=node, element=element)
        self.mesh_type='polyhedron_tet10'

    def clear(self):
        super().clear()

    def clear_geometry_derived_data(self, clear_adjacency_data=True):
        super().clear_geometry_derived_data(clear_adjacency_data)
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
        edge=np.array(edge, dtype=np.int64).reshape(-1,2)
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
        face=np.array(face, dtype=np.int64).reshape(-1,6)
        face_sorted=np.sort(face, axis=1)
        face_sorted_unique, index, inverse=np.unique(face_sorted, return_index=True, return_inverse=True, axis=0)
        self.face=torch.tensor(face[index], dtype=torch.int64)
        self.element_to_face_adj_table=inverse.reshape(-1,4).tolist()
    
    def update_element_volume(self):
        self.element_volume=Tet10Mesh.cal_element_volume(self.node, self.element)
    
    @staticmethod
    def cal_element_volume(node, element):
        #volume of the quadratic C3D10 element: the integral of det(dX/dr) over the reference tetrahedron
        #det(dX/dr) is cubic in r,s,t, and the 5-point rule below is exact up to cubic
        #so the result is exact, also for a curved element whose midside nodes are off the straight edges
        x=1.0*node[element]#shape (M,10,3), 1.0* allows integer nodes
        x=x-x[:,0:1]#relative to node 0, to keep precision far from the origin
        #integration points in barycentric coordinates (L0,L1,L2,L3), where r=L1, s=L2, t=L3
        L=torch.tensor([[1/4,1/4,1/4,1/4],
                        [1/2,1/6,1/6,1/6],
                        [1/6,1/2,1/6,1/6],
                        [1/6,1/6,1/2,1/6],
                        [1/6,1/6,1/6,1/2]], dtype=x.dtype, device=x.device)
        weight=[-4/5, 9/20, 9/20, 9/20, 9/20]#the weights add up to 1
        #dL[n] is (dLn/dr, dLn/ds, dLn/dt)
        dL=torch.tensor([[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]], dtype=x.dtype, device=x.device)
        #midside nodes 4 to 9 are between corner nodes i[k] and j[k]
        i=[0,1,0,0,1,2]
        j=[1,2,2,3,3,3]
        volume=0
        for L_k, w_k in zip(L, weight):
            #shape function: L[n]*(2*L[n]-1) for corner node n, 4*L[i]*L[j] for a midside node
            dh_corner=(4*L_k-1).unsqueeze(1)*dL#shape (4,3)
            dh_midside=4*(L_k[i].unsqueeze(1)*dL[j]+L_k[j].unsqueeze(1)*dL[i])#shape (6,3)
            dh_dr=torch.cat([dh_corner, dh_midside], dim=0)#shape (10,3)
            dx_dr=x.transpose(1,2)@dh_dr#shape (M,3,3), the columns are dx/dr, dx/ds, dx/dt
            volume=volume+w_k*(cross(dx_dr[:,:,0], dx_dr[:,:,1])*dx_dr[:,:,2]).sum(dim=-1)
        return (volume/6).abs()#shape (M,), 1/6 is the volume of the reference tetrahedron

    def subdivide(self):
        #draw a 3D figure and code this...
        raise NotImplementedError('subdivide is not implemented for Tet10Mesh')

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        sub_mesh, node_idx_list=super().get_sub_mesh(element_idx_list, return_node_idx_list=True)
        sub_mesh=Tet10Mesh(sub_mesh.node, sub_mesh.element)
        if return_node_idx_list == False:
            return sub_mesh
        else:
            return sub_mesh, node_idx_list              
#%%
