# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import torch
#import torch_scatter
#from torch_sparse import SparseTensor
import numpy as np
#import json
from torch.linalg import cross
from .polyhedron_mesh import PolyhedronMesh
#%%
class HexahedronMesh(PolyhedronMesh):
    #8-node C3D8 mesh
    def __init__(self, node=None, element=None):
        super().__init__(node=node, element=element)
        self.mesh_type='polyhedron_hex8'

    def clear(self):
        super().clear()

    def clear_geometry_derived_data(self, clear_adjacency_data=True):
        super().clear_geometry_derived_data(clear_adjacency_data)
        self.element_volume=None

    def build_edge(self):
        self.build_element_to_edge_adj_table()
        
    def build_element_to_edge_adj_table(self):
        element=self.element
        if not isinstance(element, list):
            element=element.tolist()
        edge=[]
        for m in range(0, len(element)):
            id0, id1, id2, id3, id4, id5, id6, id7=element[m]
            edge.append([id0, id1])
            edge.append([id1, id2])
            edge.append([id2, id3])
            edge.append([id3, id0])
            edge.append([id4, id5])
            edge.append([id5, id6])
            edge.append([id6, id7])
            edge.append([id7, id4])
            edge.append([id0, id4])
            edge.append([id1, id5])
            edge.append([id2, id6])
            edge.append([id3, id7])            
        edge=np.array(edge, dtype=np.int64).reshape(-1,2)
        edge=np.sort(edge, axis=1)
        edge_unique, inverse=np.unique(edge, return_inverse=True, axis=0)
        self.edge=torch.tensor(edge_unique, dtype=torch.int64)
        self.element_to_edge_adj_table=inverse.reshape(-1,12).tolist()

    def build_face(self):
        #self.face[k] is a quad: [node_idx0, node_idx1, node_idx2, node_idx3]
        self.build_element_to_face_adj_table()

    def build_element_to_face_adj_table(self):
        element=self.element
        if not isinstance(element, list):
            element=element.tolist()
        face=[]
        for m in range(0, len(element)):
            id0, id1, id2, id3, id4, id5, id6, id7=element[m]
            #face normal is from inside to outside of element[m]
            face.append([id0, id3, id2, id1])
            face.append([id4, id5, id6, id7])
            face.append([id0, id1, id5, id4])
            face.append([id1, id2, id6, id5])
            face.append([id2, id3, id7, id6])
            face.append([id3, id0, id4, id7])
        face=np.array(face, dtype=np.int64).reshape(-1,4)
        face_sorted=np.sort(face, axis=1)
        face_sorted_unique, index, inverse=np.unique(face_sorted, return_index=True, return_inverse=True, axis=0)
        self.face=torch.tensor(face[index], dtype=torch.int64)
        self.element_to_face_adj_table=inverse.reshape(-1,6).tolist()

    def update_element_volume(self):
        self.element_volume=HexahedronMesh.cal_element_volume(self.node, self.element)
    
    @staticmethod
    def cal_element_volume(node, element):
        #volume of the trilinear C3D8 element: the integral of det(dX/dr) over r,s,t in [-1,1]
        #2x2x2 Gauss integration is exact, because det(dX/dr) is quadratic in each of r,s,t
        #so the result is correct for warped faces and does not depend on the local node numbering
        x=1.0*node[element]#shape (M,8,3), 1.0* allows integer nodes
        x=x-x[:,0:1]#relative to node 0, to keep precision far from the origin
        #natural coordinates of the 8 nodes
        r=torch.tensor([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                        [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]], dtype=x.dtype, device=x.device)
        volume=0
        for r_i in r/3**0.5:#8 integration points, each with weight 1
            a=1+r*r_i#shape (8,3)
            dh_dr=(r/8)*torch.stack([a[:,1]*a[:,2], a[:,0]*a[:,2], a[:,0]*a[:,1]], dim=1)#shape (8,3)
            dx_dr=x.transpose(1,2)@dh_dr#shape (M,3,3), the columns are dx/dr, dx/ds, dx/dt
            volume=volume+(cross(dx_dr[:,:,0], dx_dr[:,:,1])*dx_dr[:,:,2]).sum(dim=-1)
        return volume.abs()#shape (M,)

    def subdivide(self):
        #draw a 3D figure and code this...
        raise NotImplementedError('subdivide is not implemented for HexahedronMesh')

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        sub_mesh, node_idx_list=super().get_sub_mesh(element_idx_list, return_node_idx_list=True)
        sub_mesh=HexahedronMesh(sub_mesh.node, sub_mesh.element)
        if return_node_idx_list == False:
            return sub_mesh
        else:
            return sub_mesh, node_idx_list
#%%
