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
        edge=np.array(edge, dtype=np.int64)
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
        face=np.array(face, dtype=np.int64)
        face_sorted=np.sort(face, axis=1)
        face_sorted_unique, index, inverse=np.unique(face_sorted, return_index=True, return_inverse=True, axis=0)
        self.face=torch.tensor(face[index], dtype=torch.int64)
        self.element_to_face_adj_table=inverse.reshape(-1,6).tolist()

    def update_element_volume(self):
        self.element_volume=HexahedronMesh.cal_element_volume(self.node, self.element)
    
    @staticmethod
    def cal_element_volume(node, element):
        #X=self.node[self.element]#shape (M,8,3)
        #r1i=get_integration_point_1i(X.dtype, X.device)
        #dX_dr=cal_dh_dr(r1i, X)
        #volumn=8*det(dX_dr)
        #---------no need for FEA element---
        #divide a hex8 to 12 tet4, and cal_vol of each tet4
        def cal_vol(node0, node1, node2, node3):
            a=node1-node0; b=node2-node0; c=node3-node0
            vol=(1/6)*(cross(a,b)*c).sum(dim=-1).abs() 
            return vol                   
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        x4=node[element[:,4]]
        x5=node[element[:,5]]
        x6=node[element[:,6]]
        x7=node[element[:,7]]
        volume=(cal_vol(x1,x2,x3,x5)
               +cal_vol(x4,x7,x5,x3)
               +cal_vol(x0,x4,x5,x3)
               +cal_vol(x2,x5,x6,x3)
               +cal_vol(x5,x7,x6,x3)
               +cal_vol(x0,x5,x1,x3))
        return volume

    def subdivide(self):
        #draw a 3D figure and code this...
        pass

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        sub_mesh, node_idx_list=super().get_sub_mesh(element_idx_list, return_node_idx_list=True)
        sub_mesh=HexahedronMesh(sub_mesh.node, sub_mesh.element)
        if return_node_idx_list == False:
            return sub_mesh
        else:
            return sub_mesh, node_idx_list
#%%
