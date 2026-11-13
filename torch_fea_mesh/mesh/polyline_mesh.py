# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:40:25 2024

@author: liang
"""
import torch
import numpy as np
from torch_fea_mesh.mesh.mesh import Mesh
#%%
class PolylineMesh(Mesh):
    def __init__(self, node=None, element=None):
        super().__init__(node=node, element=element, element_type=None, mesh_type='polyline')

    def clear(self):
        super().clear()

    def clear_geometry_derived_data(self, clear_adjacency_data=True):
        super().clear_geometry_derived_data(clear_adjacency_data)

    def initialize(self, curve):
        #curve may be
        # a single curve: a 2D numpy/torch array or a list of points
        # a list of curves: each is a 2D numpy/torch array or a list of points
        # a 3D numpy/torch array: a list of curves of the same length
        #a point in a list of points may be a 1D numpy/torch array
        if not (torch.is_tensor(curve) or isinstance(curve, (np.ndarray, list, tuple))):
            raise ValueError('unsupported type of curve: '+str(type(curve)))
        if len(curve) == 0:
            self.clear()
            return
        if torch.is_tensor(curve) or isinstance(curve, np.ndarray):
            if len(curve.shape) == 2:
                curve_list=[curve]
            elif len(curve.shape) == 3:
                curve_list=curve
            else:
                raise ValueError('a curve array must be 2D or 3D, but its shape is '+str(tuple(curve.shape)))
        else:
            try:
                first=curve[0][0]
                temp=float(first.detach() if torch.is_tensor(first) else first)
                curve_list=[curve]
            except Exception:
                curve_list=curve
        node=[]
        element=[]
        device=None #the first device that is not the cpu: a curve stays on its device and keeps its gradient
        idx_start=0
        for k in range(0, len(curve_list)):
            curve_k=curve_list[k]
            if torch.is_tensor(curve_k):
                pass
            elif isinstance(curve_k, np.ndarray):
                curve_k=torch.tensor(curve_k, dtype=torch.float64 if curve_k.dtype == np.float64 else torch.float32)
            elif isinstance(curve_k, list) or isinstance(curve_k, tuple):
                #a point may be a 1D numpy/torch array
                point_list=[point.tolist() if isinstance(point, np.ndarray) else point for point in curve_k]
                tensor_list=[point for point in point_list if torch.is_tensor(point)]
                if len(tensor_list) > 0:
                    curve_k=torch.stack([torch.as_tensor(point, dtype=torch.float64, device=tensor_list[0].device) for point in point_list])
                else:
                    curve_k=torch.tensor(point_list, dtype=torch.float64)
            else:
                raise ValueError('unsupported type of curve in a list: '+str(type(curve_k)))
            if len(curve_k) > 0:
                if len(curve_k.shape) != 2:
                    raise ValueError('a curve must be made of points, but node has shape '+str(tuple(curve_k.shape)))
                if len(node) > 0 and curve_k.shape[1] != node[0].shape[1]:
                    raise ValueError('all curves must have the same number of coordinates')
                if device is None and curve_k.device.type != 'cpu':
                    device=curve_k.device
                node.append(curve_k)
                idx_list=np.arange(idx_start, idx_start+len(curve_k)).tolist()
                idx_start=idx_start+len(curve_k)
                for n in range(1, len(idx_list)):
                    element.append([idx_list[n-1], idx_list[n]]) #each element is VTK_LINE
        if not node:
            self.clear()
            return
        #cat takes the widest dtype of the curves, so that no curve is rounded
        #and it makes a copy, so this mesh does not share memory with curve
        node=torch.cat([curve_k.to(device=device) for curve_k in node], dim=0)
        self.clear()
        self.node=node
        self.element=element
    
    @classmethod
    def from_curve(cls, curve):
        #curve may be a single curve or a list of curves, see initialize
        poly_line_mesh = cls()
        poly_line_mesh.initialize(curve)
        return poly_line_mesh
       
    def build_edge(self):
        self.build_element_to_edge_adj_table()
        
    def build_element_to_edge_adj_table(self):
        #an edge connects 2 nodes
        #an element may be a simple VTK_LINE of 2 nodes or VTK_POLY_LINE of many nodes
        element=self.element
        if not isinstance(element, list):
            element=element.tolist()
        edge=[]
        for m in range(0, len(element)):
            elm=element[m]
            for k in range(0, len(elm)-1):
                a=elm[k]; b=elm[k+1]
                edge.append([min(a,b), max(a,b)])
        edge=np.array(edge, dtype=np.int64).reshape(-1,2)
        edge_unique, inverse=np.unique(edge, return_inverse=True, axis=0)
        self.edge=torch.tensor(edge_unique, dtype=torch.int64)
        adj_table=[]
        idx=0
        for m in range(0, len(element)):
            n_edge=max(len(element[m])-1, 0)
            adj_table.append(inverse[idx:(idx+n_edge)].tolist())
            idx=idx+n_edge
        self.element_to_edge_adj_table=adj_table
    
    def find_boundary_node(self):
        #return index list of nodes on boundary
        #a boundary node has no more than one neighbor node
        if self.node_to_node_adj_table is None:
            self.build_node_to_node_adj_table()
        boundary=[]
        for k in range(0, len(self.node)):
            adj_node_idx=self.node_to_node_adj_table[k]
            if len(adj_node_idx) <= 1:
                boundary.append(k)
        return boundary

    def find_boundary_edge(self):
        if self.edge is None:
            self.build_edge()
        boundary_node=set(self.find_boundary_node())
        boundary_edge=[]
        for k in range(0, len(self.edge)):
            idx0=int(self.edge[k,0])
            idx1=int(self.edge[k,1])
            if (idx0 in boundary_node) or (idx1 in boundary_node):
                boundary_edge.append(k)
        return boundary_edge

    def find_boundary_node_and_edge(self):
        if self.edge is None:
            self.build_edge()
        boundary_node=self.find_boundary_node()
        boundary_node_set=set(boundary_node)
        boundary_edge=[]
        for k in range(0, len(self.edge)):
            idx0=int(self.edge[k,0])
            idx1=int(self.edge[k,1])
            if (idx0 in boundary_node_set) or (idx1 in boundary_node_set):
                boundary_edge.append(k)
        return boundary_node, boundary_edge

    def find_boundary_element(self):
        #return index list of elements on boundary
        boundary_node=self.find_boundary_node()
        if self.node_to_element_adj_table is None:
            self.build_node_to_element_adj_table()
        boundary=[]
        for node_idx in boundary_node:
            adj_elm_idx=self.node_to_element_adj_table[node_idx]
            boundary.extend(adj_elm_idx)
        boundary=np.unique(boundary).tolist()
        return boundary

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        new_mesh, node_idx_list=super().get_sub_mesh(element_idx_list, return_node_idx_list=True)
        new_mesh=PolylineMesh(new_mesh.node, new_mesh.element)
        if return_node_idx_list == False:
            return new_mesh
        else:
            return new_mesh, node_idx_list
