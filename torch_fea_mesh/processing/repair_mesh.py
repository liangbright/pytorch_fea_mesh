# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 16:32:29 2022

@author: liang
"""
from collections import deque
import torch
from ..mesh.polygon_mesh import PolygonMesh
#%%
def _clear_geometry_cache(mesh):
    for name in ('node_normal', 'element_normal', 'element_area',
                 'element_corner_angle', 'element_flatness', 'edge_length'):
        if hasattr(mesh, name):
            setattr(mesh, name, None)
    if hasattr(mesh, 'classify_element'):
        mesh.classify_element()


def ReordeNodeInElement(mesh, ref_element_id, verbose=False):
    #Orient the connected component consistently with the reference element.
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    if 'tri6' in mesh.mesh_type or 'quad8' in mesh.mesh_type:
        raise NotImplementedError('orientation repair requires linear polygon elements')
    element=mesh.make_element_copy('list')
    ref_element_id=int(ref_element_id)
    if not 0 <= ref_element_id < len(element):
        raise ValueError('ref_element_id is out of range')
    mesh.build_element_to_element_adj_table(adj='edge')
    element_adj_table=mesh.element_to_element_adj_table['edge']

    def direction(row, a, b):
        idx=row.index(a)
        if row[(idx+1) % len(row)] == b:
            return 1
        if row[(idx-1) % len(row)] == b:
            return -1
        raise ValueError('shared nodes do not form a polygon edge')

    #Plan all reversals before mutation, and visit each element once in cycles.
    flip={ref_element_id:False}
    active=deque([ref_element_id])
    while active:
        elm_id=active.popleft()
        for adj_id in element_adj_table[elm_id]:
            shared=sorted(set(element[elm_id]).intersection(element[adj_id]))
            if len(shared) != 2:
                raise NotImplementedError('not repairable: elements must share one edge')
            a, b=shared
            same=direction(element[elm_id], a, b) == direction(element[adj_id], a, b)
            reverse=flip[elm_id] ^ same
            if adj_id in flip:
                if flip[adj_id] != reverse:
                    raise ValueError('inconsistent orientation constraints; mesh may be nonmanifold or nonorientable')
            else:
                flip[adj_id]=reverse
                active.append(adj_id)
    for elm_id, reverse in flip.items():
        if reverse:
            row=mesh.element[elm_id]
            mesh.element[elm_id]=(row.flip(0) if torch.is_tensor(row) else row[::-1])
            if verbose:
                print('reverse node order in element', elm_id)
    if any(flip.values()):
        mesh.clear_adj_info()
        _clear_geometry_cache(mesh)
#%%
def SmoothMesh(mesh, node_idxlist=None, element_idxlist=None):
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    node_idxlist_input=[]
    if node_idxlist is not None:
        node_idxlist_input=mesh.copy_to_list(node_idxlist)
    if element_idxlist is not None:
        for idx in element_idxlist:
            node_idxlist_input.extend(mesh.copy_to_list(mesh.element[idx]))
    if not node_idxlist_input:
        return
    mesh.build_node_to_node_adj_table()
    for idx in node_idxlist_input:
        #Preserve sequential averaging without modifying the adjacency table.
        adj_node_idxlist=mesh.node_to_node_adj_table[idx]+[idx]
        avg=mesh.node[adj_node_idxlist].mean(dim=0)
        mesh.node[idx]=avg
    _clear_geometry_cache(mesh)
#%%
