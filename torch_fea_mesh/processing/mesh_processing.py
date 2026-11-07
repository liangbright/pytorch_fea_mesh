# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 00:12:58 2023

@author: liang
"""
import torch
import torch_scatter
import numpy as np
from ..mesh.mesh import Mesh, find_vtk_cell_by_element_type, get_dimension_of_mesh_type
from copy import deepcopy
from ..utility.geometry_utility import cal_angle_between_3d_vector, ComputeAngleBetweenTwoVectorIn3D
#%%
def find_connected_region(mesh, ref_element_idx, adj):
    if not isinstance(mesh, Mesh):
        raise NotImplementedError
    if adj not in ["node", "edge", "face"]:
        raise ValueError()
    if mesh.element_to_element_adj_table[adj] is None:
        mesh.build_element_to_element_adj_table(adj=adj)
    element_adj_table=mesh.element_to_element_adj_table[adj]
    region_element_list=[ref_element_idx]
    active_element_list=[ref_element_idx]
    while True:
        new_active_element_list=[]
        region_element_set=set(region_element_list)
        for elm_idx in active_element_list:
            adj_elm_idxlist=element_adj_table[elm_idx]
            adj_elm_idxlist=list(set(adj_elm_idxlist)-region_element_set)
            new_active_element_list.extend(adj_elm_idxlist)
        if len(new_active_element_list) ==0:
            break
        active_element_list=np.unique(new_active_element_list).tolist()
        region_element_list.extend(active_element_list)
    #indexes of elements
    return region_element_list
#%%
FindConnectedRegion=find_connected_region
#%%
def segment_mesh_to_connected_region(mesh, adj):
    element_list=np.arange(0, len(mesh.element)).tolist()
    region_list=[]
    while True:
        if len(element_list) == 0:
            break
        region=FindConnectedRegion(mesh, element_list[0], adj)
        region_list.append(region)
        element_list=list(set(element_list)-set(region))
    return region_list
#%%
SegmentMeshToConnectedRegion=segment_mesh_to_connected_region
#%%
def convert_smoother_mask(mask, field):
    #return 1, or a tensor of shape (N,1) with the dtype and device of field
    if mask is None:
        mask=1    
    elif isinstance(mask, list):
        mask=torch.tensor(mask, dtype=field.dtype, device=field.device)
        mask=mask.reshape(-1,1)
    elif isinstance(mask, np.ndarray):
        mask=torch.tensor(mask, dtype=field.dtype, device=field.device)
        mask=mask.reshape(-1,1)
    elif isinstance(mask, torch.Tensor):
        mask=mask.to(field.dtype).to(field.device)
        mask=mask.reshape(-1,1)
    elif isinstance(mask, int) or isinstance(mask, float):
        if mask != 1:
            raise ValueError            
    else:
        raise ValueError('python-object type of mask is not supported')
    return mask
#%%
def simple_smoother(field, adj_link, lamda, mask, inplace):
    #field.shape (N, ?)
    mask=convert_smoother_mask(mask, field)
    #the mesh builds its links on the CPU, and scatter needs them on the device of field
    adj_link=adj_link.to(field.device)
    #---------
    x_j=field[adj_link[:,0]]
    x_i=field[adj_link[:,1]]
    delta=x_j-x_i
    delta=torch_scatter.scatter(delta, adj_link[:,1], dim=0, dim_size=field.shape[0], reduce="mean")
    delta=lamda*delta*mask
    if inplace == True:
        field+=delta
    else:
        field=field+delta
    return field
#%%
SimpleSmoother=simple_smoother
#%%
def simple_smoother_for_mesh(mesh, lamda, mask, n_iter):
    #lamda: x_i = x_i + lamda*mean_j(x_j - x_i),  0<=lamda<=1
    #mesh.node is modified
    #if mask is not None: mask[k]: 1 to smooth the node-k; 0 not to smooth the node-k
    if not isinstance(mesh, Mesh):
        raise NotImplementedError
    #---------
    if mesh.node_to_node_adj_link is None:
        mesh.build_node_to_node_adj_link()
    adj_link=mesh.node_to_node_adj_link.to(mesh.node.device)
    for n in range(0, n_iter):
        SimpleSmoother(mesh.node, adj_link, lamda, mask, inplace=True)
    mesh.clear_geometry_derived_data(clear_adjacency_data=False)
#%%
SimpleSmootherForMesh=simple_smoother_for_mesh        
#%%
def is_edge_curve_closed(mesh, edge_curve):
    #edge_curve is a list/array/set of indexes into mesh.edge; order is irrelevant.
    #Return a bool: True if the selected edges include at least one closed loop.
    #None entries, duplicate indexes and edges from a node to itself are ignored.
    edge_curve={int(idx) for idx in edge_curve if idx is not None}
    if not edge_curve:
        return False
    if mesh.edge is None:
        mesh.build_edge()
    if min(edge_curve) < 0 or max(edge_curve) >= len(mesh.edge):
        raise ValueError('invalid edge index in edge_curve')
    #Union-find: a unique edge joining already connected nodes closes a loop.
    parent={}
    size={}
    for idx_n, idx_n1 in mesh.edge[list(edge_curve)].tolist():
        if idx_n == idx_n1:
            continue
        for idx in (idx_n, idx_n1):
            if idx not in parent:
                parent[idx]=idx
                size[idx]=1
        root_n=idx_n
        while parent[root_n] != root_n:
            parent[root_n]=parent[parent[root_n]]
            root_n=parent[root_n]
        root_n1=idx_n1
        while parent[root_n1] != root_n1:
            parent[root_n1]=parent[parent[root_n1]]
            root_n1=parent[root_n1]
        if root_n == root_n1:
            return True
        if size[root_n] < size[root_n1]:
            root_n, root_n1=root_n1, root_n
        parent[root_n1]=root_n
        size[root_n]+=size[root_n1]
    return False
#%%
IsEdgeCurveClosed=is_edge_curve_closed
#%%
def is_node_curve_closed(mesh, node_curve):
    #node_curve is an ordered list/array of node indexes on mesh.
    #Return a bool: True if its consecutive mesh edges include a closed loop.
    #[0,1,2] is open; [0,1,2,0] and [0,1,2,0,3,4] include a closed loop.
    #Missing edges are ignored; traversing the same edge twice is not a loop.
    node_curve=[int(x) for x in node_curve]
    edge_curve=set()
    for idx_n, idx_n1 in zip(node_curve, node_curve[1:]):
        edge_idx=mesh.get_edge_idx_from_node_pair(idx_n, idx_n1)
        if edge_idx is not None:
            edge_curve.add(edge_idx)
    return is_edge_curve_closed(mesh, edge_curve)
#%%
IsNodeCurveClosed=is_node_curve_closed
#%%
def trace_polyline(mesh, start_node_idx, next_node_idx, end_node_idx=None, angle_threshold=np.pi/2, eps=None):
    #find a smoothed polyline on mesh: start_node_idx -> next_node_idx -> ... -> end_node_idx
    #no self-interselction
    if not isinstance(mesh, Mesh):
        raise NotImplementedError
    #---------
    start_node_idx=int(start_node_idx)
    next_node_idx=int(next_node_idx)
    if end_node_idx is not None:
        end_node_idx=int(end_node_idx)
        if end_node_idx == start_node_idx:
            raise ValueError("start_node_idx is end_node_idx")
        if end_node_idx == next_node_idx:
            return [start_node_idx, next_node_idx]
    #---------
    if mesh.node_to_node_adj_table is None:
        mesh.build_node_to_node_adj_table()
    node_adj_table=mesh.node_to_node_adj_table
    polyline=[start_node_idx, next_node_idx]
    while True:
        idx_list_next=node_adj_table[polyline[-1]]
        idx_list_next=list(set(idx_list_next)-set(polyline))
        if len(idx_list_next) == 0:
            break
        if end_node_idx in idx_list_next:
            polyline.append(end_node_idx)
            break
        angel_list=[]
        for k in range(0, len(idx_list_next)):
            idxA=polyline[-2]
            idxB=polyline[-1]
            idxC=idx_list_next[k]
            vector0=mesh.node[idxB]-mesh.node[idxA]
            vector1=mesh.node[idxC]-mesh.node[idxB]
            angle_k=cal_angle_between_3d_vector(vector0, vector1, eps=eps)
            angle_k=angle_k.view(-1).item()
            angel_list.append(angle_k)
        k_min=np.argmin(angel_list)
        if angel_list[k_min] > angle_threshold:
            break
        else:
            polyline.append(idx_list_next[k_min])
    if (end_node_idx is not None) and (polyline[-1] != end_node_idx):
        print("warning: polyline[-1] != end_node_idx @ TracePolyline")
    #done
    return polyline
#%%
TracePolyline=trace_polyline
#%%
def merge_element_type(meshA, meshB):
    #element_type of a merged mesh: meshA's elements, then meshB's.
    #None if a save can take every cell type from the node count, as a reader does.
    if (meshA.element_type is None and meshB.element_type is None and type(meshA) is type(meshB)
        and meshA.mesh_type == meshB.mesh_type):
        return None
    label_list=[]
    needed=False
    for mesh in (meshA, meshB):
        if mesh.element_type is not None:
            label_list.extend(np.asarray(mesh.element_type).tolist())
            needed=True
            continue
        #a mesh_type with one shape names it: polyhedron_hex8 gives HEX8, and polygon_quad4_tri3 gives nothing
        name=mesh.mesh_type.split('_')
        label=name[1].upper() if len(name) == 2 else None
        if label is not None and find_vtk_cell_by_element_type(label) is None:
            label=None
        if label not in (None, 'TRI3', 'QUAD4', 'TET4', 'LINE2'):
            needed=True
        label_list.extend([label]*len(mesh.element))
    return np.array(label_list, dtype=object) if needed else None
#%%
def make_merged_mesh(meshA, meshB, node, element):
    #the result has the closest class that both meshes belong to: two TriangleMesh give a TriangleMesh,
    #a TriangleMesh and a QuadMesh give a PolygonMesh, and a HexahedronMesh and a TetrahedronMesh give a PolyhedronMesh
    #a polygon mesh never mixes linear and quadratic elements, so check_element_type raises on a Tri6Mesh with a TriangleMesh
    mesh_class=None
    for class_k in type(meshA).__mro__:
        if class_k is not Mesh and issubclass(class_k, Mesh) and isinstance(meshB, class_k):
            mesh_class=class_k
            break
    element_type=merge_element_type(meshA, meshB)
    if mesh_class is None:
        mesh_type=meshA.mesh_type if meshA.mesh_type == meshB.mesh_type else meshA.mesh_type.split('_')[0]
        output=Mesh(node, element, element_type, mesh_type)
    else:
        output=mesh_class(node, element)
        #mesh_class sets its own mesh_type, which can say less: load_from_torch can give a plain mesh polygon_tri6
        if meshA.mesh_type == meshB.mesh_type:
            output.mesh_type=meshA.mesh_type
        output.check_element_type(element_type)
        output.element_type=element_type
    if torch.is_tensor(output.element):
        output.element=output.element.to(node.device)
    return output
#%%
def merge_mesh(meshA, node_idx_listA, meshB, node_idx_listB, distance_threshold):
    #Merge: meshA <= meshB
    #The shared points are in node_idx_listA of meshA and node_idx_listB of meshB
    #if both node_idx_listA and node_idx_listB are None, simply do meshA + meshB - no point merging    
    #if the distance between two nodes is <= distance_threshold, then merge the two nodes
    #node_data, element_data, node_set and element_set are not carried over
    #the output is on the device of meshA.node, and meshB can be on another device
    if (not isinstance(meshA, Mesh)) or (not isinstance(meshB, Mesh)):
        #raise NotImplementedError
        print("warning: input may not be Mesh @ MergeMesh")
    if meshA.node.shape[0] == 0 or meshB.node.shape[0] == 0:
        raise ValueError("meshA or meshB is empty")
    if get_dimension_of_mesh_type(meshA.mesh_type) != get_dimension_of_mesh_type(meshB.mesh_type):
        raise ValueError('a mesh never mixes dimensions: '+meshA.mesh_type+' and '+meshB.mesh_type)
    nodeB=meshB.node.to(meshA.node.device)
    #---------------------------------------------------
    if (node_idx_listA is None) or (node_idx_listB is None):
        elementA=meshA.make_element_copy('list')
        if not isinstance(meshB.element, list):
            elementB=deepcopy(meshB.element)+len(meshA.node)
            elementB=elementB.tolist()
        else:
            elementB=deepcopy(meshB.element)
            for m in range(0, len(elementB)):
                for n in range(0, len(elementB[m])):
                    elementB[m][n]=elementB[m][n]+len(meshA.node)
        nodeAB=torch.cat([meshA.node, nodeB], dim=0)
        elementAB=elementA + elementB
        return make_merged_mesh(meshA, meshB, nodeAB, elementAB)
    #---------------------------------------------------
    node_idx_map_B_to_Out=-1*np.ones(meshB.node.shape[0], dtype=np.int64)

    A = meshA.node[node_idx_listA]
    if len(A) > 0 and len(node_idx_listB) > 0:
        counterA = np.zeros(len(A))
        for n in range(0, len(node_idx_listB)):
            node_n=nodeB[node_idx_listB[n]].view(1,-1)
            dist=((A-node_n)**2).sum(dim=1).sqrt()
            idx=dist.argmin()
            if dist[idx] <= distance_threshold:
                node_idx_map_B_to_Out[node_idx_listB[n]]=node_idx_listA[idx]
                counterA[idx]+=1
        if counterA.max() > 1:
            raise ValueError("two nodes of meshB are mapped to the same node of meshA")

    node_idx=meshA.node.shape[0]-1
    node_idx_listB_keep=[]
    for n in range(0, meshB.node.shape[0]):
        if node_idx_map_B_to_Out[n] < 0:
            node_idx+=1
            node_idx_map_B_to_Out[n]=node_idx
            node_idx_listB_keep.append(n)

    nodeAB=torch.cat([meshA.node, nodeB[node_idx_listB_keep]], dim=0)
    elementAB=meshA.make_element_copy('list') + meshB.make_element_copy('list')
    for m in range(len(meshA.element), len(elementAB)):
        elm=elementAB[m]
        for k in range(0, len(elm)):
            elm[k]=int(node_idx_map_B_to_Out[elm[k]])

    return make_merged_mesh(meshA, meshB, nodeAB, elementAB)
#%%
MergeMesh=merge_mesh
#%%
def remove_unused_node(mesh, return_node_idx_list=False, clear_adj_info=True):
    #if a node does not belong to an element, then it is unused
    #this function may change the node order in mesh.node: new indexes follow the first use in mesh.element
    #clear_adj_info has no effect, and stays so that old calls still work: the adjacency tables
    #hold node indexes, which change even when no node is unused, so they are always cleared
    if torch.is_tensor(mesh.element):
        element=None
        flat=mesh.element.detach().cpu().numpy().reshape(-1)
    else:
        element=mesh.make_element_copy("list")
        flat=np.array([idx for elm in element for idx in elm], dtype=np.int64)
    if (flat < 0).any():
        raise ValueError("idx < 0, the output may be wrong")
    used, first_use=np.unique(flat, return_index=True)
    node_idx_list=used[np.argsort(first_use)].astype(np.int64)
    map=np.full(mesh.node.shape[0], -1, dtype=np.int64)
    map[node_idx_list]=np.arange(len(node_idx_list))
    flat=map[flat]
    if element is None:
        element=torch.as_tensor(flat, dtype=mesh.element.dtype, device=mesh.element.device).reshape(mesh.element.shape)
    else:
        idx=0
        for elm in element:
            elm[:]=flat[idx:(idx+len(elm))].tolist()
            idx=idx+len(elm)
    node_idx_list=torch.from_numpy(node_idx_list)
    #Remap metadata before changing the geometry, preserving field containers.
    kept=node_idx_list.tolist()
    old_to_new=map.tolist()
    node_data={}
    for name, data in mesh.node_data.items():
        if len(data) != len(mesh.node):
            raise ValueError('node_data has wrong row count: '+name)
        if torch.is_tensor(data):
            node_data[name]=data.index_select(0, node_idx_list.to(data.device))
        elif isinstance(data, np.ndarray):
            node_data[name]=data[kept].copy()
        elif isinstance(data, (list, tuple)):
            node_data[name]=type(data)(deepcopy(data[idx]) for idx in kept)
        else:
            raise ValueError('unsupported node_data type: '+name)
    def remap_index(idx):
        idx=int(idx)
        if idx < 0 or idx >= len(old_to_new):
            raise ValueError('node metadata index out of range')
        return old_to_new[idx]
    node_set={}
    for name, values in mesh.node_set.items():
        mapped=[remap_index(idx) for idx in mesh.copy_to_list(values)]
        mapped=[idx for idx in mapped if idx >= 0]
        if torch.is_tensor(values):
            node_set[name]=torch.tensor(mapped, dtype=values.dtype, device=values.device)
        elif isinstance(values, np.ndarray):
            node_set[name]=np.array(mapped, dtype=values.dtype)
        else:
            node_set[name]=type(values)(mapped)
    node_names={name:remap_index(idx) for name, idx in mesh.mapping_from_node_name_to_index.items()}
    node_names={name:idx for name, idx in node_names.items() if idx >= 0}
    node=mesh.node[node_idx_list.to(mesh.node.device)]
    name=mesh.name
    element_type=mesh.element_type
    element_data=mesh.element_data
    element_set=mesh.element_set
    mesh_data=mesh.mesh_data
    mesh.copy(node, element)
    mesh.name=name
    mesh.element_type=element_type
    mesh.element_data=element_data
    mesh.element_set=element_set
    mesh.mesh_data=mesh_data
    mesh.node_data=node_data
    mesh.node_set=node_set
    mesh.mapping_from_node_name_to_index=node_names
    if return_node_idx_list == True:
        return node_idx_list
#%%
RemoveUnusedNode=remove_unused_node    
#%%
def find_nearest_node(mesh, point, distance_threshold=np.inf):
    #point (K, 3) or (3,)
    #return one result per point: a node index, or None if no node is within distance_threshold
    if not isinstance(point, (torch.Tensor, np.ndarray)):
        raise ValueError("unsupported type "+str(type(point)))
    if isinstance(point, np.ndarray):
        point=torch.tensor(point, dtype=mesh.node.dtype)    
    point=point.to(mesh.node.device)
    flag=(len(point.shape) == 1)
    point=point.view(-1,3)
    node_idx_list=[]
    for n in range(0, len(point)):
        p=point[n].view(1,-1)
        d=((mesh.node-p)**2).sum(dim=-1)
        if len(d) > 0 and float(d.min()) <= distance_threshold**2:
            node_idx_list.append(int(d.argmin()))
        else:
            node_idx_list.append(None)
    if flag == False:
        return node_idx_list
    else:
        return node_idx_list[0]
#%%
FindNearestNode=find_nearest_node
#%%
def find_neighbor_node(mesh, node_idx, max_n_hop):
    if isinstance(node_idx, int):
        node_idx=[node_idx]
    elif isinstance(node_idx, tuple):
        node_idx=list(node_idx)
    elif isinstance(node_idx, list):
        pass
    else:
        raise ValueError('node_idx must be int, tuple, or list')
    neighbor_node_list=node_idx.copy()
    if max_n_hop < 0:
        raise ValueError('max_n_hop must be nonnegative')
    if max_n_hop == 0:
        return neighbor_node_list
    mesh.build_node_to_node_adj_table()
    adj_table=mesh.node_to_node_adj_table
    active_node_list=neighbor_node_list
    n_hop=0
    while n_hop < max_n_hop:
        new_active_node_list=[]
        node_set=set(neighbor_node_list)
        for idx in active_node_list:
            adj_idx_list=adj_table[idx]
            adj_idx_list=list(set(adj_idx_list)-node_set)
            new_active_node_list.extend(adj_idx_list)
        if len(new_active_node_list) ==0:
            break
        active_node_list=np.unique(new_active_node_list).tolist()
        neighbor_node_list.extend(active_node_list)
        n_hop+=1
    return neighbor_node_list
#%%
FindNeighborNode=find_neighbor_node
