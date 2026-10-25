import torch
from torch.linalg import vector_norm as norm
import torch_scatter
import numpy as np
from ..mesh.polyline_mesh import PolylineMesh
from ..mesh.polygon_mesh import PolygonMesh
from ..mesh.triangle_mesh import TriangleMesh
from ..mesh.tri6_mesh import Tri6Mesh
from ..mesh.quad_mesh import QuadMesh
from ..mesh.quad_triangle_mesh import QuadTriangleMesh
from copy import deepcopy
from .mesh_processing import simple_smoother, simple_smoother_for_mesh, trace_polyline, is_edge_curve_closed, is_node_curve_closed, merge_mesh, find_connected_region, find_nearest_node, segment_mesh_to_connected_region, find_neighbor_node
from ..utility.geometry_utility import cal_angle_between_3d_vector
from .mesh_processing import SimpleSmoother, SimpleSmootherForMesh, TracePolyline, IsEdgeCurveClosed, IsNodeCurveClosed, MergeMesh, FindConnectedRegion, FindNearestNode, SegmentMeshToConnectedRegion, FindNeighborNode
from ..utility.geometry_utility import ComputeAngleBetweenTwoVectorIn3D
try:
    import vtk
except:
    print("cannot import vtk")
#%%
def trace_mesh_boundary_curve(mesh, start_node_idx, next_node_idx=None, end_node_idx=None):
    #trace boundary starting from start_node_idx -> next_node_idx -> ...
    #this function may not work well if two boundary curves share points
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    #---------
    start_node_idx=int(start_node_idx)
    if next_node_idx is not None:
        next_node_idx=int(next_node_idx)
    if end_node_idx is not None:
        end_node_idx=int(end_node_idx)
        if end_node_idx == start_node_idx:
            raise ValueError("start_node_idx is end_node_idx")
    #---------
    if mesh.node_to_edge_adj_table is None:
        mesh.build_node_to_edge_adj_table()
    #---------
    boundary_node, boundary_edge=mesh.find_boundary_node_and_edge()
    BoundaryCurve=[]
    if start_node_idx not in boundary_node:
        raise ValueError('start_node('+str(start_node_idx)+') is not a boundary node')
    if next_node_idx is not None:
        if next_node_idx not in boundary_node:
            raise ValueError('next_node('+str(next_node_idx)+') is not a boundary node')
        edge_idx=mesh.get_edge_idx_from_node_pair(start_node_idx, next_node_idx) 
        if edge_idx not in boundary_edge:
            raise ValueError('no boundary edge between node('+str(start_node_idx)+') and node('+str(next_node_idx)+')')
        else:
            BoundaryCurve.append(start_node_idx)
            start_node_idx=next_node_idx
    if end_node_idx is not None:
        if end_node_idx not in boundary_node:
            raise ValueError('end_node('+str(end_node_idx)+') is not a boundary node')
            #return BoundaryCurve
    #---------
    idx_next=start_node_idx
    while True:
        BoundaryCurve.append(idx_next)
        if idx_next == end_node_idx:
            break
        edge_idx_list=mesh.node_to_edge_adj_table[idx_next]
        flag=False
        for edge_idx in edge_idx_list:
            if edge_idx in boundary_edge:
                idx_list=mesh.edge[edge_idx].tolist()
                if idx_next == idx_list[0]:
                    node_idx=idx_list[1]
                elif idx_next == idx_list[1]:
                    node_idx=idx_list[0]
                else:
                    raise ValueError('something is wrong in edge and node_to_edge_adj_table')
                if node_idx not in BoundaryCurve:
                    idx_next=node_idx
                    flag=True
                    break
        if flag == False:
            break
    return BoundaryCurve
#%% old name
TraceMeshBoundaryCurve=trace_mesh_boundary_curve
#%%
def find_mesh_boundary_curve(mesh):
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    node_idx_list=mesh.find_boundary_node()
    BoundaryCurveList=[]
    while True:
        if len(node_idx_list) == 0:
            break
        BoundaryCurve=TraceMeshBoundaryCurve(mesh, node_idx_list[0])
        BoundaryCurveList.append(BoundaryCurve)
        node_idx_list=list(set(node_idx_list)-set(BoundaryCurve))
    return BoundaryCurveList
#%% old name
FindMeshBoundaryCurve=find_mesh_boundary_curve
#%%
def extract_region_enclosed_by_curve(mesh, node_curve_list, inner_element_idx, max_n_element=float('inf')):
    #node_curve_list[k] is an ordered list/array of node indexes on mesh.
    #Only consecutive pairs define edges: [0,1,2] is open; [0,1,2,0] closes it.
    #The combined existing edges form the region boundaries and must contain
    #at least one closed loop. Separate segments and open tails are allowed.
    #inner_element_idx is the index of an element inside the region
    #max_n_element: maximum number of elements in the region
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    #-------------------------
    if mesh.element_to_edge_adj_table is None:
        mesh.build_element_to_edge_adj_table()
    element_to_edge_adj_table=mesh.element_to_edge_adj_table    
    #-------------------------
    if mesh.edge_to_element_adj_table is None:
        mesh.build_edge_to_element_adj_table()
    edge_to_element_adj_table=mesh.edge_to_element_adj_table
    #-------------------------
    edge_curve=set()
    for k in range(0, len(node_curve_list)):
        curve_k=[int(x) for x in node_curve_list[k]]
        for idx_n, idx_n1 in zip(curve_k, curve_k[1:]):
            edge_idx=mesh.get_edge_idx_from_node_pair(idx_n, idx_n1)
            if edge_idx is not None:
                edge_curve.add(edge_idx)
    if not is_edge_curve_closed(mesh, edge_curve):
        raise ValueError('edge_curve does not contain a closed curve')
    #-------------------------
    region_element_list=[inner_element_idx]
    active_element_list=[inner_element_idx]
    counter=0
    element_flag=np.zeros(len(mesh.element))#flag 1: in the  region; 0: not in the region
    element_flag[inner_element_idx]=1
    while True:
        new_active_element_list=[]
        for act_elm_idx in active_element_list:
            edge_idx_list=element_to_edge_adj_table[act_elm_idx]
            for edge_idx in edge_idx_list:
                if edge_idx not in edge_curve:
                    adj_elm_idx=edge_to_element_adj_table[edge_idx]
                    if len(adj_elm_idx) > 2:
                        raise ValueError("len(adj_elm_idx) > 2 at edge "+str(edge_idx))
                    elif len(adj_elm_idx) == 2:
                        if adj_elm_idx[0] == act_elm_idx:
                            new_active_element_list.append(adj_elm_idx[1])
                        else:
                            new_active_element_list.append(adj_elm_idx[0])
        if len(new_active_element_list) == 0:
            break
        new_active_element_list=np.unique(new_active_element_list)
        active_element_list=new_active_element_list[element_flag[new_active_element_list]==0].tolist()
        element_flag[active_element_list]=1
        #active_element_list=list(set(new_active_element_list)-set(region_element_list))
        region_element_list.extend(active_element_list)
        counter+=1
        #if counter ==100:
        #    break
        if len(region_element_list) > max_n_element:
            print('break: len(region_element_list) > max_n_elements @ExtractRegionEnclosedByCurve')
            break
    #--------------
    # indexes of elements in the region
    return region_element_list  
#%% old name
ExtractRegionEnclosedByCurve=extract_region_enclosed_by_curve
#%%
def segment_mesh_by_curve(mesh, node_curve_list):
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    element_list=np.arange(0, len(mesh.element)).tolist()
    region_list=[]
    while True:
        if len(element_list) == 0:
            break
        region=ExtractRegionEnclosedByCurve(mesh, node_curve_list, element_list[0])
        region_list.append(region)
        element_list=list(set(element_list)-set(region))
    return region_list
#%% old name
SegmentMeshByCurve=segment_mesh_by_curve
#%%
def merge_mesh_on_boundary(mesh_list, distance_threshold):
    merged_mesh=mesh_list[0]
    for n in range(1, len(mesh_list)):
        mesh_n=mesh_list[n]
        if not isinstance(mesh_n, PolygonMesh):
            raise NotImplementedError
        merged_mesh=merge_mesh(merged_mesh, merged_mesh.find_boundary_node(),
                               mesh_n, mesh_n.find_boundary_node(),
                               distance_threshold)
        merged_mesh=PolygonMesh(merged_mesh.node, merged_mesh.element)
    return merged_mesh
#%% old name
MergeMeshOnBoundary=merge_mesh_on_boundary
#%%
def simple_smoother_for_mesh_node_normal(mesh, lamda, mask, n_iter, update_node_normal=True, eps=None):
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    if eps is None:
        if mesh.node.dtype in (torch.float32, torch.float64):
            eps=1e-12
        else:
            raise NotImplementedError()
    if update_node_normal == True:
        mesh.update_node_normal(eps=eps)
    node_normal=mesh.node_normal
    if mesh.node_to_node_adj_link is None:
        mesh.build_node_to_node_adj_link()
    adj_link=mesh.node_to_node_adj_link
    for n in range(0, n_iter):
        simple_smoother(node_normal, adj_link, lamda, mask, inplace=True)
        normal_norm=norm(node_normal, ord=2, dim=1, keepdim=True)
        normal_norm=normal_norm.clamp_min(eps)
        node_normal=node_normal/normal_norm
    node_normal=node_normal.contiguous()
    mesh.node_normal=node_normal
#%% old name
SimpleSmootherForMeshNodeNormal=simple_smoother_for_mesh_node_normal
#%%
def simple_smoother_for_mesh_element_normal(mesh, lamda, mask, n_iter, update_element_normal=True, eps=None):
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    if eps is None:
        if mesh.node.dtype in (torch.float32, torch.float64):
            eps=1e-12
        else:
            raise NotImplementedError()
    if update_element_normal == True:
        mesh.update_element_area_and_normal(eps=eps)
    element_normal =mesh.element_normal 
    if mesh.element_to_element_adj_link['node'] is None:
        mesh.build_element_to_element_adj_link('node')
    adj_link=mesh.element_to_element_adj_link['node']
    for n in range(0, n_iter):
        simple_smoother(element_normal, adj_link, lamda, mask, inplace=True)
        normal_norm=norm(element_normal, ord=2, dim=1, keepdim=True)
        normal_norm=normal_norm.clamp_min(eps)
        element_normal=element_normal/normal_norm
    element_normal=element_normal.contiguous()
    mesh.element_normal=element_normal
#%% old name
SimpleSmootherForMeshElementNormal=simple_smoother_for_mesh_element_normal
#%%
def simple_smoother_for_quad_mesh(mesh, lamda, mask, n_iter):
    if not isinstance(mesh, QuadMesh):
        raise NotImplementedError
    if mesh.node_to_element_adj_table is None:
        mesh.build_node_to_element_adj_table()
    adj_link=[]
    for n in range(0, len(mesh.node)):
        adj_elm_idx=mesh.node_to_element_adj_table[n]
        adj_elm=mesh.element[adj_elm_idx]
        adj_node_idx=torch.unique(adj_elm.reshape(-1)).tolist()
        for idx in adj_node_idx:
            if idx != n:
                adj_link.append([idx, n])
                adj_link.append([n, idx])
    adj_link=torch.tensor(adj_link, dtype=torch.int64)
    adj_link=torch.unique(adj_link, dim=0, sorted=True)
    for n in range(0, n_iter):
        simple_smoother(mesh.node, adj_link, lamda, mask, inplace=True)
#%% old name
SimpleSmootherForQuadMesh=simple_smoother_for_quad_mesh
#%%
#dtype is a necessary parameter of a function that is based on vtk functions
# function(mesh, mesh_vtk, dtype) where mesh_vtk is the output of some vtk function and mesh does not exist
#%%
def cut_mesh_by_curve(mesh, curve, point_ref, straight_cut=False, return_unselected=False, 
                      clean_output=False, eps=1e-5, triangulate_output=False, mesh_vtk=None, dtype=None,
                      threshold=0):
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    curve_vtk=vtk.vtkPoints()
    curve_vtk.SetDataTypeToDouble()
    curve_vtk.SetNumberOfPoints(len(curve))
    for n in range(0, len(curve)):
        curve_vtk.SetPoint(n, float(curve[n][0]), float(curve[n][1]), float(curve[n][2]))
    selecter=vtk.vtkSelectPolyData()
    selecter.SetInputData(mesh_vtk)
    selecter.SetLoop(curve_vtk)
    selecter.SetEdgeSearchModeToDijkstra()
    selecter.SetSelectionModeToClosestPointRegion()
    selecter.SetClosestPoint(float(point_ref[0]), float(point_ref[1]), float(point_ref[2]))    
    selecter.SetGenerateSelectionScalars(straight_cut)
    selecter.SetGenerateUnselectedOutput(return_unselected)
    selecter.Update()
    if straight_cut == False:
        output_vtk=selecter.GetOutput()
        if return_unselected == True:
            unselected_output_vtk=selecter.GetUnselectedOutput()
    else:
        clipper=vtk.vtkClipPolyData()
        clipper.SetInputData(selecter.GetOutput())
        clipper.SetValue(threshold)
        clipper.SetInsideOut(True)
        if return_unselected == True:
            clipper.GenerateClippedOutputOn()
        clipper.Update()
        #vtkClipPolyData may have weird output: isoloated points not in the selected region
        #use vtkConnectivityFilter to remove those isoloated points
        extractor=vtk.vtkConnectivityFilter() 
        extractor.SetInputData(clipper.GetOutput())
        extractor.SetExtractionModeToAllRegions()
        extractor.Update()
        output_vtk=extractor.GetOutput()        
        if return_unselected == True:            
            extractor=vtk.vtkConnectivityFilter()
            extractor.SetInputData(clipper.GetClippedOutput())
            extractor.SetExtractionModeToAllRegions()
            extractor.Update()
            unselected_output_vtk=extractor.GetOutput()
    #----------------------------------------
    if clean_output == True:
        cleaner=vtk.vtkStaticCleanPolyData()
        cleaner.SetInputData(output_vtk)
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(eps)
        cleaner.RemoveUnusedPointsOn()
        cleaner.Update()
        output_vtk=cleaner.GetOutput()
    if triangulate_output == False:
        output_mesh=PolygonMesh()
        if output_vtk.GetNumberOfCells() > 0:
            output_mesh.read_mesh_vtk(output_vtk, dtype)
    else:
        output_mesh=convert_polygon_mesh_to_triangle_mesh(None, output_vtk, dtype)
    if return_unselected == False:
        return output_mesh
    #----------------------------------------
    if clean_output == True:
        cleaner=vtk.vtkStaticCleanPolyData()
        cleaner.SetInputData(unselected_output_vtk)
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(eps)
        cleaner.RemoveUnusedPointsOn()
        cleaner.Update()
        unselected_output_vtk=cleaner.GetOutput()
    if triangulate_output == False:
        unselected_output_mesh=PolygonMesh()
        if unselected_output_vtk.GetNumberOfCells() > 0:
            unselected_output_mesh.read_mesh_vtk(unselected_output_vtk, dtype)
    else:
        unselected_output_mesh=convert_polygon_mesh_to_triangle_mesh(None, unselected_output_vtk, dtype)
    return output_mesh, unselected_output_mesh
#%% old name
CutMeshByCurve=cut_mesh_by_curve
#%%
def project_point_to_mesh(point, mesh, mesh_vtk=None, dtype=None, cell_locator_vtk=None):
    #---------------
    #the orignal function is (mesh, point, ...)
    #to be backward compatible
    arg1=point
    arg2=mesh
    if torch.is_tensor(arg2) or isinstance(arg2, np.ndarray):
        point=arg2
        mesh=arg1
    #---------------
    #point (N, 3)
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    #------------------------------------------
    if torch.is_tensor(point) or isinstance(point, np.ndarray):
        if len(point.shape) == 1:
            point=point.reshape(1,3)
        if len(point.shape) != 2:
            raise ValueError('point.shape is not supported')
        if point.shape[1] != 3:
            raise ValueError('point.shape is not supported')
    else:
        raise ValueError('point can only be torch.tensor or np.ndarray')
    #------------------------------------------
    if cell_locator_vtk is None:
        CellLocator=vtk.vtkStaticCellLocator()
        CellLocator.SetDataSet(mesh_vtk)
        CellLocator.BuildLocator()
    else:
        CellLocator=cell_locator_vtk
    point_proj=[]
    element_proj=[]
    for k in range(0, len(point)):
        testPoint=[float(point[k][0]), float(point[k][1]), float(point[k][2])]
        closestPoint=[0, 0, 0] #the coordinates of the closest point will be returned here
        closestPointDist2=vtk.reference(0) #the squared distance to the closest point will be returned here
        cellId=vtk.reference(0); #the cell id of the cell containing the closest point will be returned here
        subId=vtk.reference(0);  #this is rarely used (in triangle strips only, I believe)
        CellLocator.FindClosestPoint(testPoint, closestPoint, cellId, subId, closestPointDist2);
        point_proj.append(closestPoint)
        element_proj.append(int(cellId))
    point_proj=torch.tensor(point_proj, dtype=dtype)
    return point_proj, element_proj
#%% old name
ProjectPointToMesh=project_point_to_mesh
#%%
def smooth_and_project(mesh_move, mesh_fixed, lamda, mask, n1_iter, n2_iter, mesh_fixed_vtk=None, smooth_first=True):
    #smooth mesh_move and project it to mesh_fixed
    #mesh_move.node is modified
    #mesh_fixed must be a triangle mesh
    if mesh_fixed_vtk is None:
        mesh_fixed_vtk=mesh_fixed.convert_to_vtk()
    fixed_cell_locator_vtk=None
    if mesh_fixed_vtk is not None:
        fixed_cell_locator_vtk=vtk.vtkStaticCellLocator()
        fixed_cell_locator_vtk.SetDataSet(mesh_fixed_vtk)
        fixed_cell_locator_vtk.BuildLocator()        
    for k in range(0, n2_iter):
        if smooth_first == True:
            simple_smoother_for_mesh(mesh_move, lamda, mask, n1_iter)
        node_proj, element_proj=project_point_to_mesh(mesh_move.node, mesh_fixed, mesh_fixed_vtk, cell_locator_vtk=fixed_cell_locator_vtk)        
        temp=mask.view(-1)
        mesh_move.node[temp>0]=node_proj[temp>0]
        if smooth_first == False and k < n2_iter-1:
            simple_smoother_for_mesh(mesh_move, lamda, mask, n1_iter)
#%% old name
SmoothAndProject=smooth_and_project            
#%%
def cal_point_to_mesh_distance(point, mesh, mesh_vtk=None, dtype=None):
    #this function is not differentiable
    point_proj, __=project_point_to_mesh(point, mesh, mesh_vtk, dtype)
    if isinstance(point, np.ndarray):
        point=torch.from_numpy(point.copy())
    point=point.to(device=point_proj.device)
    dist=((point-point_proj)**2).sum(dim=1).sqrt()
    return dist
#%% old name
PointToMeshDistance=cal_point_to_mesh_distance
#%%
def cal_mesh_to_mesh_distance(meshA, meshB, reduction):
    #this function is not differentiable
    #ChamferDistance: distance between two sets of points
    nodeB_proj, __=project_point_to_mesh(meshB.node, meshA)
    nodeA_proj, __=project_point_to_mesh(meshA.node, meshB)
    distA=((meshA.node-nodeA_proj)**2).sum(dim=1).sqrt()
    distB=((meshB.node-nodeB_proj)**2).sum(dim=1).sqrt()
    if reduction == 'none':
        return distA, distB
    elif reduction == 'mean':
        return 0.5*(distA.mean()+distB.mean())
    else:
        raise ValueError
#%% old name
MeshToMeshDistance=cal_mesh_to_mesh_distance        
#%%
def convert_polygon_mesh_to_triangle_mesh(mesh, mesh_vtk=None, dtype=None):
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    trifilter=vtk.vtkTriangleFilter()
    trifilter.SetInputData(mesh_vtk)
    trifilter.Update()
    output_mesh=TriangleMesh()
    output_vtk=trifilter.GetOutput()
    if output_vtk.GetNumberOfCells() > 0:
        output_mesh.read_mesh_vtk(output_vtk, dtype)
    return output_mesh
#%%
ConvertPolygonMeshToTriangleMesh=convert_polygon_mesh_to_triangle_mesh
#%%
def clip_mesh_by_plane(mesh, origin, normal, return_clipped_output=False, clean_output=False, eps=1e-5,
                       triangulate_output=False, mesh_vtk=None, dtype=None):
    #origin and normal define the cut plane
    #Outputs with no cells are empty meshes (node=None, element=None).
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    plane=vtk.vtkPlane()
    plane.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    plane.SetNormal(float(normal[0]), float(normal[1]), float(normal[2]))
    clipper=vtk.vtkClipPolyData()
    clipper.SetInputData(mesh_vtk)
    clipper.SetClipFunction(plane)
    clipper.SetValue(0)
    if return_clipped_output == True:
        clipper.GenerateClippedOutputOn()
    clipper.Update()
    output_vtk=clipper.GetOutput()
    if clean_output == True:
        cleaner=vtk.vtkStaticCleanPolyData()
        cleaner.SetInputData(output_vtk)
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(eps)
        cleaner.RemoveUnusedPointsOn()
        cleaner.Update()
        output_vtk=cleaner.GetOutput()
    if triangulate_output == False:
        output_mesh=PolygonMesh()
        if output_vtk.GetNumberOfCells() > 0:
            output_mesh.read_mesh_vtk(output_vtk, dtype)
    else:
        output_mesh=convert_polygon_mesh_to_triangle_mesh(None, output_vtk, dtype)
    if return_clipped_output == False:
        return output_mesh
    #----------------------------------------
    clipped_output_vtk=clipper.GetClippedOutput()
    if clean_output == True:
        cleaner=vtk.vtkStaticCleanPolyData()
        cleaner.SetInputData(clipped_output_vtk)
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(eps)
        cleaner.RemoveUnusedPointsOn()
        cleaner.Update()
        clipped_output_vtk=cleaner.GetOutput()
    if triangulate_output == False:
        clipped_output_mesh=PolygonMesh()
        if clipped_output_vtk.GetNumberOfCells() > 0:
            clipped_output_mesh.read_mesh_vtk(clipped_output_vtk, dtype)
    else:
        clipped_output_mesh=convert_polygon_mesh_to_triangle_mesh(None, clipped_output_vtk, dtype)
    return output_mesh, clipped_output_mesh
#%% old name
ClipMeshByPlane=clip_mesh_by_plane
#%%
def clip_mesh_by_node_attribute(mesh, attribute_threshold, attribute_name=None, invert=True,
                                   return_clipped_output=False, clean_output=False, eps=1e-5,
                                   triangulate_output=False, mesh_vtk=None, dtype=None):
    """Clip by a named point field without changing the input's active scalars.
    The field must exist in mesh_vtk, or in mesh.node_data when mesh_vtk is
    created from mesh. When only mesh_vtk is supplied, dtype is required.
    Outputs with no cells are empty meshes (node=None, element=None).
    """
    if mesh_vtk is None:
        if mesh is None:
            raise ValueError('mesh or mesh_vtk is required')
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    if attribute_name is None or mesh_vtk.GetPointData().GetArray(attribute_name) is None:
        raise ValueError('invalid node attribute_name: '+str(attribute_name))
    selector=vtk.vtkAssignAttribute()
    selector.SetInputData(mesh_vtk)
    selector.Assign(attribute_name, vtk.vtkDataSetAttributes.SCALARS,
                    vtk.vtkAssignAttribute.POINT_DATA)
    clipper=vtk.vtkClipPolyData()
    clipper.SetInputConnection(selector.GetOutputPort())
    clipper.SetValue(attribute_threshold)
    clipper.SetInsideOut(invert)
    if return_clipped_output == True:
        clipper.GenerateClippedOutputOn()
    clipper.Update()
    output_vtk=clipper.GetOutput()
    if clean_output == True:
        cleaner=vtk.vtkStaticCleanPolyData()
        cleaner.SetInputData(output_vtk)
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(eps)
        cleaner.RemoveUnusedPointsOn()
        cleaner.Update()
        output_vtk=cleaner.GetOutput()
    if triangulate_output == False:
        output_mesh=PolygonMesh()
        if output_vtk.GetNumberOfCells() > 0:
            output_mesh.read_mesh_vtk(output_vtk, dtype)
    else:
        output_mesh=convert_polygon_mesh_to_triangle_mesh(None, output_vtk, dtype)
    if return_clipped_output == False:
        return output_mesh
    #----------------------------------------
    clipped_output_vtk=clipper.GetClippedOutput()
    if clean_output == True:
        cleaner=vtk.vtkStaticCleanPolyData()
        cleaner.SetInputData(clipped_output_vtk)
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(eps)
        cleaner.RemoveUnusedPointsOn()
        cleaner.Update()
        clipped_output_vtk=cleaner.GetOutput()
    if triangulate_output == False:
        clipped_output_mesh=PolygonMesh()
        if clipped_output_vtk.GetNumberOfCells() > 0:
            clipped_output_mesh.read_mesh_vtk(clipped_output_vtk, dtype)
    else:
        clipped_output_mesh=convert_polygon_mesh_to_triangle_mesh(None, clipped_output_vtk, dtype)
    return output_mesh, clipped_output_mesh
#%%
ClipMeshByNodeAttribute=clip_mesh_by_node_attribute
#%%
def slice_mesh_by_plane(mesh, origin, normal, clean_output=False, eps=1e-5, mesh_vtk=None, dtype=None):
    #similar to slice function in paraview
    #An output with no cells is an empty PolylineMesh.
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    plane=vtk.vtkPlane()
    plane.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    plane.SetNormal(float(normal[0]), float(normal[1]), float(normal[2]))
    if mesh is not None:
        if mesh.is_tri() == True:
            slicer=vtk.vtkPolyDataPlaneCutter()
        else:
            slicer=vtk.vtkPlaneCutter()
    else:
        slicer=vtk.vtkPlaneCutter()
    slicer.SetInputData(mesh_vtk)
    slicer.SetPlane(plane)
    slicer.Update()
    output_vtk=slicer.GetOutput()
    if clean_output == True:
        cleaner=vtk.vtkStaticCleanPolyData()
        cleaner.SetInputData(output_vtk)
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(eps)
        cleaner.RemoveUnusedPointsOn()
        cleaner.Update()
        output_vtk=cleaner.GetOutput()
    output_mesh=PolylineMesh()
    if output_vtk.GetNumberOfCells() > 0:
        output_mesh.read_mesh_vtk(output_vtk, dtype=dtype)
    return output_mesh
#%%
SliceMeshByPlane=slice_mesh_by_plane
#%%
def cal_curvature(mesh, curvature_name='mean', mesh_vtk=None, dtype=None):
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    cc = vtk.vtkCurvatures()
    cc.SetInputData(mesh_vtk)
    if 'gaussian' in curvature_name.lower():
        curvature_name='Gaussian_Curvature'
        cc.SetCurvatureTypeToGaussian()
        cc.Update()
    elif 'mean' in curvature_name.lower():
        curvature_name='Mean_Curvature'
        cc.SetCurvatureTypeToMean()
        cc.Update()
    elif 'max' in curvature_name.lower():
        curvature_name='Max_Curvature'
        cc.SetCurvatureTypeToMaximum()
        cc.Update()
    elif 'min' in curvature_name.lower():
        curvature_name='Min_Curvature'
        cc.SetCurvatureTypeToMinimum()
        cc.Update()
    else:
        raise ValueError('uknown curvature_name'+curvature_name)
    data=cc.GetOutput().GetPointData().GetAbstractArray(curvature_name)
    curvature=torch.zeros((data.GetNumberOfTuples(), ), dtype=dtype)
    for i in range(0, curvature.shape[0]):
        curvature[i]=data.GetComponent(i,0)
    return curvature
#%% old name
ComputeCurvature=cal_curvature
#%%
def fill_hole(mesh, hole_size, clean_output=False, eps=1e-5, triangulate_output=False, mesh_vtk=None, dtype=None):
    #make a watertight mesh
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    filter=vtk.vtkFillHolesFilter()
    filter.SetInputData(mesh_vtk)
    filter.SetHoleSize(hole_size)
    filter.Update()
    output_vtk=filter.GetOutput()
    if clean_output == True:
        cleaner=vtk.vtkStaticCleanPolyData()
        cleaner.SetInputData(output_vtk)
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(eps)
        cleaner.RemoveUnusedPointsOn()
        cleaner.Update()
        output_vtk=cleaner.GetOutput()
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(output_vtk)
    normal_filter.ConsistencyOn()
    normal_filter.SplittingOff()
    normal_filter.Update()    
    output_vtk=normal_filter.GetOutput()
    if triangulate_output == False:
        output_mesh=PolygonMesh()
        output_mesh.read_mesh_vtk(output_vtk, dtype)
    else:
        output_mesh=convert_polygon_mesh_to_triangle_mesh(None, output_vtk, dtype)
    return output_mesh
#%% old name
FillHole=fill_hole
#%%
def find_dijkstra_graph_geodesic_path(mesh, start_node_idx, end_node_idx, mesh_vtk=None, dtype=None):
    """Return input vertex IDs in VTK's path order (end to start).

    A supplied mesh_vtk must use the same point indexing as mesh.node.
    dtype is retained for compatibility; the returned IDs are Python ints.
    """
    if mesh.is_tri() == False:
        raise ValueError('only support triangle mesh')
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    finder=vtk.vtkDijkstraGraphGeodesicPath()
    finder.SetInputData(mesh_vtk)
    finder.SetStartVertex(start_node_idx)
    finder.SetEndVertex(end_node_idx)
    finder.Update()
    # GetIdList can contain a partial trace when no complete path exists.
    if finder.GetOutput().GetNumberOfPoints() == 0:
        return []
    path_ids=finder.GetIdList()
    return [path_ids.GetId(n) for n in range(path_ids.GetNumberOfIds())]
#%% old name
FindDijkstraGraphGeodesicPath=find_dijkstra_graph_geodesic_path
#%%
def subdivide(mesh, n_subdivision, method='linear', mesh_vtk=None, dtype=None):
    #method: 
    # linear-> vtkLinearSubdivisionFilter
    # loop -> vtkLoopSubdivisionFilter
    # butterfly -> vtkButterflySubdivisionFilter
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    
    trifilter=vtk.vtkTriangleFilter()
    trifilter.SetInputData(mesh_vtk)
    trifilter.Update()
    mesh_vtk=trifilter.GetOutput()
    
    if method == 'linear':
        filter=vtk.vtkLinearSubdivisionFilter()
    elif method == 'loop':
        filter=vtk.vtkLoopSubdivisionFilter()
    elif method == 'butterfly':
        filter=vtk.vtkButterflySubdivisionFilter()
    else:
        raise ValueError
    filter.SetNumberOfSubdivisions(n_subdivision)
    filter.SetInputData(mesh_vtk)
    filter.update()
    output_mesh=PolygonMesh()
    output_mesh.read_mesh_vtk(filter.GetOutput(), dtype)
    return output_mesh
#%% old name
Subdivision=subdivide
