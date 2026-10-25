import torch
import numpy as np
from ..mesh.polygon_mesh import PolygonMesh
from ..mesh.quad_mesh import QuadMesh
from ..processing.polygon_mesh_processing import merge_mesh_on_boundary
from ..processing.mesh_processing import simple_smoother_for_mesh
from ..mesh.hexahedron_mesh import HexahedronMesh as Hex8Mesh
#%%
def create_quad_cylinder_mesh(n_circle, n_point_per_circle, radius=1, height=1, dtype=torch.float32):
    theta=2*np.pi/n_point_per_circle
    node=torch.zeros((n_circle*n_point_per_circle, 3), dtype=dtype)
    k=-1
    for n in range(0, n_circle):
        for m in range(0, n_point_per_circle):
            x=radius*np.cos(theta*m)
            y=radius*np.sin(theta*m)
            z=height*n/(n_circle-1)
            k=k+1
            node[k,0]=x
            node[k,1]=y
            node[k,2]=z
    element=[]
    for n in range(1, n_circle):
        idxA=np.arange((n-1)*n_point_per_circle, n*n_point_per_circle)
        idxB=np.arange(n*n_point_per_circle, (n+1)*n_point_per_circle)
        for i in range(0, n_point_per_circle-1):
            element.append([idxA[i], idxA[i+1], idxB[i+1], idxB[i]])
        element.append([idxA[n_point_per_circle-1], idxA[0], idxB[0], idxB[n_point_per_circle-1]])
    cylinder=QuadMesh(node, element)
    cylinder.node_set["boundary_lower"]=np.arange(0, n_point_per_circle).tolist()
    cylinder.node_set["boundary_upper"]=np.arange((n_circle-1)*n_point_per_circle, n_circle*n_point_per_circle).tolist()
    return cylinder
#%%
def create_quad_grid_mesh(Nx, Ny, dtype=torch.float32):
    element=torch.zeros(((Nx-1)*(Ny-1), 4), dtype=torch.int64)
    node=torch.zeros((Nx*Ny, 3), dtype=dtype)
    map=torch.zeros((Ny, Nx), dtype=torch.int64)
    id=-1
    boundary=[]
    for y in range(0, Ny):
        for x in range(0, Nx):
            id+=1
            map[y,x]=id
            node[id,0]=x
            node[id,1]=y
            if y==0 or y==Ny-1 or x==0 or x==Nx-1:
                boundary.append(id)
    #-----------------------------------
    id=-1
    for y in range(0, Ny-1):
        for x in range(0, Nx-1):
            id+=1
            element[id,0]=map[y,x]
            element[id,1]=map[y,x+1]
            element[id,2]=map[y+1,x+1]
            element[id,3]=map[y+1,x]
    grid_mesh=QuadMesh(node, element)
    grid_mesh.node_set['boundary']=boundary
    return grid_mesh
#%%
def create_hex_grid_mesh(Nx, Ny, Nz, dtype=torch.float32):
    element=torch.zeros(((Nx-1)*(Ny-1)*(Nz-1), 8), dtype=torch.int64)
    grid=torch.zeros((Nx*Ny*Nz, 3), dtype=dtype)
    map=torch.zeros((Nz, Ny, Nx), dtype=torch.int64)
    id=-1
    boundary=[]
    for z in range(0,Nz):
        for y in range(0, Ny):
            for x in range(0, Nx):
                id+=1
                map[z,y,x]=id
                grid[id,0]=x
                grid[id,1]=y
                grid[id,2]=z
                if z==0 or z==Nz-1 or y==0 or y==Ny-1 or x==0 or x==Nx-1:
                    boundary.append(id)
    boundary=torch.tensor(boundary, dtype=torch.int64)
    id=-1
    for z in range(0,Nz-1):
        for y in range(0, Ny-1):
            for x in range(0, Nx-1):
                id+=1
                element[id,0]=map[z,y,x]
                element[id,1]=map[z,y,x+1]
                element[id,2]=map[z,y+1,x+1]
                element[id,3]=map[z,y+1,x]
                element[id,4]=map[z+1,y,x]
                element[id,5]=map[z+1,y,x+1]
                element[id,6]=map[z+1,y+1,x+1]
                element[id,7]=map[z+1,y+1,x]
    grid_mesh=Hex8Mesh(grid, element)
    grid_mesh.node_set['boundary']=boundary
    return grid_mesh       
#%%
def create_quad_mesh_rectangle_in_rectangle(n_ring=3, Nx=3, Ny=3, seal_hole=True):
    if n_ring < 2:
        raise ValueError("n_ring must be >= 2")
    inner_mesh=create_quad_grid_mesh(Nx, Ny)
    inner_mesh.node-=inner_mesh.node.mean(dim=0, keepdim=True)
    inner_mesh.node/=max(Nx, Ny)*n_ring    
    # y
    #/|\
    # |
    # D--------C
    # | \    / |
    # |  d__c  |
    # |  |__|  |
    # |  a  b  |
    # | /    \ |
    # A--------B--->x
    #note: A-B-C-D could deform into a circle, then the output is a quad mesh of rectangle in circle
    #------------------------------------------------------------------------------------------------
    rect0_idx=(np.arange(0,Nx).tolist()+ [Nx-1+Nx*n for n in range(1, Ny)] + [Nx*Ny-1-n for n in range(1, Nx)]
                 + [Nx*(Ny-1)-Nx*n for n in range(1,Ny-1)])
    rect0=inner_mesh.node[rect0_idx]        
    node=[]; element=[]
    node.extend(rect0.tolist())
    K=len(rect0)
    rectN=rect0*n_ring
    for n in range(1, n_ring):
        rect_n=rect0+(rectN-rect0)*n/(n_ring-1) 
        node.extend(rect_n.tolist())        
        for m in range(0, K):
            if m < K-1:
                idx0=n*K+m
                idx1=n*K+m+1
                idx2=(n-1)*K+m+1
                idx3=(n-1)*K+m
            else:
                idx0=n*K+m
                idx1=n*K
                idx2=(n-1)*K
                idx3=(n-1)*K+m
            element.append([idx0, idx1, idx2, idx3])
    output_mesh=QuadMesh(node, element)
    element_counter_no_hole=len(element)
    if seal_hole == True:
        output_mesh=merge_mesh_on_boundary([output_mesh, inner_mesh], distance_threshold=0.1/(max(Nx,Ny)*n_ring))
        output_mesh=QuadMesh(output_mesh.node, output_mesh.element)
        output_mesh.element_set['hole']=np.arange(element_counter_no_hole, len(output_mesh.element)).tolist()
    A=(2*Nx+2*Ny-4)*(n_ring-1)
    B=A+Nx-1
    C=B+Ny-1
    D=C+Nx-1
    output_mesh.node_set["ABCD"]=[A, B, C, D]
    output_mesh.node_set["lineAB"]=np.arange(A, B+1).tolist()
    output_mesh.node_set["lineBC"]=np.arange(B, C+1).tolist()
    output_mesh.node_set["lineCD"]=np.arange(C, D+1).tolist()
    output_mesh.node_set["lineDA"]=np.arange(D, D+Ny-1).tolist()+[A]
    a=0
    b=a+Nx-1
    c=b+Ny-1
    d=c+Nx-1
    output_mesh.node_set["abcd"]=[a, b, c, d]
    output_mesh.node_set["line_ab"]=np.arange(a, b+1).tolist()
    output_mesh.node_set["line_bc"]=np.arange(b, c+1).tolist()
    output_mesh.node_set["line_cd"]=np.arange(c, d+1).tolist()
    output_mesh.node_set["line_da"]=np.arange(d, d+Ny-1).tolist()+[a]    
    return output_mesh
#%%
def create_quad_mesh_rectangle_in_cirlce(radius=1, n_ring=3, Nx=3, Ny=3, seal_hole=True):
    output_mesh=create_quad_mesh_rectangle_in_rectangle(n_ring=n_ring, Nx=Nx, Ny=Ny, seal_hole=seal_hole)
    # y
    #/|\
    # |
    # D--------C
    # | \    / |
    # |  d__c  |
    # |  |__|  |
    # |  a  b  |
    # | /    \ |
    # A--------B--->x
    #-------------------------------------------------------
    output_mesh.node-=output_mesh.node.mean(dim=0, keepdim=True)
    output_mesh.node/=output_mesh.node.max()
    output_mesh.node*=radius*0.5
    lineAB=output_mesh.node_set['lineAB']
    for n in range(0, len(lineAB)):
        theta=np.pi*(5/4)+np.pi*(1/2)*(n/(len(lineAB)-1))
        x=radius*np.cos(theta)
        y=radius*np.sin(theta)
        output_mesh.node[lineAB[n],0]=x
        output_mesh.node[lineAB[n],1]=y
    lineBC=output_mesh.node_set['lineBC']
    for n in range(0, len(lineBC)):
        theta=np.pi*(7/4)+np.pi*(1/2)*(n/(len(lineBC)-1))
        x=radius*np.cos(theta)
        y=radius*np.sin(theta)
        output_mesh.node[lineBC[n],0]=x
        output_mesh.node[lineBC[n],1]=y
    lineCD=output_mesh.node_set['lineCD']
    for n in range(0, len(lineCD)):
        theta=np.pi*(1/4)+np.pi*(1/2)*(n/(len(lineCD)-1))
        x=radius*np.cos(theta)
        y=radius*np.sin(theta)
        output_mesh.node[lineCD[n],0]=x
        output_mesh.node[lineCD[n],1]=y
    lineDA=output_mesh.node_set['lineDA']
    for n in range(0, len(lineDA)):
        theta=np.pi*(3/4)+np.pi*(1/2)*(n/(len(lineDA)-1))
        x=radius*np.cos(theta)
        y=radius*np.sin(theta)
        output_mesh.node[lineDA[n],0]=x
        output_mesh.node[lineDA[n],1]=y
    #------------------------------------------------------
    mask=torch.ones(output_mesh.node.shape[0], dtype=output_mesh.node.dtype)    
    mask[lineAB]=0
    mask[lineBC]=0
    mask[lineCD]=0
    mask[lineDA]=0
    simple_smoother_for_mesh(output_mesh, 0.5, mask, n_iter=(n_ring+Nx+Ny)*10)
    return output_mesh
#%%
def create_quad_tri_mesh_circle_in_circle(n_circle=3, n_point_per_circle=11, radius=1, seal_hole=True):
    theta=2*np.pi/n_point_per_circle
    node=[]
    for n in range(0, n_circle):
        R=radius*(1-n/(n_circle))
        for m in range(0, n_point_per_circle):
            x=R*np.cos(theta*m)
            y=R*np.sin(theta*m)
            node.append([x,y,0])
    element=[]
    for n in range(1, n_circle):
        idxA=np.arange((n-1)*n_point_per_circle, n*n_point_per_circle)
        idxB=np.arange(n*n_point_per_circle, (n+1)*n_point_per_circle)
        for i in range(0, n_point_per_circle-1):
            element.append([idxA[i], idxA[i+1], idxB[i+1], idxB[i]])
        element.append([idxA[n_point_per_circle-1], idxA[0], idxB[0], idxB[n_point_per_circle-1]])
    element_counter_no_hole=len(element)
    if seal_hole == True:
        node.append([0,0,0])
        center_idx=len(node)-1
        curve=np.arange((n_circle-1)*n_point_per_circle, n_circle*n_point_per_circle)
        for k in range(0, n_point_per_circle-1):
            element.append([center_idx, curve[k], curve[k+1]])
        element.append([center_idx, curve[len(curve)-1], curve[0]])
    output_mesh=PolygonMesh(node, element)
    output_mesh.element_set['hole']=np.arange(element_counter_no_hole, len(output_mesh.element)).tolist()
    output_mesh.node_set['boundary']=np.arange(0, n_point_per_circle).tolist()
    return output_mesh
#%%
