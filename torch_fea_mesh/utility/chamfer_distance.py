import torch
from torch_geometric.nn.pool import knn
#knn(x,y) Finds for each element in y the k nearest points in x.

def scale(P1, P2):
    #single
    #P1.shape:  Nx3 or Nx2, etc, N is the number of points in P1
    #P2.shape:  Mx3 or Mx2, etc, M is the number of points in P2
    #batch: each example gets its own center and scale, so that examples can not affect each other
    #P1 (B,N,2) or (B,N,3)
    #P2 (B,M,2) or (B,M,3)
    if len(P1.shape) == 2:
        center1=P1.mean(dim=0, keepdim=True)
        center2=P2.mean(dim=0, keepdim=True)
        dim=(0,1)
    elif len(P1.shape) == 3:
        center1=P1.mean(dim=1, keepdim=True)
        center2=P2.mean(dim=1, keepdim=True)
        dim=(1,2)
    else:
        raise ValueError("P1 and P2 must have 2 or 3 dimensions")
    center=0.5*(center1+center2)
    P1=P1-center
    P2=P2-center
    s=torch.maximum(P1.abs().amax(dim=dim, keepdim=True), P2.abs().amax(dim=dim, keepdim=True))
    #if all points are at the center, then s is 0: nothing needs scaling, and P/0 would be NaN
    s=torch.where(s==0, torch.ones_like(s), s)
    P1=P1/s
    P2=P2/s
    return P1, P2

def check_input(P1, P2, ndim):
    #ndim is 2 for (N,D) and 3 for (B,N,D)
    if len(P1.shape) != ndim or len(P2.shape) != ndim:
        raise ValueError("P1 and P2 must have "+str(ndim)+" dimensions")
    if P1.shape[-1] != P2.shape[-1]:
        raise ValueError("P1.shape[-1] != P2.shape[-1]")
    if P1.device != P2.device:
        raise ValueError("P1 and P2 must be on the same device")
    #knn crashes the python process (segmentation fault) if a point cloud is empty
    if P1.numel() == 0 or P2.numel() == 0:
        raise ValueError("P1 and P2 must not be empty")
    #knn skips non-finite points without any error
    if not (P1.isfinite().all() and P2.isfinite().all()):
        raise ValueError("P1 and P2 must not contain NaN or inf")

def find_nearest_point(Px, Py, batch_x=None, batch_y=None):
    #for each point in Py, find the index of the nearest point in Px
    #contiguous: on the CPU, knn reads the memory in order and ignores the strides of its input
    Px=Px.contiguous()
    Py=Py.contiguous()
    if Px.is_cuda:
        #knn runs on the current CUDA device, which may not be the device of its input
        with torch.cuda.device(Px.device):
            index=knn(Px, Py, 1, batch_x, batch_y)
    else:
        index=knn(Px, Py, 1, batch_x, batch_y)
    #knn finds no neighbour if the squared distance is inf
    if index.shape[1] != Py.shape[0]:
        raise RuntimeError("knn did not find a neighbour for every point, try scale_input=True")
    return index[1]

def cal_distance(P1, P2, reduction, squared_distance, scale_input=True):
    #P1.shape:  Nx3 or Nx2, etc, N is the number of points in P1
    #P2.shape:  Mx3 or Mx2, etc, M is the number of points in P2
    #set squared_distance=True for loss function
    #set scale_input=True to prevent issues in knn
    check_input(P1, P2, 2)
    P1s, P2s=P1, P2
    if scale_input==True:
        P1s, P2s=scale(P1s, P2s)
    index2=find_nearest_point(P2s, P1s)
    dist=((P2[index2]-P1)**2).sum(dim=-1)
    if squared_distance == False:
        dist=dist.sqrt()
    if reduction == 'mean':
        return dist.mean()
    elif reduction == 'sum':
        return dist.sum()
    elif reduction == 'none':
        return dist
    else:
        raise ValueError('unknown reduction:'+reduction)    

def cal_chamfer_distance(P1, P2, reduction, squared_distance, scale_input=True):
    #P1.shape:  Nx3 or Nx2, etc, N is the number of points in P1
    #P2.shape:  Mx3 or Mx2, etc, M is the number of points in P2
    #set squared_distance=True for loss function
    #set scale_input=True to prevent issues in knn
    check_input(P1, P2, 2)
    P1s, P2s=P1, P2
    if scale_input==True:
        P1s, P2s=scale(P1s, P2s)
    index1=find_nearest_point(P1s, P2s)
    index2=find_nearest_point(P2s, P1s)
    dist1=((P2[index2]-P1)**2).sum(dim=-1)
    dist2=((P1[index1]-P2)**2).sum(dim=-1)
    if squared_distance == False:
        dist1=dist1.sqrt()
        dist2=dist2.sqrt()
    if reduction == 'mean':        
        return 0.5*(dist1.mean()+dist2.mean())
    elif reduction == 'sum':
        return dist1.sum()+dist2.sum()
    elif reduction == 'none':
        return dist1, dist2
    else:
        raise ValueError('unknown reduction:'+reduction)

def cal_distance_batch(P1, P2, reduction, squared_distance, scale_input=True):
    #P1 (B,N,2) or (B,N,3)
    #P2 (B,M,2) or (B,M,3)
    #set squared_distance=True for loss function
    #set scale_input=True to prevent issues in knn
    check_input(P1, P2, 3)
    if P1.shape[0] != P2.shape[0]:
        raise ValueError("P1.shape[0] != P2.shape[0]")
    B=P1.shape[0]
    N=P1.shape[1]
    M=P2.shape[1]
    P1s, P2s=P1, P2
    if scale_input==True:
        P1s, P2s=scale(P1s, P2s)
    P1=P1.reshape(B*N,-1)
    P2=P2.reshape(B*M,-1)
    P1s=P1s.reshape(B*N,-1)
    P2s=P2s.reshape(B*M,-1)
    batch_P1=[]
    batch_P2=[]
    for k in range(0, B):
        batch_P1.extend([k]*N)
        batch_P2.extend([k]*M)
    batch_P1=torch.tensor(batch_P1, dtype=torch.int64, device=P1.device)
    batch_P2=torch.tensor(batch_P2, dtype=torch.int64, device=P1.device)
    index2=find_nearest_point(P2s, P1s, batch_P2, batch_P1)
    dist=((P2[index2]-P1)**2).sum(dim=-1)
    if squared_distance == False:
        dist=dist.sqrt()
    if reduction == 'mean':        
        return dist.mean()
    elif reduction == 'sum':
        return dist.sum()
    elif reduction == 'none':
        return dist.reshape(B,-1)
    else:
        raise ValueError('unknown reduction:'+reduction)

def cal_chamfer_distance_batch(P1, P2, reduction, squared_distance, scale_input=True):
    #P1 (B,N,2) or (B,N,3)
    #P2 (B,M,2) or (B,M,3)
    #set squared_distance=True for loss function
    #set scale_input=True to prevent issues in knn
    check_input(P1, P2, 3)
    if P1.shape[0] != P2.shape[0]:
        raise ValueError("P1.shape[0] != P2.shape[0]")
    B=P1.shape[0]
    N=P1.shape[1]
    M=P2.shape[1]
    P1s, P2s=P1, P2
    if scale_input==True:
        P1s, P2s=scale(P1s, P2s)
    P1=P1.reshape(B*N,-1)
    P2=P2.reshape(B*M,-1)
    P1s=P1s.reshape(B*N,-1)
    P2s=P2s.reshape(B*M,-1)
    batch_P1=[]
    batch_P2=[]
    for k in range(0, B):
        batch_P1.extend([k]*N)
        batch_P2.extend([k]*M)
    batch_P1=torch.tensor(batch_P1, dtype=torch.int64, device=P1.device)
    batch_P2=torch.tensor(batch_P2, dtype=torch.int64, device=P1.device)
    index1=find_nearest_point(P1s, P2s, batch_P1, batch_P2)
    index2=find_nearest_point(P2s, P1s, batch_P2, batch_P1)
    dist1=((P2[index2]-P1)**2).sum(dim=-1)
    dist2=((P1[index1]-P2)**2).sum(dim=-1)
    if squared_distance == False:
        dist1=dist1.sqrt()
        dist2=dist2.sqrt()
    if reduction == 'mean':
        return 0.5*(dist1.mean()+dist2.mean())
    elif reduction == 'sum':
        return dist1.sum()+dist2.sum()
    elif reduction == 'none':
        return dist1.reshape(B,-1), dist2.reshape(B,-1)
    else:
        raise ValueError('unknown reduction:'+reduction)
#%%
