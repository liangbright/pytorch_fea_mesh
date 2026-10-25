import torch
from torch_geometric.nn.pool import knn
#knn(x,y) Finds for each element in y the k nearest points in x.

def scale(P1, P2):
    #single
    #P1.shape:  Nx3 or Nx2, etc, N is the number of points in P1
    #P2.shape:  Mx3 or Mx2, etc, M is the number of points in P2
    #batch
    #P1 (B,N,2) or (B,N,3)
    #P2 (B,M,2) or (B,M,3)
    if len(P1.shape) == 2:
        center1=P1.mean(dim=0, keepdim=True)
        center2=P2.mean(dim=0, keepdim=True)
    elif len(P1.shape) == 3:
        center1=P1.mean(dim=(0,1), keepdim=True)
        center2=P2.mean(dim=(0,1), keepdim=True)
    center=0.5*(center1+center2)
    P1=P1-center
    P2=P2-center
    s=max(P1.abs().max(), P2.abs().max())
    P1=P1/s
    P2=P2/s
    return P1, P2

def cal_distance(P1, P2, reduction, squared_distance, scale_input=True):
    #P1.shape:  Nx3 or Nx2, etc, N is the number of points in P1
    #P2.shape:  Mx3 or Mx2, etc, M is the number of points in P2
    #set squared_distance=True for loss function
    #set scale_input=True to prevent issues in knn
    P1s, P2s=P1, P2
    if scale_input==True:
        P1s, P2s=scale(P1s, P2s)
    index2=knn(P2s, P1s, 1)
    index2=index2[1]
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
    P1s, P2s=P1, P2
    if scale_input==True:
        P1s, P2s=scale(P1s, P2s)
    index1=knn(P1s, P2s, 1)
    index1=index1[1]
    index2=knn(P2s, P1s, 1)
    index2=index2[1]
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
    if P1.shape[0] != P2.shape[0]:
        raise ValueError("P1.shape[0] != P2.shape[0]")
    B=P1.shape[0]
    N=P1.shape[1]
    P1=P1.view(B*N,-1)
    M=P2.shape[1]
    P2=P2.view(B*M,-1)
    P1s, P2s=P1, P2
    if scale_input==True:
        P1s, P2s=scale(P1s, P2s)
    batch_P1=[]
    batch_P2=[]
    for k in range(0, B):
        batch_P1.extend([k]*N)
        batch_P2.extend([k]*M)
    batch_P1=torch.tensor(batch_P1, dtype=torch.int64, device=P1.device)
    batch_P2=torch.tensor(batch_P2, dtype=torch.int64, device=P1.device)
    index2=knn(P2s, P1s, 1, batch_P2,  batch_P1)
    index2=index2[1]
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
    if P1.shape[0] != P2.shape[0]:
        raise ValueError("P1.shape[0] != P2.shape[0]")
    B=P1.shape[0]
    N=P1.shape[1]
    P1=P1.view(B*N,-1)
    M=P2.shape[1]
    P2=P2.view(B*M,-1)
    P1s, P2s=P1, P2
    if scale_input==True:
        P1s, P2s=scale(P1s, P2s)
    batch_P1=[]
    batch_P2=[]
    for k in range(0, B):
        batch_P1.extend([k]*N)
        batch_P2.extend([k]*M)
    batch_P1=torch.tensor(batch_P1, dtype=torch.int64, device=P1.device)
    batch_P2=torch.tensor(batch_P2, dtype=torch.int64, device=P1.device)
    index1=knn(P1s, P2s, 1, batch_P1,  batch_P2)
    index1=index1[1]
    index2=knn(P2s, P1s, 1, batch_P2,  batch_P1)
    index2=index2[1]
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
