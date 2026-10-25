#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 21:57:45 2026

@author: liang
"""
import torch
from torch.linalg import vector_norm as norm
import numpy as np
#%%
def ComputeAngleBetweenTwoVectorIn3D_slow(VectorA, VectorB, eps=None):
    #Unsigned angle in [0, pi]; eps clamps lengths, not the acos endpoints.
    #VectorA.shape (B,3)
    #VectorB.shape (B,3)
    if len(VectorA.shape) == 1:
        VectorA=VectorA.reshape(1,3)
    if len(VectorB.shape) == 1:
        VectorB=VectorB.reshape(1,3)
    
    if VectorA.dtype != VectorB.dtype:
        raise ValueError
    
    if VectorA.dtype == np.float32 or VectorA.dtype == torch.float32:
        acos_eps=1e-7
    elif VectorA.dtype == np.float64 or VectorA.dtype == torch.float64:
        acos_eps=1e-16
    else:
        raise NotImplementedError('float32 or float64 vectors are required')
    if eps is None:
        eps=1e-12
    
    if isinstance(VectorA, np.ndarray) and isinstance(VectorB, np.ndarray):
        L2Norm_A = np.sqrt(VectorA[:,0]*VectorA[:,0]+VectorA[:,1]*VectorA[:,1]+VectorA[:,2]*VectorA[:,2])
        L2Norm_B = np.sqrt(VectorB[:,0]*VectorB[:,0]+VectorB[:,1]*VectorB[:,1]+VectorB[:,2]*VectorB[:,2])
        if np.any(L2Norm_A <= eps) or np.any(L2Norm_B <= eps):
            print("L2Norm <= eps, np.clip to eps @ ComputeAngleBetweenTwoVectorIn3D(...)")
        L2Norm_A=np.clip(L2Norm_A, min=eps)
        L2Norm_B=np.clip(L2Norm_B, min=eps)
        CosTheta = (VectorA[:,0]*VectorB[:,0]+VectorA[:,1]*VectorB[:,1]+VectorA[:,2]*VectorB[:,2])/(L2Norm_A*L2Norm_B);
        CosTheta = np.clip(CosTheta, min=-1, max=1)
        Theta = np.arccos(CosTheta) #[0, pi], acos(-1) = pi
    elif isinstance(VectorA, torch.Tensor) and isinstance(VectorB, torch.Tensor):
        L2Norm_A = norm(VectorA, ord=2, dim=-1)
        L2Norm_B = norm(VectorB, ord=2, dim=-1)
        if torch.any(L2Norm_A <= eps) or torch.any(L2Norm_B <= eps):
            print("L2Norm <= eps, torch.clamp to eps @ ComputeAngleBetweenTwoVectorIn3D(...)")
        L2Norm_A=L2Norm_A.clamp_min(eps)
        L2Norm_B=L2Norm_B.clamp_min(eps)
        CosTheta = (VectorA[:,0]*VectorB[:,0]+VectorA[:,1]*VectorB[:,1]+VectorA[:,2]*VectorB[:,2])/(L2Norm_A*L2Norm_B);              
        CosTheta = torch.clamp(CosTheta, min=-1+acos_eps, max=1-acos_eps)
        Theta = torch.acos(CosTheta) #[0, pi], acos(-1) = pi
    return Theta
    '''
    #https://github.com/pytorch/pytorch/issues/8069  
    eps=1e-16
    x=torch.tensor(-1.0, requires_grad=True, dtype=torch.float64)
    x1=torch.clamp(x, min=-1+eps, max=1-eps)
    Theta = torch.acos(x1)
    Theta.backward()
    print(x.grad)
    '''
#%%
def cal_angle_between_3d_vector(VectorA, VectorB, return_cos=False, eps=None):
    """Return the unsigned angle in [0, pi], or its cosine.

    eps clamps vector lengths and defaults to 1e-12 for float32/float64.
    The separate dtype-specific torch.acos endpoint guard is unchanged.
    """
    #VectorA.shape (B,3)
    #VectorB.shape (B,3)
    if len(VectorA.shape) == 1:
        VectorA=VectorA.reshape(1,3)
    if len(VectorB.shape) == 1:
        VectorB=VectorB.reshape(1,3)
    
    if VectorA.dtype != VectorB.dtype:
        raise ValueError
    
    if VectorA.dtype == np.float32 or VectorA.dtype == torch.float32:
        acos_eps=1e-7 # Keep torch.acos gradients finite at parallel vectors.
    elif VectorA.dtype == np.float64 or VectorA.dtype == torch.float64:
        acos_eps=1e-16
    else:
        raise NotImplementedError('float32 or float64 vectors are required')
    if eps is None:
        eps=1e-12
    
    if isinstance(VectorA, np.ndarray) and isinstance(VectorB, np.ndarray):
        L2Norm_A = np.linalg.norm(VectorA, ord=2, axis=-1)
        L2Norm_B = np.linalg.norm(VectorB, ord=2, axis=-1)
        if np.any(L2Norm_A <= eps) or np.any(L2Norm_B <= eps):
            print("L2Norm <= eps, np.clip to eps @ cal_angle_between_3d_vector(...)")
        L2Norm_A=np.clip(L2Norm_A, a_min=eps, a_max=np.inf)
        L2Norm_B=np.clip(L2Norm_B, a_min=eps, a_max=np.inf)
        CosTheta = np.sum(VectorA*VectorB, axis=-1)/(L2Norm_A*L2Norm_B);
        CosTheta = np.clip(CosTheta, a_min=-1, a_max=1)
        if return_cos == False:
            Theta = np.arccos(CosTheta) #[0, pi], acos(-1) = pi
            return Theta    
        else:
            return CosTheta
    elif isinstance(VectorA, torch.Tensor) and isinstance(VectorB, torch.Tensor):
        L2Norm_A=norm(VectorA, ord=2, dim=-1)
        L2Norm_B=norm(VectorB, ord=2, dim=-1)        
        if torch.any(L2Norm_A <= eps) or torch.any(L2Norm_B <= eps):
            print("L2Norm <= eps, torch.clamp to eps @ cal_angle_between_3d_vector(...)")
        L2Norm_A=L2Norm_A.clamp_min(eps)
        L2Norm_B=L2Norm_B.clamp_min(eps)
        CosTheta = (VectorA*VectorB).sum(dim=-1)/(L2Norm_A*L2Norm_B);                      
        if return_cos == False:
            CosTheta = torch.clamp(CosTheta, min=-1+acos_eps, max=1-acos_eps)
            Theta = torch.acos(CosTheta) #[0, pi], acos(-1) = pi
            return Theta    
        else:
            return CosTheta
    else:
        raise ValueError('invalid input')
#%%
ComputeAngleBetweenTwoVectorIn3D=cal_angle_between_3d_vector
