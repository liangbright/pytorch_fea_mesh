# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:59:46 2024

@author: liang
"""
import torch
import torch.nn as nn
from torch import cos, sin
from torch.linalg import matmul
#%% y=R*x
class RotationTransform3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.angle=nn.Parameter(torch.zeros(3))
    
    @staticmethod
    def cal_R(angle):
        #angle=[angle_x, angle_y, angle_z]
        #R=Rz*Ry*Rx
        a=angle[0]
        b=angle[1]
        c=angle[2]
        sin_a=sin(a); cos_a=cos(a)
        sin_b=sin(b); cos_b=cos(b)
        sin_c=sin(c); cos_c=cos(c)
        R=torch.zeros((3,3), dtype=a.dtype, device=a.device)
        R[0,0]=cos_b*cos_c; R[0,1]=sin_a*sin_b*cos_c-cos_a*sin_c; R[0,2]=cos_a*sin_b*cos_c+sin_a*sin_c
        R[1,0]=cos_b*sin_c; R[1,1]=sin_a*sin_b*sin_c+cos_a*cos_c; R[1,2]=cos_a*sin_b*sin_c-sin_a*cos_c
        R[2,0]=-sin_b;      R[2,1]=sin_a*cos_b;                   R[2,2]=cos_a*cos_b 
        return R
    
    @staticmethod
    def rotate(x, angle):
        #x.shape (N,3)
        R=RotationTransform3D.cal_R(angle)
        R=R.permute(1,0)#transpose
        y=matmul(x, R)
        return y
    
    def forward(self, x):
        y=RotationTransform3D.rotate(x, self.angle)
        return y
