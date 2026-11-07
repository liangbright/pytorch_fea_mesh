# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:24:02 2024

@author: liang
"""
import torch
import torch.nn as nn
from torch.linalg import matmul
from .rotation_transform import RotationTransform3D
#%% y=R*(x-t)
class RigidTransform3D(nn.Module):
    def __init__(self, dtype=None):
        super().__init__()
        self.angle=nn.Parameter(torch.zeros(3, dtype=dtype))
        self.t=nn.Parameter(torch.zeros((1,3), dtype=dtype))
        
    @staticmethod
    def cal_R(angle):
        #the same angle gives the same R in all three classes: R=Rz*Ry*Rx
        return RotationTransform3D.cal_R(angle)
    
    def forward(self, x):
        #x.shape (N,3)
        R=RigidTransform3D.cal_R(self.angle)
        R=R.permute(1,0)#transpose
        y=matmul(x-self.t, R)
        return y
