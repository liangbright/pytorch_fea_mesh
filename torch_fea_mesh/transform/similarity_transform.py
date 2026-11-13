# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 19:48:28 2024

@author: liang
"""
import torch
import torch.nn as nn
from torch.linalg import matmul
from torch_fea_mesh.transform.rotation_transform import RotationTransform3D
#%% y=s*R*(x-t)
class SimilarityTransform3D(nn.Module):
    def __init__(self, dtype=None):
        super().__init__()
        self.angle=nn.Parameter(torch.zeros(3, dtype=dtype))
        self.t=nn.Parameter(torch.zeros((1,3), dtype=dtype))
        self.s=nn.Parameter(torch.ones(1, dtype=dtype))

    @staticmethod
    def cal_R(angle):
        #the same angle gives the same R in all three classes: R=Rz*Ry*Rx
        return RotationTransform3D.cal_R(angle)
    
    def forward(self, x):
        #x.shape (N,3)
        R=SimilarityTransform3D.cal_R(self.angle)
        R=R.permute(1,0)#transpose
        y=self.s*matmul(x-self.t, R)
        return y
