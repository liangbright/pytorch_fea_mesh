# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 19:48:28 2024

@author: liang
"""
import torch
import torch.nn as nn
from torch import cos, sin
from torch.linalg import matmul
#%% y=s*R*(x-t)
class SimilarityTransform3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.angle=nn.Parameter(torch.zeros(3))
        self.t=nn.Parameter(torch.zeros((1,3)))
        self.s=nn.Parameter(torch.ones(1))

    @staticmethod
    def cal_R(angle):
        a=angle[0]
        b=angle[1]
        c=angle[2]
        sin_a=sin(a); cos_a=cos(a)
        sin_b=sin(b); cos_b=cos(b)
        sin_c=sin(c); cos_c=cos(c)
        R=torch.zeros((3,3), dtype=a.dtype, device=a.device)
        R[0,0]=cos_c*cos_b;  R[0,1]=cos_c*sin_b*sin_a+sin_c*cos_a;  R[0,2]=-cos_c*sin_b*cos_a+sin_c*sin_a
        R[1,0]=-sin_c*cos_b; R[1,1]=-sin_c*sin_b*sin_a+cos_c*cos_a; R[1,2]=sin_c*sin_b*cos_a+cos_c*sin_a
        R[2,0]=sin_b;        R[2,1]=-cos_b*sin_a;                   R[2,2]=cos_b*cos_a 
        return R
    
    def forward(self, x):
        #x.shape (N,3)
        R=SimilarityTransform3D.cal_R(self.angle)
        R=R.permute(1,0)#transpose
        y=self.s*matmul(x-self.t, R)
        return y
