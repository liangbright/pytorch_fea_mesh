# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 19:48:07 2024

@author: liang
"""

import torch
    
def save_point_as_vtk(point, filename):
    #point: (N, 3), 3D positions

    out=[]
    out.append('# vtk DataFile Version 4.2'+'\n')
    out.append('vtk output'+'\n')
    out.append('ASCII'+'\n')
    out.append('DATASET POLYDATA'+'\n')
    #------------------------------------------------------------------
    out.append('POINTS '+str(len(point))+' double'+'\n')
    if isinstance(point, torch.Tensor):
        point=point.detach().to('cpu')
    for n in range(0, len(point)):
        x=float(point[n][0])
        y=float(point[n][1])
        z=float(point[n][2])
        out.append(str(x)+' '+str(y)+' '+str(z)+'\n')
    #------------------------------------------------------------------
    with open(filename, 'w', encoding = 'utf-8') as file:
        file.writelines(out)
