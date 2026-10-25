# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:52:22 2023

@author: liang
"""
import numpy as np
import torch

def save_curve_as_vtk(curve, filename):
    #curve may be
    # a single curve: a 2D numpy/torch array or a list of points
    # a list of curves = 3D array    
    if len(curve) == 0:
        return
    if torch.is_tensor(curve) or isinstance(curve, np.ndarray):
        if len(curve.shape) == 2:
            curve_list=[curve]
        elif len(curve.shape) == 3:
            curve_list=curve
        else:
            raise ValueError
    elif isinstance(curve, list) or isinstance(curve, tuple):
        try:
            temp=float(curve[0][0])
            curve_list=[curve]
        except:            
            curve_list=curve
    node, line=convert_curve_to_polyline(curve_list)
    save_polyline_as_vtk(filename, node, line)

def convert_curve_to_polyline(curve):
    #curve may be
    # a single curve: a 2D numpy/torch array or a list of points
    # a list of curves = 3D array    
    if len(curve) == 0:
        return
    if torch.is_tensor(curve) or isinstance(curve, np.ndarray):
        if len(curve.shape) == 2:
            curve_list=[curve]
        elif len(curve.shape) == 3:
            curve_list=curve
        else:
            raise ValueError
    elif isinstance(curve, list) or isinstance(curve, tuple):
        try:
            temp=float(curve[0][0])
            curve_list=[curve]
        except:            
            curve_list=curve
    node=[]
    line=[]
    for k in range(0, len(curve_list)):
        curve=curve_list[k]        
        if isinstance(curve, torch.Tensor) or isinstance(curve, np.ndarray):
            curve=curve.tolist()
        if len(curve) > 0:
            idx_start=len(node)
            node.extend(curve)
            #poly_line.append(np.arange(idx_start, idx_start+len(curve)).tolist())
            idx_list=np.arange(idx_start, idx_start+len(curve)).tolist()
            for n in range(1, len(idx_list)):
                line.append([idx_list[n-1], idx_list[n]]) #each element is VTK_LINE
    node=torch.tensor(node, dtype=torch.float64)
    return node, line

def save_polyline_as_vtk(filename, node, line=None, node_data=None):
    #node: (N, 3), 3D positions
    #line[k] is the k-th polyline, a list of node indexes
    #node_data: {"stress":stress, "strain":strain}
    if line is None:
        #single line from node[0] to node[-1]
        line=[np.arange(0, len(node)).tolist()]
    try:
        temp=line[0][0]
    except:
        raise ValueError("line needs to be a nested list")
    
    out=[]
    out.append('# vtk DataFile Version 4.2'+'\n')
    out.append('vtk output'+'\n')
    out.append('ASCII'+'\n')
    out.append('DATASET POLYDATA'+'\n')
    #------------------------------------------------------------------
    out.append('POINTS '+str(node.shape[0])+' double'+'\n')
    if isinstance(node, torch.Tensor):
        node=node.detach().to('cpu')
    for n in range(0, node.shape[0]):
        x=float(node[n,0])
        y=float(node[n,1])
        z=float(node[n,2])
        out.append(str(x)+' '+str(y)+' '+str(z)+'\n')
    #------------------------------------------------------------------
    if isinstance(line, torch.Tensor):
        line=line.detach().to('cpu')
    offset_count=0
    for m in range(0, len(line)):
        offset_count+=1+len(line[m])
    out.append('LINES '+str(len(line))+' '+str(offset_count)+'\n')
    for m in range(0, len(line)):
        text=str(len(line[m]))
        for k in range(0, len(line[m])):
            id=int(line[m][k])
            text=text+' '+str(id)
        out.append(text+'\n')
    #------------------------------------------------------------------
    if node_data is not None:
        out.append('POINT_DATA '+str(node.shape[0])+'\n')
        out.append('FIELD FieldData '+str(len(node_data.keys()))+'\n')
    else:
        node_data={}
    for name, data in node_data.items():
        out.append(name+' '+str(data.shape[1])+' '+str(data.shape[0])+' double'+'\n')
        if isinstance(data, torch.Tensor):
            data=data.detach().to('cpu')
        for i in range(0, data.shape[0]):
            line=''
            for j in range(data.shape[1]):
                line=line+str(float(data[i,j]))+' '
            out.append(line+'\n')
    #------------------------------------------------------------------
    with open(filename, 'w', encoding = 'utf-8') as file:
        file.writelines(out)
