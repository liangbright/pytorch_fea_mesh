# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 22:29:00 2023

@author: liang
"""
import numpy as np

def read_abaqus_inp(filename, remove_unused_node):
    with open(filename, 'r') as file:
        inp=file.readlines()
        print('total number of lines in inp:', len(inp))
    #-----------
    inp=fix_inp_element(inp)
    #-----------
    node, node_id=read_node(inp)
    print('total number of nodes:', len(node))
    element, element_id, element_type, element_set=read_element(inp)
    print('total number of elements:', len(element))
    #-----------
    node=np.array(node)
    node_id=np.array(node_id)
    element=np.array(element, dtype='object')
    element_id=np.array(element_id)
    element_type=np.array(element_type, dtype='object')
    #-----------    
    elset=read_elset(inp)
    element_set=element_set | elset # merge
    #-----------
    node_set=read_nset(inp)    
    #-----------
    if remove_unused_node == True:
        node, node_id, element=clean_data_remove_unused_node(node, node_id, element)
        kept_node_id=set(node_id.tolist())
        node_set={name:[idx for idx in values if idx in kept_node_id]
                  for name, values in node_set.items()}
    out=clean_data_re_label_node_element(node, node_id, node_set, element, element_id, element_set)
    node, node_id, node_set, element, element_id, element_set=out
    #-----------
    element_orientation=read_element_orientation(inp, element_id)
    #-----------
    output={"node":node,
            "node_id":node_id,
            "node_set":node_set,
            "element":element,
            "element_id":element_id,
            "element_type":element_type,
            "element_set":element_set,
            "element_orientation":element_orientation
            }
    return output
#%%
def read_node(inp):
    node=[]
    node_id=[]
    lineindex=-1
    for k in range(0, len(inp)):
        if "*node" in inp[k].lower():
            lineindex=k
            break
    if lineindex < 0:
        print('node keyword is not found')
        return node, node_id
    k=lineindex
    while True:
        k=k+1
        if k >= len(inp):
            break
        if "**" in inp[k]:
            continue
        if "*" in inp[k]:
            break        
        temp=inp[k].replace(" ", "")
        temp=temp.split(",")
        node_id.append(int(temp[0]))
        node.append([float(temp[1]), float(temp[2]), float(temp[3])])
    return node, node_id
#%%
def fix_inp_element(inp):
    #the node indexes of an element may span multiple lines
    #this function will convert those lines of an element into one line
    for k in range(0, len(inp)):
        inp[k]=inp[k].replace(" ", "")
    end_index=0
    while True:
        start_index=None
        for k in range(end_index, len(inp)):
            if "*element" in inp[k].lower():
                start_index=k+1
                break
        if start_index is None:
            break  
        end_index=len(inp)-1
        for k in range(start_index, len(inp)):
            if "**" in inp[k]:
                continue
            if "*" in inp[k]:
                end_index=k
                break
        line_list=inp[start_index:end_index]
        flag=False
        for k in range(0, len(line_list)):
            if inp[start_index+k][-2].isdigit() == False:
                flag=True
                break
        if flag == True:
            out=[""]
            for k in range(0, len(line_list)):
                out[-1]=out[-1]+inp[start_index+k][:-1]
                if inp[start_index+k][-2].isdigit() == True:
                    if k < len(line_list)-1:
                        out.append("")
            inp=inp[:start_index]+out+inp[end_index:]
    return inp
#%%
def read_element(inp):
    element=[]
    element_id=[]
    element_type=[]
    element_set={}
    lineindexlist=[]
    for k in range(0, len(inp)):
        if "*element" in inp[k].lower():
            lineindexlist.append(k)
    if len(lineindexlist) == 0:
        print('element keyword is not found')
        return element, element_id, element_set
    for lineindex in lineindexlist:
        k=lineindex
        eltype=None
        temp=inp[k].replace(" ", "")
        temp=temp.split(",")
        for v in temp:
            if "type" in v.lower():
                eltype=v.split("=")[-1]
                eltype=eltype.replace("\n", "")
                eltype=eltype.replace(" ", "")
                eltype=eltype.upper()
                break
        elset=None
        for v in temp:
            if "elset=" in v.lower():
                elset=v.split('=')[-1]
                elset=elset.replace("\n", "")
                break
        if elset is not None:
            if elset not in element_set.keys():
                element_set[elset]=[]

        while True:
            k=k+1
            if k >= len(inp):
                break
            if "**" in inp[k]:
                continue
            if "*" in inp[k]:
                break            
            temp=inp[k].replace(" ", "")
            temp=temp.split(",")
            temp=[int(a) for a in temp]
            element_id.append(int(temp[0]))
            element.append(temp[1:])
            element_type.append(eltype)
            if elset is not None:
                element_set[elset].append(element_id[-1])
    return element, element_id, element_type, element_set
#%%
def read_elset(inp):
    # read *Elset
    element_set={}
    lineindexlist=[]
    for k in range(0, len(inp)):
        if "*elset" in inp[k].lower():
            lineindexlist.append(k)
    for lineindex in lineindexlist:
        k=lineindex
        temp=inp[k].replace(" ", "").split(",")
        name=None
        for v in temp:
            if "elset=" in v.lower():
                name=v.split('=')[-1]
                name=name.replace("\n", "")
                break
        if name not in element_set.keys():
            element_set[name]=[]
        generate=False
        for v in temp:
            if "generate" in v.lower():
                generate=True
                break
        if generate == True:
            num=inp[k+1].replace(" ", "").split(",")
            element_set[name]=np.arange(int(num[0]), int(num[1])+1).tolist()
            continue
        while True:
            k=k+1
            if k >= len(inp):
                break
            if "**" in inp[k]:
                continue
            if "*" in inp[k]:
                break            
            temp=inp[k].replace(" ", "")
            temp=temp.split(",")
            temp=[int(a) for a in temp]
            element_set[name].extend(temp)
    return element_set
#%%
def read_nset(inp):
    # read *Nset
    node_set={}
    lineindexlist=[]
    for k in range(0, len(inp)):
        if "*nset" in inp[k].lower():
            lineindexlist.append(k)
    for lineindex in lineindexlist:
        k=lineindex
        temp=inp[k].replace(" ", "")
        temp=temp.split(",")
        name=None
        for v in temp:
            if "nset=" in v.lower():
                name=v.split('=')[-1]
                name=name.replace("\n", "")
                break
        if name not in node_set.keys():
            node_set[name]=[]
        generate=False
        for v in temp:
            if "generate" in v.lower():
                generate=True
                break
        if generate == True:
            num=inp[k+1].replace(" ", "").split(",")
            node_set[name]=np.arange(int(num[0]), int(num[1])+1).tolist()
            continue
        while True:
            k=k+1
            if k >= len(inp):
                break
            if "**" in inp[k]:
                continue
            if "*" in inp[k]:
                break            
            temp=inp[k].replace(" ", "")
            temp=temp.split(",")
            temp=[int(a) for a in temp]
            node_set[name].extend(temp)
    return node_set
#%%
def clean_data_re_label_node_element(node, node_id, node_set, element, element_id, element_set):
    #node_id could start from an arbitrary number, e.g, 1000
    #sort node by id
    node_idx_sorted=np.argsort(node_id)
    node=node[node_idx_sorted]
    node_id=node_id[node_idx_sorted]
    #map node id to index in node
    map_node={}
    for n in range(0, len(node)):
        map_node[node_id[n]]=n
    for key, value in node_set.items():
        value_new=[map_node[a] for a in value]
        node_set[key]=value_new
    for m in range(0, len(element)):
        for n in range(0, len(element[m])):
            element[m][n]=map_node[element[m][n]]
    #sort element by id
    elm_idx_sorted=np.argsort(element_id)
    element=np.array(element)[elm_idx_sorted].tolist()
    element_id=np.array(element_id)[elm_idx_sorted].tolist()
    #map element id to index in element
    map_elm={}
    for n in range(0, len(element)):
        map_elm[int(element_id[n])]=n
    for key, value in element_set.items():
        value_new=[map_elm[int(a)] for a in value]
        element_set[key]=value_new    
    return node, node_id, node_set, element, element_id, element_set
#%%
def clean_data_remove_unused_node(node, node_id, element):
    #Connectivity still contains original INP node IDs, not row indices.
    used_node_id=[]
    for m in range(0, len(element)):
        used_node_id.extend(element[m])
    keep=np.isin(node_id, used_node_id)
    n_unused_node=int((~keep).sum())
    print('total number of unused nodes:', n_unused_node)
    if n_unused_node == 0:
        return node, node_id, element
    node=node[keep]
    node_id=node_id[keep]
    return node, node_id, element
#%%
def read_element_orientation(inp, element_id):    
    line_index=None
    for k in range(0, len(inp)):
        temp=inp[k].lower()
        if ("*distribution" in temp) and ("ori" in temp) and ("element" in temp):
            #*Distribution, name=Ori-1-DiscOrient, location=ELEMENT, Table=Ori-1-DiscOrient_Table
            line_index=k
            break
    if line_index is None:
        print('element orientation is not found')
        return None
    #--------------------------    
    element_id=np.array(element_id, dtype=np.int64)
    element_orientation=np.zeros((len(element_id),3,3))    
    k=line_index
    while True:
        k=k+1
        if k > len(inp)-1:
            break        
        if "**" in inp[k]:
            #** Description: Distribution generated from Discrete Orientation
            continue
        if "*" in inp[k]:
            break
        line=inp[k].split(",")
        if line[0]=='':
            #,  1., 0., 0., 0., 1., 0.
            continue
        elm_id=int(line[0])
        index=np.where(element_id==elm_id)[0]
        if len(index) == 0:
            print("elm_id", elm_id, "is missing orientation")
        else:
            index=index.item()
            d0=np.array([float(line[1]), float(line[2]), float(line[3])])            
            d1=np.array([float(line[4]), float(line[5]), float(line[6])])            
            d2=np.cross(d0, d1)            
            #update d1
            d1=np.cross(d2, d0)
            d0=d0/np.linalg.norm(d0, ord=2)
            d1=d1/np.linalg.norm(d1, ord=2)
            d2=d2/np.linalg.norm(d2, ord=2)            
            element_orientation[index,:,0]=d0
            element_orientation[index,:,1]=d1
            element_orientation[index,:,2]=d2
    return element_orientation
